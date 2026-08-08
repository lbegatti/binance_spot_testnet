import json
import logging
import math
import time
from datetime import datetime, timezone
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
from binance.lib.utils import websocket_api_signature, get_uuid
from core.order_book_state import OrderBookState
from config_parameters import (
    SYMBOL,
    CRYPTOCCY,
    CCY,
    RECV_WINDOW,
    ORDER_REPORT_LIMIT,
    BACKTEST_FEE_RATE,
    MAX_POSITION_PCT,
    MIN_CASH_RESERVE_PCT,
)


def _retry_transient(fn, *, attempts: int = 3, base_delay: float = 0.5):
    """
    Call ``fn()`` and retry on transient failures (Binance 5xx / nginx 502-504,
    network timeouts) with exponential backoff.  Re-raises the last exception
    if all attempts fail.

    Used for shutdown-critical REST calls where a single server-side blip must
    not leave session orders stranded on the book.  Any exception triggers a
    retry — safe here because the wrapped calls are shutdown-time and
    effectively idempotent (``get_open_orders`` is read-only; re-cancelling an
    already-cancelled order errors harmlessly).
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — transient guard, re-raised below
            last_exc = e
            if i < attempts - 1:
                time.sleep(base_delay * (2**i))  # 0.5 s, 1.0 s, …
    raise last_exc  # type: ignore[misc]


class OrderExecutor:
    """
    Responsible for placing LIMIT GTC orders and maintaining real-time
    balance updates via the Binance WebSocket API.

    **Order execution** — prefers the Binance WebSocket API
    (``wss://ws-api.testnet.binance.vision/ws-api/v3``) for lower latency and
    falls back transparently to the REST API when the WebSocket endpoint is
    unavailable.  No logic change is required in the caller.

    **Exchange-filter normalisation** — at construction time the executor
    calls ``exchange_info()`` once to cache the symbol's ``LOT_SIZE``
    (``stepSize``, ``minQty``) and ``(MIN_)NOTIONAL`` (``minNotional``)
    filters.  Every order's quantity is then floored down to the
    ``stepSize`` grid and rejected before dispatch if it is below
    ``minQty`` or if the notional is below ``minNotional`` —
    preventing the silent ``-1013 "Filter failure: LOT_SIZE"``
    rejection that otherwise drops every order at the Binance gateway.

    **Balance tracking** — on connection open the executor authenticates the
    WebSocket session with ``session.logon`` (HMAC-signed timestamp) and then
    calls ``userDataStream.subscribe``.  Once confirmed, Binance pushes
    ``outboundAccountPosition`` events on the **same** connection whenever a
    balance changes (e.g. after an order fill), keeping
    ``state.balance_status`` current without any listenKey.  If the testnet
    does not support these methods the executor falls back silently to the
    REST snapshot taken at session startup.

    **Routing** — ``handle_order_response`` is the single ``on_message``
    callback for the WS API connection.  It routes incoming frames by the
    presence/absence of the ``"id"`` field:

    * No ``"id"`` → push event (user data stream); forwarded to
      ``_handle_balance_update`` when ``"e" == "outboundAccountPosition"``.
    * ``"id"`` matches ``_logon_id`` → ``session.logon`` response.
    * ``"id"`` matches ``_subscribe_id`` → ``userDataStream.subscribe`` response.
    * Any other ``"id"`` → order placement response (existing logic).

    Attributes:
        ws_api_client (SpotWebsocketAPIClient | None): Authenticated WebSocket
            API client.  ``None`` when the WS endpoint was unreachable at
            startup.
        rest_client: Binance ``Spot`` REST client used as a fallback when
            ``ws_api_client`` is ``None``.
        state (OrderBookState): Shared order book and balance state.
        last_order (dict | None): Most recent order response, or ``None``.
        placed_orders (list[dict]): Accumulates every order sent this session.
    """

    def __init__(
        self,
        state: OrderBookState,
        stream_url: str,
        api_key: str,
        api_secret: str,
        rest_client=None,
    ):
        """
        Create the executor and attempt to open a WebSocket API connection.

        If the WebSocket handshake fails the executor degrades silently to
        REST-only mode and balance updates fall back to the REST snapshot
        taken at session startup.  A warning is logged so the operator is
        aware.

        Args:
            state (OrderBookState): Shared order book and balance state.
            stream_url (str): Binance WebSocket API endpoint
                (e.g. ``"wss://ws-api.testnet.binance.vision/ws-api/v3"``).
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
            rest_client: Optional ``binance.spot.Spot`` instance used as a
                fallback when the WebSocket API is unavailable.
        """
        self.state = state
        self.last_order = None
        self.rest_client = rest_client
        self.ws_api_client = None
        self.placed_orders: list[dict] = []
        # Pending GTC BUY tracking — for the 10-second stale-order cancel.
        # _pending_buy_placed_at: wall-clock time the BUY was dispatched.
        # _pending_buy_id: Binance orderId (REST: set synchronously; WS: async via handle_order_response).
        self._pending_buy_placed_at: float | None = None
        self._pending_buy_id: int | None = None
        # Pending non-urgent LIMIT GTC SELL tracking — mirrors the BUY pair.
        # A planned exit rests on the book (maker) and may not fill at once;
        # cancel_stale_sell() resolves it after the timeout.
        self._pending_sell_placed_at: float | None = None
        self._pending_sell_id: int | None = None

        # Size + price of the most recently dispatched BUY leg, exposed so the
        # strategy layer (strategy/analysis.py) can accrue the pyramiding cost
        # basis at dispatch.  Reset to 0.0 at the start of every
        # BUY branch so a skipped BUY never leaves a stale non-zero value.
        self.last_buy_qty: float = 0.0
        self.last_buy_price: float = 0.0

        # Dispatch timestamp of the most recent in-flight WS order.
        # Set in execute() at WS dispatch; consumed by handle_order_response
        # when the matching order placement response arrives and the entry is
        # appended to placed_orders with a "placed_at" datetime.  The strategy
        # enforces single-position, so at most one order is in flight at any
        # time — a single variable is sufficient (no FIFO queue needed).
        self._last_dispatch_at: datetime | None = None

        # Tracks the in-flight session.logon and userDataStream.subscribe requests
        # so handle_order_response can route their responses correctly.
        self._logon_id: str | None = None
        self._subscribe_id: str | None = None
        self._user_data_active: bool = False  # True once subscribe is confirmed

        try:
            self.ws_api_client = SpotWebsocketAPIClient(
                stream_url=stream_url,
                api_key=api_key,
                api_secret=api_secret,
                on_message=self.handle_order_response,
                on_open=self._on_ws_open,
            )
            logging.info("WebSocket API client connected: %s", stream_url)
            # _on_ws_open fires from the socket thread DURING the constructor
            # above, while self.ws_api_client is still None — so its
            # _send_session_logon() call no-ops silently and the user-data
            # stream never activates (observed all session on 2026-07-08:
            # "sending session.logon" logged, "session.logon sent" never).
            # Retry here, now that the attribute is assigned; _logon_id is
            # still None exactly when the first attempt was swallowed.
            if self._logon_id is None:
                self._send_session_logon()
        except Exception as e:
            logging.warning(
                "WebSocket API unavailable (%s: %s). "
                "Order execution will fall back to REST. "
                "Balance updates will use REST snapshot only.",
                type(e).__name__,
                e,
            )

        # Exchange filter cache — populated from Binance exchange_info() so
        # execute() can floor the requested quantity to the LOT_SIZE stepSize
        # and reject orders below minNotional, preventing the Binance
        # -1013 "Filter failure: LOT_SIZE" rejection that previously caused
        # every order to be silently dropped at the gateway.
        #
        # Defaults are BTCUSDT-correct so the executor stays operational
        # if exchange_info() fails or rest_client is None.
        self._step_size: float = 1e-5
        self._min_qty: float = 1e-5
        self._min_notional: float = 10.0
        self._qty_decimals: int = 5
        self._load_symbol_filters()

    def _load_symbol_filters(self) -> None:
        """
        Fetch the LOT_SIZE and (MIN_)NOTIONAL filters for SYMBOL once at
        session start and cache them on the executor.

        Used by :meth:`execute` to:

        1. Floor the requested quantity to an exact multiple of the
           exchange's ``stepSize`` (Binance rejects any quantity that is
           not a multiple of ``stepSize`` with error code -1013
           ``"Filter failure: LOT_SIZE"``).
        2. Reject orders whose notional (``quantity × price``) is below
           the exchange minimum (``MIN_NOTIONAL`` / ``NOTIONAL``) before
           dispatch, so we do not waste a round trip.

        No-ops when ``rest_client`` is ``None``; falls back to the
        BTCUSDT defaults set in ``__init__`` if the REST call fails so
        the executor stays operational.
        """
        if self.rest_client is None:
            return
        try:
            info = self.rest_client.exchange_info(symbol=SYMBOL)
            filters = {f["filterType"]: f for f in info["symbols"][0]["filters"]}
            lot = filters["LOT_SIZE"]
            self._min_qty = float(lot["minQty"])
            self._step_size = float(lot["stepSize"])
            # Binance Spot renamed MIN_NOTIONAL → NOTIONAL on some symbols;
            # accept either, default to 10.0 USDT if neither is present.
            notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
            self._min_notional = float(notional.get("minNotional", 10.0))
            # stepSize like "0.00001000" → 5 decimals for the qty format str.
            self._qty_decimals = max(0, int(round(-math.log10(self._step_size))))
            logging.info(
                "Loaded %s filters: minQty=%s, stepSize=%s, minNotional=%s, qtyDecimals=%d",
                SYMBOL,
                self._min_qty,
                self._step_size,
                self._min_notional,
                self._qty_decimals,
            )
        except Exception as e:
            logging.warning(
                "Failed to load %s filters (%s) — using defaults "
                "(minQty=%s, stepSize=%s, minNotional=%s).",
                SYMBOL,
                e,
                self._min_qty,
                self._step_size,
                self._min_notional,
            )

    # ------------------------------------------------------------------
    # WebSocket lifecycle — authentication and user data subscription
    # ------------------------------------------------------------------

    def _on_ws_open(self, _ws) -> None:
        """
        Called by ``SpotWebsocketAPIClient`` as soon as the socket handshake
        completes.  Immediately sends a ``session.logon`` frame to authenticate
        the session so that ``userDataStream.subscribe`` can be called next.

        Args:
            _ws: The underlying websocket-client instance (unused).
        """
        logging.info("WebSocket API connection opened — sending session.logon.")
        self._send_session_logon()

    def _send_session_logon(self) -> None:
        """
        Build and dispatch a signed ``session.logon`` request.

        The frame is signed with ``websocket_api_signature``, which appends
        ``timestamp``, ``apiKey``, and ``signature`` (HMAC-SHA256) to the
        parameter dict.  The generated ``id`` is stored in ``_logon_id`` so
        ``handle_order_response`` can match the response.

        Failures are caught and logged; order execution and the REST-snapshot
        balance fallback are unaffected.
        """
        if self.ws_api_client is None:
            return
        try:
            self._logon_id = get_uuid()
            signed_params = websocket_api_signature(
                self.ws_api_client.api_key,
                self.ws_api_client.api_secret,
                {},  # no extra params — timestamp + apiKey + signature are added by the helper
            )
            payload = {
                "id": self._logon_id,
                "method": "session.logon",
                "params": signed_params,
            }
            self.ws_api_client.send(payload)
            logging.info("session.logon sent (id=%s).", self._logon_id)
        except Exception as e:
            logging.warning(
                "session.logon could not be sent (%s: %s). "
                "Real-time balance updates unavailable; using REST snapshot.",
                type(e).__name__,
                e,
            )

    def _send_user_data_subscribe(self) -> None:
        """
        Dispatch a ``userDataStream.subscribe`` request on the already-
        authenticated WebSocket session.

        Must only be called after a successful ``session.logon`` response.
        The generated ``id`` is stored in ``_subscribe_id`` so the response
        can be matched in ``handle_order_response``.
        """
        if self.ws_api_client is None:
            return
        try:
            self._subscribe_id = get_uuid()
            payload = {
                "id": self._subscribe_id,
                "method": "userDataStream.subscribe",
            }
            self.ws_api_client.send(payload)
            logging.info("userDataStream.subscribe sent (id=%s).", self._subscribe_id)
        except Exception as e:
            logging.warning(
                "userDataStream.subscribe could not be sent (%s: %s). "
                "Real-time balance updates unavailable; using REST snapshot.",
                type(e).__name__,
                e,
            )

    def _handle_balance_update(self, data: dict) -> None:
        """
        Apply an ``outboundAccountPosition`` push event to the live balance
        state.

        Called from ``handle_order_response`` when a push event (no ``"id"``
        field) with ``"e" == "outboundAccountPosition"`` arrives on the
        WebSocket API connection after a successful ``userDataStream.subscribe``.

        Under ``state.thread_balance_lock`` the ``"f"`` (free) field for each
        tracked asset (``CRYPTOCCY``, ``CCY``) is written to
        ``state.balance_status``.  The ``"l"`` (locked) field is also recorded
        in ``state.balance_locked`` for the equity snapshot; trading reads free
        only.

        Args:
            data (dict): Parsed JSON push event from the Binance WebSocket API.
        """
        with self.state.thread_balance_lock:
            for asset_data in data.get("B", []):
                asset = asset_data.get("a")
                if asset in self.state.balance_status:
                    self.state.balance_status[asset] = float(asset_data["f"])
                    # "l" = locked; recorded only for the equity snapshot.
                    self.state.balance_locked[asset] = float(asset_data["l"])
            logging.info(
                "Balance update (WS push) — %s: %.8f | %s: %.2f",
                CRYPTOCCY,
                self.state.balance_status.get(CRYPTOCCY, 0.0),
                CCY,
                self.state.balance_status.get(CCY, 0.0),
            )

    # ------------------------------------------------------------------
    # Internal helpers — balance refresh and stale-order cancel
    # ------------------------------------------------------------------

    def _refresh_balance_rest(self) -> None:
        """
        Fetch free balances from Binance REST and update ``state.balance_status``.

        Called after every successful order placement and after a stale BUY
        cancel so that the next order sees the correct free/locked split.
        No-ops silently when ``rest_client`` is ``None``.
        """
        if self.rest_client is None:
            return
        try:
            acc = self.rest_client.account(recvWindow=RECV_WINDOW)
            with self.state.thread_balance_lock:
                for item in acc.get("balances", []):
                    asset = item.get("asset")
                    if asset in self.state.balance_status:
                        self.state.balance_status[asset] = float(item["free"])
                        self.state.balance_locked[asset] = float(item["locked"])
            logging.info(
                "Balance refreshed — %s: %.8f | %s: %.2f",
                CRYPTOCCY,
                self.state.balance_status.get(CRYPTOCCY, 0.0),
                CCY,
                self.state.balance_status.get(CCY, 0.0),
            )
        except Exception as e:
            logging.warning(
                "Balance refresh failed (%s). Next order may use stale balance.", e
            )

    def cancel_stale_buy(self, timeout_sec: float = 10.0) -> bool:
        """
        Cancel the outstanding GTC BUY order if it has been open longer than
        ``timeout_sec`` seconds, then refresh the balance so freed USDT becomes
        available for the next BUY signal.

        Returns:
            True  — caller should reset ``_position_open`` to False.
            False — keep ``_position_open`` True (order likely filled; wait for SELL).
        """
        if self._pending_buy_placed_at is None:
            return False
        elapsed = time.time() - self._pending_buy_placed_at
        if elapsed < timeout_sec:
            return False

        if self._pending_buy_id is None:
            # Order dispatched but no orderId yet (WS response still in flight).
            # Cannot cancel without an ID; clear stale state and allow re-entry.
            logging.warning(
                "BUY order timed out (%.1fs) with no orderId — clearing stale state.",
                elapsed,
            )
            self._pending_buy_placed_at = None
            return True

        if self.rest_client is None:
            logging.warning(
                "Cannot cancel stale BUY order %s: no REST client available.",
                self._pending_buy_id,
            )
            self._pending_buy_placed_at = None
            self._pending_buy_id = None
            return True

        try:
            self.rest_client.cancel_order(
                symbol=SYMBOL,
                orderId=self._pending_buy_id,
                recvWindow=RECV_WINDOW,
            )
            logging.info(
                "Stale BUY order %s cancelled after %.1fs.",
                self._pending_buy_id,
                elapsed,
            )
            self._refresh_balance_rest()
            self._pending_buy_id = None
            self._pending_buy_placed_at = None
            return True
        except Exception as e:
            # Most likely: order already filled (Binance -2011 "Unknown order").
            # Keep _position_open True so the strategy waits for a SELL signal
            # to close the position.
            logging.warning(
                "Cancel of BUY order %s failed (%s) — order may have filled; "
                "keeping position open.",
                self._pending_buy_id,
                e,
            )
            # Pull the fill into the balance state immediately so the ghost-
            # position check next tick sees the BTC and does not disarm.
            self._refresh_balance_rest()
            self._pending_buy_placed_at = None  # don't retry on every tick
            return False

    def has_pending_buy(self) -> bool:
        """True while a LIMIT GTC BUY is dispatched and unresolved."""
        return self._pending_buy_placed_at is not None

    def has_pending_sell(self) -> bool:
        """True while a non-urgent LIMIT GTC SELL is resting and unresolved."""
        return self._pending_sell_placed_at is not None

    def refresh_and_check_flat(self, min_qty: float = 0.0001) -> bool:
        """
        Refresh balances from REST, then report whether the account holds no
        BTC (free + locked below ``min_qty``).

        Used by the AnalysisEngine's ghost-position check so the position
        guard is only disarmed on a CONFIRMED-fresh flat balance — never on a
        stale snapshot (the 2026-07-08 session disarmed 161 times on balances
        up to 60 s old and re-bought the same signal 8× in 14 s).

        Args:
            min_qty (float): BTC total below which the account counts as flat.

        Returns:
            bool: ``True`` when the freshly-read BTC total is below
                ``min_qty``; ``False`` otherwise (or when only a stale
                balance is available and it shows BTC on the account).
        """
        self._refresh_balance_rest()
        with self.state.thread_balance_lock:
            btc_total = self.state.balance_status.get(
                CRYPTOCCY, 0.0
            ) + self.state.balance_locked.get(CRYPTOCCY, 0.0)
        return btc_total < min_qty

    def cancel_stale_sell(self, timeout_sec: float = 10.0) -> str | None:
        """
        Resolve the outstanding LIMIT GTC SELL once it has been open longer than
        ``timeout_sec`` seconds (mirror of :meth:`cancel_stale_buy`).

        * Cancel succeeds → order was still open, now gone; position still held.
          Returns ``"still_long"`` — caller keeps the guard armed and retries
          the exit on the next SELL signal.
        * Cancel fails (Binance -2011 "Unknown order") → order already FILLED;
          position is closed.  Returns ``"closed"`` — caller resets to flat.

        Returns ``None`` when there is no pending SELL or the timeout has not
        elapsed.  ``timeout_sec=0.0`` forces immediate resolution (used by the
        stop-loss to free locked BTC before its MARKET close).
        """
        if self._pending_sell_placed_at is None:
            return None
        elapsed = time.time() - self._pending_sell_placed_at
        if elapsed < timeout_sec:
            return None

        if self._pending_sell_id is None:
            # Dispatched but no orderId yet (WS response in flight). Cannot
            # cancel; assume still long and clear the timer to re-evaluate.
            logging.warning(
                "SELL order timed out (%.1fs) with no orderId — keeping position.",
                elapsed,
            )
            self._pending_sell_placed_at = None
            return "still_long"

        if self.rest_client is None:
            logging.warning(
                "Cannot cancel stale SELL order %s: no REST client available.",
                self._pending_sell_id,
            )
            self._pending_sell_placed_at = None
            self._pending_sell_id = None
            return "still_long"

        try:
            self.rest_client.cancel_order(
                symbol=SYMBOL, orderId=self._pending_sell_id, recvWindow=RECV_WINDOW
            )
            logging.info(
                "Stale SELL order %s cancelled after %.1fs — position still open, "
                "will retry exit on next signal.",
                self._pending_sell_id,
                elapsed,
            )
            self._refresh_balance_rest()
            self._pending_sell_id = None
            self._pending_sell_placed_at = None
            return "still_long"
        except Exception as e:
            logging.info(
                "Cancel of SELL order %s failed (%s) — order likely FILLED; "
                "treating position as closed.",
                self._pending_sell_id,
                e,
            )
            self._refresh_balance_rest()
            self._pending_sell_id = None
            self._pending_sell_placed_at = None
            return "closed"

    # ------------------------------------------------------------------
    # Callback — invoked by the WebSocket API client on every message
    # ------------------------------------------------------------------

    def handle_order_response(self, _, message: str) -> None:
        """
        Single ``on_message`` callback for the Binance WebSocket API connection.

        Routes incoming frames by the presence/absence of the ``"id"`` field:

        * **No** ``"id"`` (push event) — forwarded to ``_handle_balance_update``
          when ``"e" == "outboundAccountPosition"``; all other push events are
          silently ignored.
        * ``"id"`` **== _logon_id** — ``session.logon`` response: on success,
          immediately sends ``userDataStream.subscribe``; on failure, logs a
          warning.
        * ``"id"`` **== _subscribe_id** — ``userDataStream.subscribe`` response:
          on success, sets ``_user_data_active = True`` and logs confirmation;
          on failure, logs a warning.
        * **Any other** ``"id"`` — treated as an order placement response:
          errors are logged; successful results are stored in ``last_order``
          and appended to ``placed_orders``.

        Args:
            _: The WebSocket client instance (unused).
            message (str): Raw JSON string received from the Binance WebSocket
                API.
        """
        data = json.loads(message)

        # ── Push event: no "id" field ──────────────────────────────────────
        # Binance pushes user-data events (outboundAccountPosition, etc.) on
        # the same connection after a successful userDataStream.subscribe.
        # These frames have no "id" key — unlike request/response pairs.
        if "id" not in data:
            if data.get("e") == "outboundAccountPosition":
                self._handle_balance_update(data)
            else:
                logging.debug("Unhandled WS push event: %s", data.get("e"))
            return

        # ── session.logon response ─────────────────────────────────────────
        if data.get("id") == self._logon_id:
            if data.get("status") == 200:
                logging.info(
                    "session.logon succeeded — sending userDataStream.subscribe."
                )
                self._send_user_data_subscribe()
            else:
                err = data.get("error", data)
                logging.warning(
                    "session.logon failed (status=%s): %s. "
                    "Real-time balance updates unavailable; using REST snapshot.",
                    data.get("status"),
                    err,
                )
            return

        # ── userDataStream.subscribe response ──────────────────────────────
        if data.get("id") == self._subscribe_id:
            if data.get("status") == 200:
                self._user_data_active = True
                logging.info(
                    "userDataStream.subscribe confirmed — "
                    "real-time balance updates active on WS connection."
                )
            else:
                err = data.get("error", data)
                logging.warning(
                    "userDataStream.subscribe failed (status=%s): %s. "
                    "Real-time balance updates unavailable; using REST snapshot.",
                    data.get("status"),
                    err,
                )
            return

        # ── Order placement response ───────────────────────────────────────
        if "error" in data:
            logging.error(
                "Order WS error (code %s): %s",
                data["error"].get("code"),
                data["error"].get("msg"),
            )
            return

        result = data.get("result", data)
        # Guard against logon/subscribe success frames being mis-routed here
        # (they have a "result" key but no order fields).
        if not result.get("orderId"):
            logging.debug("WS response with no orderId — skipping order record.")
            return

        self.last_order = result
        # Capture BUY orderId for the 10-second stale-cancel mechanism.
        # The WS path sets _pending_buy_placed_at synchronously in execute();
        # the orderId only arrives here asynchronously.
        if result.get("side") == "BUY":
            self._pending_buy_id = result.get("orderId")
        # A resting LIMIT SELL: capture its id so cancel_stale_sell() can target
        # it. A MARKET SELL returns FILLED (no resting order), so only record
        # while the order is still working.
        elif result.get("side") == "SELL" and result.get("status") in (
            "NEW",
            "PARTIALLY_FILLED",
        ):
            self._pending_sell_id = result.get("orderId")
        # Pull the dispatch-time stamp from execute() (set just before the WS
        # send).  Falls back to "now" if missing (e.g. a response arrives for
        # which no dispatch was tracked — should not happen in practice).
        _placed_at = self._last_dispatch_at or datetime.now(timezone.utc)
        self._last_dispatch_at = None  # consumed
        self.placed_orders.append(
            {
                "orderId": result.get("orderId"),
                "side": result.get("side"),
                "price": result.get("price"),
                "origQty": result.get("origQty"),
                "status": result.get("status"),
                "placed_at": _placed_at,
            }
        )
        logging.info(
            "Order response: %s %s orderId=%s status=%s",
            result.get("side"),
            result.get("symbol"),
            result.get("orderId"),
            result.get("status"),
        )

    def execute(self, strategy: str, opportunity: tuple, urgent: bool = False) -> None:
        """
        Send an order via the WebSocket API (preferred) or REST (fallback):
        a LIMIT GTC BUY, or a SELL that is MARKET when ``urgent`` else LIMIT GTC.

        **Order type per side:**

        * BUY:  LIMIT GTC — dip-buy placed on the book; lives until filled or
          cancelled via ``cancel_stale_buy`` (10-second timeout).
          ``_pending_buy_placed_at`` is set on dispatch; ``_pending_buy_id`` is
          set synchronously (REST) or asynchronously via ``handle_order_response``
          (WS) so the cancel mechanism can target the correct order.
        * SELL: order type depends on ``urgent``:
          - ``urgent=True`` → MARKET — the stop-loss exit, which must close the
            position immediately. Fills against Binance's real server-side book,
            so it is immune to ``local_book`` staleness and never expires.
          - ``urgent=False`` (default) → LIMIT GTC at ``micro_price``, symmetric
            to the BUY: the planned mean-reversion exit rests on the book as a
            maker order and is resolved by ``cancel_stale_sell`` (10 s timeout)
            — filled, or cancelled-and-retried.

        **Dynamic quantity cap** — caps the requested quantity to the available
        balance rather than skipping the order entirely:

        * BUY:  ``quantity = min(aq, usdt / (micro_price × (1 + BACKTEST_FEE_RATE)))``
        * SELL: ``quantity = min(bq, btc)``

        The fee-adjusted divisor for BUY orders ensures the total debit
        (``quantity × micro_price × (1 + fee_rate)``) never exceeds the
        available USDT balance.

        **Exchange-filter floor** — after the dynamic cap the quantity is
        floored DOWN to the cached ``LOT_SIZE`` ``stepSize`` so the value
        sent to Binance is always an exact multiple of the symbol's
        precision grid (prevents -1013 ``"Filter failure: LOT_SIZE"``).
        The order is skipped before dispatch when the floored quantity
        is below ``minQty`` or when ``quantity × micro_price`` is below
        ``minNotional``.

        Args:
            strategy (str): ``"BUY"`` or ``"SELL"``.
            opportunity (tuple): 8-element tuple
                ``(level_idx, score, delta, total_depth, obi, micro_price, bq, aq)``
                returned by ``AnalysisEngine._select_best_opportunity()``.
            urgent (bool): When ``True`` and ``strategy == "SELL"``, send a
                MARKET order (immediate close, e.g. stop-loss); otherwise the
                SELL rests as a LIMIT GTC.  Ignored for BUY.
        """
        if strategy not in ("BUY", "SELL"):
            logging.error("Invalid strategy '%s'. Must be 'BUY' or 'SELL'.", strategy)
            return

        level_idx, score, delta, total_depth, obi, micro_price, bq, aq = opportunity

        with self.state.thread_balance_lock:
            usdt = self.state.balance_status.get(CCY, 0.0)
            btc = self.state.balance_status.get(CRYPTOCCY, 0.0)

        # BUY  → quantity is aq (ask-side liquidity available at this level)
        # SELL → quantity is bq (bid-side liquidity available at this level)
        # Dynamically cap the quantity to what the available balance can afford,
        # so the algo still trades at a reduced size rather than skipping entirely.
        if strategy == "BUY":
            # Reset the dispatched-leg markers so a skipped BUY (budget too low /
            # reserve floor) never leaves a stale value for the strategy layer.
            self.last_buy_qty = 0.0
            self.last_buy_price = 0.0
            # Position cap — MAX_POSITION_PCT × available USDT is the per-signal
            # budget shared with backtest/pnl.py.  Prevents all-in BUY orders
            # (e.g. 313k USDT in one trade) and keeps live order sizing aligned
            # with the simulated sizing the strategy was tuned against.
            usdt_budget = usdt * MAX_POSITION_PCT
            # Cash-reserve floor (mirrors backtest/pnl.py): clamp the budget so
            # this BUY never spends the account below MIN_CASH_RESERVE_PCT of
            # mark-to-market equity (free USDT + BTC valued at micro_price).  Live
            # legs are serialized (a new leg is dispatched only when no BUY is in
            # flight — see strategy/analysis.py), so `usdt` is the free balance
            # after prior legs settled and equity ≈ usdt + btc × micro_price.
            # Caps total invested exposure at (1 − reserve); shared with backtest
            # so live and simulated sizing stay aligned.
            equity = usdt + btc * micro_price
            spendable = usdt - MIN_CASH_RESERVE_PCT * equity
            usdt_budget = min(usdt_budget, max(0.0, spendable))
            # Divide by price × (1 + fee_rate) so the total debit
            # (notional + taker fee) never exceeds the budget.  Without the fee
            # factor Binance rejects the order with "insufficient balance"
            # because the fee is charged on top of the notional.
            max_affordable = (
                usdt_budget / (micro_price * (1.0 + BACKTEST_FEE_RATE))
                if micro_price > 0
                else 0.0
            )
            quantity = min(aq, max_affordable)
            if quantity <= 0:
                logging.warning(
                    "LIMIT BUY skipped: per-signal budget (%.2f %s = %.0f%% of "
                    "%.2f %s) too low to buy at price %.2f.",
                    usdt_budget,
                    CCY,
                    MAX_POSITION_PCT * 100,
                    usdt,
                    CCY,
                    micro_price,
                )
                return
            if quantity < aq:
                logging.info(
                    "LIMIT BUY quantity capped: requested %.6f %s → affordable %.6f %s "
                    "(budget %.2f %s = %.0f%% of %.2f %s at price %.2f).",
                    aq,
                    CRYPTOCCY,
                    quantity,
                    CRYPTOCCY,
                    usdt_budget,
                    CCY,
                    MAX_POSITION_PCT * 100,
                    usdt,
                    CCY,
                    micro_price,
                )
            # Expose the dispatched leg size/price for the strategy layer's
            # pyramiding cost-basis accrual (see analysis.py).
            self.last_buy_qty = quantity
            self.last_buy_price = micro_price
        else:  # SELL
            quantity = min(bq, btc)
            if quantity <= 0:
                logging.warning(
                    "LIMIT SELL skipped: available %s balance (%.6f) is too low.",
                    CRYPTOCCY,
                    btc,
                )
                return
            if quantity < bq:
                logging.info(
                    "LIMIT SELL quantity capped: requested %.6f %s → affordable %.6f %s "
                    "(balance %.6f %s).",
                    bq,
                    CRYPTOCCY,
                    quantity,
                    CRYPTOCCY,
                    btc,
                    CRYPTOCCY,
                )
            # SELL closes the position — clear pending BUY tracking regardless
            # of which execution path is taken below.
            self._pending_buy_placed_at = None
            self._pending_buy_id = None

        # ── SELL order type & book-health diagnostic ───────────────────
        # A SELL closes the position and MUST fill, so it is sent as a MARKET
        # order in the dispatch section below.  A marketable LIMIT/IOC priced
        # from local_book can expire unfilled when local_book has gone stale:
        # core/message_handler.py has no depth-diff gap recovery, so a single
        # missed "remove level" message leaves a phantom top-of-book bid that
        # max(bids) keeps returning.  A MARKET order fills against Binance's
        # real server-side book and is immune to that.
        # We log book health here so the staleness can be confirmed/fixed at
        # the source: a CROSSED book (best_bid >= best_ask) is direct proof
        # local_book is stale.  BUY is unaffected (LIMIT GTC at micro_price).
        if strategy == "SELL":
            with self.state.thread_lock:
                _bids = self.state.local_book.get("bids", {})
                _asks = self.state.local_book.get("asks", {})
                _best_bid = max((float(p) for p in _bids), default=0.0)
                _best_ask = min((float(p) for p in _asks), default=0.0)
                _book_id = self.state.local_book.get("lastUpdateId", 0)
            if 0.0 < _best_ask <= _best_bid:
                logging.warning(
                    "Book-health: CROSSED/STALE local_book at SELL — "
                    "best_bid=%.2f >= best_ask=%.2f (lastUpdateId=%s).",
                    _best_bid,
                    _best_ask,
                    _book_id,
                )
            else:
                logging.info(
                    "Book-health at SELL — best_bid=%.2f best_ask=%.2f lastUpdateId=%s",
                    _best_bid,
                    _best_ask,
                    _book_id,
                )

        # ── Exchange-filter normalisation ──────────────────────────────
        # Floor quantity DOWN to the LOT_SIZE stepSize so Binance does not
        # reject the order with -1013 "Filter failure: LOT_SIZE".  Round
        # DOWN (math.floor) — never UP — so the resulting quantity can
        # never exceed the budget computed above.
        quantity = math.floor(quantity / self._step_size) * self._step_size
        # round() cleans up the float-precision noise that math.floor
        # introduces (e.g. 0.00478 stored as 0.0047799999…).
        quantity = round(quantity, self._qty_decimals)
        if quantity < self._min_qty:
            logging.warning(
                "LIMIT %s skipped: qty %.8f < minQty %.8f after step-size floor.",
                strategy,
                quantity,
                self._min_qty,
            )
            return
        if quantity * micro_price < self._min_notional:
            logging.warning(
                "LIMIT %s skipped: notional %.2f < minNotional %.2f "
                "(qty=%.8f, price=%.2f).",
                strategy,
                quantity * micro_price,
                self._min_notional,
                quantity,
                micro_price,
            )
            return

        # Pre-formatted qty string — used by both WS and REST dispatch
        # paths below so the decimal width always matches stepSize
        # (sending more decimals than stepSize allows is a -1013).
        qty_str = f"{quantity:.{self._qty_decimals}f}"

        # Order type per side:
        #   BUY  → LIMIT GTC  (dip-buy rests on the book until filled/cancelled).
        #   SELL → MARKET     (closes the position against Binance's real
        #          server-side book; immune to local_book staleness that makes
        #          a marketable LIMIT/IOC expire — see book-health note above).
        if strategy == "BUY":
            order_kwargs = {
                "symbol": SYMBOL,
                "side": "BUY",
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": qty_str,
                "price": f"{micro_price:.2f}",
                "recvWindow": RECV_WINDOW,
            }
            order_desc = f"LIMIT BUY (GTC) price={micro_price:.2f}"
        else:  # SELL
            if urgent:
                # Immediate close (stop-loss): MARKET against the real book.
                order_kwargs = {
                    "symbol": SYMBOL,
                    "side": "SELL",
                    "type": "MARKET",
                    "quantity": qty_str,
                    "recvWindow": RECV_WINDOW,
                }
                order_desc = "MARKET SELL (urgent)"
            else:
                # Planned exit: resting LIMIT GTC at micro_price, like the BUY.
                order_kwargs = {
                    "symbol": SYMBOL,
                    "side": "SELL",
                    "type": "LIMIT",
                    "timeInForce": "GTC",
                    "quantity": qty_str,
                    "price": f"{micro_price:.2f}",
                    "recvWindow": RECV_WINDOW,
                }
                order_desc = f"LIMIT SELL (GTC) price={micro_price:.2f}"

        logging.info("Sending %s: level=%d qty=%s", order_desc, level_idx, qty_str)

        if self.ws_api_client is not None:
            # Preferred path — lower latency, async response via handle_order_response
            if strategy == "BUY":
                self._pending_buy_placed_at = time.time()
            elif not urgent:
                # Resting LIMIT SELL — arm the stale-resolve timer (orderId
                # captured asynchronously in handle_order_response).
                self._pending_sell_placed_at = time.time()
            else:
                # Urgent MARKET SELL force-closes — clear any resting-exit timer.
                self._pending_sell_placed_at = None
                self._pending_sell_id = None
            # Stamp dispatch wall-clock time so handle_order_response can
            # attach it to the resulting placed_orders entry.
            self._last_dispatch_at = datetime.now(timezone.utc)
            self.ws_api_client.new_order(**order_kwargs)
        elif self.rest_client is not None:
            # Fallback path — synchronous REST call (used when WS API is unavailable)
            _placed_at = datetime.now(timezone.utc)
            if strategy == "BUY":
                self._pending_buy_placed_at = time.time()
            elif not urgent:
                self._pending_sell_placed_at = time.time()
            else:
                self._pending_sell_placed_at = None
                self._pending_sell_id = None
            try:
                response = self.rest_client.new_order(**order_kwargs)
                self.last_order = response
                if strategy == "BUY":
                    self._pending_buy_id = response.get("orderId")
                elif not urgent and response.get("status") in (
                    "NEW",
                    "PARTIALLY_FILLED",
                ):
                    self._pending_sell_id = response.get("orderId")
                self.placed_orders.append(
                    {
                        "orderId": response.get("orderId"),
                        "side": response.get("side"),
                        "price": response.get("price"),
                        "origQty": response.get("origQty"),
                        "status": response.get("status"),
                        "placed_at": _placed_at,
                    }
                )
                logging.info(
                    "REST %s placed: level=%d qty=%s | orderId=%s",
                    order_desc,
                    level_idx,
                    qty_str,
                    response.get("orderId"),
                )
                self._refresh_balance_rest()
            except Exception as e:
                if strategy == "BUY":
                    self._pending_buy_placed_at = None
                elif not urgent:
                    self._pending_sell_placed_at = None
                    self._pending_sell_id = None
                logging.error(
                    "REST %s failed (level %d): %s",
                    order_desc,
                    level_idx,
                    e,
                )
        else:
            logging.error(
                "LIMIT %s skipped: no WebSocket API client and no REST client available.",
                strategy,
            )

    def cancel_session_open_orders(self) -> None:
        """
        Cancel every order placed by THIS session that is still open on the
        book, so no funds stay locked (and no unattended fills happen) after
        the session ends.

        Only the session's own orderIds are cancelled — pre-existing (foreign)
        open orders on the shared testnet account are left untouched.
        Fail-safe: every failure is logged and skipped; never raises.
        """
        if self.rest_client is None or not self.placed_orders:
            return
        try:
            open_orders = _retry_transient(
                lambda: self.rest_client.get_open_orders(
                    symbol=SYMBOL, recvWindow=RECV_WINDOW
                )
            )
        except Exception as e:
            logging.warning(
                "Shutdown cancel: could not fetch open orders after retries (%s) — "
                "session orders may remain on the book.",
                e,
            )
            return
        session_ids = {r.get("orderId") for r in self.placed_orders}
        to_cancel = [o for o in open_orders if o.get("orderId") in session_ids]
        if not to_cancel:
            logging.info("Shutdown cancel: no session orders left open.")
            return
        cancelled = failed = 0
        freed_usdt = freed_btc = 0.0
        for o in to_cancel:
            try:
                _retry_transient(
                    lambda o=o: self.rest_client.cancel_order(
                        symbol=SYMBOL, orderId=o["orderId"], recvWindow=RECV_WINDOW
                    )
                )
                remaining = float(o["origQty"]) - float(o["executedQty"])
                if o.get("side") == "BUY":
                    freed_usdt += remaining * float(o["price"])
                else:
                    freed_btc += remaining
                cancelled += 1
            except Exception as e:
                failed += 1
                logging.warning(
                    "Shutdown cancel of order %s failed (%s).", o.get("orderId"), e
                )
        logging.info(
            "Shutdown cancel: %d session order(s) cancelled (%d failed) — "
            "freed ≈ %.2f %s + %.6f %s from locked.",
            cancelled,
            failed,
            freed_usdt,
            CCY,
            freed_btc,
            CRYPTOCCY,
        )
        self._refresh_balance_rest()

    def stop(self) -> None:
        """
        Close the WebSocket API connection cleanly.

        If ``userDataStream.subscribe`` is active, unsubscription is handled
        automatically by the server when the connection closes.
        """
        if self.ws_api_client is not None:
            self.ws_api_client.stop()

    # ------------------------------------------------------------------
    # End-of-session order report
    # ------------------------------------------------------------------

    def _query_and_log_order(self, record: dict) -> tuple[int, int, int, int]:
        """
        Query a single order from Binance and log its status in the report table.

        Called exclusively by :meth:`order_status_report` for each order record
        that falls within the head or tail slice of ``placed_orders``.

        Args:
            record (dict): Entry from ``placed_orders`` containing at minimum
                ``orderId``, ``side``, ``price``, and ``origQty``.

        Returns:
            tuple[int, int, int, int]: Increments ``(filled, partial, pending, other)``
                — exactly one of the four values is 1, the rest are 0.
        """
        order_id = record.get("orderId")
        _filled = _partial = _pending = _other = 0
        try:
            result = self.rest_client.get_order(
                symbol=SYMBOL, orderId=order_id, recvWindow=RECV_WINDOW
            )
            status = result.get("status", "UNKNOWN")
            side = result.get("side", record.get("side", "?"))
            price = float(result.get("price", record.get("price", 0)))
            orig_qty = float(result.get("origQty", record.get("origQty", 0)))
            exec_qty = float(result.get("executedQty", 0))
            quote_spent = float(result.get("cummulativeQuoteQty", 0))
            # Enrich the record in place with the FINAL outcome so the
            # end-of-session P&L chart can distinguish an order that actually
            # traded from one that was placed but cancelled / never matched.
            # order_status_report() runs before the chart is built, and the
            # marker is otherwise drawn at dispatch time regardless of fill.
            record["final_status"] = status
            record["exec_qty"] = exec_qty

            if status == "FILLED":
                label = "✔ FILLED"
                _filled = 1
            elif status == "PARTIALLY_FILLED":
                label = "~ PARTIAL"
                _partial = 1
            elif status == "NEW":
                label = "○ OPEN"
                _pending = 1
            else:
                label = f"  {status}"
                _other = 1

            logging.info(
                "  orderId=%-12s  %-4s  %s  price=%-10.2f  "
                "origQty=%-12.6f  execQty=%-12.6f  quoteQty=%.4f",
                order_id,
                side,
                label,
                price,
                orig_qty,
                exec_qty,
                quote_spent,
            )
        except Exception as e:
            logging.error("  orderId=%-12s  could not query status: %s", order_id, e)
        return _filled, _partial, _pending, _other

    def order_status_report(self) -> None:
        """
        Query the final status of every order placed during the session and
        log a formatted summary table.

        To avoid flooding the console during long sessions the report is
        capped: only the first and last ``ORDER_REPORT_LIMIT`` orders (default
        100 each) are printed individually.  When the total exceeds
        ``2 * ORDER_REPORT_LIMIT`` the middle block is collapsed to a single
        summary line that states how many orders were omitted.  Raise
        ``ORDER_REPORT_LIMIT`` in ``config_parameters.py`` to expose more rows.

        For each printed order the method calls ``GET /api/v3/order`` via the
        REST client and maps the Binance ``status`` field to a human-readable
        label:

        * ``FILLED``           — fully executed
        * ``PARTIALLY_FILLED`` — partially executed (``executedQty`` < ``origQty``)
        * ``NEW``              — still open / never matched
        * ``CANCELED``         — manually or automatically cancelled
        * ``EXPIRED``          — expired (e.g. FOK/IOC that was not filled)

        If no orders were placed, or if the REST client is unavailable, a
        short informational message is logged instead.
        """
        if not self.placed_orders:
            logging.info("Order report: no orders were placed this session.")
            return
        if self.rest_client is None:
            logging.warning(
                "Order report: REST client unavailable, cannot query order statuses."
            )
            return

        logging.info(
            "\n========== END-OF-SESSION ORDER REPORT (%d order(s)) ==========",
            len(self.placed_orders),
        )

        total = len(self.placed_orders)
        # If the session generated more orders than 2 * ORDER_REPORT_LIMIT,
        # only print the first and last ORDER_REPORT_LIMIT to avoid flooding
        # the console.  The skipped middle block is summarized on one line.
        if total > 2 * ORDER_REPORT_LIMIT:
            head = self.placed_orders[:ORDER_REPORT_LIMIT]
            tail = self.placed_orders[-ORDER_REPORT_LIMIT:]
            skipped = total - 2 * ORDER_REPORT_LIMIT
        else:
            head = self.placed_orders
            tail = []
            skipped = 0

        filled = partial = pending = other = 0

        logging.info("  ── first %d order(s) ──", len(head))
        for record in head:
            f, p, pe, o = self._query_and_log_order(record)
            filled += f
            partial += p
            pending += pe
            other += o

        if skipped:
            logging.info(
                "  ── … %d order(s) omitted (set ORDER_REPORT_LIMIT in "
                "config_parameters.py to raise the cap) … ──",
                skipped,
            )

        if tail:
            logging.info("  ── last %d order(s) ──", len(tail))
            for record in tail:
                f, p, pe, o = self._query_and_log_order(record)
                filled += f
                partial += p
                pending += pe
                other += o

        logging.info(
            "  Summary → filled: %d | partial: %d | still open: %d | other: %d",
            filled,
            partial,
            pending,
            other,
        )
        logging.info("=" * 60)
