import json
import logging
import time
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
)


class OrderExecutor:
    """
    Responsible for placing LIMIT GTC orders and maintaining real-time
    balance updates via the Binance WebSocket API.

    **Order execution** — prefers the Binance WebSocket API
    (``wss://testnet.binance.vision/ws-api/v3``) for lower latency and
    falls back transparently to the REST API when the WebSocket endpoint is
    unavailable.  No logic change is required in the caller.

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
                (e.g. ``"wss://testnet.binance.vision/ws-api/v3"``).
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
        except Exception as e:
            logging.warning(
                "WebSocket API unavailable (%s: %s). "
                "Order execution will fall back to REST. "
                "Balance updates will use REST snapshot only.",
                type(e).__name__,
                e,
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
        ``state.balance_status``.  Locked quantities are intentionally ignored.

        Args:
            data (dict): Parsed JSON push event from the Binance WebSocket API.
        """
        with self.state.thread_balance_lock:
            for asset_data in data.get("B", []):
                asset = asset_data.get("a")
                if asset in self.state.balance_status:
                    self.state.balance_status[asset] = float(asset_data["f"])
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
            self._pending_buy_placed_at = None  # don't retry on every tick
            return False

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
        self.placed_orders.append(
            {
                "orderId": result.get("orderId"),
                "side": result.get("side"),
                "price": result.get("price"),
                "origQty": result.get("origQty"),
                "status": result.get("status"),
            }
        )
        logging.info(
            "Order response: %s %s orderId=%s status=%s",
            result.get("side"),
            result.get("symbol"),
            result.get("orderId"),
            result.get("status"),
        )

    def execute(self, strategy: str, opportunity: tuple) -> None:
        """
        Send a LIMIT GTC (BUY) or LIMIT IOC (SELL) order via the WebSocket API
        (preferred) or the REST API (fallback).

        **Order type per side:**

        * BUY:  LIMIT GTC — dip-buy placed on the book; lives until filled or
          cancelled via ``cancel_stale_buy`` (10-second timeout).
          ``_pending_buy_placed_at`` is set on dispatch; ``_pending_buy_id`` is
          set synchronously (REST) or asynchronously via ``handle_order_response``
          (WS) so the cancel mechanism can target the correct order.
        * SELL: LIMIT IOC — fills at the best available bid immediately or
          cancels; never locks BTC beyond a single matching cycle, eliminating
          the -2010 "insufficient balance" failure caused by stale locked BTC.

        **Dynamic quantity cap** — caps the requested quantity to the available
        balance rather than skipping the order entirely:

        * BUY:  ``quantity = min(aq, usdt / (micro_price × (1 + BACKTEST_FEE_RATE)))``
        * SELL: ``quantity = min(bq, btc)``

        The fee-adjusted divisor for BUY orders ensures the total debit
        (``quantity × micro_price × (1 + fee_rate)``) never exceeds the
        available USDT balance.

        Args:
            strategy (str): ``"BUY"`` or ``"SELL"``.
            opportunity (tuple): 8-element tuple
                ``(level_idx, score, delta, total_depth, obi, micro_price, bq, aq)``
                returned by ``AnalysisEngine._select_best_opportunity()``.
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
            # Divide by price × (1 + fee_rate) so the total debit
            # (notional + taker fee) never exceeds the available USDT balance.
            # Without the fee factor Binance rejects the order with
            # "insufficient balance" because the fee is charged on top of the
            # notional and pushes the total spend over the account balance.
            max_affordable = (
                usdt / (micro_price * (1.0 + BACKTEST_FEE_RATE))
                if micro_price > 0
                else 0.0
            )
            quantity = min(aq, max_affordable)
            if quantity <= 0:
                logging.warning(
                    "LIMIT BUY skipped: available %s balance (%.2f) is too low "
                    "to buy at price %.2f.",
                    CCY,
                    usdt,
                    micro_price,
                )
                return
            if quantity < aq:
                logging.info(
                    "LIMIT BUY quantity capped: requested %.6f %s → affordable %.6f %s "
                    "(balance %.2f %s at price %.2f).",
                    aq,
                    CRYPTOCCY,
                    quantity,
                    CRYPTOCCY,
                    usdt,
                    CCY,
                    micro_price,
                )
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

        # BUY → GTC (dip-buy needs time on the book).
        # SELL → IOC (fill at best bid immediately or cancel; never locks BTC).
        time_in_force = "GTC" if strategy == "BUY" else "IOC"

        logging.info(
            "Sending LIMIT %s (%s): level=%d price=%.2f qty=%.6f",
            strategy,
            time_in_force,
            level_idx,
            micro_price,
            quantity,
        )

        if self.ws_api_client is not None:
            # Preferred path — lower latency, async response via handle_order_response
            if strategy == "BUY":
                self._pending_buy_placed_at = time.time()
            self.ws_api_client.new_order(
                symbol=SYMBOL,
                side=strategy,
                type="LIMIT",
                timeInForce=time_in_force,
                quantity=f"{quantity:.6f}",
                price=f"{micro_price:.2f}",
                recvWindow=RECV_WINDOW,
            )
        elif self.rest_client is not None:
            # Fallback path — synchronous REST call (used when WS API is unavailable)
            if strategy == "BUY":
                self._pending_buy_placed_at = time.time()
            try:
                response = self.rest_client.new_order(
                    symbol=SYMBOL,
                    side=strategy,
                    type="LIMIT",
                    timeInForce=time_in_force,
                    quantity=f"{quantity:.6f}",
                    price=f"{micro_price:.2f}",
                    recvWindow=RECV_WINDOW,
                )
                self.last_order = response
                if strategy == "BUY":
                    self._pending_buy_id = response.get("orderId")
                self.placed_orders.append(
                    {
                        "orderId": response.get("orderId"),
                        "side": response.get("side"),
                        "price": response.get("price"),
                        "origQty": response.get("origQty"),
                        "status": response.get("status"),
                    }
                )
                logging.info(
                    "REST LIMIT %s placed: level=%d price=%.2f qty=%s | orderId=%s",
                    strategy,
                    level_idx,
                    micro_price,
                    quantity,
                    response.get("orderId"),
                )
                self._refresh_balance_rest()
            except Exception as e:
                if strategy == "BUY":
                    self._pending_buy_placed_at = None
                logging.error(
                    "REST LIMIT %s failed (level %d, price %.2f): %s",
                    strategy,
                    level_idx,
                    micro_price,
                    e,
                )
        else:
            logging.error(
                "LIMIT %s skipped: no WebSocket API client and no REST client available.",
                strategy,
            )

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
