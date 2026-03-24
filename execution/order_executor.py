import json
import logging
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
from core.order_book_state import OrderBookState
from config_parameters import SYMBOL, CRYPTOCCY, CCY, RECV_WINDOW


class OrderExecutor:
    """
    Responsible for placing LIMIT GTC orders via the Binance WebSocket API
    and tracking their responses asynchronously.

    Order requests are sent as JSON frames over an already-open WebSocket
    connection, avoiding the per-request HTTP overhead of the REST API.
    Responses arrive in ``handle_order_response``, which logs the result
    and stores it in ``last_order``.

    The WebSocket API client is created internally during ``__init__`` so
    that ``self`` is available as the callback target — no external wiring
    or circular-dependency workarounds are needed.

    Attributes:
        ws_api_client (SpotWebsocketAPIClient): Authenticated WebSocket API
            client used to send order requests.  Created and owned by this
            instance.
        state (OrderBookState): Shared order book and balance state, injected at
            construction and also consumed by ``MessageHandler`` and
            ``AnalysisEngine``.
        last_order (dict | None): Most recent order response received from
            Binance, or ``None`` if no order has been acknowledged yet.
    """

    def __init__(
        self, state: OrderBookState, stream_url: str, api_key: str, api_secret: str
    ):
        """
        Create the executor and open a WebSocket API connection to Binance.

        The ``SpotWebsocketAPIClient`` is constructed here so that
        ``self.handle_order_response`` can be passed directly as the
        ``on_message`` callback — ``self`` already exists at this point.

        Args:
            state (OrderBookState): Shared order book and balance state.
            stream_url (str): Binance WebSocket API endpoint
                (e.g. ``"wss://testnet.binance.vision/ws-api/v3"``).
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
        """
        self.state = state
        self.last_order = None
        self.ws_api_client = SpotWebsocketAPIClient(
            stream_url=stream_url,
            api_key=api_key,
            api_secret=api_secret,
            on_message=self.handle_order_response,
        )

    def stop(self):
        """Close the underlying WebSocket API connection."""
        self.ws_api_client.stop()

    # ------------------------------------------------------------------
    # Callback — invoked by the WebSocket API client on every response
    # ------------------------------------------------------------------
    def handle_order_response(self, _, message):
        """
        WebSocket callback that processes order placement responses from
        the Binance WebSocket API.

        On success the response is stored in ``last_order`` and logged.
        On failure (``status`` != 200 or presence of an error payload)
        the error is logged.

        Args:
            _: The WebSocket client instance (unused).
            message (str): Raw JSON response string from the WebSocket API.
        """
        data = json.loads(message)

        # Binance WS API wraps the result under a "result" key on success
        # and returns an "error" key on failure.
        if "error" in data:
            logging.error(
                "Order WS error (code %s): %s",
                data["error"].get("code"),
                data["error"].get("msg"),
            )
            return

        result = data.get("result", data)
        self.last_order = result
        logging.info(
            "Order response: %s %s orderId=%s status=%s",
            result.get("side"),
            result.get("symbol"),
            result.get("orderId"),
            result.get("status"),
        )

    # ------------------------------------------------------------------
    # Fire-and-forget order placement
    # ------------------------------------------------------------------
    def execute(self, strategy: str, opportunity: tuple) -> None:
        """
        Send a LIMIT GTC order request over the WebSocket API.

        The method validates the strategy, checks available balances, and
        sends the order frame.  The actual Binance response is handled
        asynchronously in ``handle_order_response``.

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
        quantity = aq if strategy == "BUY" else bq

        # BUY  → spending USDT to acquire BTC.
        # SELL → selling BTC to acquire USDT.
        if strategy == "BUY" and quantity * micro_price > usdt:
            logging.warning(
                "LIMIT BUY skipped: cost %.2f %s exceeds available %.2f %s",
                quantity * micro_price,
                CCY,
                usdt,
                CCY,
            )
            return
        if strategy == "SELL" and quantity > btc:
            logging.warning(
                "LIMIT SELL skipped: qty %.6f %s exceeds available %.6f %s.",
                quantity,
                CRYPTOCCY,
                btc,
                CRYPTOCCY,
            )
            return

        logging.info(
            "Sending LIMIT %s: level=%d price=%.2f qty=%.6f",
            strategy,
            level_idx,
            micro_price,
            quantity,
        )
        self.ws_api_client.new_order(
            symbol=SYMBOL,
            side=strategy,
            type="LIMIT",
            timeInForce="GTC",
            quantity=f"{quantity:.6f}",
            price=f"{micro_price:.2f}",
            recvWindow=RECV_WINDOW,
        )
