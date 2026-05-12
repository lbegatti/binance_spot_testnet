import json
import logging

from strategy.best_quote_calculator import calculate_best_quote
from core.order_book_state import OrderBookState
from config_parameters import QUOTE_EVERY_N_TICKS


class MessageHandler:
    """
    WebSocket callback handler responsible for maintaining the local order book
    in real time.

    Exposes one active WebSocket callback:

    * ``handle_depth_message`` — registered on the production diff-depth stream
      (``ws_client``).  Applies every bid/ask delta to ``state.local_book``,
      appends best-bid/ask snapshots to ``state.history_order_book``, and
      triggers a throttled best-quote calculation every
      ``QUOTE_EVERY_N_TICKS`` ticks (~1 s at 100 ms tick rate).

    A second callback, ``handle_balance_message``, is preserved but **no longer
    wired** to any stream.  Real-time balance updates now flow through
    ``OrderExecutor._handle_balance_update`` on the same WebSocket API
    connection used for order placement (via ``session.logon`` →
    ``userDataStream.subscribe`` → ``outboundAccountPosition`` push events).
    The old listenKey / User Data Stream was discontinued by Binance in
    February 2026.

    Both callbacks communicate exclusively through the injected
    ``OrderBookState`` instance and their respective threading locks
    (``thread_lock`` for order-book data, ``thread_balance_lock`` for balance
    data), keeping the two data paths fully independent.

    Attributes:
        state (OrderBookState): Shared order book and balance state, injected at
            construction and also consumed by ``AnalysisEngine``.
    """

    def __init__(self, state: OrderBookState):
        """
        Args:
            state (OrderBookState): The shared order book and balance state
                instance that provides ``local_book``, ``history_order_book``,
                ``thread_lock``, ``balance_status``, and
                ``thread_balance_lock``.
        """
        self.state = state
        self._tick_count: int = 0

    def handle_depth_message(self, _, message):
        """
        Parse and apply a single diff-depth WebSocket message to the local book.

        Processing steps:

        1. **Subscription confirmation** — if the message contains only
           ``"id"`` and ``"result"`` (no bid/ask data), log the confirmation
           and return immediately.
        2. **Stale-update guard** — discard any update whose ``u`` (final
           update ID) is less than or equal to the last known
           ``lastUpdateId``, keeping the book monotonically consistent.
        3. **Bid/ask merge** — for every price level in ``"b"`` (bids) and
           ``"a"`` (asks): remove the level if quantity is 0, otherwise
           insert/update it in ``state.local_book``.
        4. **Snapshot append** — record the current best bid price and ask price
           (float), their respective quantities (float), together with the event
           timestamp and update ID into ``state.history_order_book`` as
           ``{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}``.
        5. **Quote calculation (throttled)** — call ``calculate_best_quote``
           every ``QUOTE_EVERY_N_TICKS`` ticks (default 10 ≈ once per second)
           rather than on every 100 ms tick, reducing CPU work and console
           noise.  The local book itself is still updated on every tick.

        Args:
            _: The WebSocket client instance (unused).
            message (str): Raw JSON string received from the Binance diff-depth
                stream.
        """
        data = json.loads(message)

        # Skip the subscription confirmation message (only has 'id' and 'result')
        if "result" in data and "b" not in data:
            logging.info("WebSocket subscription confirmed.")
            return
        # SYNC LOGIC: Ignore updates that are older than our snapshot
        if data["u"] <= self.state.local_book["lastUpdateId"]:
            return
        # 1. Logic to sync with lastUpdateId goes here
        # 2. Update local_book dictionary — bids ('b') and asks ('a')
        with self.state.thread_lock:
            # update bids
            for price, qty in data.get("b", []):
                if float(qty) == 0:
                    self.state.local_book["bids"].pop(price, None)
                else:
                    self.state.local_book["bids"][price] = qty
            # update asks
            for price, qty in data.get("a", []):
                if float(qty) == 0:
                    self.state.local_book["asks"].pop(price, None)
                else:
                    self.state.local_book["asks"][price] = qty

            self.state.local_book["lastUpdateId"] = data["u"]
            # 3. Append the best [bid/ask] to history_order_book for strategy evaluation
            if self.state.local_book["bids"] and self.state.local_book["asks"]:
                best_bid_key = max(
                    self.state.local_book["bids"].keys(), key=float
                )  # str key for dict lookup
                best_ask_key = min(
                    self.state.local_book["asks"].keys(), key=float
                )  # str key for dict lookup
                self.state.history_order_book.append(
                    {
                        "timestamp": data["E"],
                        "lastUpdateId": data["u"],
                        "best_bid": float(best_bid_key),
                        "best_ask": float(best_ask_key),
                        "volume_best_bid": float(
                            self.state.local_book["bids"][best_bid_key]
                        ),
                        "volume_best_ask": float(
                            self.state.local_book["asks"][best_ask_key]
                        ),
                    }
                )

            # Now your strategy logic can always read from 'local_book'
            # which is updated in real-time (no 1-second lag!)
            self._tick_count += 1
            if self._tick_count % QUOTE_EVERY_N_TICKS == 0:
                calculate_best_quote(self.state.local_book)

    # def handle_balance_message(self, _, message):
    #     """
    #     Parse and apply an ``outboundAccountPosition`` event to the live balance state.
    #
    #     .. deprecated::
    #         **Superseded by** ``OrderExecutor._handle_balance_update``.
    #
    #         Real-time balance updates now arrive on the same WebSocket API
    #         connection used for order placement (``session.logon`` →
    #         ``userDataStream.subscribe`` → ``outboundAccountPosition`` push).
    #         This method is preserved for reference but is **not wired** to any
    #         stream in the current architecture.
    #
    #     Processing steps (when active):
    #
    #     1. **Event filter** — silently ignores any event whose ``"e"`` field
    #        is not ``"outboundAccountPosition"``.
    #     2. **Balance update** — under ``state.thread_balance_lock``, iterates
    #        over the ``"B"`` (balances) array and updates
    #        ``state.balance_status[asset]`` for ``CRYPTOCCY`` and ``CCY``.
    #        Only the ``"f"`` (free) field is stored.
    #     3. **Logging** — logs the refreshed free balances while holding the
    #        lock.
    #
    #     Args:
    #         _: The WebSocket client instance (unused).
    #         message (str): Raw JSON string from a Binance User Data Stream.
    #     """
    #     data = json.loads(message)
    #     if "e" in data and data["e"] == "outboundAccountPosition":
    #         with self.state.thread_balance_lock:
    #             for asset_data in data.get("B", []):
    #                 asset = asset_data["a"]
    #                 if asset in self.state.balance_status:
    #                     self.state.balance_status[asset] = float(asset_data["f"])
    #             logging.info(
    #                 "Balance status - %s: %s | %s: %s",
    #                 CRYPTOCCY,
    #                 self.state.balance_status[CRYPTOCCY],
    #                 CCY,
    #                 self.state.balance_status[CCY],
    #             )
