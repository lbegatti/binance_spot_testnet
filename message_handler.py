import json
import logging

from best_quote_calculator import calculate_best_quote
from order_book_state import OrderBookState


class MessageHandler:
    """
    WebSocket callback handler responsible for maintaining the local order book
    in real time.

    Every diff-depth message received from the Binance stream is processed here:
    stale updates are discarded, live bids/asks are merged into
    ``state.local_book``, and each valid update produces a new snapshot that is
    appended to ``state.history_order_book`` for downstream analysis.

    Attributes:
        state (OrderBookState): Shared order book state, injected at
            construction and also consumed by ``AnalysisEngine``.
    """

    def __init__(self, state: OrderBookState):
        """
        Args:
            state (OrderBookState): The shared order book state instance that
                provides ``local_book``, ``history_order_book``, and
                ``thread_lock``.
        """
        self.state = state

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
        4. **Snapshot append** — record the current best bid and ask together
           with the event timestamp and update ID into
           ``state.history_order_book``.
        5. **Quote calculation** — call ``calculate_best_quote`` so the
           strategy layer always has an up-to-date quote after every message.

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
                self.state.history_order_book.append(
                    {
                        "timestamp": data["E"],
                        "lastUpdateId": data["u"],
                        "best_bids": max(
                            self.state.local_book["bids"].keys(), key=float
                        ),
                        "best_asks": min(
                            self.state.local_book["asks"].keys(), key=float
                        ),
                    }
                )

            # Now your strategy logic can always read from 'local_book'
            # which is updated in real-time (no 1-second lag!)
            calculate_best_quote(self.state.local_book)
