from order_book_state import OrderBookState
import threading
import logging

from config import HFT_INTERVAL, HIST_INTERVAL, MIN_SNAPSHOTS


class AnalysisEngine:
    """
    Strategy engine that consumes shared ``OrderBookState`` to run both
    high-frequency and historical analyses in separate background threads.

    The class is intentionally decoupled from ``MessageHandler``: the two
    communicate exclusively through the injected ``OrderBookState`` instance,
    which guarantees that both threads always operate on the same data and
    the same lock.

    Both loops respect a shared ``threading.Event`` (``stop_event``) so that
    ``websocket_spot_main.py`` can terminate the session cleanly after the
    user-defined duration has elapsed.

    Attributes:
        state (OrderBookState): Shared order book state, injected at construction.
        stop_event (threading.Event): Shared event; when set, both loops exit
            at their next scheduled wake-up.
    """

    def __init__(self, state: OrderBookState, stop_event: threading.Event):
        """
        Args:
            state (OrderBookState): The shared order book state instance that
                provides ``local_book``, ``history_order_book``, and
                ``thread_lock``.
            stop_event (threading.Event): Set by the session driver
                (``websocket_spot_main.py``) when the session duration has
                elapsed.  Both analysis loops check this event on every
                iteration and exit gracefully when it is set.
        """
        self.state = state
        self.stop_event = stop_event

    def htf_analysis(self):
        """
        Periodically inspect the live order book and apply a high-frequency
        trading strategy.

        Runs every 5 seconds until ``stop_event`` is set.  On each wake-up it
        acquires ``state.thread_lock``, takes a local copy of the current bids
        and asks, then releases the lock before executing any heavier strategy
        logic — keeping the critical section as short as possible.

        Logs a message and skips the iteration if no bids are present yet
        (e.g. the WebSocket snapshot has not arrived yet).

        Exits cleanly when ``stop_event`` is set, logging how many iterations
        were completed.
        """
        iteration = 0

        logging.info("HFT analysis loop started (interval: %ds).", HFT_INTERVAL)
        while not self.stop_event.is_set():
            self.stop_event.wait(HFT_INTERVAL)  # interruptible sleep
            if self.stop_event.is_set():
                break

            with self.state.thread_lock:
                if not self.state.local_book["bids"]:
                    logging.info("HFT: no bids available yet, skipping iteration.")
                    continue
                snaps_bids = self.state.local_book["bids"].copy()
                snaps_asks = self.state.local_book["asks"].copy()

            # Lock is released — do heavier work outside the critical section.
            best_bid = max(snaps_bids.keys(), key=float)
            best_ask = min(snaps_asks.keys(), key=float)
            iteration += 1

            # TODO: implement full strategy logic (metrics → indicators → scores → quote)
            logging.info(
                "HFT Analysis #%d — Best Bid: %s | Best Ask: %s",
                iteration,
                best_bid,
                best_ask,
            )

        logging.info("HFT analysis loop stopped after %d iteration(s).", iteration)

    def historical_analysis(self):
        """
        Periodically analyse the rolling window of order book snapshots stored
        in ``state.history_order_book`` to identify longer-term patterns.

        Runs every 10 minutes until ``stop_event`` is set.  If fewer than 100
        snapshots have accumulated the iteration is skipped and a log message
        is emitted so the operator knows the engine is still warming up.

        Intended use-cases include backtesting sub-strategies, computing
        spread distributions, or detecting regime changes over the last 10
        minutes of data.

        Exits cleanly when ``stop_event`` is set, logging how many iterations
        were completed.
        """
        iteration = 0

        logging.info(
            "Historical analysis loop started (interval: %ds / %.0f min).",
            HIST_INTERVAL,
            HIST_INTERVAL / 60,
        )
        while not self.stop_event.is_set():
            self.stop_event.wait(HIST_INTERVAL)  # interruptible sleep
            if self.stop_event.is_set():
                break

            snap_count = len(self.state.history_order_book)
            if snap_count < MIN_SNAPSHOTS:
                logging.info(
                    "Historical: only %d snapshots available, need ≥%d — skipping.",
                    snap_count,
                    MIN_SNAPSHOTS,
                )
                continue

            iteration += 1
            # TODO: implement historical analysis logic using history_order_book.
            logging.info(
                "Historical Analysis #%d — %d snapshots in window.",
                iteration,
                snap_count,
            )

        logging.info(
            "Historical analysis loop stopped after %d iteration(s).", iteration
        )
