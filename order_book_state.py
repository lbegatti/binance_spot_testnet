from collections import deque
import threading

from config_parameters import HISTORY_MAXLEN


class OrderBookState:
    """
    Central shared state container for the real-time order book.

    Holds the current best bids and asks (``local_book``), a rolling history of
    order book snapshots (``history_order_book``), and a threading lock so that
    both ``MessageHandler`` and ``AnalysisEngine`` can safely read/write the same
    data without race conditions.

    Attributes:
        local_book (dict): Live order book with keys ``"bids"``, ``"asks"``, and
            ``"lastUpdateId"``.  Bids and asks are stored as ``{price: qty}`` dicts.
        history_order_book (deque): Rolling window of snapshots, each containing
            ``timestamp``, ``lastUpdateId``, ``best_bids``, and ``best_asks``.
        thread_lock (threading.Lock): Mutex shared by all consumers of this state.
    """

    def __init__(self, maxlen=HISTORY_MAXLEN):
        """
        Initialize the order book state.

        Args:
            maxlen (int): Maximum number of snapshots to keep in
                ``history_order_book``.  At 100 ms update intervals, 6000 entries
                correspond to roughly 10 minutes of history.  Defaults to 6000.
        """
        self.local_book = {"bids": {}, "asks": {}, "lastUpdateId": 0}
        self.history_order_book = deque(
            maxlen=maxlen
        )  # Store recent order book snapshots
        self.thread_lock = (
            threading.Lock()
        )  # Ensure thread-safe updates to the local book
