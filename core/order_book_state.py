from collections import deque
import threading

from config_parameters import HISTORY_MAXLEN, CRYPTOCCY, CCY


class OrderBookState:
    """
    Central shared state container for the real-time order book and account balances.

    Holds the current best bids and asks (``local_book``), a rolling history of
    order book snapshots (``history_order_book``), the live account balances for
    the traded pair (``balance_status``), and two dedicated threading locks so
    that all consumers can safely read/write without race conditions.

    Two locks are used intentionally to avoid contention between the
    high-frequency WebSocket order-book path (updating every 100 ms) and the
    lower-frequency balance-update path (driven by trade fills via the User
    Data Stream):

    * ``thread_lock`` — serializes access to ``local_book`` and
      ``history_order_book``.
    * ``thread_balance_lock`` — serializes access to ``balance_status``
      independently of order-book updates.

    Attributes:
        local_book (dict): Live order book with keys ``"bids"``, ``"asks"``, and
            ``"lastUpdateId"``.  Bids and asks are stored as ``{price: qty}`` dicts.
        history_order_book (deque): Rolling window of snapshots, each containing
            ``timestamp`` (int, Unix ms), ``lastUpdateId`` (int),
            ``best_bid`` (float, highest bid price), ``best_ask`` (float, lowest ask price),
            ``volume_best_bid`` (float, quantity at best bid), and
            ``volume_best_ask`` (float, quantity at best ask).
        balance_status (dict): Live free balances keyed by ``CRYPTOCCY`` and
            ``CCY`` (e.g. ``{"BTC": 0.0, "USDT": 0.0}``).  Seeded from the
            initial REST balance fetch in ``websocket_main.py`` and kept
            up to date by ``OrderExecutor._handle_balance_update`` via
            ``outboundAccountPosition`` push events on the WebSocket API
            connection (``session.logon`` → ``userDataStream.subscribe``).
            Holds the *free* (non-locked) quantity only; the locked quantity is
            kept separately in ``balance_locked`` (below).
        balance_locked (dict): Live *locked* balances keyed by ``CRYPTOCCY`` and
            ``CCY`` (quantity tied up in resting LIMIT orders).  Maintained
            alongside ``balance_status`` by the same two update paths.  Read
            ONLY by the end-of-session equity snapshot so a position locked in a
            resting order (e.g. BTC in a LIMIT SELL) still counts toward equity;
            trading logic never reads it.
        thread_lock (threading.Lock): Mutex that serializes access to
            ``local_book`` and ``history_order_book``.  ``MessageHandler``
            acquires it to write; ``AnalysisEngine`` acquires it to take a
            read-only snapshot before any heavy computation.
        thread_balance_lock (threading.Lock): Mutex that serializes access to
            ``balance_status``.  ``OrderExecutor._handle_balance_update``
            acquires it to write; ``AnalysisEngine.low_latency_analysis``
            acquires it to read before each iteration.
    """

    def __init__(self, maxlen=HISTORY_MAXLEN):
        """
        Initialize the order book state with empty books and zero balances.

        ``balance_status`` is initialized to zero for both assets and must be
        seeded by the caller (``websocket_main.py``) immediately after
        construction using the balances returned by the Binance REST API,
        before any analysis threads are started.

        Args:
            maxlen (int): Maximum number of snapshots to keep in
                ``history_order_book``.  At 100 ms update intervals, 3000 entries
                correspond to roughly 5 minutes of history.  Defaults to
                ``HISTORY_MAXLEN`` (3000).
        """
        self.local_book = {"bids": {}, "asks": {}, "lastUpdateId": 0}
        self.balance_status = {
            CRYPTOCCY: 0.0,
            CCY: 0.0,
        }  # FREE balances — seeded by websocket_main.py before threads start
        # LOCKED balances (BTC/USDT tied up in resting LIMIT orders).  Tracked
        # in parallel ONLY so the end-of-session equity snapshot can mark the
        # full position (free + locked) to market.  Trading still reads
        # balance_status (free) — you can only spend what is free.
        self.balance_locked = {
            CRYPTOCCY: 0.0,
            CCY: 0.0,
        }
        self.history_order_book = deque(
            maxlen=maxlen
        )  # Store recent order book snapshots
        self.thread_lock = (
            threading.Lock()
        )  # Serializes local_book and history_order_book
        self.thread_balance_lock = threading.Lock()  # Serializes balance_status
