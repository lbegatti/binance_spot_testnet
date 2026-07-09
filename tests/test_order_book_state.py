"""Tier B tests for core/order_book_state.py — the shared state container."""

from collections import deque

from config_parameters import CCY, CRYPTOCCY, HISTORY_MAXLEN
from core.order_book_state import OrderBookState


def test_default_init_shapes():
    """A fresh state starts with an empty book and zeroed free/locked balance
    dicts keyed by the configured crypto and quote currencies."""
    s = OrderBookState()
    assert s.local_book == {"bids": {}, "asks": {}, "lastUpdateId": 0}
    assert s.balance_status == {CRYPTOCCY: 0.0, CCY: 0.0}
    assert s.balance_locked == {CRYPTOCCY: 0.0, CCY: 0.0}


def test_history_is_deque_with_default_maxlen():
    """The order-book history is a bounded deque sized by HISTORY_MAXLEN, so
    memory stays capped over a long session."""
    s = OrderBookState()
    assert isinstance(s.history_order_book, deque)
    assert s.history_order_book.maxlen == HISTORY_MAXLEN


def test_custom_maxlen_evicts_oldest():
    """Once the deque is full, appending a new snapshot evicts the oldest one
    (FIFO), confirming the bound is enforced."""
    s = OrderBookState(maxlen=2)
    s.history_order_book.append("a")
    s.history_order_book.append("b")
    s.history_order_book.append("c")  # evicts "a"
    assert list(s.history_order_book) == ["b", "c"]


def test_locks_are_usable_and_distinct():
    """The book lock and the balance lock are separate, working lock objects,
    so book updates and balance updates can be guarded independently."""
    s = OrderBookState()
    assert s.thread_lock is not s.thread_balance_lock
    for lock in (s.thread_lock, s.thread_balance_lock):
        assert hasattr(lock, "acquire") and hasattr(lock, "release")
        assert lock.acquire(blocking=False) is True  # fresh lock is free
        lock.release()


def test_instances_do_not_share_mutable_state():
    """Each instance gets its own balance dicts (no shared class-level
    mutable default), so mutating one instance never leaks into another."""
    a = OrderBookState()
    b = OrderBookState()
    a.balance_status[CCY] = 123.0
    assert b.balance_status[CCY] == 0.0  # no shared dict between instances
