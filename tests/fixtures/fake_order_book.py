"""Deterministic order-book builders for the book_utils tests."""

from __future__ import annotations


def make_book(
    bids: dict[float, float], asks: dict[float, float]
) -> tuple[dict[str, str], dict[str, str]]:
    """Convert ``{price: qty}`` float dicts to the ``{str: str}`` shape the live
    order book actually stores.

    ``strategy.book_utils.build_levels`` calls ``float(p)`` / ``float(q)``
    internally, so string keys/values faithfully mirror what the Binance
    diff-depth WebSocket stream writes into ``local_book``.
    """
    bids_str = {f"{p}": f"{q}" for p, q in bids.items()}
    asks_str = {f"{p}": f"{q}" for p, q in asks.items()}
    return bids_str, asks_str
