"""Tier B tests for core/message_handler.py — diff-depth application and the
gap-recovery / resync path.

The WebSocket callback ``handle_depth_message(self, _, message)`` takes a raw
JSON string, so every event below is built with ``_event(...)`` and dumped.
The REST client used for resync is mocked via pytest-mock — no network calls.
"""

import json

from config_parameters import DEPTH_RESYNC_MIN_INTERVAL_SEC
from core.message_handler import MessageHandler
from core.order_book_state import OrderBookState


def _event(U, u, bids=None, asks=None, E=1):
    """Build a Binance diff-depth event JSON string (U=first id, u=final id)."""
    return json.dumps(
        {
            "e": "depthUpdate",
            "E": E,
            "s": "BTCUSDT",
            "U": U,
            "u": u,
            "b": bids or [],
            "a": asks or [],
        }
    )


def _seeded_state(last_update_id=100):
    """A state whose local_book is seeded like websocket_main's startup snapshot."""
    state = OrderBookState()
    state.local_book = {
        "bids": {"100.0": "1.0"},
        "asks": {"101.0": "1.0"},
        "lastUpdateId": last_update_id,
    }
    return state


def _snapshot(last_update_id=500):
    """A REST depth() snapshot payload (bids/asks are [price, qty] string pairs)."""
    return {
        "lastUpdateId": last_update_id,
        "bids": [["200.0", "2.0"]],
        "asks": [["201.0", "3.0"]],
    }


# ── happy path ─────────────────────────────────────────────────────────────


def test_contiguous_event_applies_and_advances_last_update_id():
    """A diff whose first id == last+1 is contiguous: it is applied to the
    book and lastUpdateId advances, with no resync."""
    state = _seeded_state(last_update_id=100)
    handler = MessageHandler(state=state, rest_client=None)
    # U == last + 1 → contiguous
    handler.handle_depth_message(None, _event(U=101, u=105, bids=[["100.0", "2.5"]]))
    assert state.local_book["lastUpdateId"] == 105
    assert state.local_book["bids"]["100.0"] == "2.5"  # applied
    assert handler._resync_count == 0


def test_zero_quantity_removes_level():
    """Quantity "0" is Binance's convention for "this price level is gone",
    so the level is deleted from the book rather than stored as zero."""
    state = _seeded_state(last_update_id=100)
    handler = MessageHandler(state=state, rest_client=None)
    handler.handle_depth_message(None, _event(U=101, u=102, bids=[["100.0", "0"]]))
    assert "100.0" not in state.local_book["bids"]  # level removed


# ── stale / duplicate ──────────────────────────────────────────────────────


def test_stale_event_is_dropped(mocker):
    """A diff whose final id <= lastUpdateId was already applied, so it is
    ignored: the book stays unchanged and no REST resync is triggered."""
    state = _seeded_state(last_update_id=100)
    client = mocker.Mock()
    handler = MessageHandler(state=state, rest_client=client)
    # u <= last → already applied, dropped without touching the book or REST
    handler.handle_depth_message(None, _event(U=90, u=100, bids=[["100.0", "9.9"]]))
    assert state.local_book["lastUpdateId"] == 100
    assert state.local_book["bids"]["100.0"] == "1.0"  # unchanged
    client.depth.assert_not_called()


# ── gap → resync ───────────────────────────────────────────────────────────


def test_gap_triggers_resync_from_rest_snapshot(mocker):
    """A gap (first id > last+1) means frames were missed: the handler
    rebuilds the whole book from a fresh REST snapshot and drops the
    gapped event. The REST client is mocked, so no network call happens."""
    state = _seeded_state(last_update_id=100)
    client = mocker.Mock()
    client.depth.return_value = _snapshot(last_update_id=500)
    handler = MessageHandler(state=state, rest_client=client)

    # U (150) > last + 1 (101) → gap → resync, then drop this event
    handler.handle_depth_message(None, _event(U=150, u=160))

    client.depth.assert_called_once()
    assert state.local_book["lastUpdateId"] == 500  # rebuilt to snapshot
    assert state.local_book["bids"] == {"200.0": "2.0"}  # snapshot bids
    assert state.local_book["asks"] == {"201.0": "3.0"}
    assert handler._resync_count == 1


def test_resync_cooldown_prevents_second_rest_call(mocker):
    """After one resync, a second gap within DEPTH_RESYNC_MIN_INTERVAL_SEC
    is suppressed (no repeat REST call). Rewinding _last_resync_ts past the
    cooldown lets the next gap resync again — proving the timer gates it."""
    state = _seeded_state(last_update_id=100)
    client = mocker.Mock()
    client.depth.return_value = _snapshot(last_update_id=500)
    handler = MessageHandler(state=state, rest_client=client)

    # First gap → resync (last becomes 500).
    handler.handle_depth_message(None, _event(U=150, u=160))
    # Second gap immediately after (U 700 > 501) → within cooldown → no 2nd call.
    handler.handle_depth_message(None, _event(U=700, u=710))

    assert client.depth.call_count == 1
    assert handler._resync_count == 1
    # Force the cooldown to have elapsed → next gap resyncs again.
    handler._last_resync_ts -= DEPTH_RESYNC_MIN_INTERVAL_SEC + 1
    handler.handle_depth_message(None, _event(U=700, u=710))
    assert client.depth.call_count == 2


def test_gap_without_rest_client_does_not_crash():
    """A gap with no REST client available is logged and the event dropped:
    the book is left unchanged and no exception escapes."""
    state = _seeded_state(last_update_id=100)
    handler = MessageHandler(state=state, rest_client=None)
    # Gap but no client → logs + drops the event, book left as-is, no exception.
    handler.handle_depth_message(None, _event(U=150, u=160))
    assert state.local_book["lastUpdateId"] == 100  # unchanged
    assert handler._resync_count == 0


# ── subscription confirmation (regression guard) ───────────────────────────


def test_subscription_confirmation_is_ignored():
    """The "{result, id}" subscription-ack frame Binance sends on connect is
    not a depth update and must leave the book untouched (regression guard)."""
    state = _seeded_state(last_update_id=100)
    handler = MessageHandler(state=state, rest_client=None)
    handler.handle_depth_message(None, json.dumps({"result": None, "id": 1}))
    assert state.local_book["lastUpdateId"] == 100  # untouched
