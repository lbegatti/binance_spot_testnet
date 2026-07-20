"""Tier B tests for the live pyramiding logic in ``strategy/analysis.py``.

These lock in the multi-leg position accounting added to the live path — the
piece a live session cannot reliably exercise on demand
(it only fires when a real VWAP dip lands while the regime allows a BUY, which
did not happen in the quiet 2026-07-12 evening validation run).

Scope: the four cost-basis helpers that the exposure gate drives
(``_add_leg_to_basis``, ``_back_out_pending_leg``, ``_reset_position_flat``,
``_recompute_avg_entry``), the inherited-BTC pre-arm, and a simulation of the
executor's per-leg cash sizing that validates the ``MAX_PYRAMID_LEGS`` /
cash-reserve-floor interaction (the leg cap must be high enough for the floor
to bind).

Every number below is verifiable by hand.  ``AnalysisEngine.__init__`` only
reads ``state.balance_status`` under a lock and stores its collaborators, so we
build it with light fakes — no network, no order book, no HMM.
"""

import threading
import types

import pytest

from config_parameters import (
    MAX_POSITION_PCT,
    MAX_PYRAMID_LEGS,
    MIN_CASH_RESERVE_PCT,
    CRYPTOCCY,
)
from strategy.analysis import AnalysisEngine


def _make_engine(initial_btc: float = 0.0, initial_avg_entry_price: float = 0.0):
    """Build an AnalysisEngine with fakes for every collaborator.

    ``__init__`` only touches ``state.balance_status`` (under
    ``thread_balance_lock``) and otherwise just stores references, so a
    ``SimpleNamespace`` state + unittest ``Mock`` executor/regime_director are
    enough to exercise the pyramiding helpers in isolation.
    """
    state = types.SimpleNamespace(
        balance_status={CRYPTOCCY: initial_btc},
        thread_balance_lock=threading.Lock(),
    )
    return AnalysisEngine(
        state=state,
        stop_event=threading.Event(),
        executor=types.SimpleNamespace(),  # never called by the helpers
        regime_director=types.SimpleNamespace(),
        initial_avg_entry_price=initial_avg_entry_price,
    )


# ── _add_leg_to_basis ───────────────────────────────────────────────────────


def test_single_leg_sets_basis_and_avg_entry():
    """One dispatched leg opens the position: qty, cost, leg count, the pending
    slot, and the stop-loss anchor (avg entry) all reflect that single leg."""
    eng = _make_engine()
    eng._add_leg_to_basis(price=60_000.0, qty=0.5)

    assert eng._position_open is True
    assert eng._pyramid_legs == 1
    assert eng._position_qty_btc == pytest.approx(0.5)
    assert eng._position_cost_usdt == pytest.approx(30_000.0)
    assert eng._avg_entry_price == pytest.approx(60_000.0)
    # The single-slot pending record mirrors this leg so it can be backed out.
    assert eng._pending_leg_qty == pytest.approx(0.5)
    assert eng._pending_leg_cost == pytest.approx(30_000.0)


def test_two_legs_volume_weight_the_avg_entry():
    """Stacking a second leg at a different price makes the anchor the
    volume-weighted average, NOT the last leg's price — the property the
    stop-loss depends on.  (0.5·60k + 1.0·63k) / 1.5 = 62 000."""
    eng = _make_engine()
    eng._add_leg_to_basis(price=60_000.0, qty=0.5)
    eng._add_leg_to_basis(price=63_000.0, qty=1.0)

    assert eng._pyramid_legs == 2
    assert eng._position_qty_btc == pytest.approx(1.5)
    assert eng._position_cost_usdt == pytest.approx(93_000.0)
    assert eng._avg_entry_price == pytest.approx(62_000.0)


def test_leg_count_reaches_the_cap_at_MAX_PYRAMID_LEGS():
    """Stacking legs one-by-one reaches exactly ``MAX_PYRAMID_LEGS``; at that
    point the exposure gate's ``_pyramid_legs >= MAX_PYRAMID_LEGS`` guard is
    satisfied, so the next BUY would be skipped rather than dispatched."""
    eng = _make_engine()
    for _ in range(MAX_PYRAMID_LEGS):
        eng._add_leg_to_basis(price=60_000.0, qty=0.01)

    assert eng._pyramid_legs == MAX_PYRAMID_LEGS
    assert eng._pyramid_legs >= MAX_PYRAMID_LEGS  # gate would now block


# ── _back_out_pending_leg ───────────────────────────────────────────────────


def test_back_out_last_leg_keeps_earlier_legs_open():
    """A stale/unfilled cancel reverses ONLY the most recent (pending) leg.
    After stacking two legs and backing the second out, the first leg's basis
    survives, the position stays open, and the anchor returns to leg 1."""
    eng = _make_engine()
    eng._add_leg_to_basis(price=60_000.0, qty=0.5)
    eng._add_leg_to_basis(price=63_000.0, qty=1.0)

    eng._back_out_pending_leg()

    assert eng._position_open is True
    assert eng._pyramid_legs == 1
    assert eng._position_qty_btc == pytest.approx(0.5)
    assert eng._position_cost_usdt == pytest.approx(30_000.0)
    assert eng._avg_entry_price == pytest.approx(60_000.0)
    # Pending slot cleared so a second back-out cannot double-subtract.
    assert eng._pending_leg_qty == 0.0
    assert eng._pending_leg_cost == 0.0


def test_back_out_only_leg_goes_flat():
    """Backing out the sole open leg empties the book → full flat reset."""
    eng = _make_engine()
    eng._add_leg_to_basis(price=60_000.0, qty=0.5)

    eng._back_out_pending_leg()

    assert eng._position_open is False
    assert eng._pyramid_legs == 0
    assert eng._position_qty_btc == 0.0
    assert eng._position_cost_usdt == 0.0
    assert eng._avg_entry_price == 0.0


# ── _reset_position_flat ────────────────────────────────────────────────────


def test_reset_position_flat_clears_everything():
    """A full close (SELL fill / stop-loss) wipes all pyramiding state so the
    next session-tick starts genuinely flat."""
    eng = _make_engine()
    eng._add_leg_to_basis(price=60_000.0, qty=0.5)
    eng._add_leg_to_basis(price=61_000.0, qty=0.5)

    eng._reset_position_flat()

    assert eng._position_open is False
    assert eng._pyramid_legs == 0
    assert eng._position_qty_btc == 0.0
    assert eng._position_cost_usdt == 0.0
    assert eng._pending_leg_qty == 0.0
    assert eng._pending_leg_cost == 0.0
    assert eng._avg_entry_price == 0.0


# ── _recompute_avg_entry edge case ──────────────────────────────────────────


def test_recompute_avg_entry_is_zero_when_flat():
    """With no BTC held the anchor is 0.0 (guards a divide-by-zero and disables
    the stop-loss floor rather than emitting a bogus entry price)."""
    eng = _make_engine()
    eng._position_qty_btc = 0.0
    eng._position_cost_usdt = 0.0
    eng._recompute_avg_entry()
    assert eng._avg_entry_price == 0.0


# ── inherited-BTC pre-arm ───────────────────────────────────────────────────


def test_inherited_btc_prearms_one_leg():
    """When the account already holds BTC at startup (FLATTEN_ON_START=False),
    __init__ folds it in as a single pre-existing leg anchored at the supplied
    cost basis, so the exposure gate and stop-loss treat it like any other leg."""
    eng = _make_engine(initial_btc=0.25, initial_avg_entry_price=58_000.0)

    assert eng._position_open is True
    assert eng._pyramid_legs == 1
    assert eng._position_qty_btc == pytest.approx(0.25)
    assert eng._position_cost_usdt == pytest.approx(0.25 * 58_000.0)
    assert eng._avg_entry_price == pytest.approx(58_000.0)


def test_dust_btc_below_threshold_does_not_prearm():
    """A sub-0.0001 BTC dust balance (e.g. the un-sellable 1e-05 seen live) is
    ignored — the session starts flat, not pre-armed on un-tradeable dust."""
    eng = _make_engine(initial_btc=1e-05)

    assert eng._position_open is False
    assert eng._pyramid_legs == 0
    assert eng._position_qty_btc == 0.0


# ── reserve floor / leg-cap interaction ─────────────────────────────────────


def test_pyramid_leg_cap_is_high_enough_for_the_reserve_floor_to_bind():
    """Simulate the executor's "MAX_POSITION_PCT of REMAINING free cash" leg
    sizing that the exposure gate dispatches against, and confirm the leg cap is
    high enough that the cash-reserve floor is the binding constraint:

      invested-after-n-legs / starting-cash  ≈  1 − (1 − MAX_POSITION_PCT)ⁿ

    At 20 % legs this reaches ~93 % after MAX_PYRAMID_LEGS = 12 legs — past the
    (1 − MIN_CASH_RESERVE_PCT) ceiling — so the reserve floor binds.  Written
    against the live config so it stays correct as the reserve moves: it derives
    the minimum legs needed to cross the ceiling and confirms the leg cap clears
    it while one leg fewer still falls short.
    """
    starting_cash = 100_000.0
    price = 60_000.0
    eng = _make_engine()

    cash = starting_cash
    for _ in range(MAX_PYRAMID_LEGS):
        leg_usdt = cash * MAX_POSITION_PCT  # of remaining cash
        eng._add_leg_to_basis(price=price, qty=leg_usdt / price)
        cash -= leg_usdt

    invested_frac = eng._position_cost_usdt / starting_cash

    # Raw (unclamped) decay: 1 − (1 − MAX_POSITION_PCT)^legs
    assert invested_frac == pytest.approx(
        1.0 - (1.0 - MAX_POSITION_PCT) ** MAX_PYRAMID_LEGS, abs=1e-9
    )
    # Cap is high enough that the raw decay passes the invested ceiling, so the
    # reserve floor (not the leg cap) is the true constraint live.
    assert invested_frac >= (1.0 - MIN_CASH_RESERVE_PCT)
    # Derive the minimum legs needed to cross the invested ceiling; confirm the
    # leg cap clears it and that one leg fewer would fall short (floor inert).
    import math

    legs_to_reach = math.ceil(
        math.log(MIN_CASH_RESERVE_PCT) / math.log(1.0 - MAX_POSITION_PCT)
    )
    assert MAX_PYRAMID_LEGS >= legs_to_reach
    assert (1.0 - (1.0 - MAX_POSITION_PCT) ** (legs_to_reach - 1)) < (
        1.0 - MIN_CASH_RESERVE_PCT
    )
