"""Tier B tests for backtest/pnl.py — buy-and-hold, the P&L simulator, and the
round-trip pairing helper.

All scenarios are hand-built so every number below is verifiable by hand.
BUY sizing is capped at ``MAX_POSITION_PCT`` (0.20) of available USDT — with a
1000-USDT start and a 100.0 entry price this yields up to 2.0 BTC per leg (or
less when the synthetic book quantity ``buy_qty`` is the binding constraint).
"""

import numpy as np
import pandas as pd
import pytest

from backtest.pnl import _pair_round_trips, compute_buy_and_hold, simulate_pnl
from tests.fixtures.fake_klines import make_signals


# ── compute_buy_and_hold ───────────────────────────────────────────────────


def test_buy_and_hold_simple_no_fee():
    """The B&H benchmark buys everything at the first close and marks to the
    last close; every field (entry, exit, BTC held, equity, return) is
    checked against hand math with fees off."""
    sig = pd.DataFrame({"close": [100.0, 150.0, 200.0]})
    bnh = compute_buy_and_hold(sig, initial_usdt=1000.0, initial_btc=0.0, fee_rate=0.0)
    assert bnh["bnh_entry_price"] == 100.0
    assert bnh["bnh_exit_price"] == 200.0
    assert bnh["bnh_btc_held"] == pytest.approx(10.0)  # 1000 / 100
    assert bnh["bnh_final_equity_usdt"] == pytest.approx(2000.0)  # 10 BTC × 200
    assert bnh["bnh_total_return_pct"] == pytest.approx(100.0)


def test_buy_and_hold_all_nan_close_returns_nan():
    """With no valid closes there is no price to buy or mark against, so the
    B&H return is NaN rather than a bogus number."""
    sig = pd.DataFrame({"close": [np.nan, np.nan]})
    bnh = compute_buy_and_hold(sig, initial_usdt=1000.0, initial_btc=0.0, fee_rate=0.0)
    assert np.isnan(bnh["bnh_total_return_pct"])


# ── simulate_pnl ───────────────────────────────────────────────────────────


def test_simulate_pnl_single_round_trip():
    """A BUY then SELL forms one round trip: the simulator records both
    trades, ends with the expected equity, and reports one 100%-win round
    trip — the end-to-end happy path."""
    sig = make_signals(
        [
            {"signal": 1, "close": 100.0, "buy_qty": 1.0, "best_buy_micro": 100.0},
            {"signal": 0, "close": 110.0},
            {"signal": -1, "close": 120.0, "sell_qty": 1.0, "best_sell_micro": 120.0},
        ]
    )
    trades, equity, stats = simulate_pnl(
        sig, initial_usdt=1000.0, initial_btc=0.0, fee_rate=0.0
    )
    # BUY 1 BTC @ 100 (usdt 900, btc 1), SELL 1 BTC @ 120 (usdt 1020, btc 0)
    assert list(trades["side"]) == ["BUY", "SELL"]
    assert equity["equity"].iloc[-1] == pytest.approx(1020.0)
    assert stats["n_round_trips"] == 1
    assert stats["total_return_pct"] == pytest.approx(2.0)  # (1020-1000)/1000
    assert stats["win_rate_pct"] == pytest.approx(100.0)


def test_simulate_pnl_pyramids_second_buy():
    """Two consecutive BUY signals STACK (pyramid): each leg is sized at
    MAX_POSITION_PCT (20%) of the *remaining* cash, so both execute and no
    reserve-floor skip is recorded. A large book qty makes the budget bind:
    leg 1 spends 20% of 1000 (= 2.0 BTC @ 100); leg 2 spends 20% of the
    remaining 800 (= 160/101 BTC @ 101). The 20% reserve floor does not bind
    (cash stays above 20% of equity for both legs)."""
    sig = make_signals(
        [
            {"signal": 1, "close": 100.0, "buy_qty": 100.0, "best_buy_micro": 100.0},
            {"signal": 1, "close": 101.0, "buy_qty": 100.0, "best_buy_micro": 101.0},
        ]
    )
    trades, _equity, stats = simulate_pnl(
        sig, initial_usdt=1000.0, initial_btc=0.0, fee_rate=0.0
    )
    assert list(trades["side"]) == ["BUY", "BUY"]  # both legs pyramid
    assert stats["n_position_guard_skips"] == 0
    assert trades["quantity"].iloc[0] == pytest.approx(2.0)  # 200 / 100
    assert trades["quantity"].iloc[1] == pytest.approx(160.0 / 101.0)  # 160 / 101


def test_simulate_pnl_reserve_floor_blocks_buy():
    """When cash is already at/below MIN_CASH_RESERVE_PCT (20%) of mark-to-market
    equity, a BUY is suppressed (counted in n_position_guard_skips) so the book
    never invests past 80%. Here 100 USDT cash + 10 BTC @ 100 → equity 1100, and
    the 220 reserve floor (20% × 1100) already exceeds the 100 cash, so the BUY is
    refused and no trade fires."""
    sig = make_signals(
        [
            {"signal": 1, "close": 100.0, "buy_qty": 1.0, "best_buy_micro": 100.0},
        ]
    )
    trades, _equity, stats = simulate_pnl(
        sig, initial_usdt=100.0, initial_btc=10.0, fee_rate=0.0
    )
    assert stats["n_position_guard_skips"] == 1
    assert len(trades) == 0  # reserve floor blocked the only BUY


def test_simulate_pnl_stop_loss_fires():
    """When price drops below entry × (1 − stop_loss_pct) the adaptive stop
    forces an exit even with no SELL signal, logged as a SELL_STOP_LOSS
    trade and counted in the stop-loss stat."""
    sig = make_signals(
        [
            {"signal": 1, "close": 100.0, "buy_qty": 1.0, "best_buy_micro": 100.0},
            # 90 < entry 100 × (1 - 0.05) = 95 → adaptive stop-loss forces the exit
            {"signal": 0, "close": 90.0, "stop_loss_pct": 0.05},
        ]
    )
    trades, _equity, stats = simulate_pnl(
        sig, initial_usdt=1000.0, initial_btc=0.0, fee_rate=0.0
    )
    assert stats["n_stop_loss_fires"] == 1
    assert "SELL_STOP_LOSS" in set(trades["side"])


# ── _pair_round_trips ──────────────────────────────────────────────────────


def test_pair_round_trips_matches_buy_then_sell():
    """A BUY followed by a SELL is paired into one round trip with the
    correct entry/exit prices and realised P&L."""
    trades = pd.DataFrame(
        {
            "side": ["BUY", "SELL"],
            "fill_price": [100.0, 120.0],
            "quantity": [1.0, 1.0],
        },
        index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05"], utc=True),
    )
    rts = _pair_round_trips(trades, last_close=120.0, fee_rate=0.0)
    assert len(rts) == 1
    assert rts[0]["entry_price"] == 100.0
    assert rts[0]["exit_price"] == 120.0
    assert rts[0]["pnl_usdt"] == pytest.approx(20.0)


def test_pair_round_trips_orphan_sell_produces_no_round_trip():
    """A SELL with no preceding BUY has nothing to pair against, so it yields
    no round trip (guards against counting phantom trades)."""
    trades = pd.DataFrame(
        {"side": ["SELL"], "fill_price": [120.0], "quantity": [1.0]},
        index=pd.to_datetime(["2026-01-01 00:00"], utc=True),
    )
    rts = _pair_round_trips(trades, last_close=120.0, fee_rate=0.0)
    assert rts == []
