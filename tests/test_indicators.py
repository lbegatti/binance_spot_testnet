"""Tier B tests for strategy/indicators.py — VWAP, trend-pause flag, adaptive
stop-loss, and the REST-path indicator helper."""

import numpy as np
import pandas as pd
import pytest

from strategy.indicators import (
    add_strategy_indicators,
    add_trend_pause_flag,
    compute_live_stop_loss_pct,
    volume_weighted_average_price,
)


# ── volume_weighted_average_price ──────────────────────────────────────────


def test_vwap_basic():
    """VWAP weights each price by its volume, so a larger trade pulls the
    average toward its price — checked against the hand-computed value."""
    price = np.array([100.0, 102.0])
    volume = np.array([1.0, 3.0])
    assert volume_weighted_average_price(price, volume) == pytest.approx(
        (100 * 1 + 102 * 3) / 4
    )


def test_vwap_zero_volume_returns_nan():
    """Zero total volume would divide by zero, so VWAP returns NaN instead of
    raising — a signal the caller can filter on."""
    price = np.array([100.0, 102.0])
    volume = np.array([0.0, 0.0])
    assert np.isnan(volume_weighted_average_price(price, volume))


# ── add_trend_pause_flag ───────────────────────────────────────────────────


def test_trend_pause_flag_streak_and_cooldown():
    """The pause flag turns on when closes rise n bars in a row and stays on
    for the cooldown window afterwards, so entries are held back right after
    a run-up — the expected on/off pattern is asserted bar by bar."""
    # closes rise for 3 bars (up-streak reaches n=3 at index 3), then flatten.
    close = pd.Series([10, 11, 12, 13, 12, 12, 12, 12], dtype=float)
    df = pd.DataFrame({"close": close})
    paused = add_trend_pause_flag(df, n=3, cooldown=1)
    # index 3 = streak hits 3; index 4 = 1-bar cooldown carry-over.
    assert paused.tolist() == [False, False, False, True, True, False, False, False]


def test_trend_pause_no_sustained_trend_all_false():
    """Alternating closes never form an n-bar up-streak, so the pause flag
    stays False throughout (no spurious pauses on choppy price)."""
    close = pd.Series(
        [10, 11, 10, 11, 10], dtype=float
    )  # alternating, never 3 in a row
    df = pd.DataFrame({"close": close})
    paused = add_trend_pause_flag(df, n=3, cooldown=2)
    assert paused.tolist() == [False] * 5


# ── compute_live_stop_loss_pct (Binance REST client mocked) ────────────────


def _klines_rows(closes):
    # Binance kline rows are 12-element lists; close is column index 4.
    return [[0, 0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0] for c in closes]


def test_stop_loss_empty_klines_returns_zero(mocker):
    """With no kline history to measure volatility, the adaptive stop-loss
    disables itself by returning 0.0. The REST client is mocked (no network)."""
    client = mocker.Mock()
    client.klines.return_value = []
    assert compute_live_stop_loss_pct(client, "BTCUSDT", 2, 3.0) == 0.0


def test_stop_loss_exception_returns_zero(mocker):
    """If the kline REST call raises, the stop-loss calc swallows it and
    returns 0.0 rather than crashing the live loop (mock raises on purpose)."""
    client = mocker.Mock()
    client.klines.side_effect = RuntimeError("boom")
    assert compute_live_stop_loss_pct(client, "BTCUSDT", 2, 3.0) == 0.0


def test_stop_loss_happy_path(mocker):
    """The stop-loss pct is the rolling std of absolute pct-changes times the
    multiplier; the mocked closes make every step verifiable by hand."""
    # closes 100,200,100 → abs pct-change [nan, 1.0, 0.5]
    # rolling(2).std() on the last window = std([1.0, 0.5]) = 0.353553 ; × mult 1.0
    client = mocker.Mock()
    client.klines.return_value = _klines_rows([100.0, 200.0, 100.0])
    result = compute_live_stop_loss_pct(client, "BTCUSDT", rolling_days=2, std_mult=1.0)
    assert result == pytest.approx(0.353553, rel=1e-4)


# ── add_strategy_indicators (REST path) ────────────────────────────────────


def test_add_strategy_indicators_invalid_strategy_raises():
    """Only "buy"/"sell" are valid; an unknown strategy name (e.g. "hold")
    raises ValueError so a typo fails loudly instead of silently no-op-ing."""
    df = pd.DataFrame(
        {
            "micro_vs_mid": [True],
            "micro_price": [1.0],
            "mid_price": [1.0],
            "total_depth": [1.0],
        }
    )
    with pytest.raises(ValueError):
        add_strategy_indicators(df, strategy="hold")


def test_add_strategy_indicators_buy_adds_columns():
    """The buy path adds the micro-vs-mid delta plus the thin-micro and
    depth-fraction flag columns the signal generator downstream relies on."""
    df = pd.DataFrame(
        {
            "micro_vs_mid": [True, False],
            "micro_price": [101.0, 99.0],
            "mid_price": [100.0, 100.0],
            "total_depth": [10.0, 5.0],
        }
    )
    out = add_strategy_indicators(df, strategy="buy")
    assert out["micro_mid_delta"].tolist() == [1.0, -1.0]  # micro - mid
    assert "is_thin_micro_effect" in out.columns
    assert "is_total_depth_50pct_l0" in out.columns
