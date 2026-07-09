"""Tier B tests for backtest/signals.py.

Only the pure feature-engineering helper ``_add_hmm_features`` is covered here.
The full ``run_signals`` walk-forward fits a GaussianHMM and stitches two kline
frames — that is integration-level (Tier C) and out of scope for the Tier B
deterministic suite.
"""

import numpy as np
import pytest

from backtest.signals import _add_hmm_features
from tests.fixtures.fake_klines import make_ohlcv


def test_add_hmm_features_drops_first_row_and_computes():
    """The first bar has no prior close so its log-return is NaN and the row
    is dropped; the surviving row carries the four HMM features computed by
    hand here (return, volatility, obi_proxy, trade_density)."""
    df = make_ohlcv(
        close=[100.0, 110.0],
        high=[105.0, 115.0],
        low=[95.0, 108.0],
        volume=[10.0, 20.0],
        taker_buy_base_vol=[6.0, 12.0],
        num_trades=[50, 80],
    )
    out = _add_hmm_features(df)

    # first row dropped (log-return is NaN on shift(1))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["return"] == pytest.approx(np.log(110 / 100))
    assert row["volatility"] == pytest.approx((115 - 108) / 110)
    assert row["obi_proxy"] == pytest.approx((12 / 20) * 2 - 1)
    assert row["trade_density"] == pytest.approx(80 / 20)


def test_add_hmm_features_appends_four_columns_no_nan():
    """The helper appends exactly the four feature columns in a fixed order
    and, after the first-row drop, leaves no NaNs for the HMM to choke on."""
    df = make_ohlcv(
        close=[100.0, 110.0, 120.0],
        high=[101.0, 111.0, 121.0],
        low=[99.0, 109.0, 119.0],
        volume=[10.0, 10.0, 10.0],
        taker_buy_base_vol=[5.0, 5.0, 5.0],
        num_trades=[10, 10, 10],
    )
    out = _add_hmm_features(df)
    assert list(out.columns[-4:]) == [
        "return",
        "volatility",
        "obi_proxy",
        "trade_density",
    ]
    assert not out.isna().any().any()


def test_add_hmm_features_does_not_mutate_input():
    """The helper must work on a copy: the caller's original frame gains no
    feature columns, so upstream data stays clean (no look-ahead leakage)."""
    df = make_ohlcv(
        close=[100.0, 110.0],
        high=[105.0, 115.0],
        low=[95.0, 108.0],
        volume=[10.0, 20.0],
        taker_buy_base_vol=[6.0, 12.0],
        num_trades=[50, 80],
    )
    _add_hmm_features(df)
    assert "return" not in df.columns  # helper works on a copy
