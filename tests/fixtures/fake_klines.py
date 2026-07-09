"""Deterministic kline / signal DataFrame builders for the signals + pnl tests."""

from __future__ import annotations

import pandas as pd


def make_ohlcv(
    *,
    close: list[float],
    high: list[float],
    low: list[float],
    volume: list[float],
    taker_buy_base_vol: list[float],
    num_trades: list[int],
    start: str = "2026-01-01",
    freq: str = "5min",
) -> pd.DataFrame:
    """Build a minimal OHLCV frame with a UTC ``DatetimeIndex``.

    Columns match the raw klines frame that
    ``backtest.signals._add_hmm_features`` consumes.
    """
    n = len(close)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "taker_buy_base_vol": taker_buy_base_vol,
            "num_trades": num_trades,
        },
        index=idx,
    )


# Columns that ``backtest.pnl.simulate_pnl`` reads (directly or via getattr),
# with the defaults it treats as "not present / inactive".
_SIGNAL_DEFAULTS: dict = {
    "signal": 0,
    "half_spread": 0.0,
    "buy_qty": float("nan"),
    "sell_qty": float("nan"),
    "regime": "neutral",
    "best_buy_micro": float("nan"),
    "best_sell_micro": float("nan"),
    "stop_loss_pct": 0.0,
    "trend_pause": False,
}


def make_signals(
    rows: list[dict], start: str = "2026-01-01", freq: str = "1min"
) -> pd.DataFrame:
    """Build a ``simulate_pnl`` input frame from a list of per-bar dicts.

    Any column omitted from a row is filled with the production default from
    ``_SIGNAL_DEFAULTS`` so the ``getattr`` / ``pd.notna`` guards in
    ``simulate_pnl`` behave exactly as they do on a real signals frame.
    """
    n = len(rows)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    for col, default in _SIGNAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df
