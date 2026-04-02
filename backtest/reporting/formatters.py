"""
backtest/reporting/formatters.py
---------------------------------
Console report formatting and optional CSV export helpers for the backtest
pipeline.

NOTE
----
This module was written entirely by AI (GitHub Copilot / GPT-4o) and is
provided as-is for educational and testing purposes only.  It has not been
independently audited.  See the project-level disclaimer in README.md.

Public symbols
--------------
fmt(value, fmt, fallback)      — safe float formatter
print_report(signals, trades, equity, stats)
                               — prints the structured backtest summary
save_csv(trades, equity)       — persists trade log & equity curve to CSV
"""

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config_parameters import (
    BACKTEST_FEE_RATE,
    BACKTEST_INITIAL_BTC,
    BACKTEST_INITIAL_CAPITAL,
    SYMBOL,
)

log = logging.getLogger(__name__)

# ── Console formatting constants ──────────────────────────────────────────────
REPORT_WIDTH = 68  # total report width (characters)
HEAVY = "═" * REPORT_WIDTH
LIGHT = "─" * REPORT_WIDTH
PREVIEW = 10  # max trade rows shown in head / tail preview


# ---------------------------------------------------------------------------
# Float formatter
# ---------------------------------------------------------------------------


def fmt(value: float, spec: str = ".4f", fallback: str = "n/a") -> str:
    """
    Format *value* as a string using *spec*.

    Returns *fallback* when *value* is ``None``, ``NaN``, or ``inf`` so that
    the report never crashes on missing statistics.

    Parameters
    ----------
    value : float
        The number to format.
    spec : str
        ``format()``-style format spec, e.g. ``".2f"`` or ``"+.4f"``.
    fallback : str
        String returned for missing / non-finite values.  Default ``"n/a"``.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    if value == math.inf:
        return "∞"
    return format(value, spec)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    stats: dict[str, Any],
) -> None:
    """
    Print a structured backtest summary report to *stdout*.

    Sections
    --------
    1. Session        — date range, candle count, cost assumptions.
    2. Signals        — raw candidates vs executed, filter hit rates.
    3. P&L Summary    — total return, win rate, profit factor, holding period.
    4. Risk Metrics   — max drawdown, Sharpe, Sortino.
    5. Trade Preview  — first / last ``PREVIEW`` rows of the trade log.

    Parameters
    ----------
    signals : pd.DataFrame
        Signal DataFrame produced by ``run_signals()``.
    trades : pd.DataFrame
        Executed trade log produced by ``simulate_pnl()``.
    equity : pd.DataFrame
        Mark-to-market equity curve produced by ``simulate_pnl()``.
    stats : dict
        Performance metrics dict produced by ``simulate_pnl()``.
    """
    print()
    print(HEAVY)
    print(f"  BACKTEST REPORT  —  {SYMBOL} @ 1 m")
    print(HEAVY)

    # ── 1. Session ────────────────────────────────────────────────────────
    start_ts = signals.index[0]
    end_ts = signals.index[-1]
    n_candles = len(signals)

    print()
    print("  SESSION")
    print(LIGHT)
    print(f"  Period          :  {start_ts}  →  {end_ts}")
    print(f"  Candles         :  {n_candles:>10,}")
    print(f"  Initial USDT    :  {BACKTEST_INITIAL_CAPITAL:>12,.2f}  USDT")
    print(f"  Initial BTC     :  {BACKTEST_INITIAL_BTC:>12.6f}  BTC")
    print(f"  Taker fee       :  {BACKTEST_FEE_RATE * 100:.2f} % per side")
    print(
        "  Slippage        :  (high − low) / 2  per candle"
        "  (half-spread embedded in fill price)"
    )

    # ── 2. Signals ────────────────────────────────────────────────────────
    n_raw_buy = stats["n_raw_buy_candidates"]
    n_raw_sell = stats["n_raw_sell_candidates"]
    n_buy = stats["n_buy_signals"]
    n_sell = stats["n_sell_signals"]
    n_hold = int((signals["signal"] == 0).sum())
    conf_hit = stats.get("confidence_filter_hit_rate_pct", float("nan"))
    regime_hit = stats["regime_filter_hit_rate_pct"]
    vwap_hit = stats["vwap_filter_hit_rate_pct"]

    print()
    print("  SIGNALS")
    print(LIGHT)
    print(f"  Raw BUY  candidates        :  {n_raw_buy:>8,}")
    print(f"  Raw SELL candidates        :  {n_raw_sell:>8,}")
    print(f"  Executed BUY               :  {n_buy:>8,}")
    print(f"  Executed SELL              :  {n_sell:>8,}")
    print(f"  HOLD (no signal)           :  {n_hold:>8,}")
    print(f"  Confidence filter blocked  :  {fmt(conf_hit, '.1f')} % of raw candidates")
    print(
        f"  Regime    filter blocked   :  {fmt(regime_hit, '.1f')} % of raw candidates"
    )
    print(f"  VWAP      filter blocked   :  {fmt(vwap_hit, '.1f')} % of raw candidates")

    # ── 3. P&L Summary ────────────────────────────────────────────────────
    final_eq = stats["final_equity_usdt"]
    total_ret = stats["total_return_pct"]
    n_trips = stats["n_round_trips"]
    win_rate = stats["win_rate_pct"]
    avg_pnl = stats["avg_trade_pnl_usdt"]
    pf = stats["profit_factor"]
    avg_hold = stats["avg_holding_minutes"]

    ret_sign = "+" if (isinstance(total_ret, float) and total_ret >= 0) else ""

    print()
    print("  P&L SUMMARY")
    print(LIGHT)
    print(f"  Final equity    :  {final_eq:>12,.2f}  USDT")
    print(f"  Total return    :  {ret_sign}{fmt(total_ret, '.4f')} %")
    print(f"  Round trips     :  {n_trips:>8,}  (BUY → next SELL pairs)")
    print(f"  Win rate        :  {fmt(win_rate, '.1f')} %")
    print(f"  Avg trade PnL   :  {fmt(avg_pnl, '+.4f')}  USDT")
    if pf == math.inf:
        print("  Profit factor   :  ∞  (no losing round trips)")
    else:
        print(f"  Profit factor   :  {fmt(pf, '.3f')}")
    print(f"  Avg hold        :  {fmt(avg_hold, '.1f')} min")

    # ── 4. Risk Metrics ───────────────────────────────────────────────────
    dd = stats["max_drawdown_pct"]
    sharpe = stats["sharpe_ratio"]
    sortino = stats["sortino_ratio"]

    print()
    print("  RISK METRICS")
    print(LIGHT)
    print(f"  Max drawdown    :  {fmt(dd, '.4f')} %")
    print(f"  Sharpe ratio    :  {fmt(sharpe, '.4f')}")
    print(f"  Sortino ratio   :  {fmt(sortino, '.4f')}")
    print(
        "  Note: Sharpe / Sortino annualised with √365 "
        "(crypto trades 24/7; no weekend gaps)"
    )

    # ── 5. Trade log preview ──────────────────────────────────────────────
    print()
    print(f"  TRADE LOG PREVIEW  (first / last {PREVIEW})")
    print(LIGHT)

    if trades.empty:
        print("  No trades were executed in this backtest window.")
    else:
        _cols = [
            c
            for c in ("side", "fill_price", "quantity", "gross", "fee")
            if c in trades.columns
        ]
        n = len(trades)

        pd.set_option("display.float_format", "{:.6f}".format)
        pd.set_option("display.max_columns", len(_cols))
        pd.set_option("display.width", REPORT_WIDTH)

        if n <= PREVIEW * 2:
            print(trades[_cols].to_string(index=True))
        else:
            head = trades[_cols].head(PREVIEW)
            tail = trades[_cols].tail(PREVIEW)
            print(head.to_string(index=True))
            print(f"\n  ... {n - PREVIEW * 2:,} rows omitted ...\n")
            print(tail.to_string(index=True))

        pd.reset_option("display.float_format")
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")

    print()
    print(HEAVY)
    print("  End of report")
    print(HEAVY)
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def save_csv(trades: pd.DataFrame, equity: pd.DataFrame) -> None:
    """
    Save the trade log and equity curve to ``backtest/results/``.

    Files are timestamped so successive runs do not overwrite each other::

        backtest/results/trades_YYYYMMDD_HHMMSS.csv
        backtest/results/equity_YYYYMMDD_HHMMSS.csv

    Parameters
    ----------
    trades : pd.DataFrame
        Executed trade log produced by ``simulate_pnl()``.
    equity : pd.DataFrame
        Mark-to-market equity curve produced by ``simulate_pnl()``.
    """
    # Resolve the results/ directory relative to this file's parent (backtest/)
    results_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    os.makedirs(results_dir, exist_ok=True)

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if not trades.empty:
        path = os.path.join(results_dir, f"trades_{ts_str}.csv")
        trades.to_csv(path)
        log.info("Trade log  saved → %s", path)

    path = os.path.join(results_dir, f"equity_{ts_str}.csv")
    equity.to_csv(path)
    log.info("Equity curve saved → %s", path)
