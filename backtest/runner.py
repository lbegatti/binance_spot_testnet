"""
backtest/runner.py
------------------
Step 5 — Top-level backtest runner.

Chains the four backtest modules into a single end-to-end run and prints a
formatted summary report to the console.

Pipeline
--------
    run_signals()       ← backtest/signals.py      (Steps 1–3)
         │
         ▼
    simulate_pnl()      ← backtest/pnl.py           (Step 4)
         │
         ▼
    _print_report()     ← this file                 (Step 5)

Usage
-----
    # As a script:
    python -m backtest.runner

    # Programmatic — returns all intermediate artefacts:
    from backtest.runner import run_backtest
    signals, trades, equity, stats = run_backtest(save_csv=True)
"""

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from backtest.pnl import simulate_pnl
from backtest.signals import run_signals
from config_parameters import (
    BACKTEST_FEE_RATE,
    BACKTEST_INITIAL_BTC,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_SLIPPAGE,
    SYMBOL,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Console formatting ────────────────────────────────────────────────────────
_W = 68  # report width
_HEAVY = "═" * _W
_LIGHT = "─" * _W
_PREVIEW = 10  # max trades shown in head / tail


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
        save_csv: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Execute the full backtest pipeline and print a summary report.

    Steps executed in order:

    1. ``run_signals()``   — fetches klines, builds synthetic order books,
       runs the production pipeline (Flows A/B/C), and returns a signal
       DataFrame.
    2. ``simulate_pnl()``  — walks the signal DataFrame, executes BUY/SELL
       trades with the balance guard, and computes the equity curve and
       Step 5 metrics.
    3. ``_print_report()`` — formats and prints the summary to stdout.

    Parameters
    ----------
    save_csv : bool
        If ``True``, saves ``trades_<timestamp>.csv`` and
        ``equity_<timestamp>.csv`` to ``backtest/results/``.
        Default ``False``.

    Returns
    -------
    signals : pd.DataFrame
        Signal DataFrame from ``run_signals()``.
    trades : pd.DataFrame
        Executed trade log from ``simulate_pnl()``.
    equity : pd.DataFrame
        Mark-to-market equity curve from ``simulate_pnl()``.
    stats : dict
        Step 5 performance metrics from ``simulate_pnl()``.
    """
    log.info(_HEAVY)
    log.info(
        "Backtest starting  —  %s UTC",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )
    log.info(_HEAVY)

    # ── Steps 1–3: fetch klines → synthetic books → signals ──────────────
    signals = run_signals()

    # ── Step 4: simulate P&L ─────────────────────────────────────────────
    trades, equity, stats = simulate_pnl(signals)

    # ── Step 5: print summary report ─────────────────────────────────────
    _print_report(signals, trades, equity, stats)

    if save_csv:
        _save_csv(trades, equity)

    return signals, trades, equity, stats


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(value: float, fmt: str = ".4f", fallback: str = "n/a") -> str:
    """Format a float safely — returns ``fallback`` when value is NaN/inf."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    if value == math.inf:
        return "∞"
    return format(value, fmt)


def _print_report(
        signals: pd.DataFrame,
        trades: pd.DataFrame,
        equity: pd.DataFrame,
        stats: dict[str, Any],
) -> None:
    """
    Print a structured backtest summary report to stdout.

    Sections
    --------
    1. Session        — date range, candle count, cost assumptions.
    2. Signals        — raw candidates vs executed, filter hit rates.
    3. P&L Summary    — return, win rate, profit factor, holding period.
    4. Risk Metrics   — drawdown, Sharpe, Sortino.
    5. Trade Preview  — first / last ``_PREVIEW`` rows of the trade log.
    """
    print()
    print(_HEAVY)
    print(f"  BACKTEST REPORT  —  {SYMBOL} @ 1 m")
    print(_HEAVY)

    # ── 1. Session ────────────────────────────────────────────────────────
    start_ts = signals.index[0]
    end_ts = signals.index[-1]
    n_candles = len(signals)

    print()
    print("  SESSION")
    print(_LIGHT)
    print(f"  Period          :  {start_ts}  →  {end_ts}")
    print(f"  Candles         :  {n_candles:>10,}")
    print(f"  Initial USDT    :  {BACKTEST_INITIAL_CAPITAL:>12,.2f}  USDT")
    print(f"  Initial BTC     :  {BACKTEST_INITIAL_BTC:>12.6f}  BTC")
    print(f"  Taker fee       :  {BACKTEST_FEE_RATE * 100:.2f} % per side")
    print(
        f"  Extra slippage  :  {BACKTEST_SLIPPAGE * 100:.3f} %"
        "  (spread cost already embedded in fill price)"
    )

    # ── 2. Signals ────────────────────────────────────────────────────────
    n_raw_buy = stats["n_raw_buy_candidates"]
    n_raw_sell = stats["n_raw_sell_candidates"]
    n_buy = stats["n_buy_signals"]
    n_sell = stats["n_sell_signals"]
    n_hold = int((signals["signal"] == 0).sum())
    regime_hit = stats["regime_filter_hit_rate_pct"]
    vwap_hit = stats["vwap_filter_hit_rate_pct"]

    print()
    print("  SIGNALS")
    print(_LIGHT)
    print(f"  Raw BUY  candidates   :  {n_raw_buy:>8,}")
    print(f"  Raw SELL candidates   :  {n_raw_sell:>8,}")
    print(f"  Executed BUY          :  {n_buy:>8,}")
    print(f"  Executed SELL         :  {n_sell:>8,}")
    print(f"  HOLD (no signal)      :  {n_hold:>8,}")
    print(f"  Regime filter blocked :  {_fmt(regime_hit, '.1f')} % of raw candidates")
    print(f"  VWAP   filter blocked :  {_fmt(vwap_hit, '.1f')} % of raw candidates")

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
    print(_LIGHT)
    print(f"  Final equity    :  {final_eq:>12,.2f}  USDT")
    print(f"  Total return    :  {ret_sign}{_fmt(total_ret, '.4f')} %")
    print(f"  Round trips     :  {n_trips:>8,}  (BUY → next SELL pairs)")
    print(f"  Win rate        :  {_fmt(win_rate, '.1f')} %")
    print(f"  Avg trade PnL   :  {_fmt(avg_pnl, '+.4f')}  USDT")
    if pf == math.inf:
        print("  Profit factor   :  ∞  (no losing round trips)")
    else:
        print(f"  Profit factor   :  {_fmt(pf, '.3f')}")
    print(f"  Avg hold        :  {_fmt(avg_hold, '.1f')} min")

    # ── 4. Risk Metrics ───────────────────────────────────────────────────
    dd = stats["max_drawdown_pct"]
    sharpe = stats["sharpe_ratio"]
    sortino = stats["sortino_ratio"]

    print()
    print("  RISK METRICS")
    print(_LIGHT)
    print(f"  Max drawdown    :  {_fmt(dd, '.4f')} %")
    print(f"  Sharpe ratio    :  {_fmt(sharpe, '.4f')}")
    print(f"  Sortino ratio   :  {_fmt(sortino, '.4f')}")
    print(
        "  Note: Sharpe / Sortino annualised with √365 "
        "(crypto trades 24/7; no weekend gaps)"
    )

    # ── 5. Trade log preview ──────────────────────────────────────────────
    print()
    print(f"  TRADE LOG PREVIEW  (first / last {_PREVIEW})")
    print(_LIGHT)

    if trades.empty:
        print("  No trades were executed in this backtest window.")
    else:
        _cols = [c for c in ("side", "fill_price", "quantity", "gross", "fee") if c in trades.columns]
        n = len(trades)

        pd.set_option("display.float_format", "{:.6f}".format)
        pd.set_option("display.max_columns", len(_cols))
        pd.set_option("display.width", _W)

        if n <= _PREVIEW * 2:
            print(trades[_cols].to_string(index=True))
        else:
            head = trades[_cols].head(_PREVIEW)
            tail = trades[_cols].tail(_PREVIEW)
            print(head.to_string(index=True))
            print(f"\n  ... {n - _PREVIEW * 2:,} rows omitted ...\n")
            print(tail.to_string(index=True))

        pd.reset_option("display.float_format")
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")

    print()
    print(_HEAVY)
    print("  End of report")
    print(_HEAVY)
    print()


# ---------------------------------------------------------------------------
# Optional CSV export
# ---------------------------------------------------------------------------

def _save_csv(trades: pd.DataFrame, equity: pd.DataFrame) -> None:
    """
    Save the trade log and equity curve to ``backtest/results/``.

    Files are timestamped so successive runs do not overwrite each other:
        backtest/results/trades_YYYYMMDD_HHMMSS.csv
        backtest/results/equity_YYYYMMDD_HHMMSS.csv
    """
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if not trades.empty:
        path = os.path.join(results_dir, f"trades_{ts_str}.csv")
        trades.to_csv(path)
        log.info("Trade log  saved → %s", path)

    path = os.path.join(results_dir, f"equity_{ts_str}.csv")
    equity.to_csv(path)
    log.info("Equity curve saved → %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_backtest(save_csv=False)
