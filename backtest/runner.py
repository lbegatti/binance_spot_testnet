"""
backtest/runner.py
------------------
Step 5 — Top-level backtest runner.

Chains the backtest modules into a single end-to-end run and prints a
formatted summary report to the console.

Pipeline
--------
    fetch_macro_klines()    ← backtest/data.py  (5-minute bars, HMM frame)
    fetch_micro_klines()    ← backtest/data.py  (1-minute bars, execution frame)
         │
         ▼
    run_signals()           ← backtest/signals.py              (Steps 1–3)
         │  Phase 1: HMM walk-forward on 5-minute macro bars
         │  Phase 2: merge_asof stitch onto 1-minute micro bars
         │  Phase 3: execution loop (book + VWAP + regime gate)
         ▼
    simulate_pnl()          ← backtest/pnl.py                  (Step 4)
         │  Intra-candle whipsaw guard on open positions
         ▼
    print_report()          ← backtest/reporting/formatters.py (Step 5)
         │
         ▼
    plot_backtest()         ← backtest/visualization.py        (Step 7, default ON)

Usage
-----
    # As a script (no chart — headless):
    python -m backtest.runner --no-plot

    # As a script (with interactive chart, default):
    python -m backtest.runner

    # Export CSV artefacts:
    python -m backtest.runner --csv

    # Flush Parquet cache before running:
    python -m backtest.runner --flush-cache

    # Programmatic — returns all intermediate artefacts:
    from backtest.runner import run_backtest
    signals, trades, equity, stats = run_backtest(export_csv=True, plot=True, save_png=True)
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from backtest.data import fetch_macro_klines, fetch_micro_klines, flush_kline_cache
from backtest.pnl import compute_buy_and_hold, simulate_pnl
from backtest.reporting.formatters import (
    HEAVY,
    print_bnh_comparison,
    print_report,
    save_csv,
)
from backtest.signals import run_signals
from backtest.visualization import plot_backtest
from config_parameters import (
    BACKTEST_FEE_RATE,
    BACKTEST_INITIAL_BTC,
    BACKTEST_OOS_START,
)
from strategy.param_loader import load_best_params_for_backtest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def run_backtest(
    export_csv: bool = False,
    plot: bool = False,
    save_png: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Execute the full backtest pipeline and print a summary report.

    Steps executed in order:

    1. ``run_signals()``    — fetches klines, builds synthetic order books,
       runs the production pipeline (Flows A/B/C), and returns a signal
       DataFrame.
    2. ``simulate_pnl()``   — walks the signal DataFrame, executes BUY/SELL
       trades with the balance guard, and computes the equity curve and
       Step 5 metrics.
    3. ``print_report()``   — formats and prints the summary to stdout.
       Implemented in ``backtest/reporting/formatters.py``.
    4. ``plot_backtest()``  — (optional, Step 7) generates the interactive
       six-panel Plotly figure.  Only runs when ``plot=True``.  For headless
       environments pass ``show=False`` inside ``plot_backtest()`` directly.

    Parameters
    ----------
    export_csv : bool
        If ``True``, saves ``trades_<timestamp>.csv`` and
        ``equity_<timestamp>.csv`` to ``backtest/results/``.
        Default ``False``.
    plot : bool
        If ``True``, generate the Step 7 Plotly visualisation after the
        summary report.  When invoked via ``python -m backtest.runner`` the
        plot is **shown by default**; pass ``--no-plot`` to suppress.
        When called programmatically the default is ``False`` (no side-effects
        in library use).  Pass ``plot=True`` explicitly for programmatic
        visualisation.
    save_png : bool
        If ``True`` (and ``plot=True``), persist the figure as a timestamped
        PNG in ``backtest/results/`` (requires ``kaleido``; falls back to
        HTML if not installed).  Has no effect when ``plot=False``.
        When invoked via ``python -m backtest.runner`` this defaults to
        ``True`` so that every run automatically saves a chart; pass
        ``--no-plot`` to suppress both display and saving.
        Default ``False`` for programmatic/library use.

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
    log.info(HEAVY)
    log.info(
        "Backtest starting  —  %s UTC",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )
    log.info(HEAVY)

    # Load tuned parameters from sensitivity.py output (falls back to
    # config_parameters.py defaults if best_params.json is absent).
    best = load_best_params_for_backtest()

    # OOS window: BACKTEST_OOS_START → today (90 days).
    # sensitivity.py was trained on the IS window (days −360 → −90) and never
    # saw this data — genuine out-of-sample validation.
    #
    # Pre-fetch BOTH frames here so:
    #   • The Parquet cache is checked exactly once per frame (no repeated API calls).
    #   • Callers can inspect df_macro / df_micro before signals are generated.
    #   • run_signals() receives raw OHLCV — no pre-computed features passed in.
    log.info("Fetching OOS klines — window: '%s' → today", BACKTEST_OOS_START)
    df_macro = fetch_macro_klines(lookback=BACKTEST_OOS_START)
    df_micro = fetch_micro_klines(lookback=BACKTEST_OOS_START)
    log.info(
        "OOS frames ready: macro=%d 5-min bars, micro=%d 1-min bars.",
        len(df_macro),
        len(df_micro),
    )

    # Steps 1–3: HMM walk-forward (5 m) → stitch → execution loop (1 m).
    # best.get() returns None when a key is absent → run_signals() uses its
    # config_parameters.py default for that parameter automatically.
    signals = run_signals(
        prefetched_macro=df_macro,
        prefetched_micro=df_micro,
        hmm_lookback_rows=best.get("hmm_lookback_rows"),
        hmm_max_regimes=best.get("hmm_max_regimes"),
        vwap_window=best.get("vwap_window"),
        vwap_threshold=best.get("vwap_threshold"),
    )

    # Step 4: simulate P&L
    # fee_rate is loaded from best_params.json when present (always written as
    # SENSITIVITY_FEE_RATE = 0.001 by sensitivity.py — fee_rate is NOT a tunable
    # Optuna parameter, so it will never be an artificially low value).
    # Falls back to BACKTEST_FEE_RATE from config_parameters.py when absent.
    trades, equity, stats = simulate_pnl(
        signals,
        fee_rate=best.get("fee_rate", BACKTEST_FEE_RATE),
    )

    # Step 5: print summary report
    print_report(signals, trades, equity, stats)

    # Buy-and-hold comparison — pass the same initial_btc as simulate_pnl so
    # both strategy and BnH are measured against the same starting portfolio
    # (initial_usdt USDT + initial_btc BTC valued at the first close).
    # Without this, BnH uses only initial_usdt as the denominator while the
    # strategy uses the full portfolio — the two percentages are incomparable
    # whenever BACKTEST_INITIAL_BTC > 0.
    bnh_stats = compute_buy_and_hold(df_micro, initial_btc=BACKTEST_INITIAL_BTC)
    stats.update(bnh_stats)
    print_bnh_comparison(pd.Series(stats))

    if export_csv:
        save_csv(trades, equity)

    # Step 7: plotly visualisation (optional)
    if plot:
        plot_backtest(signals, trades, equity, stats, save_png=save_png, show=True)

    return signals, trades, equity, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Full OOS backtest runner — two-resolution architecture.\n\n"
            "Fetches 5-minute macro klines (HMM regime) and 1-minute micro klines\n"
            "(VWAP / PnL execution), runs the end-to-end pipeline, and prints a\n"
            "formatted summary report.\n\n"
            "Data window: BACKTEST_OOS_START → today  (~90 days, ~25 920 rows at 5 m).\n"
            "Best params: loaded from backtest/results/best_params.json if present;\n"
            "             falls back to config_parameters.py defaults."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the interactive six-panel Plotly chart (headless / CI mode).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save trades_<ts>.csv and equity_<ts>.csv to backtest/results/.",
    )
    # --save-png kept for backwards-compatibility but is now a no-op:
    # the chart is always saved when plot is active (i.e. unless --no-plot).
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help=(
            "Delete all cached kline Parquet files under cache/klines/ and exit. "
            "Use this whenever BACKTEST_LOOKBACK or BACKTEST_OOS_START change "
            "to force a fresh download on the next run."
        ),
    )
    args = parser.parse_args()

    if args.flush_cache:
        flush_kline_cache()
        print("Cache flushed. Run runner.py again to fetch fresh klines.")
        sys.exit(0)

    run_backtest(
        export_csv=args.csv,
        plot=not args.no_plot,
        save_png=not args.no_plot,  # always save chart to results/ unless headless
    )
