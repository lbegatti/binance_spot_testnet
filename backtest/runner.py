"""
backtest/runner.py
------------------
Step 5 — Top-level backtest runner.

Chains the backtest modules into a single end-to-end run and prints a
formatted summary report to the console.

Pipeline
--------
    run_signals()       ← backtest/signals.py              (Steps 1–3)
         │
         ▼
    simulate_pnl()      ← backtest/pnl.py                  (Step 4)
         │
         ▼
    print_report()      ← backtest/reporting/formatters.py (Step 5)
         │
         ▼
    plot_backtest()     ← backtest/visualization.py        (Step 7, opt-in)

Usage
-----
    # As a script (no chart):
    python -m backtest.runner

    # As a script (with interactive chart):
    python -m backtest.runner   # then call run_backtest(plot=True) programmatically

    # Programmatic — returns all intermediate artefacts:
    from backtest.runner import run_backtest
    signals, trades, equity, stats = run_backtest(export_csv=True, plot=True, save_png=True)
"""

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from backtest.pnl import simulate_pnl
from backtest.reporting.formatters import HEAVY, print_report, save_csv
from backtest.signals import run_signals
from backtest.visualization import plot_backtest
from config_parameters import BACKTEST_FEE_RATE
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
        summary report.  Default ``False`` (opt-in so CLI runs are
        unaffected).
    save_png : bool
        If ``True`` (and ``plot=True``), persist the figure as a timestamped
        PNG in ``backtest/results/`` (requires ``kaleido``; falls back to
        HTML if not installed).  Has no effect when ``plot=False``.
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
    log.info(HEAVY)
    log.info(
        "Backtest starting  —  %s UTC",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )
    log.info(HEAVY)

    # Load tuned parameters from sensitivity.py output (falls back to
    # config_parameters.py defaults if best_params.json is absent).
    best = load_best_params_for_backtest()

    # Steps 1–3: fetch klines → synthetic books → signals
    # best.get() returns None when a key is absent → run_signals() uses its
    # config_parameters.py default for that parameter automatically.
    signals = run_signals(
        hmm_lookback_rows=best.get("hmm_lookback_rows"),
        hmm_max_regimes=best.get("hmm_max_regimes"),
        vwap_window=best.get("vwap_window"),
        vwap_threshold=best.get(
            "vwap_threshold"
        ),  # from best_params.json (Bayesian-optimised)
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

    if export_csv:
        save_csv(trades, equity)

    # Step 7: plotly visualisation (optional)
    if plot:
        plot_backtest(signals, trades, equity, stats, save_png=save_png, show=True)

    return signals, trades, equity, stats


if __name__ == "__main__":
    run_backtest(export_csv=False, plot=True, save_png=True)
