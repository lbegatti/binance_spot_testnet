"""
backtest/runner.py
------------------
Step 5 — Top-level backtest runner.

Chains the four backtest modules into a single end-to-end run and prints a
formatted summary report to the console.

Pipeline
--------
    run_signals()       ← backtest/signals.py              (Steps 1–3)
         │
         ▼
    simulate_pnl()      ← backtest/pnl.py                  (Step 4)
         │
         ▼
    _print_report()     ← backtest/reporting/formatters.py (Step 5)

Usage
-----
    # As a script:
    python -m backtest.runner

    # Programmatic — returns all intermediate artefacts:
    from backtest.runner import run_backtest
    signals, trades, equity, stats = run_backtest(save_csv=True)
"""

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from backtest.pnl import simulate_pnl
from backtest.reporting.formatters import HEAVY, print_report, save_csv
from backtest.signals import run_signals

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    export_csv: bool = False,
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
    3. ``_print_report()``  — formats and prints the summary to stdout.
       Implemented in ``backtest/reporting/formatters.py``.

    Parameters
    ----------
    export_csv : bool
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
    log.info(HEAVY)
    log.info(
        "Backtest starting  —  %s UTC",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )
    log.info(HEAVY)

    # ── Steps 1–3: fetch klines → synthetic books → signals ──────────────
    signals = run_signals()

    # ── Step 4: simulate P&L ─────────────────────────────────────────────
    trades, equity, stats = simulate_pnl(signals)

    # ── Step 5: print summary report ─────────────────────────────────────
    print_report(signals, trades, equity, stats)

    if export_csv:
        save_csv(trades, equity)

    return signals, trades, equity, stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_backtest(export_csv=False)
