"""
backtest/sensitivity.py
-----------------------
Step 8 — Sensitivity Analysis  (Use Case A: live parameter tuning)

Runs the full backtest pipeline (run_signals → simulate_pnl → _compute_stats)
across a grid of parameter values to answer one question:

    "Does the strategy's performance degrade gracefully as each tunable
     parameter shifts away from its current default value?"

Two execution modes
-------------------
  OAT sweep (default, Phase 1):
      Holds all parameters at their defaults and varies ONE parameter at a time.
      6 runs total (1 baseline + 5 non-default).
      Typical wall time: ~36–108 min on a laptop (30-day window).
      Run with:  python -m backtest.sensitivity

  Full factorial grid (Phase 2):
      All 24 parameter combinations.
      Triggered only if any single parameter shows |ΔSharpe| > 0.5 in Phase 1.
      Run with:  python -m backtest.sensitivity --full-grid

Use Case A vs Use Case B
------------------------
  USE CASE A — Live tuning  [IMPLEMENTED HERE]
      Window: SENSITIVITY_LOOKBACK = "30 days ago UTC" (~43,200 rows).
      Purpose: find parameter values that work well in RECENT conditions so
      the live system starts with a tuned configuration rather than hard-coded
      defaults.  Output written to backtest/results/best_params.json.

  USE CASE B — Backtest robustness validation  [DEFERRED]
      Window: must match the main backtest ("180 days ago UTC", ~259,200 rows)
      to avoid window-mismatch bias.  Runtime on a laptop: ~4–12 h for OAT
      alone → impractical without dedicated compute.  Revisit if resources allow.

Output
------
  Console: sorted summary table (one row per combination).
  File:    backtest/results/sensitivity_<timestamp>.csv  (all metrics, all runs).
  File:    backtest/results/best_params.json             (winning parameter set).
           Loaded by websocket_main.py at startup via _load_best_params().

Notes
-----
- config_parameters.py and the live system are NEVER modified by this script.
  All overrides are passed as keyword arguments to run_signals() / simulate_pnl().
- SENSITIVITY_REFIT_EVERY (480 iterations = 8 h at 1 m) is used instead of
  REFIT_EVERY (120) to cut per-run cost from ~360 to ~90 refits (~4× speedup)
  while preserving the relative ranking of parameter combinations.
- Do NOT commit best_params.json to git — it is sample-specific.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from backtest.pnl import simulate_pnl
from backtest.signals import run_signals
from config_parameters import (
    SENSITIVITY_REFIT_EVERY,
    SENSITIVITY_LOOKBACK,
    SENSITIVITY_PREDICT_EVERY,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
_RESULTS_DIR = pathlib.Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

_BEST_PARAMS_PATH = _RESULTS_DIR / "best_params.json"

# ---------------------------------------------------------------------------
# Parameter grid definition
# ---------------------------------------------------------------------------
# Each entry is (config_constant_name, list_of_values_to_test).
# The first value in every list is the DEFAULT — used as the baseline run
# and held fixed while other parameters are varied in the OAT sweep.
_PARAM_GRID: dict[str, list[Any]] = {
    "hmm_lookback_rows": [120, 60, 240],    # default=120 (2 h); test 1 h, 4 h
    "hmm_max_regimes":   [3, 2],            # default=3;   test 2
    "vwap_window":       [5, 2],            # default=5 min; test 2 min
    "fee_rate":          [0.001, 0.0005],   # default=0.10 %; test 0.05 %
}

# Metric used to rank combinations and select best_params.json.
_RANK_METRIC = "sharpe_ratio"

# Sharpe-change threshold that triggers Phase 2 (full factorial grid).
_OAT_SENSITIVITY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def _build_oat_grid() -> list[dict[str, Any]]:
    """
    Build the One-At-a-Time (OAT) sweep grid.

    Returns a list of parameter dicts:
      - Run 0:   all defaults (baseline).
      - Runs 1–N: each non-default value for one parameter while all others
                  remain at their default.

    Total: 1 + sum(len(v) - 1 for v in grid) = 1 + 2 + 1 + 1 + 1 = 6 runs.
    """
    defaults = {k: v[0] for k, v in _PARAM_GRID.items()}
    grid: list[dict[str, Any]] = [defaults.copy()]  # run 0: baseline

    for param, values in _PARAM_GRID.items():
        for val in values[1:]:  # skip the default (index 0)
            run = defaults.copy()
            run[param] = val
            grid.append(run)

    return grid


def _build_full_grid() -> list[dict[str, Any]]:
    """
    Build the full factorial grid (all combinations).

    Total: 3 × 2 × 2 × 2 = 24 combinations.
    """
    keys = list(_PARAM_GRID.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*_PARAM_GRID.values())
    ]


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def _run_one(params: dict[str, Any]) -> dict[str, Any]:
    """
    Execute one full backtest with the given parameter overrides.

    Parameters
    ----------
    params : dict
        Must contain: ``hmm_lookback_rows``, ``hmm_max_regimes``,
        ``vwap_window``, ``fee_rate``.

    Returns
    -------
    dict
        The ``params`` dict merged with the stats dict from ``simulate_pnl()``.
    """
    log.info(
        "Running: lookback=%d  max_regimes=%d  vwap=%d  fee=%.4f%%",
        params["hmm_lookback_rows"],
        params["hmm_max_regimes"],
        params["vwap_window"],
        params["fee_rate"] * 100,
    )

    # run_signals() uses SENSITIVITY_REFIT_EVERY (480) for a ~4× speedup.
    # This cuts refits from ~360 to ~90 per 30-day run without changing
    # the relative ranking of parameter combinations.
    # SENSITIVITY_LOOKBACK ("30 days ago UTC") is the key speedup:
    # without it, each run fetches 180 days (~259,200 rows), making
    # sensitivity 6× slower than a single run_backtest.py call.
    signals = run_signals(
        hmm_lookback_rows=params["hmm_lookback_rows"],
        hmm_max_regimes=params["hmm_max_regimes"],
        vwap_window=params["vwap_window"],
        refit_every=SENSITIVITY_REFIT_EVERY,
        predict_every=SENSITIVITY_PREDICT_EVERY,
        lookback=SENSITIVITY_LOOKBACK,
    )

    _, _, stats = simulate_pnl(signals, fee_rate=params["fee_rate"])

    return {**params, **stats}


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

_DISPLAY_COLS = [
    "hmm_lookback_rows",
    "hmm_max_regimes",
    "vwap_window",
    "fee_rate",
    "total_return_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate_pct",
    "profit_factor",
    "n_round_trips",
    "avg_holding_minutes",
]


def _print_table(results_df: pd.DataFrame, mode: str) -> None:
    """Print a formatted summary table to the console."""
    n = len(results_df)
    print()
    print("═" * 100)
    print(f"  SENSITIVITY ANALYSIS — BTCUSDT  ({mode}, {n} combinations)")
    print("═" * 100)

    # Mark default row
    defaults = {k: v[0] for k, v in _PARAM_GRID.items()}
    display = results_df[_DISPLAY_COLS].copy()
    display.insert(0, "note", "")
    for idx, row in display.iterrows():
        is_default = all(
            row[k] == defaults[k]
            for k in ("hmm_lookback_rows", "hmm_max_regimes", "vwap_window", "fee_rate")
        )
        display.at[idx, "note"] = "← baseline" if is_default else ""

    print(display.to_string(index=False, float_format="{:.3f}".format))
    print("═" * 100)
    print()


def _oat_sensitivity_report(results_df: pd.DataFrame, baseline_sharpe: float) -> bool:
    """
    Print per-parameter sensitivity vs the baseline Sharpe.

    Returns True if any parameter breaches the threshold (triggers Phase 2).
    """
    defaults = {k: v[0] for k, v in _PARAM_GRID.items()}
    trigger_phase2 = False

    print("── OAT Sensitivity Report (vs baseline Sharpe = {:.3f}) ──".format(baseline_sharpe))
    for param in _PARAM_GRID.keys():
        non_default = results_df[results_df[param] != defaults[param]]
        for _, row in non_default.iterrows():
            delta = row[_RANK_METRIC] - baseline_sharpe
            flag = "  ⚠️  |ΔSharpe| > 0.5 — consider Phase 2!" if abs(delta) > _OAT_SENSITIVITY_THRESHOLD else ""
            print(
                f"  {param}={row[param]!r:>6}  →  Sharpe={row[_RANK_METRIC]:.3f}"
                f"  ΔSharpe={delta:+.3f}{flag}"
            )
            if abs(delta) > _OAT_SENSITIVITY_THRESHOLD:
                trigger_phase2 = True
    print()
    return trigger_phase2


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_results(results_df: pd.DataFrame, mode: str) -> pathlib.Path:
    """Write all results to a timestamped CSV file."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = _RESULTS_DIR / f"sensitivity_{mode}_{ts}.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("Results saved → %s", csv_path)
    return csv_path


def _save_best_params(best_row: pd.Series) -> None:
    """
    Write the winning parameter set to best_params.json.

    The live system loads this file at startup via _load_best_params()
    in websocket_main.py.  If the file is missing the live system falls
    back silently to config_parameters.py defaults.

    IMPORTANT: do NOT commit best_params.json to git — it is sample-specific.
    Add 'backtest/results/best_params.json' to .gitignore.
    """
    payload = {
        "hmm_lookback_rows": int(best_row["hmm_lookback_rows"]),
        "hmm_max_regimes":   int(best_row["hmm_max_regimes"]),
        "vwap_window":       int(best_row["vwap_window"]),
        "fee_rate":          float(best_row["fee_rate"]),
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "source_metric":     _RANK_METRIC,
        "source_value":      float(best_row[_RANK_METRIC]),
    }
    with _BEST_PARAMS_PATH.open("w") as fh:
        json.dump(payload, fh, indent=2)
    log.info(
        "Best params saved → %s  (%s = %.4f)",
        _BEST_PARAMS_PATH,
        _RANK_METRIC,
        payload["source_value"],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sensitivity(full_grid: bool = False) -> pd.DataFrame:
    """
    Execute the sensitivity sweep and return the results DataFrame.

    Parameters
    ----------
    full_grid : bool
        If ``True``, run the full 24-combination factorial grid (Phase 2).
        If ``False`` (default), run the OAT sweep (Phase 1, 6 combinations).

    Returns
    -------
    pd.DataFrame
        One row per combination, sorted by ``_RANK_METRIC`` descending.
    """
    if full_grid:
        grid = _build_full_grid()
        mode = "full-grid"
        log.info("Phase 2 — full factorial grid: %d combinations.", len(grid))
    else:
        grid = _build_oat_grid()
        mode = "OAT"
        log.info("Phase 1 — OAT sweep: %d combinations.", len(grid))

    log.info(
        "NOTE: Use Case A only (30-day window).  "
        "Use Case B (180-day backtest validation) is DEFERRED — "
        "runtime would be ~4–12 h for OAT alone on a laptop."
    )

    all_results: list[dict[str, Any]] = []
    for i, params in enumerate(grid, 1):
        log.info("── Run %d / %d ──", i, len(grid))
        try:
            result = _run_one(params)
            all_results.append(result)
        except Exception as exc:
            # Broad catch is intentional: lets the sweep continue even if one
            # combination fails (e.g. HMM fit error on a flat overnight window,
            # Binance API timeout, or empty signals DataFrame).
            # exc_info=True prints the full traceback so the root cause is clear.
            log.error(
                "Run %d / %d failed — skipping.  params=%s",
                i, len(grid), params,
                exc_info=True,
            )
            # Store the params with NaN metrics so the row still appears in the
            # CSV and the OAT sensitivity report, visibly marked as failed.
            all_results.append({**params, _RANK_METRIC: float("nan")})

    results_df = pd.DataFrame(all_results).sort_values(
        _RANK_METRIC, ascending=False, na_position="last"
    )

    _print_table(results_df, mode)

    if not full_grid:
        # OAT: compute per-parameter sensitivity vs baseline
        defaults = {k: v[0] for k, v in _PARAM_GRID.items()}
        baseline_rows = results_df[
            (results_df["hmm_lookback_rows"] == defaults["hmm_lookback_rows"])
            & (results_df["hmm_max_regimes"] == defaults["hmm_max_regimes"])
            & (results_df["vwap_window"] == defaults["vwap_window"])
            & (results_df["fee_rate"] == defaults["fee_rate"])
        ]
        if not baseline_rows.empty:
            baseline_sharpe = float(baseline_rows.iloc[0][_RANK_METRIC])
            trigger = _oat_sensitivity_report(results_df, baseline_sharpe)
            if trigger:
                log.warning(
                    "At least one parameter exceeds the |ΔSharpe| > %.1f threshold. "
                    "Consider running Phase 2 with --full-grid.",
                    _OAT_SENSITIVITY_THRESHOLD,
                )

    _save_results(results_df, mode)

    # Save best params (top row after sorting by _RANK_METRIC)
    valid = results_df[results_df[_RANK_METRIC].notna()]
    if not valid.empty:
        _save_best_params(valid.iloc[0])
    else:
        log.warning("No valid results to save as best_params.json.")

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis for the BTCUSDT backtesting pipeline.\n"
            "Default: OAT sweep (Phase 1, 6 combinations, ~36–108 min).\n"
            "Use --full-grid for the full 24-combination factorial grid (Phase 2).\n\n"
            "NOTE — Use Case B (180-day window) is DEFERRED due to runtime:\n"
            "  OAT 6 runs × ~45–120 min/run = ~4–12 hours on a laptop.\n"
            "  Only Use Case A (30-day window) is run here."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the full 24-combination factorial grid (Phase 2) instead of OAT.",
    )
    args = parser.parse_args()
    run_sensitivity(full_grid=args.full_grid)

