"""
backtest/sensitivity.py
-----------------------
Step 8 — Sensitivity Analysis  (Use Case A: live parameter tuning)

Runs the full backtest pipeline (run_signals → simulate_pnl → _compute_stats)
across a grid of parameter values to answer one question:

    "Does the strategy's performance degrade gracefully as each tunable
     parameter shifts away from its current default value?"

Three execution modes
---------------------
  Bayesian optimisation (DEFAULT, Phase 2 replacement):
      Uses Optuna TPE sampler to intelligently search the parameter space.
      Default 40 trials (~5–8 h on a laptop, ~8–12 min per trial).
      Optuna TPE warm-up uses 10 random trials; exploitation begins at trial 11.
      The equivalent exhaustive grid would be 3×2×12×9 = 648 combos (weeks of compute);
      Bayes explores that same space intelligently in 40 trials.
        ↳ klines fetched ONCE then shared across all trials.
      Run with:  python -m backtest.sensitivity
             or: python -m backtest.sensitivity --bayes
             or: python -m backtest.sensitivity --bayes --n-trials 40

  OAT sweep (Phase 1 — quick sanity check):
      Holds all parameters at their defaults and varies ONE parameter at a time.
      8 runs total (1 baseline + 7 non-default).  Use this first to confirm the
      pipeline runs cleanly and to spot obviously sensitive parameters.
      Typical wall time: ~1–2 h on a laptop (90-day window, ~8–12 min per run).
        ↳ klines fetched ONCE (~30–90 s) then shared across all 8 runs.
      Run with:  python -m backtest.sensitivity --oat

  Full factorial grid (DEPRECATED — Phase 2 legacy):
      All combinations exhaustively.  Superseded by --bayes.
      30 combinations (3×2×5). Typical wall time: ~4–6 h on a laptop.
      Run with:  python -m backtest.sensitivity --full-grid

Use Case A vs Use Case B
------------------------
  USE CASE A — Live tuning  [IMPLEMENTED HERE]
      Window: SENSITIVITY_LOOKBACK = "90 days ago UTC" (~129,600 rows at 1 m).
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
  File:    backtest/reporting/sensitivity_<timestamp>.csv  (all metrics, all runs).
  File:    backtest/reporting/optuna_*.html                (Optuna diagnostic charts).
  File:    backtest/results/best_params.json               (winning parameter set).
           Loaded by websocket_main.py at startup via _load_best_params().
  File:    backtest/results/optuna.db                      (Optuna study state — resumable).

Notes
-----
- config_parameters.py and the live system are NEVER modified by this script.
  All overrides are passed as keyword arguments to run_signals() / simulate_pnl().
- SENSITIVITY_REFIT_EVERY (480 iterations = 40 h at 5 m) equals REFIT_EVERY so the IS
  optimisation and OOS validation (runner.py) share the same HMM refit cadence — ~162
  refits over the 270-day IS window, ~54 over the 90-day OOS window.  IS↔OOS Sharpe
  figures are therefore directly comparable.
- Do NOT commit best_params.json to git — it is sample-specific.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

import optuna
import pandas as pd

from backtest.data import fetch_macro_klines, fetch_micro_klines, flush_kline_cache
from backtest.pnl import compute_buy_and_hold, simulate_pnl
from backtest.reporting.formatters import (
    print_bnh_comparison,
    print_oat_sensitivity_report,
    print_sensitivity_table,
)
from backtest.signals import run_signals
from backtest.visualization import plot_backtest
from config_parameters import (
    BACKTEST_INITIAL_BTC,
    BACKTEST_LOOKBACK,
    BACKTEST_OOS_START,
    SENSITIVITY_REFIT_EVERY,
    SENSITIVITY_PREDICT_EVERY,
    SENSITIVITY_FEE_RATE,
    SENSITIVITY_RANK_METRIC,
    SENSITIVITY_OAT_THRESHOLD,
    VWAP_THRESHOLD_MULTIPLIER,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
# _RESULTS_DIR  — machine-readable artefacts consumed by the live system and
#                 Optuna (best_params.json, optuna.db).  NOT for human reports.
# _REPORTING_DIR — human-readable summaries: sensitivity CSVs and Optuna HTML
#                  charts.  Lives under backtest/reporting/ alongside formatters.py.
_RESULTS_DIR = pathlib.Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

_REPORTING_DIR = pathlib.Path(__file__).parent / "reporting"
_REPORTING_DIR.mkdir(exist_ok=True)

_BEST_PARAMS_PATH = _RESULTS_DIR / "best_params.json"

# ---------------------------------------------------------------------------
# Parameter grid definition
# ---------------------------------------------------------------------------
# Each entry is (config_constant_name, list_of_values_to_test).
# The first value in every list is the DEFAULT — used as the baseline run
# and held fixed while other parameters are varied in the OAT sweep.
#
# fee_rate and vwap_threshold are NOT in this grid:
#   fee_rate      — fixed at SENSITIVITY_FEE_RATE (0.001 = standard Binance taker).
#                   Testing different fee tiers is meaningless: Binance charges
#                   what it charges regardless of what value we use here.
#   vwap_threshold — fixed at VWAP_THRESHOLD_MULTIPLIER (0.003) for OAT / full-grid.
#                    Tuned by Bayesian search (_OPTUNA_SPACE); not varied in OAT/full-grid
#                    because a single non-default value per sweep is already informative.

_PARAM_GRID: dict[str, list[Any]] = {
    "hmm_lookback_rows": [120, 60, 30],  # default 120 (2 h)
    "hmm_max_regimes": [3, 2],  # default=3; test 2
    "vwap_window": [
        20,
        5,
        10,
        40,
        60,
    ],  # default=20 min; tests short (5,10) and long (40,60)
}

_OPTUNA_SPACE: dict[str, tuple] = {
    "hmm_lookback_rows": ("int", 30, 240, 10),
    "hmm_max_regimes": ("int", 2, 4, 1),
    "vwap_window": (
        "int",
        5,
        60,
        5,
    ),  # 12 values: 5,10,15…60 min; step=5 keeps trials tractable
    "vwap_threshold": ("float", 0.001, 0.005, 0.0005),
    # trend_consecutive_bars and trend_cooldown_bars removed from Optuna search space
    # (2026-05-24): fixed at TREND_CONSECUTIVE_BARS=3 / TREND_COOLDOWN_BARS=4 in
    # config_parameters.py based on the best values found in the first Optuna run.
    # Iterative optimisation: tune 4 core params first, revisit trend params later.
}


# ---------------------------------------------------------------------------
# Existing-results guard
# ---------------------------------------------------------------------------


def _check_existing_best_params(mode: str, extra_note: str = "") -> bool:
    """
    Warn the user if ``best_params.json`` already exists from a previous
    run, then ask whether to continue.

    Called at the start of every execution path (OAT, full-grid, and Bayes).
    Returns ``True`` if the caller should proceed, ``False`` if the user
    chose to abort.

    Parameters
    ----------
    mode : str
        Label shown in the warning box (e.g. ``"OAT"``, ``"full-grid"``,
        ``"bayes (adding 30 more trials)"``).
    extra_note : str
        Optional one-line note appended inside the box before the prompt
        (e.g. to explain resume behaviour for ``--bayes``).
    """
    if not _BEST_PARAMS_PATH.exists():
        return True  # no existing output — proceed unconditionally

    try:
        with _BEST_PARAMS_PATH.open() as fh:
            best = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return True  # unreadable file — proceed normally

    generated_at_str = best.get("generated_at", "unknown")
    sharpe = best.get("source_value", float("nan"))
    metric = best.get("source_metric", "sharpe_ratio")

    # Compute age in days if the timestamp is parseable
    age_str = ""
    try:
        generated_dt = datetime.fromisoformat(generated_at_str)
        age_days = (datetime.now(timezone.utc) - generated_dt).days
        age_str = f" ({age_days} day{'s' if age_days != 1 else ''} ago)"
    except Exception:
        pass

    print()
    print("╔" + "═" * 78 + "╗")
    print("║  ⚠️   EXISTING BEST PARAMS DETECTED" + " " * 41 + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  Generated : {generated_at_str}{age_str}".ljust(79) + "║")
    print(f"║  {metric:<18}: {sharpe:.4f}".ljust(79) + "║")
    print(
        f"║  hmm_lookback_rows : {best.get('hmm_lookback_rows', '?')}".ljust(79) + "║"
    )
    print(f"║  hmm_max_regimes   : {best.get('hmm_max_regimes', '?')}".ljust(79) + "║")
    print(f"║  vwap_window       : {best.get('vwap_window', '?')}".ljust(79) + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  You are about to run: {mode:<54}║")
    print("║  This will overwrite best_params.json if it finds a better result.  ║")
    if extra_note:
        print(f"║  {extra_note:<76}║")
    print("╚" + "═" * 78 + "╝")
    print()

    try:
        answer = input("  Continue? [y/N]: \n").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Non-interactive environment detected — aborting.")
        return False

    if answer in ("y", "yes"):
        log.info("User confirmed — proceeding with %s.", mode)
        return True

    log.info("User aborted — keeping existing best_params.json.")
    return False


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

    Total: 1 + sum(len(v) - 1 for v in grid) = 1 + 2 + 1 + 4 = 7 non-default → 8 runs.
      hmm_lookback_rows: 2 non-default  [60, 30]
      hmm_max_regimes:   1 non-default  [2]
      vwap_window:       4 non-default  [5, 10, 40, 60]
    fee_rate and vwap_threshold are held fixed at SENSITIVITY_FEE_RATE / VWAP_THRESHOLD_MULTIPLIER.
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

    Total: 3 × 2 × 5 = 30 combinations.
      hmm_lookback_rows: [120, 60, 30]           → 3
      hmm_max_regimes:   [3, 2]                  → 2
      vwap_window:       [20, 5, 10, 40, 60]     → 5
    fee_rate and vwap_threshold are held fixed at SENSITIVITY_FEE_RATE / VWAP_THRESHOLD_MULTIPLIER.
    """
    keys = list(_PARAM_GRID.keys())
    return [
        dict(zip(keys, combo)) for combo in itertools.product(*_PARAM_GRID.values())
    ]


def _make_objective(prefetched_macro: pd.DataFrame, prefetched_micro: pd.DataFrame):
    """
    Factory that closes over ``prefetched_macro`` and ``prefetched_micro`` and
    returns the Optuna objective.

    The inner ``objective`` function is called once per trial by
    ``study.optimize()``.  It builds the parameter dict from Optuna's
    suggestions (driven by ``_OPTUNA_SPACE``), runs the full backtest via
    ``_run_one()``, and returns the Sharpe ratio.

    Parameters tuned by Optuna
    --------------------------
    ``hmm_lookback_rows`` — rolling HMM warm-up window (5-minute bars).
    ``hmm_max_regimes``   — upper bound on hidden states in BIC search.
    ``vwap_window``       — rolling VWAP window (1-minute bars).
    ``vwap_threshold``    — dead-zone half-width around VWAP (fraction; e.g. 0.003 = 0.30 %).
                            Controls how selective the BUY/SELL gate is.  Higher values →
                            fewer but higher-quality signals → less fee drag in trending markets.

    ``fee_rate`` is always fixed at ``SENSITIVITY_FEE_RATE`` regardless.

    Error handling
    --------------
    If ``_run_one()`` raises (e.g. HMM EM divergence, empty signals from an
    unusual market period) the exception is caught, a WARNING is logged, and
    ``-inf`` is returned so Optuna marks the trial as complete but bad rather
    than crashing the whole study.  This mirrors the ``except`` block in
    ``run_sensitivity()`` that appends ``NaN`` for failed grid runs.
    """

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for name, (kind, low, high, step) in _OPTUNA_SPACE.items():
            if kind == "int":
                params[name] = trial.suggest_int(
                    name=name, low=low, high=high, step=step
                )
            else:  # "float"
                params[name] = trial.suggest_float(
                    name=name, low=low, high=high, step=step
                )
        # fee_rate is fixed; vwap_threshold is now tuned by Optuna (in params above).
        log.info("Trial %d starting: %s", trial.number, params)
        try:
            result = _run_one(params, prefetched_macro, prefetched_micro)
        except Exception:
            log.warning(
                "Trial %d failed (params=%s) — returning -inf so Optuna skips "
                "this region.  Full traceback follows.",
                trial.number,
                params,
                exc_info=True,
            )
            return float("-inf")
        sharpe = result.get("sharpe_ratio", float("nan"))
        # NaN → -inf so Optuna treats it as a failed trial without crashing.
        return float("-inf") if (sharpe != sharpe) else sharpe

    return objective


def _compute_bnh(df_micro: pd.DataFrame) -> dict[str, Any]:
    """
    Compute the buy-and-hold benchmark exactly once for a fixed price window.

    Parameters
    ----------
    df_micro : pd.DataFrame
        Raw 1-minute klines DataFrame (execution frame).  The ``close``
        column is used for entry/exit prices.  Passed to
        ``compute_buy_and_hold()`` which only requires ``close``.

    Returns
    -------
    dict
        Keys: ``bnh_entry_price``, ``bnh_exit_price``, ``bnh_btc_held``,
        ``bnh_final_equity_usdt``, ``bnh_total_return_pct``.
    """
    # Pass the same initial_btc as simulate_pnl so BnH and strategy share
    # the same initial-portfolio denominator — makes the % figures comparable.
    bnh = compute_buy_and_hold(df_micro, initial_btc=BACKTEST_INITIAL_BTC)
    log.info(
        "Buy-and-hold benchmark: entry=%.2f  exit=%.2f  return=%.2f%%",
        bnh["bnh_entry_price"],
        bnh["bnh_exit_price"],
        bnh["bnh_total_return_pct"],
    )
    return bnh


def _save_optuna_plots(study: optuna.Study) -> None:
    """
    Save the three standard Optuna diagnostic charts as self-contained HTML
    files to ``backtest/results/``.  Called automatically after every Bayesian
    study completes.

    Charts produced
    ---------------
    *  ``optuna_history_<ts>.html``    — objective value vs trial number.
       Shows whether the study is converging or still exploring.
    *  ``optuna_importance_<ts>.html`` — parameter importance (fANOVA).
       Answers "which parameter matters most for Sharpe?".
    *  ``optuna_contour_<ts>.html``    — 2-D contour of the two most
       important parameters (``hmm_lookback_rows`` × ``vwap_window``).
       Shows where the optimum lies in the joint space.

    Requires ``plotly`` (already a project dependency via ``visualization.py``).
    If fewer than 2 completed trials exist the importance / contour plots are
    skipped with a WARNING (fANOVA needs at least 2 data points).
    """
    try:
        from optuna.visualization import (
            plot_contour,
            plot_optimization_history,
            plot_param_importances,
        )
    except ImportError:
        log.warning(
            "optuna.visualization requires plotly — "
            "run 'pip install plotly' to enable Step 7 charts."
        )
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # IS window label for chart titles
    _is_label = f"IS: {BACKTEST_LOOKBACK} → {BACKTEST_OOS_START}  |  run: {run_date}  |  {len(completed)} trials"

    # 1. Optimisation history — always available after ≥ 1 trial
    try:
        fig = plot_optimization_history(study)
        fig.update_layout(title=f"Optuna — Optimisation History  |  {_is_label}")
        path = _REPORTING_DIR / f"optuna_history_{ts}.html"
        fig.write_html(str(path))
        log.info("Optuna history chart → %s", path)
    except Exception as exc:
        log.warning("Could not save optimisation history chart: %s", exc)

    if len(completed) < 2:
        log.warning(
            "Fewer than 2 completed trials — skipping importance / contour charts "
            "(fANOVA requires at least 2 data points)."
        )
        return

    # 2. Parameter importance
    try:
        fig = plot_param_importances(study)
        fig.update_layout(title=f"Optuna — Parameter Importance  |  {_is_label}")
        path = _REPORTING_DIR / f"optuna_importance_{ts}.html"
        fig.write_html(str(path))
        log.info("Optuna importance chart → %s", path)
    except Exception as exc:
        log.warning("Could not save parameter importance chart: %s", exc)

    # 3. Contour — hmm_lookback_rows × vwap_window (the two continuous knobs)
    try:
        fig = plot_contour(study, params=["hmm_lookback_rows", "vwap_window"])
        fig.update_layout(
            title=f"Optuna — Contour (hmm_lookback_rows × vwap_window)  |  {_is_label}"
        )
        path = _REPORTING_DIR / f"optuna_contour_{ts}.html"
        fig.write_html(str(path))
        log.info("Optuna contour chart → %s", path)
    except Exception as exc:
        log.warning("Could not save contour chart: %s", exc)


def _run_sensitivity_optuna_study(
    n_trials: int = 40,
    lookback: str | None = None,
    force_save: bool = False,
) -> pd.DataFrame:
    """
    Run the Bayesian sensitivity study using Optuna TPE and return results.

    This is the DEFAULT execution path (used when no CLI flag is given or
    when ``--bayes`` is passed).  Replaces the deprecated full factorial grid.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials to run.  Each trial calls ``_run_one()`` once.
        Default 40 (~5–8 h on a laptop).
        Reduced from 60 to 40 after trend_consecutive_bars / trend_cooldown_bars
        were removed from the search space (now 4 params, was 6).
        TPE warm-up is 10 random trials; exploitation begins at trial 11.
        Use 60 for a thorough deep search.
    lookback : str | None
        dateutil string overriding ``SENSITIVITY_LOOKBACK`` for this run only.
        Example: ``"180 days ago UTC"`` for a deep-calibration run.
        ``None`` (default) → use ``SENSITIVITY_LOOKBACK`` from
        ``config_parameters.py``.

    Returns
    -------
    pd.DataFrame
        One row per completed trial, columns:
        ``hmm_lookback_rows``, ``hmm_max_regimes``, ``vwap_window``,
        ``fee_rate``, ``sharpe_ratio``.
        Sorted by ``sharpe_ratio`` descending.  Persisted to
        ``backtest/reporting/sensitivity_bayes_<ts>.csv`` and
        ``backtest/results/best_params.json``.

    Notes
    -----
    The Optuna study is persisted to ``backtest/results/optuna.db`` (SQLite).
    ``load_if_exists=True`` means the study resumes automatically if it was
    interrupted — previously completed trials are not re-run.
    """
    effective_lookback = lookback if lookback is not None else BACKTEST_LOOKBACK
    log.info(
        "Bayesian optimisation — %d trials  (IS window: '%s' → '%s')",
        n_trials,
        effective_lookback,
        BACKTEST_OOS_START,
    )

    # Warn if a previous result exists — bayes resumes the study so it adds
    # trials on top rather than restarting, but it still commits hours of
    # compute and may overwrite best_params.json with a new best.
    if not _check_existing_best_params(
        mode=f"bayes (adding {n_trials} more Optuna trials)",
        extra_note=(
            "The study resumes — prior trials are kept and result can only improve.  "
            "Use --force-save if the stored params are stale (negative OOS Sharpe)."
        ),
    ):
        return pd.DataFrame()

    # ── Pre-fetch data ONCE — shared across all n_trials calls to _run_one ──
    log.info(
        "Pre-fetching IS macro (5 m) + micro (1 m) klines for %d trials "
        "(window: '%s' → '%s')…",
        n_trials,
        effective_lookback,
        BACKTEST_OOS_START,
    )
    df_macro = fetch_macro_klines(
        lookback=effective_lookback, end_str=BACKTEST_OOS_START
    )
    df_micro = fetch_micro_klines(
        lookback=effective_lookback, end_str=BACKTEST_OOS_START
    )
    log.info(
        "IS data ready: macro=%d 5-min bars, micro=%d 1-min bars "
        "(window: '%s' → '%s'). Shared across all %d trials — no further API calls.",
        len(df_macro),
        len(df_micro),
        effective_lookback,
        BACKTEST_OOS_START,
        n_trials,
    )

    # ── Buy-and-hold benchmark — computed ONCE (price window is fixed) ────
    bnh = _compute_bnh(df_micro)

    # ── Create / resume study ─────────────────────────────────────────────
    # The study name encodes the data window's START date so that every unique
    # lookback window gets its own isolated study in the DB.
    #
    # WHY THIS MATTERS:
    #   Optuna stores Sharpe values per trial. If the same study is resumed two
    #   days later, the old trials carry Sharpe values computed on a *different*
    #   price window (e.g. Feb 7–May 8) while new trials use today's window
    #   (Feb 9–May 10). study.best_value then picks the best across mixed windows
    #   — the winning trial's Sharpe is stale and the saved best_params.json
    #   reflects parameters that were optimal for a window that no longer exists.
    #
    # FIX: name the study after the window start date. Running on May 8 creates
    #   "btcusdt_sensitivity_20260208". Running on May 10 creates a FRESH study
    #   "btcusdt_sensitivity_20260210". Interrupted runs on the same day still
    #   resume correctly (same study name). Old per-day studies stay in the DB
    #   as an audit trail but never pollute a new day's optimisation.
    import re as _re

    _m = _re.match(r"(\d+)\s+days?\s+ago", effective_lookback, _re.IGNORECASE)
    if _m:
        window_start_dt = pd.Timestamp.now("UTC") - pd.Timedelta(days=int(_m.group(1)))
    else:
        # Fallback: use today as tag (study is still isolated per calendar day)
        window_start_dt = pd.Timestamp.now("UTC")
    window_tag = window_start_dt.strftime("%Y%m%d")
    study_name = f"btcusdt_sensitivity_{window_tag}"

    _optuna_db = (_RESULTS_DIR / "optuna.db").as_posix()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{_optuna_db}",
        load_if_exists=True,  # resume if interrupted ON THE SAME DAY — trials not re-run
        sampler=optuna.samplers.TPESampler(seed=42),  # type: ignore[arg-type]
    )

    already_done = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    if already_done:
        log.info(
            "Resuming study '%s' — %d trials already complete on this window, running %d more.",
            study_name,
            already_done,
            n_trials,
        )
    else:
        log.info("Fresh study '%s' — no prior trials for this window.", study_name)

    study.optimize(
        _make_objective(prefetched_macro=df_macro, prefetched_micro=df_micro),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ── Step 7: diagnostic charts (always saved) ─────────────────────────
    _save_optuna_plots(study)

    # ── Convert completed trials → DataFrame ─────────────────────────────
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append(
                {
                    **t.params,  # hmm_lookback_rows, hmm_max_regimes, vwap_window, vwap_threshold
                    # fee_rate is fixed — recorded for CSV self-documentation but never varied.
                    # vwap_threshold is now a Bayesian knob — its value comes from t.params above.
                    "fee_rate": SENSITIVITY_FEE_RATE,
                    SENSITIVITY_RANK_METRIC: t.value,
                }
            )

    if not rows:
        log.warning("No completed trials — returning empty DataFrame.")
        return pd.DataFrame()

    results_df = (
        pd.DataFrame(rows)
        .sort_values(SENSITIVITY_RANK_METRIC, ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    # Broadcast the B&H benchmark columns onto every row — same value for all
    # trials since the price window is identical across the whole study.
    for k, v in bnh.items():
        results_df[k] = v

    log.info(
        "Best trial: %s = %.4f  (params: %s)",
        SENSITIVITY_RANK_METRIC,
        study.best_value,
        study.best_params,
    )

    # ── Reuse shared helpers ──────────────────────────────────────────────
    print_sensitivity_table(
        results_df,
        mode="bayes",
        param_grid=_PARAM_GRID,
        display_cols=_DISPLAY_COLS,
        rank_metric=SENSITIVITY_RANK_METRIC,
    )
    _save_results(results_df, mode="bayes")

    valid = results_df[results_df[SENSITIVITY_RANK_METRIC].notna()]
    if not valid.empty:
        # Re-run the best trial's params on current data BEFORE saving.
        # This ensures best_params.json always reflects today's window, not a
        # stale Sharpe from a previous day that still lives in the Optuna DB.
        # _run_one_full also returns signals/trades/equity for the IS chart.
        log.info(
            "Re-running best trial params to compute full stats for B&H comparison and IS chart…"
        )
        best_signals = best_trades = best_equity = None
        try:
            best_full, best_signals, best_trades, best_equity = _run_one_full(
                study.best_params, df_macro, df_micro
            )
            best_series = pd.Series({**best_full, **bnh})
            # Save using the CURRENT-data stats so source_value is honest.
            _save_best_params(best_series, force_save=force_save)
        except Exception:
            log.warning(
                "Could not re-run best trial — falling back to trial row "
                "(B&H comparison may be skipped and IS chart not saved).",
                exc_info=True,
            )
            best_series = valid.iloc[0]
            _save_best_params(best_series, force_save=force_save)
        print_bnh_comparison(best_series)
        if best_signals is not None:
            assert (
                best_trades is not None and best_equity is not None
            )  # set together with best_signals
            _plot_is_chart(
                best_params=study.best_params,
                best_stats=best_series.to_dict(),
                signals=best_signals,
                trades=best_trades,
                equity=best_equity,
            )
    else:
        log.warning("No valid Optuna trials to save as best_params.json.")

    return results_df


# ---------------------------------------------------------------------------


def _run_one(
    params: dict[str, Any],
    prefetched_macro: pd.DataFrame,
    prefetched_micro: pd.DataFrame,
) -> dict[str, Any]:
    """
    Execute one full backtest with the given parameter overrides.

    Parameters
    ----------
    params : dict
        Must contain: ``hmm_lookback_rows``, ``hmm_max_regimes``, ``vwap_window``.
        May optionally contain ``vwap_threshold`` (Bayes trials supply it;
        OAT / full-grid runs do not — defaults to ``VWAP_THRESHOLD_MULTIPLIER``).
        ``fee_rate`` is always fixed at ``SENSITIVITY_FEE_RATE`` regardless.
    prefetched_macro : pd.DataFrame
        Raw 5-minute OHLCV klines pre-fetched once before the grid loop.
        Passed to ``run_signals()`` via ``prefetched_macro=`` to avoid
        one Binance API call per combination.
    prefetched_micro : pd.DataFrame
        Raw 1-minute OHLCV klines pre-fetched once before the grid loop.
        Passed to ``run_signals()`` via ``prefetched_micro=``.

    Returns
    -------
    dict
        The ``params`` dict merged with the stats dict from ``simulate_pnl()``.
    """
    _fee = SENSITIVITY_FEE_RATE
    _threshold = params.get("vwap_threshold", VWAP_THRESHOLD_MULTIPLIER)
    log.info(
        "Running: lookback=%d  max_regimes=%d  vwap=%d  threshold=%.4f%%  fee=%.4f%%",
        params["hmm_lookback_rows"],
        params["hmm_max_regimes"],
        params["vwap_window"],
        _threshold * 100,
        _fee * 100,
    )

    # run_signals() uses SENSITIVITY_REFIT_EVERY for a speedup: fewer full BIC
    # refits on the 5-minute macro walk-forward without changing relative rankings.
    # prefetched_macro / prefetched_micro are passed directly — no API calls.
    # trend_consecutive_bars / trend_cooldown_bars are NOT passed here — they are
    # fixed in config_parameters.py and run_signals() uses the config defaults.
    signals = run_signals(
        hmm_lookback_rows=params["hmm_lookback_rows"],
        hmm_max_regimes=params["hmm_max_regimes"],
        vwap_window=params["vwap_window"],
        vwap_threshold=_threshold,
        refit_every=SENSITIVITY_REFIT_EVERY,
        predict_every=SENSITIVITY_PREDICT_EVERY,
        prefetched_macro=prefetched_macro,
        prefetched_micro=prefetched_micro,
    )

    _, _, stats = simulate_pnl(signals, fee_rate=_fee)

    # Always record the actual fee and threshold used so the CSV/JSON are
    # self-documenting even when they weren't part of the grid sweep.
    return {**params, "fee_rate": _fee, "vwap_threshold": _threshold, **stats}


def _run_one_full(
    params: dict[str, Any],
    prefetched_macro: pd.DataFrame,
    prefetched_micro: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Like ``_run_one`` but also returns ``(signals, trades, equity)`` for
    visualisation.  Used only for the final best-params re-run at the end of
    the Bayesian / OAT / full-grid study so that ``plot_backtest()`` can draw
    the IS sensitivity chart without an extra API call.

    Returns
    -------
    result_dict : dict
        Same as ``_run_one()`` — params merged with stats.
    signals : pd.DataFrame
    trades : pd.DataFrame
    equity : pd.DataFrame
    """
    _fee = SENSITIVITY_FEE_RATE
    _threshold = params.get("vwap_threshold", VWAP_THRESHOLD_MULTIPLIER)
    log.info(
        "Re-running best params for IS chart: lookback=%d  max_regimes=%d  "
        "vwap=%d  threshold=%.4f%%  fee=%.4f%%",
        params["hmm_lookback_rows"],
        params["hmm_max_regimes"],
        params["vwap_window"],
        _threshold * 100,
        _fee * 100,
    )
    signals = run_signals(
        hmm_lookback_rows=params["hmm_lookback_rows"],
        hmm_max_regimes=params["hmm_max_regimes"],
        vwap_window=params["vwap_window"],
        vwap_threshold=_threshold,
        refit_every=SENSITIVITY_REFIT_EVERY,
        predict_every=SENSITIVITY_PREDICT_EVERY,
        prefetched_macro=prefetched_macro,
        prefetched_micro=prefetched_micro,
    )
    trades, equity, stats = simulate_pnl(signals, fee_rate=_fee)
    result_dict = {**params, "fee_rate": _fee, "vwap_threshold": _threshold, **stats}
    return result_dict, signals, trades, equity


# ---------------------------------------------------------------------------
# IS chart helper
# ---------------------------------------------------------------------------


def _plot_is_chart(
    best_params: dict[str, Any],
    best_stats: dict[str, Any],
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
) -> None:
    """
    Generate, save, and open the IS sensitivity chart using the best-params re-run data.
    The chart is saved to ``backtest/results/sensitivity_chart_<ts>.html`` (or ``.png``
    if kaleido is installed) **and** opened in a new browser tab via ``fig.show()``,
    matching the behaviour of ``runner.py`` so both charts can be compared side-by-side.

    Labelled ``"IS (Sensitivity)"`` in the title so it is visually distinct
    from the OOS chart produced by ``runner.py`` (labelled ``"Backtest"``).
    The date range in the title comes from the IS window (BACKTEST_LOOKBACK
    → BACKTEST_OOS_START), so both charts can be compared at a glance.
    """
    try:
        plot_backtest(
            signals,
            trades,
            equity,
            best_stats,
            save_png=True,  # triggers _save_figure → HTML fallback (kaleido absent) or PNG
            show=True,  # open a new browser tab — mirrors runner.py behaviour
            title_prefix="IS (Sensitivity)",
            file_prefix="sensitivity_chart",
        )
    except Exception:
        log.warning(
            "IS chart generation failed — chart not saved.  "
            "The best_params.json and sensitivity CSV are unaffected.",
            exc_info=True,
        )


_DISPLAY_COLS = [
    "hmm_lookback_rows",
    "hmm_max_regimes",
    "vwap_window",
    "vwap_threshold",  # fixed for OAT/grid; Bayes-optimised for Bayesian runs
    "total_return_pct",
    "bnh_total_return_pct",  # passive buy-and-hold benchmark
    "strategy_vs_bnh_pct",  # total_return_pct − bnh_total_return_pct (derived)
    "max_drawdown_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate_pct",
    "profit_factor",
    "n_round_trips",
    "avg_holding_minutes",
]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_results(results_df: pd.DataFrame, mode: str) -> pathlib.Path:
    """Write all results to a timestamped CSV file under backtest/reporting/."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = _REPORTING_DIR / f"sensitivity_{mode}_{ts}.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("Results saved → %s", csv_path)
    return csv_path


def _save_best_params(best_row: pd.Series, force_save: bool = False) -> None:
    """
    Write the winning parameter set to best_params.json.

    By default only overwrites the file if the new Sharpe ratio is strictly
    better than the value already stored.  This prevents a noisy re-run from
    destroying a previously healthy result.

    Pass ``force_save=True`` (via ``--force-save`` CLI flag) to bypass the
    guard when the market regime has shifted and the current IS window's best
    Sharpe is lower than the stored one — in that case the stale params are
    actively harmful and should be replaced with the current best.

    The live system loads this file at startup via _load_best_params()
    in websocket_main.py.  If the file is missing the live system falls
    back silently to config_parameters.py defaults.

    IMPORTANT: do NOT commit best_params.json to git — it is sample-specific.
    Add 'backtest/results/best_params.json' to .gitignore.
    """
    new_sharpe = float(best_row[SENSITIVITY_RANK_METRIC])

    # ── Guard: never overwrite a better existing result (unless forced) ──
    existing_sharpe = float("-inf")
    if _BEST_PARAMS_PATH.exists():
        try:
            with _BEST_PARAMS_PATH.open() as fh:
                existing = json.load(fh)
            existing_sharpe = float(existing.get("source_value", float("-inf")))
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # unreadable — treat as no prior

    if new_sharpe <= existing_sharpe:
        if force_save:
            log.warning(
                "--force-save active: overwriting existing %s (%.4f) with new (%.4f). "
                "Use this only when the market regime has shifted and stale params are "
                "actively harmful.",
                SENSITIVITY_RANK_METRIC,
                existing_sharpe,
                new_sharpe,
            )
        else:
            log.warning(
                "New %s (%.4f) ≤ existing (%.4f) — best_params.json NOT overwritten. "
                "Use --force-save to override when market conditions have changed.",
                SENSITIVITY_RANK_METRIC,
                new_sharpe,
                existing_sharpe,
            )
            return

    payload = {
        "hmm_lookback_rows": int(best_row["hmm_lookback_rows"]),
        "hmm_max_regimes": int(best_row["hmm_max_regimes"]),
        "vwap_window": int(best_row["vwap_window"]),
        "vwap_threshold": float(
            best_row.get("vwap_threshold", VWAP_THRESHOLD_MULTIPLIER)
        ),
        "fee_rate": float(best_row.get("fee_rate", SENSITIVITY_FEE_RATE)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_metric": SENSITIVITY_RANK_METRIC,
        "source_value": new_sharpe,
    }
    with _BEST_PARAMS_PATH.open("w") as fh:
        json.dump(payload, fh, indent=2)
    log.info(
        "Best params saved → %s  (%s = %.4f, improved from %.4f)",
        _BEST_PARAMS_PATH,
        SENSITIVITY_RANK_METRIC,
        new_sharpe,
        existing_sharpe,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_sensitivity(
    full_grid: bool = True, lookback: str | None = None, force_save: bool = False
) -> pd.DataFrame:
    """
    Execute the sensitivity sweep and return the results DataFrame.

    Parameters
    ----------
    full_grid : bool
        If ``True`` (default), run the full 36-combination factorial grid.
        If ``False``, run the OAT sweep (8 combinations, ~15–35 min).
    lookback : str | None
        dateutil string overriding ``BACKTEST_LOOKBACK`` (IS start) for this
        run only.  ``None`` → use ``BACKTEST_LOOKBACK`` from
        ``config_parameters.py``.  The IS end is always ``BACKTEST_OOS_START``.
    force_save: bool | None
        If ``True``, bypass the guard in ``_save_best_params()`` that prevents
        overwriting a better existing result.  Use this when the market regime
        has shifted and the current IS window's best Sharpe is lower than the
        stored one — in that case the stale params are actively harmful and
        should be replaced with the current best.  Default ``False`` (safe mode).
    Returns
    -------
    pd.DataFrame
        One row per combination, sorted by ``SENSITIVITY_RANK_METRIC`` descending.
    """
    effective_lookback = lookback if lookback is not None else BACKTEST_LOOKBACK
    if full_grid:
        grid = _build_full_grid()
        mode = "full-grid"
        log.info("Phase 2 — full factorial grid: %d combinations.", len(grid))
    else:
        grid = _build_oat_grid()
        mode = "OAT"
        log.info("Phase 1 — OAT sweep: %d combinations.", len(grid))

    # Warn early if a Bayesian result already exists — let user decide whether
    # the sweep is still necessary before committing to hours of compute.
    if not _check_existing_best_params(mode):
        return pd.DataFrame()

    log.info(
        "IS window: '%s' → '%s'.  OOS validation is runner.py's responsibility.",
        effective_lookback,
        BACKTEST_OOS_START,
    )

    log.info(
        "Pre-fetching IS macro (5 m) + micro (1 m) klines for all %d runs "
        "(window: '%s' → '%s')…",
        len(grid),
        effective_lookback,
        BACKTEST_OOS_START,
    )
    df_macro = fetch_macro_klines(
        lookback=effective_lookback, end_str=BACKTEST_OOS_START
    )
    df_micro = fetch_micro_klines(
        lookback=effective_lookback, end_str=BACKTEST_OOS_START
    )
    log.info(
        "IS data ready: macro=%d 5-min bars, micro=%d 1-min bars. "
        "Shared across all %d runs — no further API calls.",
        len(df_macro),
        len(df_micro),
        len(grid),
    )

    # ── Buy-and-hold benchmark — computed ONCE (price window is fixed) ────
    bnh = _compute_bnh(df_micro)

    all_results: list[dict[str, Any]] = []
    for i, params in enumerate(grid, 1):
        log.info("── Run %d / %d ──", i, len(grid))
        try:
            result = _run_one(
                params, prefetched_macro=df_macro, prefetched_micro=df_micro
            )
            all_results.append(result)
        except Exception:
            # Broad catch is intentional: lets the sweep continue even if one
            # combination fails (e.g. HMM fit error on a flat overnight window,
            # or empty signals DataFrame).
            # exc_info=True prints the full traceback so the root cause is clear.
            log.error(
                "Run %d / %d failed — skipping.  params=%s",
                i,
                len(grid),
                params,
                exc_info=True,
            )
            # Store the params with NaN metrics so the row still appears in the
            # CSV and the OAT sensitivity report, visibly marked as failed.
            all_results.append(
                {
                    **params,
                    "fee_rate": SENSITIVITY_FEE_RATE,
                    "vwap_threshold": VWAP_THRESHOLD_MULTIPLIER,
                    SENSITIVITY_RANK_METRIC: float("nan"),
                }
            )

    results_df = pd.DataFrame(all_results).sort_values(
        SENSITIVITY_RANK_METRIC, ascending=False, na_position="last"
    )

    # Broadcast the B&H benchmark columns onto every row — same value for all
    # runs since the price window is identical across the whole sweep.
    for k, v in bnh.items():
        results_df[k] = v

    print_sensitivity_table(
        results_df,
        mode=mode,
        param_grid=_PARAM_GRID,
        display_cols=_DISPLAY_COLS,
        rank_metric=SENSITIVITY_RANK_METRIC,
    )

    if not full_grid:
        # OAT: compute per-parameter sensitivity vs baseline
        defaults = {k: v[0] for k, v in _PARAM_GRID.items()}
        baseline_rows = results_df[
            (results_df["hmm_lookback_rows"] == defaults["hmm_lookback_rows"])
            & (results_df["hmm_max_regimes"] == defaults["hmm_max_regimes"])
            & (results_df["vwap_window"] == defaults["vwap_window"])
        ]
        if not baseline_rows.empty:
            baseline_sharpe = float(baseline_rows.iloc[0][SENSITIVITY_RANK_METRIC])
            trigger = print_oat_sensitivity_report(
                results_df,
                baseline_sharpe=baseline_sharpe,
                param_grid=_PARAM_GRID,
                rank_metric=SENSITIVITY_RANK_METRIC,
                sensitivity_threshold=SENSITIVITY_OAT_THRESHOLD,
            )
            if trigger:
                log.warning(
                    "At least one parameter exceeds the |ΔSharpe| > %.1f threshold. "
                    "Consider running Bayesian optimisation with --bayes for a wider search.",
                    SENSITIVITY_OAT_THRESHOLD,
                )

    _save_results(results_df, mode)

    # Save best params (top row after sorting by SENSITIVITY_RANK_METRIC)
    valid = results_df[results_df[SENSITIVITY_RANK_METRIC].notna()]
    if not valid.empty:
        _save_best_params(valid.iloc[0], force_save=force_save)
        print_bnh_comparison(valid.iloc[0])
        # Re-run best for IS chart — extract only the keys _run_one_full needs.
        _param_keys = list(_PARAM_GRID.keys()) + ["vwap_threshold"]
        best_row = valid.iloc[0]
        best_params_dict = {k: best_row[k] for k in _param_keys if k in best_row.index}
        try:
            best_full, best_signals, best_trades, best_equity = _run_one_full(
                best_params_dict, df_macro, df_micro
            )
            best_stats = {**best_row.to_dict(), **best_full}
            _plot_is_chart(
                best_params=best_params_dict,
                best_stats=best_stats,
                signals=best_signals,
                trades=best_trades,
                equity=best_equity,
            )
        except Exception:
            log.warning(
                "Could not generate IS chart for OAT/grid best run.", exc_info=True
            )
    else:
        log.warning("No valid results to save as best_params.json.")

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis for the BTCUSDT backtesting pipeline.\n\n"
            "DEFAULT (no flags): Bayesian optimisation via Optuna — 40 trials by default (~5–8 h).\n"
            "  Explores 3×2×12×9 = 648-combination-equivalent space intelligently.\n"
            "  Use --n-trials 20 for a faster ~2.5–4 h run.\n"
            "  Optuna diagnostic charts are always saved to backtest/reporting/.\n\n"
            "--oat        Phase 1 OAT sweep (8 combinations, ~1–2 h).\n"
            "--full-grid  Deprecated full factorial grid (30 combinations, ~4–6 h).\n"
            "--bayes      Explicit Bayesian search (same as default, accepts --n-trials).\n\n"
            "Data window (Option B — IS/OOS split at 5 m resolution):\n"
            "  IS  (optimisation): BACKTEST_LOOKBACK → BACKTEST_OOS_START\n"
            "      (360 days ago → 90 days ago = 270 days, ~77,760 rows at 5 m).\n"
            "  OOS (validation):   runner.py uses BACKTEST_OOS_START → today\n"
            "      (90 days, ~25,920 rows at 5 m) — sensitivity.py never sees it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--oat",
        action="store_true",
        help="Run the OAT sweep (Phase 1, 8 combinations, ~1–2 h).",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="[DEPRECATED] Run the full factorial grid (30 combinations, ~4–6 h). Use --bayes instead.",
    )
    parser.add_argument(
        "--bayes",
        action="store_true",
        help=(
            "Run Bayesian optimisation via Optuna (default when no flag is given). "
            "Combine with --n-trials to control the evaluation budget."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help=(
            "Number of Optuna trials for --bayes (default: 40, ~5–8 h). "
            "Reduced to 40 from 60 after trend_consecutive_bars / trend_cooldown_bars "
            "were removed from the search space (now 4 params: hmm_lookback_rows, "
            "hmm_max_regimes, vwap_window, vwap_threshold). "
            "TPE warm-up is 10 random trials; exploitation begins at trial 11. "
            "The study is resumable — interrupted runs continue from where they left off."
        ),
    )
    parser.add_argument(
        "--lookback",
        type=str,
        default=None,
        help=(
            "Override the data window for this run only. "
            "Accepts any dateutil string, e.g. '180 days ago UTC' for a deep-calibration "
            "run (~2× slower but more robust). "
            f"Default: BACKTEST_LOOKBACK from config_parameters.py ('{BACKTEST_LOOKBACK}')."
        ),
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help=(
            "Delete all cached kline Parquet files under cache/klines/ and exit. "
            "Use this whenever BACKTEST_LOOKBACK or BACKTEST_OOS_START change "
            "to force a fresh download on the next run."
        ),
    )
    parser.add_argument(
        "--force-save",
        action="store_true",
        help=(
            "Bypass the Sharpe-improvement guard in _save_best_params and always "
            "overwrite best_params.json with the current run's best result, even if "
            "its IS Sharpe is lower than the stored value.  Use this when the market "
            "regime has shifted and the stored params are producing negative OOS "
            "performance — keeping stale params is worse than accepting a lower IS Sharpe."
        ),
    )
    args = parser.parse_args()

    # --flush-cache: clear Parquet cache and exit immediately (no backtest run).
    if args.flush_cache:
        flush_kline_cache()
        print("Cache flushed. Run sensitivity.py again to fetch fresh klines.")
        sys.exit(0)

    if args.oat:
        run_sensitivity(full_grid=False, lookback=args.lookback, force_save=args.force_save)
    elif args.full_grid:
        log.warning(
            "--full-grid is deprecated. Consider using --bayes for smarter, "
            "faster optimisation with a wider search space."
        )
        run_sensitivity(full_grid=True, lookback=args.lookback, force_save=args.force_save)
    else:
        # Default path: --bayes or no flag at all → Bayesian optimisation
        _run_sensitivity_optuna_study(
            n_trials=args.n_trials,
            lookback=args.lookback,
            force_save=args.force_save,
        )
