"""
backtest/diagnostics/regime_validation.py
-----------------------------------------
Step 6b — Offline Long-Horizon Regime Validation.

Standalone diagnostic script that tests whether the HMM regime labels
produced by ``RegimeDirector`` remain statistically meaningful over a
**fully out-of-sample** horizon (3-day test set, model frozen after a
one-time fit on the preceding 7 days).

**Not wired into ``run_backtest.py``** — this is a standalone diagnostic tool.
Run manually with:

    python -m backtest.diagnostics.regime_validation

Why is this in ``backtest/diagnostics/`` and not ``backtest/``?
    All files directly in ``backtest/`` (``signals.py``, ``pnl.py``,
    ``run_backtest.py``, etc.) form the main pipeline orchestrated by
    ``run_backtest()``.  This script is a separate, one-off health check
    for the HMM model and must be invoked independently.

See ``BACKTESTING.md`` (Step 6b) for the full rationale, dataset split,
and interpretation guidance.

NOTE: This script **bypasses** ``BACKTEST_MAX_ROWS`` intentionally.
The full ~14,400-row dataset (10 days at 1 m) is required for the
7-day / 3-day split to be statistically meaningful.

When to re-run
--------------
Re-run this tool whenever ``strategy/regime_director.py`` is modified
(feature columns, BIC search range, label-assignment rules, confidence
threshold, etc.) to confirm the frozen model still produces statistically
meaningful labels on out-of-sample data.

Written with the assistance of AI models — results should be reviewed
critically before drawing conclusions.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from backtest.data import fetch_klines
from backtest.signals import _add_hmm_features
from backtest.reporting.formatters import print_regime_validation_report
from strategy.regime_director import RegimeDirector

from config_parameters import (
    HMM_LOOKBACK_ROWS,
    HMM_MIN_CONFIDENCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Constants local to this diagnostic — not in config_parameters.py because
# they are specific to the one-off validation, not the live system.
# ---------------------------------------------------------------------------
_TRAIN_DAYS = 7
_CANDLES_PER_DAY = 1440  # 24 h × 60 min
_SPLIT_IDX = _TRAIN_DAYS * _CANDLES_PER_DAY  # 10,080


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Data Fetch + One-time Training Fit
# ═══════════════════════════════════════════════════════════════════════════


def _fetch_and_prepare() -> pd.DataFrame:
    """
    Download the full 10-day kline dataset and add HMM features.

    Bypasses ``BACKTEST_MAX_ROWS`` — the entire ~14,400-row DataFrame is
    returned so that the 7/3 split has enough data to be meaningful.

    Returns:
        pd.DataFrame: Feature-enriched klines (``return``, ``volatility``,
            ``obi_proxy``, ``trade_density`` columns appended; NaN rows
            dropped).
    """
    klines = fetch_klines()
    features_df = _add_hmm_features(klines)
    logging.info(
        "Regime validation: fetched %d raw klines → %d rows after features.",
        len(klines),
        len(features_df),
    )
    return features_df


def _train_model(features_df: pd.DataFrame) -> RegimeDirector:
    """
    Fit the HMM **once** on the first 7 days (train set) and freeze it.

    The model, scaler, and label mapping are frozen after this call.
    They are **never re-fitted** during Phase 2.

    Args:
        features_df: Full 10-day feature DataFrame.

    Returns:
        RegimeDirector: Fitted director with ``model``, ``scaler``,
            ``regime_label``, and ``regime_confidence`` populated.
    """
    train_df = features_df.iloc[:_SPLIT_IDX]
    logging.info(
        "Phase 1: training on first %d days (%d rows).  "
        "Using last %d rows (2 h window) for the HMM fit.",
        _TRAIN_DAYS,
        len(train_df),
        HMM_LOOKBACK_ROWS,
    )

    rd = RegimeDirector()
    # Pass only the last HMM_LOOKBACK_ROWS (120) rows of the train period.
    # select_hmm_model() internally splits these into:
    #   fit     → rows[:HMM_TRAIN_ROWS]  (first 80 — in-sample)
    #   predict → rows[HMM_TRAIN_ROWS:]  (last ~40 — out-of-sample)
    # The 7-day boundary guarantees no test-set candle leaks into the model;
    # within the train period we use the most recent 2 h window, identical
    # to how the live system and signals.py feed RegimeDirector.
    rd.klines_df = train_df.iloc[-HMM_LOOKBACK_ROWS:]
    rd.select_hmm_model()
    rd.assign_regime_labels()

    logging.info(
        "Phase 1 complete — best n=%d, regime='%s', confidence=%.2f.  "
        "Model and scaler FROZEN from this point onward.",
        rd.model.n_components if rd.model else -1,
        rd.regime_label,
        rd.regime_confidence or 0.0,
    )
    return rd


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Rolling Label Assignment on Test Set (frozen model)
# ═══════════════════════════════════════════════════════════════════════════


def _rolling_predict(
    features_df: pd.DataFrame,
    rd: RegimeDirector,
) -> pd.DataFrame:
    """
    Assign regime labels to every candle in the 3-day test set using the
    **frozen** model from Phase 1 — no ``select_hmm_model()`` refit.

    For each candle ``t`` starting at ``_SPLIT_IDX + HMM_LOOKBACK_ROWS``:

    1. Slice ``features_df[t - HMM_LOOKBACK_ROWS : t]`` — the same 120-row
       rolling window the live system uses.
    2. Assign the slice to ``rd.klines_df``.
    3. Call ``rd.predict_current_regime()`` (cheap Viterbi pass, frozen model).
    4. Call ``rd.assign_regime_labels()`` to map the state index → label.
    5. Record ``(timestamp, regime_label, regime_confidence)``.

    The first ``HMM_LOOKBACK_ROWS`` candles of the test set are skipped
    (warm-up) so the rolling window never reaches into the training period.

    Args:
        features_df: Full 10-day feature DataFrame (from ``_fetch_and_prepare``).
        rd: ``RegimeDirector`` with frozen ``model`` and ``scaler``
            (from ``_train_model``).

    Returns:
        pd.DataFrame: One row per labelled test candle, indexed by
            ``timestamp``, with columns:
            ``regime_label`` (str), ``regime_confidence`` (float),
            ``close`` (candle close price for forward-return calculation).
    """
    start = _SPLIT_IDX + HMM_LOOKBACK_ROWS  # first valid index in the test set
    total = len(features_df)
    n_candles = total - start

    logging.info(
        "Phase 2: rolling predict on %d test candles (rows %d … %d).  "
        "Model is FROZEN — predict_current_regime() only, no refit.",
        n_candles,
        start,
        total - 1,
    )

    # Temporarily silence per-iteration RegimeDirector logs to avoid
    # flooding stdout with ~4,200 lines.  Progress is logged every 500 iter.
    rd_logger = logging.getLogger("strategy.regime_director")
    original_level = rd_logger.level
    rd_logger.setLevel(logging.WARNING)

    records: list[dict] = []

    for t in range(start, total):
        # 120-row rolling window — identical to the live system's feed.
        rd.klines_df = features_df.iloc[t - HMM_LOOKBACK_ROWS : t]

        # Cheap Viterbi pass on the frozen model — NO refit.
        rd.predict_current_regime()
        rd.assign_regime_labels()

        records.append(
            {
                "timestamp": features_df.index[t],
                "regime_label": rd.regime_label,
                "regime_confidence": rd.regime_confidence,
                "close": float(features_df.iloc[t]["close"]),
            }
        )

        # Progress indicator every 500 iterations.
        done = t - start + 1
        if done % 500 == 0 or done == n_candles:
            logging.info(
                "Phase 2 progress: %d / %d candles (%.0f %%)",
                done,
                n_candles,
                done / n_candles * 100,
            )

    # Restore original logging level.
    rd_logger.setLevel(original_level)

    result = pd.DataFrame(records).set_index("timestamp")
    logging.info(
        "Phase 2 complete: %d candles labelled.  Label distribution:\n%s",
        len(result),
        result["regime_label"].value_counts().to_string(),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Validation Checks
# ═══════════════════════════════════════════════════════════════════════════


def _run_checks(
    test_labels: pd.DataFrame,
    features_df: pd.DataFrame,
) -> dict:
    """
    Run the six statistical validation checks defined in Step 6b of
    ``BACKTESTING.md``.

    **Check 1 — Direction test:**
        ``mean(fwd_return | trending_up) > mean(fwd_return | neutral)
        > mean(fwd_return | trending_down)``

    **Check 2 — Welch's t-test:**
        Two-sample t-test on forward returns between ``trending_up`` and
        ``trending_down``.  Pass if p < 0.05.

    **Check 3 — Volatility check:**
        ``mean(volatility | high_volatility) > mean(volatility | neutral)``.
        Confirms the label is stable on out-of-sample data.

    **Check 4 — Confidence floor:**
        Median ``regime_confidence`` per label ≥ ``HMM_MIN_CONFIDENCE``.

    **Check 5 — Label frequency:**
        No regime has < 1 % relative frequency (regime collapse guard).

    **Check 6 — Hit-rate alignment:**
        Percentage of test candles where regime ∈ {``trending_down``,
        ``high_volatility``} (i.e. blocked).  Reported for comparison
        with ``run_backtest.py``'s ``regime_filter_hit_rate_pct``.

    Args:
        test_labels: DataFrame from ``_rolling_predict`` (indexed by
            ``timestamp``; columns: ``regime_label``, ``regime_confidence``,
            ``close``).
        features_df: Full 10-day feature DataFrame (needed for the
            ``volatility`` column in Check 3).

    Returns:
        dict: Keys are check names; values are dicts with ``pass`` (bool)
            and human-readable ``detail`` (str).
    """
    results: dict = {}

    # --- forward return: close(t+1) / close(t) - 1 ---
    test_labels = test_labels.copy()
    test_labels["fwd_return"] = test_labels["close"].pct_change().shift(-1)
    test_labels.dropna(subset=["fwd_return"], inplace=True)

    # Map volatility feature from features_df onto test_labels by timestamp.
    test_labels["volatility"] = features_df.loc[test_labels.index, "volatility"].values

    # Group by regime label.
    grouped = test_labels.groupby("regime_label")

    # ── Check 1 — Direction test ──────────────────────────────────────────
    mean_fwd = grouped["fwd_return"].mean()
    tu = mean_fwd.get("trending_up", np.nan)
    td = mean_fwd.get("trending_down", np.nan)
    ne = mean_fwd.get("neutral", np.nan)

    ordering_ok = True
    if not np.isnan(tu) and not np.isnan(ne):
        ordering_ok = ordering_ok and (tu > ne)
    if not np.isnan(ne) and not np.isnan(td):
        ordering_ok = ordering_ok and (ne > td)
    if not np.isnan(tu) and not np.isnan(td):
        ordering_ok = ordering_ok and (tu > td)

    results["direction_test"] = {
        "pass": ordering_ok,
        "detail": (
            f"trending_up={tu:+.6f}  neutral={ne:+.6f}  "
            f"trending_down={td:+.6f}  ordering={'OK' if ordering_ok else 'VIOLATED'}"
        ),
    }

    # ── Check 2 — Welch's t-test ─────────────────────────────────────────
    fwd_tu = test_labels.loc[test_labels["regime_label"] == "trending_up", "fwd_return"]
    fwd_td = test_labels.loc[
        test_labels["regime_label"] == "trending_down", "fwd_return"
    ]

    if len(fwd_tu) > 1 and len(fwd_td) > 1:
        t_stat, p_val = stats.ttest_ind(fwd_tu, fwd_td, equal_var=False)
        ttest_pass = p_val < 0.05
        results["welch_ttest"] = {
            "pass": ttest_pass,
            "detail": f"t={t_stat:.3f}  p={p_val:.6f}",
        }
    else:
        results["welch_ttest"] = {
            "pass": False,
            "detail": (
                f"Insufficient samples: trending_up={len(fwd_tu)}, "
                f"trending_down={len(fwd_td)}"
            ),
        }

    # ── Check 3 — Volatility check ───────────────────────────────────────
    vol_hv = grouped["volatility"].mean().get("high_volatility", np.nan)
    vol_ne = grouped["volatility"].mean().get("neutral", np.nan)

    if not np.isnan(vol_hv) and not np.isnan(vol_ne):
        vol_pass = vol_hv > vol_ne
        results["volatility_check"] = {
            "pass": vol_pass,
            "detail": f"high_volatility={vol_hv:.6f} > neutral={vol_ne:.6f}  →  {vol_pass}",
        }
    else:
        results["volatility_check"] = {
            "pass": False,
            "detail": f"Missing regime: high_volatility={vol_hv}, neutral={vol_ne}",
        }

    # ── Check 4 — Confidence floor ───────────────────────────────────────
    median_conf = grouped["regime_confidence"].median()
    conf_pass = bool((median_conf >= HMM_MIN_CONFIDENCE).all())
    results["confidence_floor"] = {
        "pass": conf_pass,
        "detail": "  ".join(f"{label}={val:.2f}" for label, val in median_conf.items()),
    }

    # ── Check 5 — Label frequency ────────────────────────────────────────
    counts = test_labels["regime_label"].value_counts()
    freq_pct = counts / counts.sum() * 100
    freq_pass = bool((freq_pct >= 1.0).all())
    results["label_frequency"] = {
        "pass": freq_pass,
        "detail": "  ".join(f"{label}={pct:.1f}%" for label, pct in freq_pct.items()),
    }

    # ── Check 6 — Hit-rate alignment ─────────────────────────────────────
    # Mirrors the gating logic in analysis.py:
    #   BUY  blocked when regime ∈ {trending_down, high_volatility}
    #   SELL blocked when regime ∈ {trending_up,   high_volatility}
    n = len(test_labels)
    buy_blocked = (
        test_labels["regime_label"].isin({"trending_down", "high_volatility"}).sum()
    )
    sell_blocked = (
        test_labels["regime_label"].isin({"trending_up", "high_volatility"}).sum()
    )
    both_blocked = test_labels["regime_label"].isin({"high_volatility"}).sum()
    results["hit_rate_alignment"] = {
        "pass": True,  # informational — compare with run_backtest.py regime_filter_hit_rate_pct
        "detail": (
            f"BUY  blocked (trending_down|high_vol): {buy_blocked / n * 100:.1f}%  ({buy_blocked}/{n})  |  "
            f"SELL blocked (trending_up|high_vol):   {sell_blocked / n * 100:.1f}%  ({sell_blocked}/{n})  |  "
            f"Both sides blocked (high_vol only):    {both_blocked / n * 100:.1f}%  ({both_blocked}/{n})"
        ),
    }

    logging.info("Phase 3 complete — %d checks executed.", len(results))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point — full pipeline
# ═══════════════════════════════════════════════════════════════════════════


def run_validation() -> None:
    """
    Run the complete offline regime validation pipeline (Phases 1–4).

    Phases:
        1. Fetch 10-day kline dataset and fit HMM on last 120 rows of
           the 7-day train period.  Model and scaler are frozen.
        2. Roll the frozen model over the 3-day test set, recording
           regime labels and confidence for each candle.
        3. Run six statistical checks on the labelled test set.
        4. Print a formatted report to stdout via
           ``backtest.reporting.formatters.print_regime_validation_report``.
    """
    features_df = _fetch_and_prepare()
    rd = _train_model(features_df)
    test_labels = _rolling_predict(features_df, rd)
    checks = _run_checks(test_labels, features_df)
    print_regime_validation_report(test_labels, checks, _TRAIN_DAYS, _SPLIT_IDX)


if __name__ == "__main__":
    run_validation()
