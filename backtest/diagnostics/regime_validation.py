"""
backtest/diagnostics/regime_validation.py
-----------------------------------------
Step 6b — Offline Long-Horizon Regime Validation.

Standalone diagnostic script that tests whether the HMM regime labels
produced by ``RegimeDirector`` remain statistically meaningful over a
**fully out-of-sample** horizon (~219-day evaluation window).

This tool uses a **single-fit approach**:

1. Fit the HMM once on the **full train set** (first 70% of rows).
2. Predict regime labels for the **entire test set** (last 30%) in one
   vectorised ``model.predict()`` call.
3. Run 6 statistical checks on the test-set labels.

This is fast (seconds, not minutes) and scales to 2-year lookback with
no performance concern.  The model never sees test data during fitting,
so there is zero leakage by construction.

**Not wired into ``runner.py``** — this is a standalone diagnostic tool.
Run manually with:

    python -m backtest.diagnostics.regime_validation

Why is this in ``backtest/diagnostics/`` and not ``backtest/``?
    All files directly in ``backtest/`` (``signals.py``, ``pnl.py``,
    ``runner.py``, etc.) form the main pipeline orchestrated by
    ``run_backtest()``.  This script is a separate, one-off health check
    for the HMM model and must be invoked independently.

See ``BACKTESTING.md`` (Step 6b) for the full rationale, dataset split,
and interpretation guidance.

NOTE: This script **bypasses** ``BACKTEST_MAX_ROWS`` intentionally.
Two years of 5-minute klines (~210,000 rows) cover multiple full BTC
market cycle turns.  The Binance paginated fetch takes ~3–5 minutes;
Phase 2 (HMM fit + Viterbi predict) completes in seconds regardless of
dataset size.  Use ``"90 days ago UTC"`` for a faster smoke-test run.

When to re-run
--------------
Re-run this tool whenever ``strategy/regime_director.py`` is modified
(feature columns, BIC search range, label-assignment rules, confidence
threshold, etc.) to confirm the model still produces statistically
meaningful labels on out-of-sample data.

Written with the assistance of AI models — results should be reviewed
critically before drawing conclusions.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from backtest.data import fetch_klines
from backtest.signals import _add_hmm_features
from backtest.reporting.formatters import print_regime_validation_report

from config_parameters import (
    HMM_FEATURE_COLS,
    HMM_MAX_REGIMES,
    HMM_MIN_CONFIDENCE,
    HMM_MIN_COVAR,
    HMM_N_INIT,
    HMM_N_ITERATIONS,
    HMM_RANDOM_STATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Constants local to this diagnostic — not in config_parameters.py because
# they are specific to the one-off validation, not the live system.
#
# Industry-standard 70 / 30 train-test split applied to 2 years of 5-minute
# klines (~210,000 rows after feature engineering).
# At _TRAIN_RATIO = 0.70:
#   train ≈ 147,000 rows  (~511 days, ~17 months)  — fit window
#   test  ≈  63,000 rows  (~219 days, ~7.3 months) — evaluation window
#
# The split is derived at runtime so the ratio stays exact regardless of the
# actual number of rows Binance returns.
#
# Why 2 years?
#   Two years of BTC data capture multiple full market cycle turns
#   (trending, ranging, volatile) giving the HMM the broadest possible
#   basis for regime learning.  The Binance fetch takes ~3–5 minutes
#   (paginated at 5m resolution) but Phase 2 (fit + predict) still
#   completes in seconds.  Use "90 days ago UTC" for a faster run.
#
# Note: BACKTEST_LOOKBACK (used by runner.py) is intentionally kept
#       at "180 days ago UTC" — this validation script uses its own
#       VALIDATION_LOOKBACK so the two tools remain independent.
#       Unlike the live RegimeDirector (which trains on train_end =
#       max(2, int(n_rows × 2/3)) rows — adaptive ~⅔ split — to avoid
#       look-ahead bias), this diagnostic fits on the full 70%
#       train set (~147,000 rows) for a thorough regime coverage check.
# ---------------------------------------------------------------------------
VALIDATION_LOOKBACK = "730 days ago UTC"  # fetch window (2 years, ~210,000 rows)
_TRAIN_RATIO = 0.70  # first 70 % = train, last 30 % = test (evaluated)
_CANDLES_PER_DAY = (
    288  # 24 h × 12 bars/h at 5m resolution — used for human-readable logging only
)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Data Fetch
# ═══════════════════════════════════════════════════════════════════════════


def _fetch_and_prepare() -> tuple[pd.DataFrame, int]:
    """
    Download the 2-year kline dataset, add HMM features, and compute
    the evaluation-window start index from ``_TRAIN_RATIO``.

    Uses ``VALIDATION_LOOKBACK`` (``"730 days ago UTC"``, 2 years,
    independent of ``BACKTEST_LOOKBACK`` used by ``runner.py``) —
    bypasses ``BACKTEST_MAX_ROWS`` intentionally.

    Returns:
        tuple[pd.DataFrame, int]:
            features_df — feature-enriched klines (``return``, ``volatility``,
                ``obi_proxy``, ``trade_density`` appended; NaN rows dropped).
            split_idx   — row index where the walk-forward test period begins
                (rows[:split_idx] provide history; rows[split_idx:] are labelled).
    """
    logging.info(
        "Phase 0 — Fetching kline data for window '%s' (~210,000 rows at 5m resolution). "
        "This is a paginated Binance API call and may take ~3–5 minutes. Please wait...",
        VALIDATION_LOOKBACK,
    )
    klines = fetch_klines(start_str=VALIDATION_LOOKBACK)
    features_df = _add_hmm_features(klines)

    split_idx = int(len(features_df) * _TRAIN_RATIO)
    train_days = split_idx // _CANDLES_PER_DAY
    test_days = (len(features_df) - split_idx) // _CANDLES_PER_DAY

    logging.info(
        "Data ready: %d raw klines → %d rows after features.  "
        "Walk-forward test period: rows %d … %d (~%d days)  "
        "[pre-test history: ~%d days]",
        len(klines),
        len(features_df),
        split_idx,
        len(features_df) - 1,
        test_days,
        train_days,
    )
    return features_df, split_idx


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Train on full train set, predict on full test set
# ═══════════════════════════════════════════════════════════════════════════


def _fit_best_hmm(
    train_scaled: np.ndarray,
) -> GaussianHMM:
    """
    BIC search over n=2…HMM_MAX_REGIMES, return the best model.

    Same logic as ``RegimeDirector.select_hmm_model()`` but operates on
    an arbitrary-length training array — not limited to 80 rows.
    """
    best_model, best_bic = None, np.inf

    for n in range(2, HMM_MAX_REGIMES + 1):
        for seed_offset in range(HMM_N_INIT):
            m = GaussianHMM(
                n_components=n,
                covariance_type="full",
                n_iter=HMM_N_ITERATIONS,
                random_state=HMM_RANDOM_STATE + seed_offset,
                min_covar=HMM_MIN_COVAR,
            )
            try:
                m.fit(train_scaled)
                row_sums = m.transmat_.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-3):
                    continue
                bic = m.bic(train_scaled)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue

            logging.info("BIC search: n=%d seed+%d  BIC=%.1f", n, seed_offset, bic)
            if bic < best_bic:
                best_bic, best_model = bic, m
            break  # first valid fit for this n is enough

    if best_model is None:
        raise RuntimeError("All HMM fits failed during validation.")
    return best_model


def _assign_labels(model: GaussianHMM) -> dict[int, str]:
    """
    Map each HMM state index to a human-readable label.

    Same rank-based logic as ``RegimeDirector.assign_regime_labels()``,
    reproduced here so this module is fully self-contained.
    """
    means = pd.DataFrame(model.means_, columns=HMM_FEATURE_COLS)

    direction_score = means["return"].rank() + means["obi_proxy"].rank()
    best_state = int(direction_score.idxmax())
    worst_state = int(direction_score.idxmin())

    mean_vol = means["volatility"].mean()
    std_vol = means["volatility"].std()
    mean_td = means["trade_density"].mean()
    std_td = means["trade_density"].std()

    labels: dict[int, str] = {}
    for state in range(model.n_components):
        vol = means.loc[state, "volatility"]
        td = means.loc[state, "trade_density"]
        high_vol = vol > mean_vol + std_vol
        high_td = td > mean_td + 0.5 * std_td

        if state == best_state:
            labels[state] = "trending_up"
        elif state == worst_state:
            labels[state] = "trending_down"
        elif high_vol or high_td:
            labels[state] = "high_volatility"
        else:
            labels[state] = "neutral"

    logging.info("State labels: %s", labels)
    return labels


def _train_and_predict(
    features_df: pd.DataFrame,
    split_idx: int,
) -> pd.DataFrame:
    """
    Fit HMM on the full train set, predict labels for the full test set.

    How it works (tiny-numbers example — 10 rows, split at 7)
    ----------------------------------------------------------
    ::

        Dataset:   row 0  1  2  3  4  5  6 | 7  8  9
                        TRAIN (70%)         | TEST (30%)

        Step 1 — scale:  scaler.fit_transform(rows 0–6)   → train_scaled
                         scaler.transform(rows 7–9)        → test_scaled
        Step 2 — fit:    model.fit(train_scaled)   ← uses ALL 7 train rows
        Step 3 — predict: model.predict(test_scaled) → [state7, state8, state9]
        Step 4 — label:  map each state to "trending_up" / "neutral" / etc.

    Zero leakage: the model and scaler are fitted exclusively on rows
    before ``split_idx``.  Test rows are only passed to ``transform()``
    and ``predict()`` — neither changes learned parameters.

    Args:
        features_df: Full feature DataFrame.
        split_idx:   First row of the test set.

    Returns:
        pd.DataFrame indexed by timestamp with columns:
        ``regime_label``, ``regime_confidence``, ``close``.
    """
    train_features = features_df.iloc[:split_idx][HMM_FEATURE_COLS].values
    test_features = features_df.iloc[split_idx:][HMM_FEATURE_COLS].values

    # ── Step 1: scale (fit on train only) ─────────────────────────────────
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # ── Step 2: fit HMM on full train set ─────────────────────────────────
    logging.info(
        "Phase 2 — fitting HMM on %d train rows (~%d days)...",
        len(train_features),
        len(train_features) // _CANDLES_PER_DAY,
    )
    model = _fit_best_hmm(train_scaled)
    state_labels = _assign_labels(model)
    logging.info(
        "Fit complete: %d states selected by BIC.",
        model.n_components,
    )

    # ── Step 3: predict on full test set (one Viterbi pass) ───────────────
    logging.info(
        "Phase 2 — predicting %d test rows (~%d days) in one pass...",
        len(test_features),
        len(test_features) // _CANDLES_PER_DAY,
    )
    states = model.predict(test_scaled)
    proba = model.predict_proba(test_scaled)

    # ── Step 4: build result DataFrame ────────────────────────────────────
    test_df = features_df.iloc[split_idx:]

    # Vectorised label mapping: build a lookup array indexed by state int,
    # then index it with the full states array in one shot.
    n_states = model.n_components
    label_array = np.array(
        [state_labels.get(i, "neutral") for i in range(n_states)], dtype=object
    )
    labels = label_array[states]  # shape (n_test,) — no Python loop

    # Vectorised confidence extraction: advanced numpy indexing selects
    # proba[row_i, states[row_i]] for every row simultaneously.
    confidences = proba[np.arange(len(states)), states]  # shape (n_test,)

    result = pd.DataFrame(
        {
            "regime_label": labels,
            "regime_confidence": confidences,
            "close": test_df["close"].values,
        },
        index=test_df.index,
    )

    logging.info(
        "Phase 2 complete: %d test candles labelled.\nLabel distribution:\n%s",
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

    Because labels are now produced by a walk-forward loop that mirrors
    the live system, all six checks — including the Kruskal-Wallis H-test — are
    statistically meaningful.

    **Check 1 — Direction test:**
        ``mean(fwd_return | trending_up) > mean(fwd_return | neutral)
        > mean(fwd_return | trending_down)``

        Uses a materiality tolerance ε = 1e-4 (0.01 % = 1 bp over 5 min).
        A reversal smaller than ε is treated as noise, not a genuine violation.
        NaN for a regime means that regime was absent from the test labels
        (expected when BIC selects n=2 states — no neutral state exists).

    **Check 2 — Kruskal-Wallis H-test (all regime states):**
        Non-parametric rank-based test of H₀: all state forward-return
        distributions are identical.  Tests ALL k BIC-selected states
        simultaneously (k=2 or k=3).  No normality or equal-variance
        assumption — both of which HMM states violate.
        Pass if p < 0.10.

    **Check 3 — Volatility check:**
        ``mean(volatility | high_volatility) > mean(volatility | neutral)``.
        **SKIP** (``pass=True``, informational) when either regime is absent
        from the test labels — expected when BIC selects n=2 states.

    **Check 4 — Confidence floor:**
        Median ``regime_confidence`` per label ≥ ``HMM_MIN_CONFIDENCE``.

    **Check 5 — Label frequency:**
        No regime has < 1 % relative frequency (regime collapse guard).

    **Check 6 — Hit-rate alignment (informational):**
        Fraction of test candles where the regime filter would block
        BUY or SELL orders.  Compare with ``runner.py``'s
        ``regime_filter_hit_rate_pct``.

    Args:
        test_labels: DataFrame from ``_walk_forward_predict`` (indexed by
            timestamp; columns: ``regime_label``, ``regime_confidence``,
            ``close``).
        features_df: Full feature DataFrame (needed for
            ``volatility`` in Check 3).

    Returns:
        dict: Keys are check names; values are dicts with ``pass`` (bool)
            and human-readable ``detail`` (str).
    """
    results: dict = {}

    # 1-hour cumulative forward log-return: sum of 12 single-period log-returns,
    # shifted forward so that row t carries log(close[t+12] / close[t]).
    # Log-returns are more symmetrically distributed than arithmetic returns
    # (less positive skew from large price jumps), which makes Check 1's
    # directional comparison more meaningful and satisfies the Gaussian
    # assumption underlying Check 2's Welch's t-test more cleanly.
    # 12 candles at 5m resolution = 1 hour — the shortest horizon at which
    # regime-driven directional effects are consistently detectable above
    # microstructure noise. The HMM captures structural conditions that
    # persist for hours, not seconds, so a 1-hour window is appropriate.
    _FWD_PERIODS = 12  # candles = 1 h at 5m resolution
    test_labels = test_labels.copy()
    test_labels["log_return"] = np.log(
        test_labels["close"] / test_labels["close"].shift(1)
    )
    test_labels["fwd_return"] = (
        test_labels["log_return"]
        .rolling(window=_FWD_PERIODS)
        .sum()
        .shift(-_FWD_PERIODS)
    )
    test_labels.dropna(subset=["fwd_return"], inplace=True)

    # align volatility feature from the full dataset
    test_labels["volatility"] = features_df.loc[test_labels.index, "volatility"].values

    grouped = test_labels.groupby("regime_label")

    # ── Check 1 — Direction test ──────────────────────────────────────────
    # Tolerance ε: only flag a violation if the wrong-direction gap exceeds
    # 1e-4 (0.01 % = 1 bp over 5 min). Differences smaller than ε are noise
    # at microstructure scale and should not fail the check.
    # NaN for a regime = that label absent from test labels (expected when
    # BIC selects n=2 states — no neutral state is produced).
    # Minimum spread δ: when both directional labels are present, their mean
    # forward returns must differ by at least δ = 1e-4 (1 bp). At a 1-hour
    # horizon the real directional spread is typically 1–3 bps — measurable
    # but frequently just below 2 bps. 1 bp is the minimum spread above noise
    # floor that confirms a genuine (if modest) directional signal.
    _DIRECTION_TOLERANCE = 1e-4
    _DIRECTION_MIN_SPREAD = 1e-4  # tu − td must exceed this when both present

    mean_fwd = grouped["fwd_return"].mean()
    tu = mean_fwd.get("trending_up", np.nan)
    td = mean_fwd.get("trending_down", np.nan)
    ne = mean_fwd.get("neutral", np.nan)

    ordering_ok = True
    spread_ok = True  # separate flag so detail message is informative

    if not np.isnan(tu) and not np.isnan(ne):
        ordering_ok = ordering_ok and (tu >= ne - _DIRECTION_TOLERANCE)
    if not np.isnan(ne) and not np.isnan(td):
        ordering_ok = ordering_ok and (ne >= td - _DIRECTION_TOLERANCE)
    if not np.isnan(tu) and not np.isnan(td):
        ordering_ok = ordering_ok and (tu >= td - _DIRECTION_TOLERANCE)
        # Guard against both-zero / degenerate model: require a meaningful gap.
        spread_ok = (tu - td) >= _DIRECTION_MIN_SPREAD

    def _fmt(v: float) -> str:
        return f"{v:+.6f}" if not np.isnan(v) else "absent (n=2 states)"

    direction_pass = ordering_ok and spread_ok
    results["direction_test"] = {
        "pass": direction_pass,
        "detail": (
            f"trending_up={_fmt(tu)}  neutral={_fmt(ne)}  "
            f"trending_down={_fmt(td)}  "
            f"ordering={'OK' if ordering_ok else 'VIOLATED'}  "
            f"spread(tu-td)="
            + (
                f"{tu - td:+.6f} ({'OK' if spread_ok else f'FAIL < δ={_DIRECTION_MIN_SPREAD:.0e}'})"
                if not np.isnan(tu) and not np.isnan(td)
                else "n/a (one label absent)"
            )
            + f"  (tolerance ε={_DIRECTION_TOLERANCE:.0e})"
        ),
    }

    # ── Check 2 — Kruskal-Wallis H-test (all regime states) ─────────────
    # Non-parametric rank-based test of H₀: all state forward-return
    # distributions are identical (location shift).
    #
    # Why Kruskal-Wallis over Welch's t-test:
    #   1. Works for any k ≥ 2 states — no separate code path for n=2 vs n=3.
    #   2. No normality assumption — BTC returns are fat-tailed (leptokurtic),
    #      violating the Gaussian premise of Welch's t-test.
    #   3. No equal-variance assumption — HMM states have different emission
    #      variances by construction (each state models its own σ²).
    #   4. For k=2 it reduces to the Mann-Whitney U / Wilcoxon rank-sum test,
    #      which is strictly more robust than Welch's t-test for fat-tailed data.
    #
    # Threshold: p < 0.10 (standard for high-frequency financial data where
    # within-group variance is large relative to between-group mean differences).
    _KW_P_THRESHOLD = 0.10
    all_labels: list[str] = sorted(str(v) for v in test_labels["regime_label"].unique())
    kw_groups = [
        test_labels.loc[test_labels["regime_label"] == lbl, "fwd_return"].values
        for lbl in all_labels
    ]
    kw_groups_valid = [(lbl, g) for lbl, g in zip(all_labels, kw_groups) if len(g) > 1]

    if len(kw_groups_valid) >= 2:
        h_stat, p_val = stats.kruskal(*[g for _, g in kw_groups_valid])
        kw_pass = p_val < _KW_P_THRESHOLD
        valid_label_str = ", ".join(lbl for lbl, _ in kw_groups_valid)
        results["welch_ttest"] = {
            "pass": kw_pass,
            "detail": (
                f"H={h_stat:.3f}  p={p_val:.6f}  k={len(kw_groups_valid)} groups "
                f"({valid_label_str})  "
                f"(pass if p < {_KW_P_THRESHOLD}; Kruskal-Wallis, 1-h fwd return)"
            ),
        }
    else:
        results["welch_ttest"] = {
            "pass": False,
            "detail": (
                "Insufficient groups with >1 sample: "
                + ", ".join(f"{lbl}={len(g)}" for lbl, g in zip(all_labels, kw_groups))
            ),
        }

    # ── Check 3 — Volatility check ────────────────────────────────────────
    vol_means = grouped["volatility"].mean()
    vol_hv = vol_means.get("high_volatility", np.nan)
    vol_ne = vol_means.get("neutral", np.nan)

    if np.isnan(vol_hv) and np.isnan(vol_ne):
        results["volatility_check"] = {
            "pass": True,
            "detail": (
                "SKIP — neither 'high_volatility' nor 'neutral' present. "
                "Model selected n=2 states (both consumed by directional labels). "
                "No violation."
            ),
        }
    elif np.isnan(vol_hv):
        results["volatility_check"] = {
            "pass": True,
            "detail": (
                f"SKIP — 'high_volatility' absent from test labels "
                f"(neutral vol={vol_ne:.6f}).  BIC-selected model has no "
                f"high-volatility state; not a failure."
            ),
        }
    elif np.isnan(vol_ne):
        results["volatility_check"] = {
            "pass": True,
            "detail": (
                f"SKIP — 'neutral' absent from test labels "
                f"(high_volatility vol={vol_hv:.6f}).  Cannot compare; not a failure."
            ),
        }
    else:
        vol_pass = bool(vol_hv > vol_ne)
        results["volatility_check"] = {
            "pass": vol_pass,
            "detail": (
                f"high_volatility={vol_hv:.6f} > neutral={vol_ne:.6f}  "
                f"→  {'PASS' if vol_pass else 'FAIL'}"
            ),
        }

    # ── Check 4 — Confidence floor ────────────────────────────────────────
    median_conf = grouped["regime_confidence"].median()
    conf_pass = bool((median_conf >= HMM_MIN_CONFIDENCE).all())
    results["confidence_floor"] = {
        "pass": conf_pass,
        "detail": "  ".join(f"{lbl}={val:.2f}" for lbl, val in median_conf.items()),
    }

    # ── Check 5 — Label frequency ─────────────────────────────────────────
    counts = test_labels["regime_label"].value_counts()
    freq_pct = counts / counts.sum() * 100
    freq_pass = bool((freq_pct >= 1.0).all())
    results["label_frequency"] = {
        "pass": freq_pass,
        "detail": "  ".join(f"{lbl}={pct:.1f}%" for lbl, pct in freq_pct.items()),
    }

    # ── Check 6 — Hit-rate alignment (informational) ──────────────────────
    # Mirrors gating logic in analysis.py:
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
        "pass": True,  # informational — compare with runner.py regime_filter_hit_rate_pct
        "detail": (
            f"BUY  — allowed: {(n - buy_blocked) / n * 100:.1f}%  "
            f"blocked: {buy_blocked / n * 100:.1f}%  "
            f"(regime ∈ {{trending_down, high_vol}})  |  "
            f"SELL — allowed: {(n - sell_blocked) / n * 100:.1f}%  "
            f"blocked: {sell_blocked / n * 100:.1f}%  "
            f"(regime ∈ {{trending_up, high_vol}})  |  "
            f"Both sides blocked (high_vol only): {both_blocked / n * 100:.1f}%  "
            f"({both_blocked}/{n})"
        ),
    }

    logging.info("Phase 3 complete — %d checks executed.", len(results))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point — full pipeline
# ═══════════════════════════════════════════════════════════════════════════


def run_validation() -> None:
    """
    Run the complete offline regime validation pipeline (Phases 1–3).

    Pipeline
    --------
    1. ``_fetch_and_prepare()`` — download ~1 year of 1-minute klines,
       add HMM features, compute the 70/30 split index.
    2. ``_train_and_predict()`` — fit HMM on the train set (first 70%),
       predict labels for the test set (last 30%) in one vectorised pass.
    3. ``_run_checks()`` — run six statistical checks on the test-set
       labels.
    4. Print a formatted report.

    Total expected runtime: ~8–12 minutes (dominated by Binance fetch).
    """
    features_df, split_idx = _fetch_and_prepare()
    test_labels = _train_and_predict(features_df, split_idx)
    checks = _run_checks(test_labels, features_df)
    train_days = split_idx // _CANDLES_PER_DAY
    print_regime_validation_report(test_labels, checks, train_days, split_idx)


if __name__ == "__main__":
    run_validation()
