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
print_regime_validation_report(test_labels, checks, train_days, split_idx)
                               — prints the Step 6b regime validation report
save_csv(trades, equity)       — persists trade log & equity curve to CSV

Sensitivity report helpers (called by backtest/sensitivity.py)
--------------------------------------------------------------
print_sensitivity_table(results_df, mode, param_grid, display_cols, rank_metric)
                               — prints the per-run summary table
print_oat_sensitivity_report(results_df, baseline_sharpe, param_grid,
                              rank_metric, sensitivity_threshold)
                               — prints the OAT ΔSharpe report
print_bnh_comparison(best_row) — prints the strategy vs buy-and-hold box
"""

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config_parameters import (
    BACKTEST_FEE_RATE,
    BACKTEST_INITIAL_BTC,
    BACKTEST_INITIAL_CAPITAL,
    HMM_MIN_CONFIDENCE,
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
    init_total = stats.get("initial_equity_total_usdt")
    if init_total is not None:
        print(
            f"  Initial equity  :  {init_total:>12,.2f}  USDT  (USDT + BTC @ first close)"
        )
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
    n_guard = stats.get("n_position_guard_skips", 0)
    n_whipsaw = stats.get("n_whipsaw_exits", 0)
    n_stop_loss = stats.get("n_stop_loss_fires", 0)
    n_trend_pause = stats.get("n_trend_pause_skips", 0)
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
    print(
        f"  HOLD (cash reserve)        :  {n_guard:>8,}  ← BUY suppressed by cash-reserve floor"
    )
    print(
        f"  Whipsaw exits              :  {n_whipsaw:>8,}  ← forced SELL (same bar hit BUY+SELL zone)"
    )
    print(
        f"  Trend-pause skips          :  {n_trend_pause:>8,}  ← bars suppressed during sustained trend"
    )
    print(
        f"  Stop-loss fires            :  {n_stop_loss:>8,}  ← positions force-closed by adaptive SL"
    )
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
        "  Note: Sharpe / Sortino adaptively annualised "
        "(√365 daily, √8760 hourly, √105120 5-min — "
        "crypto trades 24/7; Rf = BACKTEST_RISK_FREE_RATE)"
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
# Regime validation report
# ---------------------------------------------------------------------------


def print_regime_validation_report(
    test_labels: pd.DataFrame,
    checks: dict,
    train_days: int,
    split_idx: int,
) -> None:
    """
    Print the Step 6b offline regime validation report to stdout.

    Produces two sections:

    1. **Per-regime statistics** — count, frequency, mean/std forward return
       and median confidence for each label over the 3-day test set.
    2. **Statistical tests** — one line per check with PASS / FAIL verdict.

    Called by ``backtest.regime_validation.run_validation()`` at the end of
    the pipeline.  Separated here so all console formatting lives in one place.

    Parameters
    ----------
    test_labels : pd.DataFrame
        Output of ``_rolling_predict()`` — must contain columns
        ``regime_label``, ``regime_confidence``, ``close``.
    checks : dict
        Output of ``_run_checks()`` — keys are check names, values are
        ``{"pass": bool, "detail": str}``.
    train_days : int
        Number of training days used (e.g. 7).
    split_idx : int
        Row index where the train/test split occurs (e.g. 10,080).
    """
    # Re-compute 1-period log forward return for the statistics table.
    # Log-returns are more symmetric and Gaussian than arithmetic returns.
    tl = test_labels.copy()
    tl["fwd_return"] = np.log(tl["close"] / tl["close"].shift(1)).shift(-1)
    tl.dropna(subset=["fwd_return"], inplace=True)

    n_total = len(tl)
    counts = tl["regime_label"].value_counts()
    freq_pct = counts / n_total * 100
    mean_fwd = tl.groupby("regime_label")["fwd_return"].mean()
    std_fwd = tl.groupby("regime_label")["fwd_return"].std()
    med_conf = tl.groupby("regime_label")["regime_confidence"].median()

    _CANDLES_PER_DAY = 1440
    test_days = round(
        n_total / _CANDLES_PER_DAY
    )  # derived from actual test candles, not hardcoded total

    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines = [
        "",
        HEAVY,
        f" REGIME VALIDATION REPORT — {test_days}-day test set ({n_total:,} candles)",
        HEAVY,
        "",
        f" Train : {train_days} days  (~{split_idx:,} rows)   model frozen after initial fit",
        f" Test  : {test_days} days   (~{n_total:,} rows)   vectorised single Viterbi pass (frozen model)",
        "",
        LIGHT,
        " PER-REGIME STATISTICS",
        LIGHT,
        f" {'Regime':<18} {'Count':>6}  {'Freq%':>6}  "
        f"{'Mean fwd-ret':>13}  {'Std fwd-ret':>12}  {'Med conf':>9}",
    ]

    for label in ["trending_up", "trending_down", "high_volatility", "neutral"]:
        if label not in counts:
            continue
        lines.append(
            f" {label:<18} {counts[label]:>6,}  {freq_pct[label]:>5.1f}%"
            f"  {mean_fwd[label] * 100:>+12.5f} %"
            f"  {std_fwd[label] * 100:>10.5f} %"
            f"  {med_conf.get(label, float('nan')):>9.2f}"
        )

    lines += ["", LIGHT, " STATISTICAL TESTS", LIGHT]

    check_order = [
        (
            "direction_test",
            "Check 1 — Direction test (trending_up > neutral > trending_down):",
        ),
        ("welch_ttest", "Check 2 — Kruskal-Wallis H-test (all regime states):"),
        (
            "volatility_check",
            "Check 3 — Volatility check (high_vol mean vol > neutral mean vol):",
        ),
        (
            "confidence_floor",
            f"Check 4 — Confidence floor (all median conf ≥ {HMM_MIN_CONFIDENCE:.2f}):",
        ),
        ("label_frequency", "Check 5 — Label frequency (all regimes ≥ 1 %):"),
        (
            "hit_rate_alignment",
            "Check 6 — Hit-rate alignment (informational — compare with runner.py):",
        ),
    ]

    auto_checks = [k for k, _ in check_order if k != "hit_rate_alignment"]

    for key, title in check_order:
        c = checks[key]
        lines += [f" {title}", f"   {c['detail']}"]
        if key != "hit_rate_alignment":
            suffix = "  (pass if p < 0.10)" if key == "welch_ttest" else ""
            lines.append(f"   → {verdict(c['pass'])}{suffix}")
        lines.append("")

    n_pass = sum(checks[k]["pass"] for k in auto_checks)
    n_checks = len(auto_checks)
    overall = (
        "  ✓ model is statistically valid"
        if n_pass == n_checks
        else "  ✗ review failing checks before relying on regime filter"
    )
    lines += [
        LIGHT,
        f" OVERALL: {n_pass}/{n_checks} checks passed{overall}",
        HEAVY,
        "",
    ]

    print("\n".join(lines))


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


# ---------------------------------------------------------------------------
# Sensitivity analysis report helpers
# ---------------------------------------------------------------------------


def print_sensitivity_table(
    results_df: pd.DataFrame,
    mode: str,
    param_grid: dict,
    display_cols: list[str],
    rank_metric: str = "sharpe_ratio",
) -> None:
    """
    Print the per-run sensitivity summary table to stdout.

    Parameters
    ----------
    results_df : pd.DataFrame
        One row per parameter combination produced by ``run_sensitivity()``
        or ``_run_sensitivity_optuna_study()``.
    mode : str
        Execution mode label shown in the header (e.g. ``"bayes"``, ``"OAT"``).
    param_grid : dict[str, list]
        The ``_PARAM_GRID`` dict from ``sensitivity.py``.  Used to identify
        the baseline (first value of each parameter list) and to mark it
        with ``"← baseline"`` in the ``note`` column.
    display_cols : list[str]
        Ordered list of column names to show.  Columns absent from
        ``results_df`` are silently skipped so the table works for all three
        execution modes (Bayes stores only Sharpe; OAT/grid store all metrics).
    rank_metric : str
        Name of the ranking metric column (default ``"sharpe_ratio"``).
    """
    n = len(results_df)
    print()
    print("═" * 110)
    print(f"  SENSITIVITY ANALYSIS — BTCUSDT  ({mode}, {n} combinations)")
    print("═" * 110)

    # Derive strategy_vs_bnh_pct if both source columns are present.
    display_df = results_df.copy()
    if (
        "total_return_pct" in display_df.columns
        and "bnh_total_return_pct" in display_df.columns
    ):
        display_df["strategy_vs_bnh_pct"] = (
            display_df["total_return_pct"] - display_df["bnh_total_return_pct"]
        )

    defaults = {k: v[0] for k, v in param_grid.items()}
    available_cols = [c for c in display_cols if c in display_df.columns]
    display = display_df[available_cols].copy()
    display.insert(0, "note", "")
    for idx, row in display.iterrows():
        is_default = all(
            row[k] == defaults[k]
            for k in ("hmm_lookback_rows", "hmm_max_regimes", "vwap_window", "fee_rate")
            if k in row.index
        )
        display.at[idx, "note"] = "← baseline" if is_default else ""

    print(display.to_string(index=False, float_format="{:.3f}".format))
    print("═" * 110)
    print()


def print_oat_sensitivity_report(
    results_df: pd.DataFrame,
    baseline_sharpe: float,
    param_grid: dict,
    rank_metric: str = "sharpe_ratio",
    sensitivity_threshold: float = 0.5,
) -> bool:
    """
    Print the per-parameter OAT sensitivity report vs the baseline Sharpe.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full OAT results (one row per combination).
    baseline_sharpe : float
        Sharpe of the all-defaults baseline run.
    param_grid : dict[str, list]
        The ``_PARAM_GRID`` dict from ``sensitivity.py``.
    rank_metric : str
        Column name of the ranking metric (default ``"sharpe_ratio"``).
    sensitivity_threshold : float
        |ΔSharpe| beyond which a ⚠️ flag is displayed and ``True`` is returned
        (default ``0.5``).

    Returns
    -------
    bool
        ``True`` if any parameter exceeds *sensitivity_threshold* — signals
        that Bayesian optimisation (``--bayes``) is advisable.
    """
    defaults = {k: v[0] for k, v in param_grid.items()}
    trigger_phase2 = False

    print(
        "── OAT Sensitivity Report (vs baseline Sharpe = {:.3f}) ──".format(
            baseline_sharpe
        )
    )
    for param in param_grid.keys():
        non_default = results_df[results_df[param] != defaults[param]]
        for _, row in non_default.iterrows():
            delta = row[rank_metric] - baseline_sharpe
            flag = (
                f"  ⚠️  |ΔSharpe| > {sensitivity_threshold} — consider running --bayes!"
                if abs(delta) > sensitivity_threshold
                else ""
            )
            print(
                f"  {param}={row[param]!r:>6}  →  Sharpe={row[rank_metric]:.3f}"
                f"  ΔSharpe={delta:+.3f}{flag}"
            )
            if abs(delta) > sensitivity_threshold:
                trigger_phase2 = True
    print()
    return trigger_phase2


def print_bnh_comparison(best_row: pd.Series) -> None:
    """
    Print a bordered summary box comparing the best strategy result against
    the passive buy-and-hold benchmark for the same window.

    Called after every execution path (Bayes, OAT, full-grid) once the best
    row has been identified.  If the B&H columns are absent the box is skipped
    with an INFO log.

    Interpretation
    --------------
    * ``outperformance > 0`` → strategy beat passive holding (gained more or
      lost less than simply holding BTC).
    * ``outperformance < 0`` → strategy underperformed even passive holding —
      review signal quality or the lookback window.

    Parameters
    ----------
    best_row : pd.Series
        Must contain ``total_return_pct``, ``bnh_total_return_pct``, and
        optionally ``sharpe_ratio``.
    """
    strat = best_row.get("total_return_pct", float("nan"))
    bnh_ret = best_row.get("bnh_total_return_pct", float("nan"))
    sharpe = best_row.get("sharpe_ratio", float("nan"))

    if math.isnan(strat) or math.isnan(bnh_ret):
        log.info("B&H comparison skipped — columns not present in best row.")
        return

    w = 62  # inner width
    print()
    print("╔" + "═" * w + "╗")
    print(f"║  {'STRATEGY vs BUY-AND-HOLD  (best result)':<{w - 2}}║")
    print("╠" + "═" * w + "╣")
    sharpe_str = f"{sharpe:+.4f}" if not math.isnan(sharpe) else "n/a"
    print(
        f"║  Strategy  return : {strat:>+8.2f}%   (Sharpe: {sharpe_str}){'':>{w - 48}}║"
    )
    print(f"║  Buy-and-hold     : {bnh_ret:>+8.2f}%{'':>{w - 30}}║")
    print("╠" + "═" * w + "╣")
    print()
