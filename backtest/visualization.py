"""
backtest/visualization.py
--------------------------
Step 7 — Backtest Visualisation (Plotly).

Generates an interactive six-row Plotly figure from the four artefacts
produced by ``backtest.runner.run_backtest()``.  No new data is fetched;
all inputs flow directly from the pipeline.

Panels
------
1a. Equity curve — portfolio value over time with initial-equity reference.
1b. Drawdown (%) — red fill showing peak-to-trough decline (shared x-axis).
2.  BTC close price + BUY ▲ / SELL ▼ markers at fill prices + VWAP lines.
3.  Regime timeline — colour-coded vrect bands per HMM label with a
    ``regime_confidence`` overlay line and a dashed ``HMM_MIN_CONFIDENCE``
    threshold.
4.  VWAP vs micro-price — ``bid_vwap``, ``ask_vwap``, ``best_buy_micro`` and
    ``best_sell_micro``, plus grey dot markers where the VWAP gate specifically
    blocked a raw candidate (confidence + regime gates both passed).
5a. Signal funnel — stacked horizontal bar: executed / confidence-blocked /
    regime-blocked / VWAP-blocked per side.
5b. Signals by regime — stacked vertical bar: BUY / SELL / HOLD composition
    for each of the four HMM regime labels.

All rows 1–5 share a synchronised datetime x-axis (zoom/pan one → all move).
Rows 6-col-1 and 6-col-2 (bar charts) have independent categorical x-axes.

Public API
----------
plot_backtest(signals, trades, equity, stats, save_png=False, show=True)

Usage
-----
    from backtest.runner import run_backtest
    signals, trades, equity, stats = run_backtest()

    from backtest.visualization import plot_backtest
    plot_backtest(signals, trades, equity, stats, save_png=True)

    # or via runner directly:
    run_backtest(plot=True, save_png=True)

Saving
------
*  ``show=True``     → ``fig.show()`` (opens the default browser, interactive).
*  ``save_png=True`` → writes a PNG via ``kaleido``.  If ``kaleido`` is not
   installed (``pip install kaleido``), falls back to a self-contained
   interactive HTML file.  Both are written to ``backtest/results/``.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config_parameters import HMM_MIN_CONFIDENCE, SYMBOL

log = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
_BUY_COLOUR = "#2ca02c"  # medium green  — executed BUY signals
_SELL_COLOUR = "#d62728"  # medium red    — executed SELL signals

# Maps regime label → integer position on the step-line y-axis in Panel 3.
# Order (bottom→top): trending_down(0) → high_volatility(1) → neutral(2) → trending_up(3)
_REGIME_NUMERIC: dict[str, int] = {
    "trending_down": 0,
    "high_volatility": 1,
    "neutral": 2,
    "trending_up": 3,
}

# Regime band fill colours (rgba with alpha, used for vrects)
_REGIME_COLOURS: dict[str, str] = {
    "trending_up": "rgba(200, 230, 201, 0.55)",  # pale green
    "trending_down": "rgba(255, 205, 210, 0.55)",  # pale red / salmon
    "high_volatility": "rgba(255, 224, 178, 0.55)",  # pale orange
    "neutral": "rgba(180, 180, 180, 0.40)",  # medium grey — visible on white background
}

# Opaque hex variants used for the legend dummy markers in Panel 3
_REGIME_LEGEND_COLOURS: dict[str, str] = {
    "trending_up": "#c8e6c9",
    "trending_down": "#ffcdd2",
    "high_volatility": "#ffe0b2",
    "neutral": "#e0e0e0",
}

# Funnel segment colours (Panel 5a)
_C_EXECUTED = "#1f77b4"  # blue         — passed all three gates
_C_CONFIDENCE = "#9467bd"  # purple       — blocked by confidence gate (1st)
_C_REGIME = "#ff7f0e"  # orange       — blocked by regime direction (2nd)
_C_VWAP = "#bcbd22"  # yellow-grn   — blocked by VWAP momentum (3rd)

# Regime sets that block each side — must stay in sync with analysis.py / pnl.py
_BUY_BLOCKED: frozenset[str] = frozenset({"trending_down", "high_volatility"})
_SELL_BLOCKED: frozenset[str] = frozenset({"trending_up", "high_volatility"})


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def plot_backtest(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    stats: dict[str, Any],
    save_png: bool = False,
    show: bool = True,
) -> None:
    """
    Build the interactive five-panel backtest figure using Plotly.

    Parameters
    ----------
    signals : pd.DataFrame
        Output of ``backtest.signals.run_signals()``.  Required columns:
        ``close``, ``signal``, ``half_spread``, ``regime``,
        ``regime_confidence``, ``bid_vwap``, ``ask_vwap``,
        ``best_buy_micro``, ``best_sell_micro``.
    trades : pd.DataFrame
        Executed trade log from ``backtest.pnl.simulate_pnl()``.  Required
        columns: ``side``, ``fill_price``.  May be empty if no trades fired.
    equity : pd.DataFrame
        Mark-to-market equity curve from ``backtest.pnl.simulate_pnl()``.
        Required columns: ``equity``, ``drawdown_pct``.
    stats : dict
        Step 5 performance metrics from ``backtest.pnl.simulate_pnl()``.
    save_png : bool
        If ``True``, persist the chart as a PNG (requires ``kaleido``) or as
        an HTML file if ``kaleido`` is not installed.  Files are written to
        ``backtest/results/``.  Default ``False`` (opt-in).
    show : bool
        If ``True``, call ``fig.show()`` to open an interactive browser window.
        Default ``True``.  Set to ``False`` for headless / CI environments.
    """
    log.info("Building backtest visualisation (Plotly)…")

    # ── Build subplot grid ────────────────────────────────────────────────────
    # Rows 1–5: full-width single subplot (colspan=2).
    # Row 6:    two side-by-side bar-chart subplots.
    fig = make_subplots(
        rows=6,
        cols=2,
        specs=[
            [{"colspan": 2}, None],  # Row 1: equity curve
            [{"colspan": 2}, None],  # Row 2: drawdown
            [{"colspan": 2}, None],  # Row 3: close + signals + VWAP
            [{"colspan": 2}, None],  # Row 4: regime bands + confidence
            [{"colspan": 2}, None],  # Row 5: VWAP vs micro-price
            [{}, {}],  # Row 6: funnel | regime dist
        ],
        row_heights=[0.19, 0.07, 0.19, 0.13, 0.13, 0.29],
        vertical_spacing=0.038,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Panel 1a — Equity Curve",
            "Panel 1b — Drawdown (%)",
            "Panel 2 — BTC Close + BUY ▲ / SELL ▼ at Fill Price + VWAP",
            "Panel 3 — HMM Regime Timeline + Confidence",
            "Panel 4 — VWAP vs Micro-Price  (● = VWAP gate near-miss)",
            "Panel 5a — Signal Funnel",
            "Panel 5b — Signals by Regime",
        ],
    )

    # Link x-axes of rows 2–5 to row 1 for synchronised zoom/pan.
    # Row 6 bar charts are intentionally excluded (categorical x-axes).
    fig.update_layout(
        xaxis2=dict(matches="x"),
        xaxis3=dict(matches="x"),
        xaxis4=dict(matches="x"),
        xaxis5=dict(matches="x"),
    )

    # ── Draw panels ───────────────────────────────────────────────────────────
    _panel_equity(fig, equity, stats)
    _panel_price_signals(fig, signals, trades)
    _panel_regime(fig, signals)
    _panel_vwap(fig, signals)
    _panel_funnel(fig, signals)
    _panel_regime_dist(fig, signals)

    # Hide datetime x-tick labels on rows 1–4 (shared axis; only show on row 5)
    for r in range(1, 5):
        fig.update_xaxes(showticklabels=False, row=r, col=1)

    # ── Figure-level title with headline stats ────────────────────────────────
    def _fmt(v: float, f: str) -> str:
        """Format a float with a printf-style format string; return 'N/A' for NaN."""
        return (f % v) if not (isinstance(v, float) and np.isnan(v)) else "N/A"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{SYMBOL} Backtest</b> &nbsp;|&nbsp; "
                f"Return: {_fmt(stats.get('total_return_pct', float('nan')), '%+.2f%%')} &nbsp; "
                f"Sharpe: {_fmt(stats.get('sharpe_ratio', float('nan')), '%.3f')} &nbsp; "
                f"Win-rate: {_fmt(stats.get('win_rate_pct', float('nan')), '%.1f%%')} &nbsp; "
                f"Max-DD: {_fmt(stats.get('max_drawdown_pct', float('nan')), '%.2f%%')} &nbsp; "
                f"Round-trips: {stats.get('n_round_trips', 0)}"
            ),
            x=0.5,
            font=dict(size=14),
        ),
        height=1900,
        template="plotly_white",
        hovermode="x unified",  # single tooltip across all shared-x traces
        barmode="stack",  # stacked bars for both funnel and regime dist
        legend=dict(
            orientation="v",
            x=1.01,
            y=1.0,
            font=dict(size=10),
        ),
    )

    if save_png:
        _save_figure(fig)

    if show:
        fig.show()

    log.info("Visualisation complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Panel helpers
# ──────────────────────────────────────────────────────────────────────────────


def _panel_equity(
    fig: go.Figure,
    equity: pd.DataFrame,
    stats: dict[str, Any],
) -> None:
    """
    Panel 1 — equity curve (row 1) and drawdown fill (row 2).

    The equity curve shows the continuous mark-to-market portfolio value
    including HOLD candles (no trade fired).  A dashed horizontal reference
    line marks the starting equity so over- or under-performance is
    immediately visible.

    The drawdown sub-panel uses a red ``tozeroy`` fill so the worst
    peak-to-trough period is immediately visible without reading numbers.
    Both panels share the x-axis with all other datetime panels (rows 1–5).
    """
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["equity"],
            mode="lines",
            name="Portfolio equity",
            line=dict(color="steelblue", width=1.5),
            legendgroup="equity",
        ),
        row=1,
        col=1,
    )

    init_eq = stats.get("initial_equity_total_usdt", float(equity["equity"].iloc[0]))
    if isinstance(init_eq, (int, float)) and not np.isnan(float(init_eq)):
        fig.add_hline(
            y=float(init_eq),
            line_dash="dash",
            line_color="grey",
            line_width=1,
            annotation_text=f"Initial equity: {float(init_eq):,.0f} USDT",
            annotation_position="top right",
            row=1,
            col=1,
        )

    fig.update_yaxes(title_text="Equity (USDT)", row=1, col=1)

    # Drawdown fill
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["drawdown_pct"],
            fill="tozeroy",
            mode="lines",
            name="Drawdown %",
            line=dict(color="crimson", width=0.8),
            fillcolor="rgba(220, 20, 60, 0.30)",
            legendgroup="drawdown",
        ),
        row=2,
        col=1,
    )

    max_dd = stats.get("max_drawdown_pct", float("nan"))
    if not np.isnan(float(max_dd)):
        fig.add_hline(
            y=float(max_dd),
            line_dash="dot",
            line_color="darkred",
            line_width=1,
            annotation_text=f"Max DD: {float(max_dd):.2f}%",
            annotation_position="bottom right",
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)


def _panel_price_signals(
    fig: go.Figure,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    """
    Panel 2 — BTC close price with BUY ▲ / SELL ▼ fill-price markers and
    rolling VWAP lines (row 3).

    Markers are placed at the actual fill price (``close ± half_spread``)
    rather than at the candle close so they represent the real taker cost.
    ``bid_vwap`` and ``ask_vwap`` are shown as dashed lines once the VWAP
    rolling window is full (first ``VWAP_WINDOW`` candles produce ``None``).
    """
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals["close"],
            mode="lines",
            name="Close",
            line=dict(color="black", width=0.8),
            opacity=0.85,
            legendgroup="price",
        ),
        row=3,
        col=1,
    )

    vwap_mask = signals["bid_vwap"].notna()
    if vwap_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signals.index[vwap_mask],
                y=signals["bid_vwap"][vwap_mask],
                mode="lines",
                name="bid_vwap",
                line=dict(color="dodgerblue", width=1.0, dash="dash"),
                opacity=0.85,
                legendgroup="vwap",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=signals.index[vwap_mask],
                y=signals["ask_vwap"][vwap_mask],
                mode="lines",
                name="ask_vwap",
                line=dict(color="coral", width=1.0, dash="dash"),
                opacity=0.85,
                legendgroup="vwap",
            ),
            row=3,
            col=1,
        )

    if trades is not None and not trades.empty:
        buys = trades[trades["side"] == "BUY"]
        sells = trades[trades["side"] == "SELL"]

        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys["fill_price"],
                    mode="markers",
                    name=f"BUY ({len(buys)})",
                    marker=dict(symbol="triangle-up", color=_BUY_COLOUR, size=9),
                    legendgroup="trades",
                ),
                row=3,
                col=1,
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells["fill_price"],
                    mode="markers",
                    name=f"SELL ({len(sells)})",
                    marker=dict(symbol="triangle-down", color=_SELL_COLOUR, size=9),
                    legendgroup="trades",
                ),
                row=3,
                col=1,
            )

    fig.update_yaxes(title_text="Price (USDT)", row=3, col=1)


def _panel_regime(fig: go.Figure, signals: pd.DataFrame) -> None:
    """
    Panel 3 — HMM regime step-line, colour bands, and confidence overlay (row 4).

    Three layers are drawn on the same subplot:

    1. **Background vrect bands** (``_draw_regime_bands``) — colour-coded fills
       for each contiguous regime run.  Drawn first so they sit below all traces.
    2. **Regime step-line** (dark-slate step trace) — the regime label mapped to
       a numeric position on the y-axis (0=trending_down → 3=trending_up).  This
       is the primary "readable" layer: regime transitions appear as vertical
       jumps, making the timeline immediately obvious.
    3. **Confidence dotted line** — ``regime_confidence`` (posterior probability)
       scaled by 3 so it uses the full vertical range.  Hover shows the real
       0–1 value.  A dashed threshold line marks ``HMM_MIN_CONFIDENCE × 3``.

    The y-axis tick labels show the regime names (not raw integers), and the
    right-hand annotation on the threshold line includes the real confidence
    value so there is no ambiguity about the scaling.

    Why scale confidence by 3?
        The regime axis range is [−0.3, 3.5].  If the confidence line were
        plotted at its raw 0–1 value it would be squeezed into the bottom 25 %
        of the panel and look like a flat line near zero — the original bug.
        Multiplying by 3 spreads it across the full panel height while keeping
        the hover tooltip accurate.
    """
    _draw_regime_bands(fig, signals, row=4)

    # ── 1. Regime step-line ───────────────────────────────────────────────
    # Map each label to an integer position; unknown / NaN → neutral (2).
    regime_num = signals["regime"].map(_REGIME_NUMERIC).fillna(2)
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=regime_num,
            mode="lines",
            name="Regime",
            line=dict(shape="hv", color="darkslategray", width=2.5),
            legendgroup="regime_step",
        ),
        row=4,
        col=1,
    )

    # ── 2. Legend colour swatches (square marker per regime, no data) ─────
    for label, colour in _REGIME_LEGEND_COLOURS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=12,
                    color=colour,
                    symbol="square",
                    line=dict(color="grey", width=0.5),
                ),
                name=label,
                legendgroup=f"regime_legend_{label}",
                showlegend=True,
            ),
            row=4,
            col=1,
        )

    # ── 3. Confidence dotted overlay (scaled ×3 to fill panel height) ─────
    if "regime_confidence" in signals.columns:
        conf_series = pd.to_numeric(signals["regime_confidence"], errors="coerce")
        # Scale to the same [0, 3] space as the regime axis so the line is not
        # squashed into the bottom 25 % of the panel.  Hover shows real value.
        conf_scaled = conf_series * 3.0
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=conf_scaled,
                mode="lines",
                name=f"Confidence (×3, threshold={HMM_MIN_CONFIDENCE:.2f})",
                line=dict(color="navy", width=1.0, dash="dot"),
                opacity=0.70,
                customdata=conf_series,
                hovertemplate="%{x}<br>confidence: %{customdata:.3f}<extra></extra>",
                legendgroup="confidence",
            ),
            row=4,
            col=1,
        )
        fig.add_hline(
            y=HMM_MIN_CONFIDENCE * 3.0,
            line_dash="dash",
            line_color="navy",
            line_width=0.9,
            annotation_text=f"Min conf {HMM_MIN_CONFIDENCE:.2f} (×3={HMM_MIN_CONFIDENCE * 3:.2f})",
            annotation_position="top right",
            row=4,
            col=1,
        )

    # Y-axis: show regime names as tick labels, not raw integers.
    fig.update_yaxes(
        tickvals=[0, 1, 2, 3],
        ticktext=["trending_down", "high_volatility", "neutral", "trending_up"],
        range=[-0.3, 3.5],
        title_text="Regime  ·  Confidence (×3)",
        row=4,
        col=1,
    )


def _panel_vwap(fig: go.Figure, signals: pd.DataFrame) -> None:
    """
    Panel 4 — VWAP reference prices vs micro-prices with near-miss markers
    (row 5).

    Plots four series:
    *  ``bid_vwap`` / ``ask_vwap`` — rolling momentum reference prices (lines).
    *  ``best_buy_micro`` / ``best_sell_micro`` — micro-prices of the best
       scored candidate at each candle (dot scatter to avoid connecting gaps).

    Grey dot markers (●) highlight VWAP gate near-misses: candles where the
    confidence gate AND regime gate both passed but the VWAP condition failed:
    *  BUY near-miss:  ``best_buy_micro ≤ ask_vwap``
    *  SELL near-miss: ``best_sell_micro ≥ bid_vwap``

    A dense cluster of near-miss dots suggests the VWAP window may be too
    wide (catching too many candidates just below/above the threshold).
    """
    vwap_mask = signals["bid_vwap"].notna()
    micro_buy_mask = signals["best_buy_micro"].notna()
    micro_sell_mask = signals["best_sell_micro"].notna()

    if vwap_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signals.index[vwap_mask],
                y=signals["bid_vwap"][vwap_mask],
                mode="lines",
                name="bid_vwap (P4)",
                line=dict(color="dodgerblue", width=1.0),
                opacity=0.9,
                legendgroup="vwap_p4",
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=signals.index[vwap_mask],
                y=signals["ask_vwap"][vwap_mask],
                mode="lines",
                name="ask_vwap (P4)",
                line=dict(color="coral", width=1.0),
                opacity=0.9,
                legendgroup="vwap_p4",
            ),
            row=5,
            col=1,
        )

    if micro_buy_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signals.index[micro_buy_mask],
                y=signals["best_buy_micro"][micro_buy_mask],
                mode="markers",
                name="best_buy_micro",
                marker=dict(size=3, color=_BUY_COLOUR, opacity=0.5),
                legendgroup="micro",
            ),
            row=5,
            col=1,
        )
    if micro_sell_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signals.index[micro_sell_mask],
                y=signals["best_sell_micro"][micro_sell_mask],
                mode="markers",
                name="best_sell_micro",
                marker=dict(size=3, color=_SELL_COLOUR, opacity=0.5),
                legendgroup="micro",
            ),
            row=5,
            col=1,
        )

    # ── VWAP gate near-miss markers ───────────────────────────────────────────
    has_conf = "regime_confidence" in signals.columns
    conf_ok: pd.Series = (
        signals["regime_confidence"].isna()
        | (signals["regime_confidence"] >= HMM_MIN_CONFIDENCE)
        if has_conf
        else pd.Series(True, index=signals.index)
    )
    regime_ok_buy = ~signals["regime"].isin(_BUY_BLOCKED)
    regime_ok_sell = ~signals["regime"].isin(_SELL_BLOCKED)

    if vwap_mask.any():
        nm_buy = (
            micro_buy_mask
            & conf_ok
            & regime_ok_buy
            & signals["ask_vwap"].notna()
            & (signals["best_buy_micro"] <= signals["ask_vwap"])
        ).fillna(False)

        nm_sell = (
            micro_sell_mask
            & conf_ok
            & regime_ok_sell
            & signals["bid_vwap"].notna()
            & (signals["best_sell_micro"] >= signals["bid_vwap"])
        ).fillna(False)

        if nm_buy.any():
            fig.add_trace(
                go.Scatter(
                    x=signals.index[nm_buy],
                    y=signals["best_buy_micro"][nm_buy],
                    mode="markers",
                    name=f"VWAP near-miss BUY ({int(nm_buy.sum())})",
                    marker=dict(size=6, color="grey", opacity=0.40, symbol="circle"),
                    legendgroup="nearmiss",
                ),
                row=5,
                col=1,
            )
        if nm_sell.any():
            fig.add_trace(
                go.Scatter(
                    x=signals.index[nm_sell],
                    y=signals["best_sell_micro"][nm_sell],
                    mode="markers",
                    name=f"VWAP near-miss SELL ({int(nm_sell.sum())})",
                    marker=dict(size=6, color="silver", opacity=0.40, symbol="circle"),
                    legendgroup="nearmiss",
                ),
                row=5,
                col=1,
            )

    fig.update_yaxes(title_text="Price (USDT)", row=5, col=1)


def _panel_funnel(fig: go.Figure, signals: pd.DataFrame) -> None:
    """
    Panel 5a — Signal funnel stacked horizontal bar (row 6, col 1).

    For each side (BUY and SELL) the stacked bar is split into four segments:
    *  **Blue**         — executed: passed all three gates.
    *  **Purple**       — confidence-blocked: model posterior < ``HMM_MIN_CONFIDENCE``.
    *  **Orange**       — regime-blocked: label in the blocked set for that side.
    *  **Yellow-green** — VWAP-blocked: micro-price did not confirm momentum.

    Total bar length = ``n_raw_candidates`` for each side.
    ``barmode="stack"`` is set globally so bars stack without extra config.
    """
    counts = _compute_filter_counts(signals)

    labels = ["BUY", "SELL"]
    executed = [counts["exec_buy"], counts["exec_sell"]]
    conf_b = [counts["conf_buy"], counts["conf_sell"]]
    regime_b = [counts["regime_buy"], counts["regime_sell"]]
    vwap_b = [counts["vwap_buy"], counts["vwap_sell"]]

    for values, colour, name in [
        (executed, _C_EXECUTED, "Executed"),
        (conf_b, _C_CONFIDENCE, "Confidence blocked"),
        (regime_b, _C_REGIME, "Regime blocked"),
        (vwap_b, _C_VWAP, "VWAP blocked"),
    ]:
        fig.add_trace(
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                name=name,
                marker=dict(color=colour),
                legendgroup=f"funnel_{name}",
            ),
            row=6,
            col=1,
        )

    fig.update_xaxes(title_text="# Candidates", row=6, col=1)


def _panel_regime_dist(fig: go.Figure, signals: pd.DataFrame) -> None:
    """
    Panel 5b — Stacked bar showing BUY / SELL / HOLD signal composition per
    HMM regime label (row 6, col 2).

    Bar height = total candles labelled with that regime.
    Colour breakdown shows how many fired as BUY / SELL or remained HOLD.
    Confirms that regime gates work as designed:
    *  ``trending_up``     → BUYs dominate; SELLs suppressed.
    *  ``trending_down``   → SELLs dominate; BUYs suppressed.
    *  ``high_volatility`` → near-all HOLD (both sides suppressed).
    *  ``neutral``         → mixed, gated by VWAP momentum only.
    """
    regime_order = ["trending_up", "trending_down", "high_volatility", "neutral"]
    buy_c, sell_c, hold_c = [], [], []

    for r in regime_order:
        mask = signals["regime"] == r
        buy_c.append(int((signals.loc[mask, "signal"] == 1).sum()))
        sell_c.append(int((signals.loc[mask, "signal"] == -1).sum()))
        hold_c.append(int((signals.loc[mask, "signal"] == 0).sum()))

    x_labels = ["trending_up", "trending_down", "high_vol", "neutral"]

    for values, colour, name in [
        (buy_c, _BUY_COLOUR, "BUY (dist)"),
        (sell_c, _SELL_COLOUR, "SELL (dist)"),
        (hold_c, "#aec7e8", "HOLD (dist)"),
    ]:
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=values,
                name=name,
                marker=dict(color=colour),
                opacity=0.85,
                legendgroup=f"regdist_{name}",
            ),
            row=6,
            col=2,
        )

    fig.update_yaxes(title_text="# Candles", row=6, col=2)


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────


def _draw_regime_bands(
    fig: go.Figure,
    signals: pd.DataFrame,
    row: int,
) -> None:
    """
    Add one ``add_vrect`` per contiguous same-regime run.

    Iterates over regime *transitions* only — O(transitions), not O(rows).
    For a 14,400-row dataset there are typically only a few hundred
    transitions, so this is significantly faster than calling ``add_vrect``
    14,400 times and keeps the resulting HTML file small.

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    signals : pd.DataFrame
        Must contain a ``regime`` column (string labels or NaN).
        NaN entries are treated as ``"neutral"``.
    row : int
        Subplot row to draw the bands on.
    """
    if signals.empty:
        return

    regimes = signals["regime"].fillna("neutral")

    # Identify only the rows where the regime changes (pandas shift trick)
    transitions = regimes[regimes != regimes.shift()]
    starts = list(transitions.index)
    labels = list(transitions.values)
    # Pair each transition start with the next one (or last index) as its end
    ends = starts[1:] + [signals.index[-1]]

    for ts_start, ts_end, label in zip(starts, ends, labels):
        colour = _REGIME_COLOURS.get(str(label), "rgba(224, 224, 224, 0.55)")
        fig.add_vrect(
            x0=ts_start,
            x1=ts_end,
            fillcolor=colour,
            layer="below",
            line_width=0,
            row=row,
            col=1,
        )


def _compute_filter_counts(signals: pd.DataFrame) -> dict[str, int]:
    """
    Compute per-side block counts for the funnel chart (Panel 5a).

    The three gates are applied sequentially, mirroring ``backtest/signals.py``
    and ``backtest/pnl.py``:

    1. **Confidence gate** — ``regime_confidence < HMM_MIN_CONFIDENCE``
    2. **Regime direction gate** — regime label in the blocked set for that side
    3. **VWAP gate** (residual) — passed gates 1+2 but was not executed

    Returns
    -------
    dict
        Keys: ``exec_buy``, ``exec_sell``, ``conf_buy``, ``conf_sell``,
        ``regime_buy``, ``regime_sell``, ``vwap_buy``, ``vwap_sell``.
    """
    raw_buy = signals["best_buy_micro"].notna()
    raw_sell = signals["best_sell_micro"].notna()
    exec_buy = int((signals["signal"] == 1).sum())
    exec_sell = int((signals["signal"] == -1).sum())

    has_conf = "regime_confidence" in signals.columns
    if has_conf:
        _low = signals["regime_confidence"].notna() & (
            signals["regime_confidence"] < HMM_MIN_CONFIDENCE
        )
        conf_buy = int((raw_buy & _low).sum())
        conf_sell = int((raw_sell & _low).sum())
        _conf_ok = ~_low
    else:
        conf_buy = conf_sell = 0
        _conf_ok = pd.Series(True, index=signals.index)

    regime_buy = int((raw_buy & _conf_ok & signals["regime"].isin(_BUY_BLOCKED)).sum())
    regime_sell = int(
        (raw_sell & _conf_ok & signals["regime"].isin(_SELL_BLOCKED)).sum()
    )

    # VWAP-blocked is the residual.  max(0, …) guards against rounding edge cases.
    vwap_buy = max(0, int(raw_buy.sum()) - exec_buy - conf_buy - regime_buy)
    vwap_sell = max(0, int(raw_sell.sum()) - exec_sell - conf_sell - regime_sell)

    return {
        "exec_buy": exec_buy,
        "exec_sell": exec_sell,
        "conf_buy": conf_buy,
        "conf_sell": conf_sell,
        "regime_buy": regime_buy,
        "regime_sell": regime_sell,
        "vwap_buy": vwap_buy,
        "vwap_sell": vwap_sell,
    }


def _save_figure(fig: go.Figure) -> None:
    """
    Save the figure as PNG (preferred) or HTML fallback.

    PNG export requires the ``kaleido`` package (``pip install kaleido``).
    If ``kaleido`` is not installed, the figure is saved as a self-contained
    interactive HTML file instead and a warning is logged.

    Files are written to ``backtest/results/`` (created if absent) with a
    UTC timestamp in the filename.
    """
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    try:
        import kaleido  # noqa: F401  (import only to verify availability)

        path = results_dir / f"backtest_chart_{ts}.png"
        fig.write_image(str(path), width=1800, height=1900, scale=1.5)
        log.info("Chart (PNG) saved → %s", path)
    except ImportError:
        path = results_dir / f"backtest_chart_{ts}.html"
        fig.write_html(str(path))
        log.warning(
            "kaleido not installed — chart saved as interactive HTML: %s  "
            "(install with: pip install kaleido)",
            path,
        )
