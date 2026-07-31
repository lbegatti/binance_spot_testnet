"""
visualization/session_chart.py
-------------------------------
End-of-session P&L chart generator for live trading sessions.

Renders a 2-panel HTML chart from the equity snapshots and order log captured
during a live trading session:

  Panel 1 (65%) — Strategy equity index vs. Buy & Hold baseline (both = 100 at t0).
                  Orders overlaid: filled BUY (solid green ▲) / SELL (solid
                  orange ▼); unfilled orders (placed but cancelled / never
                  matched) as hollow green ▲ / red ▼ markers, so an order that
                  moved the equity line is visibly distinct from one that did
                  nothing.
  Panel 2 (35%) — Component balances: USDT free vs. BTC value (= btc × mid_price).

Pure function — no global state, no side effects beyond writing the output
file.  Called from websocket_main.py's finally block after the session ends.

Scope boundary
--------------
The chart deliberately does NOT show:
  * Slippage vs. limit price
  * Intra-second sub-tick moves
  * Free vs. locked balance split.  Equity marks the STRATEGY-CONTROLLED
    position to market: free balance plus whatever the strategy itself locked
    this session (e.g. BTC resting in its own LIMIT SELL still counts).  Locked
    balances that existed at session start (foreign resting orders not placed by
    this strategy) are subtracted via locked_*_at_start, so they neither inflate
    the curve nor cause a jump against the free-only start_total denominator.
  * Cumulative trade count or win/loss ratio
"""

import logging
import os
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)


def generate_session_pnl_chart(
    snapshots: list[tuple],
    orders: list[dict],
    start_total_usdt: float,
    btc_start_price: float,
    out_path: str,
    locked_usdt_at_start: float = 0.0,
    locked_btc_at_start: float = 0.0,
) -> None:
    """
    Render and write the session P&L HTML chart.

    Parameters
    ----------
    snapshots : list[tuple]
        Each entry: (datetime_utc, usdt_total, btc_total, mid_price), where
        *_total = free + locked so a holding resting in a LIMIT order still
        counts.  Captured every HFT tick (~1 s) by
        AnalysisEngine.low_latency_analysis().  The foreign-locked baselines
        (locked_*_at_start) are subtracted from these totals before plotting.
    orders : list[dict]
        OrderExecutor.placed_orders.  Each dict carries "placed_at" (datetime),
        "side" ("BUY"/"SELL"), and "price".  After order_status_report() runs,
        each record is also enriched with "exec_qty" (final executed quantity):
        orders with exec_qty == 0 (cancelled / never matched) are drawn as
        hollow green ▲ / red ▼ markers, filled orders as solid.  When "exec_qty" is absent
        the order is assumed filled (a real fill is never hidden).  Entries
        without "placed_at" are skipped (legacy frames).
    start_total_usdt : float
        Initial portfolio value (usdt + btc × btc_start_price) — denominator of
        the strategy index.
    btc_start_price : float
        BTC mid price at session start — anchors the B&H baseline.
    out_path : str
        Absolute output path for the HTML file.
    locked_usdt_at_start, locked_btc_at_start : float
        Locked USDT / BTC on the account at session start — i.e. tied up in
        PRE-EXISTING orders this strategy did not place (shared testnet
        account).  Subtracted from every snapshot's free+locked total so the
        equity curve reflects only what the strategy controls.  Defaults to 0.0
        (no foreign locks).  Assumes these baselines stay constant during the
        session; a foreign order filling/cancelling mid-session can briefly make
        the adjusted balance dip — rare, and far better than counting foreign
        locks as strategy equity.

    The function silently returns (with a log line) if fewer than 2 snapshots
    were captured or if start_total_usdt / btc_start_price are non-positive.
    """
    if len(snapshots) < 2:
        log.info(
            "Session P&L chart skipped: only %d snapshot(s) captured (need ≥ 2).",
            len(snapshots),
        )
        return
    if start_total_usdt <= 0 or btc_start_price <= 0:
        log.info(
            "Session P&L chart skipped: start_total_usdt=%.2f, btc_start_price=%.2f.",
            start_total_usdt,
            btc_start_price,
        )
        return

    timestamps = [s[0] for s in snapshots]
    price_series = [s[3] for s in snapshots]
    # Snapshots mark free+locked.  Subtract the locked balances that existed at
    # session start (foreign resting orders not placed by this strategy) so the
    # curve reflects only the equity the STRATEGY controls: its free balance
    # plus whatever IT locked during the session (e.g. BTC in its own LIMIT
    # SELL).  Without this, foreign locked inflates the curve while start_total
    # (free-only) does not, producing a spurious jump.  The baselines are
    # constant, so subtracting them also makes the t0 equity equal start_total
    # (index starts at 100 and reconciles with the text report).
    usdt_series = [s[1] - locked_usdt_at_start for s in snapshots]
    btc_series = [s[2] - locked_btc_at_start for s in snapshots]

    # Strategy index — strategy-controlled equity at each tick, normalised to 100.
    strategy_equity = [
        u + b * p for u, b, p in zip(usdt_series, btc_series, price_series)
    ]
    strategy_index = [100.0 * e / start_total_usdt for e in strategy_equity]
    # B&H index — start_total fully held as BTC at btc_start_price.
    bnh_index = [100.0 * p / btc_start_price for p in price_series]
    btc_value_usdt = [b * p for b, p in zip(btc_series, price_series)]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.07,
        subplot_titles=(
            "Equity index (start = 100) — Strategy vs. Buy & Hold",
            "Component balances (USDT)",
        ),
    )

    # Panel 1 — equity index lines
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=strategy_index,
            mode="lines",
            name="Strategy",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=bnh_index,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#888888", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Panel 1 — order markers, split by outcome.  A marker is plotted at the
    # order's DISPATCH time regardless of whether it traded, so a LIMIT order
    # that was later cancelled without filling would otherwise look identical
    # to one that actually opened/closed a position (and moved the equity
    # line).  exec_qty is enriched onto each record by
    # OrderExecutor.order_status_report() (final Binance status, queried just
    # before this chart is built); when it is absent — e.g. a long session
    # where the order fell outside the report's head/tail cap — the order is
    # treated as filled so a genuine fill is never hidden.
    # Y-position uses the nearest strategy_index point so markers sit on the line.
    buy_fill_ts, buy_fill_y = [], []
    buy_open_ts, buy_open_y = [], []
    sell_fill_ts, sell_fill_y = [], []
    sell_open_ts, sell_open_y = [], []
    for o in orders:
        ts = o.get("placed_at")
        if ts is None or not isinstance(ts, datetime):
            continue
        # Linear scan is fine — order count is O(10–100) per session.
        nearest = min(
            range(len(timestamps)),
            key=lambda i: abs((timestamps[i] - ts).total_seconds()),
        )
        y_val = strategy_index[nearest]
        executed = o.get("exec_qty", 1.0) > 0  # absent ⇒ assume filled
        if o.get("side") == "BUY":
            if executed:
                buy_fill_ts.append(ts)
                buy_fill_y.append(y_val)
            else:
                buy_open_ts.append(ts)
                buy_open_y.append(y_val)
        elif o.get("side") == "SELL":
            if executed:
                sell_fill_ts.append(ts)
                sell_fill_y.append(y_val)
            else:
                sell_open_ts.append(ts)
                sell_open_y.append(y_val)

    # Filled BUY — solid green ▲
    if buy_fill_ts:
        fig.add_trace(
            go.Scatter(
                x=buy_fill_ts,
                y=buy_fill_y,
                mode="markers",
                name="BUY (filled)",
                marker=dict(
                    color="#2ca02c",
                    symbol="triangle-up",
                    size=11,
                    line=dict(width=1, color="#1b5e20"),
                ),
            ),
            row=1,
            col=1,
        )
    # Filled SELL — solid orange ▼
    if sell_fill_ts:
        fig.add_trace(
            go.Scatter(
                x=sell_fill_ts,
                y=sell_fill_y,
                mode="markers",
                name="SELL (filled)",
                marker=dict(
                    color="#ff7f0e",
                    symbol="triangle-down",
                    size=11,
                    line=dict(width=1, color="#b35900"),
                ),
            ),
            row=1,
            col=1,
        )
    # Unfilled BUY — hollow green ▲ (placed but cancelled / never matched; no
    # position change, so the equity line stays flat at this marker).  Coloured
    # green like the filled BUY but left hollow, so filled vs unfilled stays
    # distinguishable while the side (BUY) is obvious at a glance.
    if buy_open_ts:
        fig.add_trace(
            go.Scatter(
                x=buy_open_ts,
                y=buy_open_y,
                mode="markers",
                name="BUY (unfilled)",
                marker=dict(
                    color="#2ca02c",
                    symbol="triangle-up-open",
                    size=10,
                    line=dict(width=1.5, color="#2ca02c"),
                ),
            ),
            row=1,
            col=1,
        )
    # Unfilled SELL — hollow red ▼ (coloured red so the side is obvious; left
    # hollow so filled vs unfilled stays distinguishable — and red vs the filled
    # SELL's orange further separates the two).
    if sell_open_ts:
        fig.add_trace(
            go.Scatter(
                x=sell_open_ts,
                y=sell_open_y,
                mode="markers",
                name="SELL (unfilled)",
                marker=dict(
                    color="#d62728",
                    symbol="triangle-down-open",
                    size=10,
                    line=dict(width=1.5, color="#d62728"),
                ),
            ),
            row=1,
            col=1,
        )

    # Panel 2 — component balances
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=usdt_series,
            mode="lines",
            name="USDT free",
            line=dict(color="#17becf", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=btc_value_usdt,
            mode="lines",
            name="BTC × price",
            line=dict(color="#bcbd22", width=1.5),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=(
            f"Live session P&L — {timestamps[0].strftime('%Y-%m-%d %H:%M')} "
            f"to {timestamps[-1].strftime('%H:%M')} UTC<br>"
            "<sup>B&amp;H baseline: all starting equity held as BTC at session-start price. "
            "Equity marks the full position (free + locked) to market.</sup>"
        ),
        hovermode="x unified",
        showlegend=True,
        height=720,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Index (start = 100)", row=1, col=1)
    fig.update_yaxes(title_text="USDT", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    log.info(
        "Session P&L chart written: %s (%d snapshots, %d BUY [%d filled] / "
        "%d SELL [%d filled] markers).",
        out_path,
        len(snapshots),
        len(buy_fill_ts) + len(buy_open_ts),
        len(buy_fill_ts),
        len(sell_fill_ts) + len(sell_open_ts),
        len(sell_fill_ts),
    )
