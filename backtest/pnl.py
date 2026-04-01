"""
backtest/pnl.py
---------------
Step 4 — Simulated P&L

Converts the signal DataFrame produced by ``backtest/signals.py`` into three
outputs used for strategy evaluation:

1. **Trade log** (``trades_df``) — one row per *executed* BUY or SELL.
   HOLD rows (``signal == 0``) carry no cash flow and never appear here.

2. **Equity curve** (``equity_df``) — portfolio value at *every* candle,
   including HOLD candles.  Required for drawdown and Sharpe computation
   because the portfolio is marked to market continuously even when no
   trade fires.

3. **Summary statistics** (``stats``) — Step 5 metrics from BACKTESTING.md:
   total return, win rate, max drawdown, Sharpe, Sortino, profit factor,
   average holding period, and filter hit rates.

Fill assumption
---------------
Fill prices are derived from the ``half_spread`` column stored in the signals
DataFrame (``(high - low) / 2`` — the same quantity computed in
``synthetic_book.py`` Step 1).  This is the natural taker cost:

    BUY  fill = close + half_spread   (≡ synthetic_best_ask — you cross the spread)
    SELL fill = close - half_spread   (≡ synthetic_best_bid — you receive the bid)

An optional extra ``slippage`` fraction can be added on top to model
queue/latency effects (defaults to ``BACKTEST_SLIPPAGE = 0.0``).

Usage
-----
    from backtest.signals import run_signals
    from backtest.pnl import simulate_pnl

    signals  = run_signals()
    trades, equity, stats = simulate_pnl(signals)
    print(stats)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from config_parameters import (
    BACKTEST_FEE_RATE,
    BACKTEST_INITIAL_BTC,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_SLIPPAGE,
)

log = logging.getLogger(__name__)

# Regimes that block a BUY / SELL signal (must stay in sync with analysis.py)
_BUY_BLOCKED_REGIMES = {"trending_down", "high_volatility"}
_SELL_BLOCKED_REGIMES = {"trending_up", "high_volatility"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_pnl(
    signals: pd.DataFrame,
    initial_usdt: float = BACKTEST_INITIAL_CAPITAL,
    initial_btc:  float = BACKTEST_INITIAL_BTC,
    fee_rate: float     = BACKTEST_FEE_RATE,
    slippage: float     = BACKTEST_SLIPPAGE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Simulate P&L on the signal DataFrame from ``run_signals()``.

    Walks through every candle in chronological order, executing BUY and
    SELL trades where ``signal != 0``, applying a balance guard identical
    to the live ``OrderExecutor``, and marking the portfolio to market at
    every candle (including HOLD rows) to build a continuous equity curve.

    Parameters
    ----------
    signals : pd.DataFrame
        Output of ``backtest.signals.run_signals()``.  Required columns:
        ``close``, ``signal`` (+1 / -1 / 0), ``buy_qty``, ``sell_qty``,
        ``half_spread``, ``regime``, ``best_buy_micro``, ``ask_vwap``.
    initial_usdt : float
        Starting USDT balance.  Defaults to ``BACKTEST_INITIAL_CAPITAL``.
    initial_btc : float
        Starting BTC balance.  Defaults to ``BACKTEST_INITIAL_BTC`` (0.0).
        Set to a non-zero value to simulate starting with an existing BTC
        position (e.g. 0.1 BTC already held before the backtest window).
    fee_rate : float
        Taker fee fraction per side (e.g. 0.001 = 0.10 %).
        Defaults to ``BACKTEST_FEE_RATE``.
    slippage : float
        Additional slippage fraction applied on top of the half_spread fill
        cost.  Defaults to ``BACKTEST_SLIPPAGE`` (0.0 = spread cost only).

    Returns
    -------
    trades_df : pd.DataFrame
        Indexed by ``timestamp``.  One row per executed trade with columns:
        ``side``, ``fill_price``, ``quantity``, ``gross``, ``fee``,
        ``net_cost`` (BUY) or ``net_proceeds`` (SELL), ``regime``.

    equity_df : pd.DataFrame
        Indexed by ``timestamp``.  One row per candle (including HOLDs)
        with columns: ``usdt``, ``btc``, ``close``, ``equity``,
        ``drawdown_pct``.

    stats : dict
        Step 5 summary metrics — see ``_compute_stats()`` for the full key
        list.
    """
    usdt = float(initial_usdt)
    btc  = float(initial_btc)

    trade_rows: list[dict] = []
    equity_rows: list[dict] = []

    for ts, row in signals.iterrows():
        close = float(row["close"])
        sig = int(row["signal"])

        # ── BUY ─────────────────────────────────────────────────────────────
        if sig == 1:
            raw_qty = float(row["buy_qty"]) if pd.notna(row["buy_qty"]) else 0.0
            half_spread = float(row["half_spread"]) if pd.notna(row["half_spread"]) else 0.0

            # Fill at the synthetic ask: close + half_spread.
            # This is the natural taker cost — you cross the spread when buying.
            # An optional extra slippage fraction covers queue/latency effects.
            eff_price = close + half_spread + close * slippage

            # Balance guard: cannot spend more USDT than available.
            # Total debit per unit = eff_price × (1 + fee_rate).
            max_affordable = (
                usdt / (eff_price * (1.0 + fee_rate)) if eff_price > 0 else 0.0
            )
            qty = min(raw_qty, max_affordable)

            if qty > 0:
                gross = qty * eff_price
                fee = gross * fee_rate
                net_cost = gross + fee  # total USDT debited
                usdt -= net_cost
                btc += qty
                trade_rows.append({
                    "timestamp": ts,
                    "side": "BUY",
                    "fill_price": eff_price,
                    "quantity": qty,
                    "gross": gross,
                    "fee": fee,
                    "net_cost": net_cost,
                    "net_proceeds": None,
                    "regime": row.get("regime"),
                })
                log.debug(
                    "BUY  %s | qty=%.6f | price=%.2f | cost=%.2f USDT | fee=%.4f",
                    ts, qty, eff_price, net_cost, fee,
                )
            else:
                log.warning(
                    "BUY skipped at %s — USDT %.2f insufficient at price %.2f",
                    ts, usdt, close,
                )

        # ── SELL ─────────────────────────────────────────────────────────────
        elif sig == -1:
            raw_qty = float(row["sell_qty"]) if pd.notna(row["sell_qty"]) else 0.0
            half_spread = float(row["half_spread"]) if pd.notna(row["half_spread"]) else 0.0

            # Fill at the synthetic bid: close - half_spread.
            # You receive less than mid when selling — the spread is the cost.
            # The optional extra slippage fraction is subtracted on top.
            eff_price = close - half_spread - close * slippage

            # Balance guard: cannot sell more BTC than held.
            qty = min(raw_qty, btc)

            if qty > 0:
                gross = qty * eff_price
                fee = gross * fee_rate
                net_proceeds = gross - fee  # USDT credited after fee
                usdt += net_proceeds
                btc -= qty
                trade_rows.append({
                    "timestamp": ts,
                    "side": "SELL",
                    "fill_price": eff_price,
                    "quantity": qty,
                    "gross": gross,
                    "fee": fee,
                    "net_cost": None,
                    "net_proceeds": net_proceeds,
                    "regime": row.get("regime"),
                })
                log.debug(
                    "SELL %s | qty=%.6f | price=%.2f | proceeds=%.2f USDT | fee=%.4f",
                    ts, qty, eff_price, net_proceeds, fee,
                )
            else:
                log.warning(
                    "SELL skipped at %s — BTC %.8f insufficient",
                    ts, btc,
                )

        # ── Mark-to-market at every candle (BUY, SELL, and HOLD) ────────────
        # The equity curve must be continuous so that drawdown and Sharpe
        # are computed correctly.  HOLD rows carry no cash flow but the
        # portfolio value still changes as the BTC price moves.
        equity_rows.append({
            "timestamp": ts,
            "usdt": usdt,
            "btc": btc,
            "close": close,
            "equity": usdt + btc * close,
        })

    # ── Post-loop bookkeeping ────────────────────────────────────────────────
    final_close = float(signals["close"].iloc[-1])
    final_equity = usdt + btc * final_close

    if btc > 0:
        log.info(
            "Session ended with %.8f BTC open (≈ %.2f USDT at last close %.2f).",
            btc, btc * final_close, final_close,
        )

    # ── Build DataFrames ─────────────────────────────────────────────────────
    trades_df = (
        pd.DataFrame(trade_rows).set_index("timestamp")
        if trade_rows
        else pd.DataFrame()
    )

    equity_df = pd.DataFrame(equity_rows).set_index("timestamp")
    # Annotate the equity curve with a drawdown column for convenience.
    peak = equity_df["equity"].cummax()
    equity_df["drawdown_pct"] = (equity_df["equity"] - peak) / peak * 100

    # ── Summary statistics ───────────────────────────────────────────────────
    round_trips = _pair_round_trips(trades_df, final_close)
    stats = _compute_stats(
        signals, equity_df, round_trips,
        initial_usdt, final_equity,
    )

    log.info(
        "P&L: return=%.2f%%  trades=%d  win_rate=%.1f%%  "
        "max_dd=%.2f%%  sharpe=%.3f",
        stats["total_return_pct"],
        stats["n_round_trips"],
        stats["win_rate_pct"],
        stats["max_drawdown_pct"],
        stats["sharpe_ratio"],
    )

    return trades_df, equity_df, stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pair_round_trips(
        trades_df: pd.DataFrame,
        last_close: float,
) -> list[dict]:
    """
    Pair each BUY with the subsequent SELL to form round-trip trades.

    Uses FIFO matching: the oldest open BUY is closed by the first SELL
    that follows it.  If a BUY is still open at session end it is closed
    at ``last_close`` (mark-to-market exit).

    Parameters
    ----------
    trades_df : pd.DataFrame
        Output of ``simulate_pnl()`` — one row per executed trade.
    last_close : float
        Final candle close price used to mark open positions to market.

    Returns
    -------
    list[dict]
        Each dict contains: ``entry_ts``, ``exit_ts``, ``entry_price``,
        ``exit_price``, ``quantity``, ``pnl_usdt``, ``holding_minutes``.
    """
    if trades_df.empty:
        return []

    round_trips: list[dict] = []

    # Track an open long position using explicit typed scalars so the type
    # checker can narrow them without warnings — avoids dict[str, Any] issues.
    has_open_buy:   bool                 = False
    open_buy_ts:    pd.Timestamp | None  = None
    open_buy_price: float                = 0.0
    open_buy_qty:   float                = 0.0

    for ts, row in trades_df.iterrows():
        if row["side"] == "BUY":
            # If a previous BUY is still open, treat the new BUY as a
            # separate position (back-to-back BUYs can happen when the
            # regime and VWAP both allow repeated buy signals).
            if not has_open_buy:
                has_open_buy   = True
                open_buy_ts    = pd.Timestamp(str(ts))
                open_buy_price = float(row["fill_price"])
                open_buy_qty   = float(row["quantity"])

        elif row["side"] == "SELL" and has_open_buy:
            exit_price: float = float(row["fill_price"])
            qty:        float = min(open_buy_qty, float(row["quantity"]))

            # P&L in USDT: (exit − entry) × qty.  Fees are already
            # embedded in the fill prices via the slippage / fee_rate
            # adjustments made in simulate_pnl().
            pnl_usdt: float = (exit_price - open_buy_price) * qty

            # Holding period in minutes (requires a datetime index).
            try:
                holding_min: float = (
                    pd.Timestamp(str(ts)) - open_buy_ts
                ).total_seconds() / 60.0
            except (TypeError, ValueError):
                holding_min = float("nan")

            round_trips.append({
                "entry_ts": open_buy_ts,
                "exit_ts": ts,
                "entry_price": open_buy_price,
                "exit_price": exit_price,
                "quantity": qty,
                "pnl_usdt": pnl_usdt,
                "holding_minutes": holding_min,
            })
            has_open_buy = False  # position closed

    # ── Close any remaining open BUY at session-end mark-to-market ──────────
    if has_open_buy:
        pnl_usdt = (last_close - open_buy_price) * open_buy_qty
        round_trips.append({
            "entry_ts": open_buy_ts,
            "exit_ts": None,  # session close — no explicit SELL
            "entry_price": open_buy_price,
            "exit_price": last_close,
            "quantity": open_buy_qty,
            "pnl_usdt": pnl_usdt,
            "holding_minutes": float("nan"),
        })

    return round_trips


def _compute_stats(
        signals: pd.DataFrame,
        equity_df: pd.DataFrame,
        round_trips: list[dict],
        initial_usdt: float,
        final_equity: float,
) -> dict[str, Any]:
    """
    Compute Step 5 performance metrics from the equity curve and round trips.

    Parameters
    ----------
    signals : pd.DataFrame
        Original signal DataFrame (needed for filter hit-rate computation).
    equity_df : pd.DataFrame
        Output of ``simulate_pnl()`` — one row per candle with ``equity``.
    round_trips : list[dict]
        Output of ``_pair_round_trips()``.
    initial_usdt : float
        Starting capital.
    final_equity : float
        Terminal portfolio value (USDT + BTC mark-to-market).

    Returns
    -------
    dict
        Keys: ``total_return_pct``, ``n_round_trips``, ``win_rate_pct``,
        ``avg_trade_pnl_usdt``, ``max_drawdown_pct``, ``sharpe_ratio``,
        ``sortino_ratio``, ``profit_factor``, ``avg_holding_minutes``,
        ``n_buy_signals``, ``n_sell_signals``,
        ``regime_filter_hit_rate_pct``, ``vwap_filter_hit_rate_pct``.
    """
    total_return_pct = (final_equity - initial_usdt) / initial_usdt * 100

    # ── Per-trade metrics ────────────────────────────────────────────────────
    n_trades = len(round_trips)
    pnls = [rt["pnl_usdt"] for rt in round_trips]
    n_wins = sum(1 for p in pnls if p > 0)
    win_rate = n_wins / n_trades if n_trades > 0 else float("nan")
    avg_pnl = float(np.mean(pnls)) if pnls else float("nan")

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    holdings = [rt["holding_minutes"] for rt in round_trips
                if not np.isnan(rt["holding_minutes"])]
    avg_hold = float(np.mean(holdings)) if holdings else float("nan")

    # ── Drawdown ─────────────────────────────────────────────────────────────
    max_drawdown = float(equity_df["drawdown_pct"].min())

    # ── Sharpe & Sortino (annualised for 24/7 crypto at 1 m resolution) ─────
    # Aggregate equity returns to daily to reduce noise, then annualise
    # with sqrt(365) — standard for crypto (no weekend gaps).
    daily_equity = equity_df["equity"].resample("1D").last().dropna()
    daily_ret = daily_equity.pct_change().dropna()

    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365))
    else:
        sharpe = float("nan")

    downside_ret = daily_ret[daily_ret < 0]
    if len(downside_ret) > 1 and downside_ret.std() > 0:
        sortino = float(daily_ret.mean() / downside_ret.std() * np.sqrt(365))
    else:
        sortino = float("nan")

    # ── Filter hit rates ─────────────────────────────────────────────────────
    # raw_buy_candidates  = candles where a buy opportunity was scored
    #                       (best_buy_micro is not None/NaN).
    # regime_blocked_buy  = candidates suppressed by the regime gate.
    # vwap_blocked_buy    = candidates that passed the regime gate but
    #                       were suppressed by the VWAP momentum filter.
    # executed_buys       = candidates that passed both gates → signal == +1.
    raw_buy = signals["best_buy_micro"].notna().sum()
    raw_sell = signals["best_sell_micro"].notna().sum()
    exec_buy = (signals["signal"] == 1).sum()
    exec_sell = (signals["signal"] == -1).sum()

    regime_blocked_buy = (
            signals["best_buy_micro"].notna()
            & signals["regime"].isin(_BUY_BLOCKED_REGIMES)
    ).sum()
    regime_blocked_sell = (
            signals["best_sell_micro"].notna()
            & signals["regime"].isin(_SELL_BLOCKED_REGIMES)
    ).sum()

    # VWAP-blocked = had a candidate, passed regime gate, but not executed.
    vwap_blocked_buy = int(raw_buy - exec_buy - regime_blocked_buy)
    vwap_blocked_sell = int(raw_sell - exec_sell - regime_blocked_sell)

    regime_hit_rate = (
        (regime_blocked_buy + regime_blocked_sell)
        / (raw_buy + raw_sell) * 100
        if (raw_buy + raw_sell) > 0 else float("nan")
    )
    vwap_hit_rate = (
        (vwap_blocked_buy + vwap_blocked_sell)
        / (raw_buy + raw_sell) * 100
        if (raw_buy + raw_sell) > 0 else float("nan")
    )

    return {
        # Overall return
        "initial_equity_usdt": initial_usdt,
        "final_equity_usdt": final_equity,
        "total_return_pct": total_return_pct,
        # Trade-level
        "n_round_trips": n_trades,
        "win_rate_pct": win_rate * 100 if not np.isnan(win_rate) else float("nan"),
        "avg_trade_pnl_usdt": avg_pnl,
        "profit_factor": profit_factor,
        "avg_holding_minutes": avg_hold,
        # Risk
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        # Signal counts
        "n_buy_signals": int(exec_buy),
        "n_sell_signals": int(exec_sell),
        "n_raw_buy_candidates": int(raw_buy),
        "n_raw_sell_candidates": int(raw_sell),
        # Filter hit rates
        "regime_filter_hit_rate_pct": regime_hit_rate,
        "vwap_filter_hit_rate_pct": vwap_hit_rate,
    }
