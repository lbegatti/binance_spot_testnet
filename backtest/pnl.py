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

No additional slippage fraction is applied — the half-spread already captures
the round-trip cost of crossing the synthetic bid/ask.

Usage
-----
    from backtest.signals import run_signals
    from backtest.pnl import simulate_pnl

    signals  = run_signals()
    trades, equity, stats = simulate_pnl(signals)
    print(stats)
"""

import logging
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from config_parameters import (
    BACKTEST_FEE_RATE,
    BACKTEST_INITIAL_BTC,
    BACKTEST_INITIAL_CAPITAL,
    HMM_MIN_CONFIDENCE,
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
    initial_btc: float = BACKTEST_INITIAL_BTC,
    fee_rate: float = BACKTEST_FEE_RATE,
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
    btc = float(initial_btc)

    trade_rows: list[dict] = []
    equity_rows: list[dict] = []

    for ts, row in signals.iterrows():
        close = float(row["close"])
        sig = int(row["signal"])

        # ── BUY ─────────────────────────────────────────────────────────────
        if sig == 1:
            raw_qty = float(row["buy_qty"]) if pd.notna(row["buy_qty"]) else 0.0
            half_spread = (
                float(row["half_spread"]) if pd.notna(row["half_spread"]) else 0.0
            )

            # Fill at the synthetic ask: close + half_spread.
            # This is the natural taker cost — you cross the spread when buying.
            # half_spread = (high - low) / 2 already captures the round-trip cost.
            eff_price = close + half_spread

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
                trade_rows.append(
                    {
                        "timestamp": ts,
                        "side": "BUY",
                        "fill_price": eff_price,
                        "quantity": qty,
                        "gross": gross,
                        "fee": fee,
                        "net_cost": net_cost,
                        "net_proceeds": None,
                        "regime": row.get("regime"),
                    }
                )
                log.debug(
                    "BUY  %s | qty=%.6f | price=%.2f | cost=%.2f USDT | fee=%.4f",
                    ts,
                    qty,
                    eff_price,
                    net_cost,
                    fee,
                )
            else:
                log.warning(
                    "BUY skipped at %s — USDT %.2f insufficient at price %.2f",
                    ts,
                    usdt,
                    close,
                )

        # ── SELL ─────────────────────────────────────────────────────────────
        elif sig == -1:
            raw_qty = float(row["sell_qty"]) if pd.notna(row["sell_qty"]) else 0.0
            half_spread = (
                float(row["half_spread"]) if pd.notna(row["half_spread"]) else 0.0
            )

            # Fill at the synthetic bid: close - half_spread.
            # You receive less than mid when selling — the spread is the cost.
            # half_spread = (high - low) / 2 already captures the round-trip cost.
            eff_price = close - half_spread

            # Balance guard: cannot sell more BTC than held.
            qty = min(raw_qty, btc)

            if qty > 0:
                gross = qty * eff_price
                fee = gross * fee_rate
                net_proceeds = gross - fee  # USDT credited after fee
                usdt += net_proceeds
                btc -= qty
                trade_rows.append(
                    {
                        "timestamp": ts,
                        "side": "SELL",
                        "fill_price": eff_price,
                        "quantity": qty,
                        "gross": gross,
                        "fee": fee,
                        "net_cost": None,
                        "net_proceeds": net_proceeds,
                        "regime": row.get("regime"),
                    }
                )
                log.debug(
                    "SELL %s | qty=%.6f | price=%.2f | proceeds=%.2f USDT | fee=%.4f",
                    ts,
                    qty,
                    eff_price,
                    net_proceeds,
                    fee,
                )
            else:
                log.warning(
                    "SELL skipped at %s — BTC %.8f insufficient",
                    ts,
                    btc,
                )

        # ── Mark-to-market at every candle (BUY, SELL, and HOLD) ────────────
        # The equity curve must be continuous so that drawdown and Sharpe
        # are computed correctly.  HOLD rows carry no cash flow but the
        # portfolio value still changes as the BTC price moves.
        equity_rows.append(
            {
                "timestamp": ts,
                "usdt": usdt,
                "btc": btc,
                "close": close,
                "equity": usdt + btc * close,
            }
        )

    # ── Post-loop bookkeeping ────────────────────────────────────────────────
    final_close = float(signals["close"].iloc[-1])
    final_equity = usdt + btc * final_close

    if btc > 0:
        log.info(
            "Session ended with %.8f BTC open (≈ %.2f USDT at last close %.2f).",
            btc,
            btc * final_close,
            final_close,
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
        signals,
        equity_df,
        round_trips,
        initial_usdt,
        final_equity,
    )

    log.info(
        "P&L: return=%.2f%%  trades=%d  win_rate=%.1f%%  max_dd=%.2f%%  sharpe=%.3f",
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

    Uses a FIFO queue so that multiple concurrent open legs are supported.
    This correctly models three real-world entry strategies:

    * **Scaling in**  — several BUYs at descending prices before one SELL.
    * **Layering**    — BUYs placed at regular intervals (grid-style).
    * **Pyramiding**  — adding to a winning position before exiting.

    Each SELL closes the **oldest** open BUY leg (FIFO).  Any legs still
    open at session end are closed at ``last_close`` (mark-to-market exit).

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

    # FIFO queue used purely for accounting — NOT a live order tracker.
    # All trades in trades_df are already settled (simulate_pnl executed and
    # recorded every BUY and SELL before this function is called).  This queue
    # is just a cursor that pairs each BUY entry with its matching SELL so that
    # per-round-trip P&L, win rate, and holding time can be computed.
    # Each entry: {"ts": pd.Timestamp, "price": float, "qty": float}
    # A new BUY always appends to the right; a SELL pops from the left (oldest).
    open_buys: deque[dict] = deque()

    for ts, row in trades_df.iterrows():
        if row["side"] == "BUY":
            # All three strategies (scale-in, layering, pyramiding) land here.
            # Every BUY opens a new independent leg in the queue — no skipping.
            open_buys.append(
                {
                    "ts": pd.Timestamp(str(ts)),
                    "price": float(row["fill_price"]),
                    "qty": float(row["quantity"]),
                }
            )

        elif row["side"] == "SELL" and not open_buys:
            # Orphan SELL — no open BUY leg to match against.
            # Most likely cause: initial_btc > 0 and the first signal fired
            # is a SELL.  The cash flow is already correct in the equity curve
            # (simulate_pnl updated usdt/btc before _pair_round_trips runs);
            # only the round-trip stats miss this leg.
            log.warning(
                "\n"
                "  ╔══════════════════════════════════════════════════╗\n"
                "  ║  ORPHAN SELL — no matching open BUY leg found.  ║\n"
                "  ║  ts=%-44s  ║\n"
                "  ║  qty=%-10.6f  price=%-10.2f                 ║\n"
                "  ║  Equity curve is correct; round-trip stats skip  ║\n"
                "  ║  this leg.  Set initial_btc=0 to avoid this.    ║\n"
                "  ╚══════════════════════════════════════════════════╝\n",
                str(ts),
                float(row["quantity"]),
                float(row["fill_price"]),
            )

        elif row["side"] == "SELL" and open_buys:
            exit_price: float = float(row["fill_price"])
            remaining_sell: float = float(row["quantity"])

            # Exhaust as many open BUY legs as the SELL quantity allows.
            # Two sub-cases are handled cleanly:
            #
            #   Partial close  — SELL qty < oldest leg qty:
            #     matched_qty  = remaining_sell (full sell consumed)
            #     leftover     = entry["qty"] - matched_qty > 0
            #     → push leftover back to the FRONT of the queue (appendleft)
            #       so the next SELL can continue closing the same leg.
            #
            #   Over-sell      — SELL qty > oldest leg qty:
            #     matched_qty  = entry["qty"] (full leg consumed)
            #     remaining_sell reduced; loop continues to the next leg.
            while remaining_sell > 1e-10 and open_buys:
                entry = open_buys.popleft()
                matched_qty: float = min(entry["qty"], remaining_sell)

                # If only part of the leg was consumed, push the remainder
                # back to the front so the next SELL picks it up first.
                leftover: float = entry["qty"] - matched_qty
                if leftover > 1e-10:
                    open_buys.appendleft(
                        {"ts": entry["ts"], "price": entry["price"], "qty": leftover}
                    )

                remaining_sell -= matched_qty

                # P&L in USDT: (exit − entry) × matched_qty.  Fees are
                # already embedded in fill prices via half_spread / fee_rate.
                pnl_usdt: float = (exit_price - entry["price"]) * matched_qty

                try:
                    holding_min: float = (
                        pd.Timestamp(str(ts)) - entry["ts"]
                    ).total_seconds() / 60.0
                except (TypeError, ValueError):
                    holding_min = float("nan")

                round_trips.append(
                    {
                        "entry_ts": entry["ts"],
                        "exit_ts": ts,
                        "entry_price": entry["price"],
                        "exit_price": exit_price,
                        "quantity": matched_qty,
                        "pnl_usdt": pnl_usdt,
                        "holding_minutes": holding_min,
                    }
                )

    # ── Drain remaining open legs at session-end mark-to-market ─────────────
    # Any BUY that never found a matching SELL is closed at the last close. So basically a Market Order.
    # price so that unrealized P&L is captured in the statistics.
    while open_buys:
        entry = open_buys.popleft()
        pnl_usdt = (last_close - entry["price"]) * entry["qty"]
        round_trips.append(
            {
                "entry_ts": entry["ts"],
                "exit_ts": None,  # session close — no explicit SELL
                "entry_price": entry["price"],
                "exit_price": last_close,
                "quantity": entry["qty"],
                "pnl_usdt": pnl_usdt,
                "holding_minutes": float("nan"),
            }
        )

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
        ``regime_filter_hit_rate_pct``, ``vwap_filter_hit_rate_pct``,
        ``confidence_filter_hit_rate_pct``.
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

    holdings = [
        rt["holding_minutes"]
        for rt in round_trips
        if not np.isnan(rt["holding_minutes"])
    ]
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
    # confidence_blocked  = candidates suppressed because regime_confidence
    #                       < HMM_MIN_CONFIDENCE (model too uncertain).
    # regime_blocked      = candidates that passed the confidence gate but
    #                       were suppressed by the regime direction gate.
    # vwap_blocked        = candidates that passed both confidence and regime
    #                       gates but were suppressed by the VWAP filter.
    # executed            = candidates that passed all three gates → signal.
    raw_buy = signals["best_buy_micro"].notna().sum()
    raw_sell = signals["best_sell_micro"].notna().sum()
    exec_buy = (signals["signal"] == 1).sum()
    exec_sell = (signals["signal"] == -1).sum()

    # Confidence-blocked: had a candidate but model posterior < threshold.
    # regime_confidence column is present from signals.py v2+; fall back to 0
    # if running against an older signal DataFrame that lacks the column.
    has_confidence_col = "regime_confidence" in signals.columns
    if has_confidence_col:
        _conf_low = signals["regime_confidence"].notna() & (
            signals["regime_confidence"] < HMM_MIN_CONFIDENCE
        )
        confidence_blocked_buy = int(
            (signals["best_buy_micro"].notna() & _conf_low).sum()
        )
        confidence_blocked_sell = int(
            (signals["best_sell_micro"].notna() & _conf_low).sum()
        )
        # Passed-confidence mask — needed to correctly attribute regime blocks
        _conf_passed = ~_conf_low
    else:
        confidence_blocked_buy = 0
        confidence_blocked_sell = 0
        _conf_passed = pd.Series(True, index=signals.index)

    # Regime-blocked: passed confidence gate but regime was unfavourable.
    regime_blocked_buy = int(
        (
            signals["best_buy_micro"].notna()
            & _conf_passed
            & signals["regime"].isin(_BUY_BLOCKED_REGIMES)
        ).sum()
    )
    regime_blocked_sell = int(
        (
            signals["best_sell_micro"].notna()
            & _conf_passed
            & signals["regime"].isin(_SELL_BLOCKED_REGIMES)
        ).sum()
    )

    # VWAP-blocked: residual after confidence and regime.
    # max(0, …) guards against floating-point rounding edge cases.
    vwap_blocked_buy = max(
        0, int(raw_buy - exec_buy - confidence_blocked_buy - regime_blocked_buy)
    )
    vwap_blocked_sell = max(
        0, int(raw_sell - exec_sell - confidence_blocked_sell - regime_blocked_sell)
    )

    total_raw = raw_buy + raw_sell
    confidence_hit_rate = (
        (confidence_blocked_buy + confidence_blocked_sell) / total_raw * 100
        if total_raw > 0
        else float("nan")
    )
    regime_hit_rate = (
        (regime_blocked_buy + regime_blocked_sell) / total_raw * 100
        if total_raw > 0
        else float("nan")
    )
    vwap_hit_rate = (
        (vwap_blocked_buy + vwap_blocked_sell) / total_raw * 100
        if total_raw > 0
        else float("nan")
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
        "confidence_filter_hit_rate_pct": confidence_hit_rate,
    }
