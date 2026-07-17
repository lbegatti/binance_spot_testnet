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
DataFrame.  This is a bps-based taker cost, NOT the candle range:

    half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20 000
    BUY  fill   = close + half_spread   (≡ synthetic ask — you cross the spread)
    SELL fill   = close − half_spread   (≡ synthetic bid — you receive the bid)

Why NOT ``(high − low) / 2``:
    A 1-min BTC candle range of $50–$300 gives half_spread $25–$150 — 10–100×
    larger than the real Binance BTCUSDT spread of ~1–5 bps.  ``(high-low)/2``
    is used in ``synthetic_book.py`` ONLY for constructing synthetic level
    prices; it is NOT the fill-cost model here.

No additional slippage fraction is applied — the half-spread already captures
the round-trip cost of crossing the synthetic bid/ask.

Usage
-----
    from backtest.signals import run_signals
    from backtest.pnl import simulate_pnl

    signals  = run_signals()
    # fee_rate defaults to BACKTEST_FEE_RATE; override to use best_params.json value:
    trades, equity, stats = simulate_pnl(signals, fee_rate=0.001)
    print(stats["n_position_guard_skips"])   # BUY signals suppressed by cash-reserve floor
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
    BACKTEST_RISK_FREE_RATE,
    MAX_POSITION_PCT,
    MIN_CASH_RESERVE_PCT,
    HMM_MIN_CONFIDENCE,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Position guard constant
# ---------------------------------------------------------------------------
# BTC amounts below this are treated as "flat" (position closed).
# Prevents floating-point dust from keeping the position guard permanently
# engaged after a full close (e.g. 1e-15 BTC from rounding).
_POSITION_DUST_BTC: float = 1e-6


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------


def compute_buy_and_hold(
    signals: pd.DataFrame,
    initial_usdt: float = BACKTEST_INITIAL_CAPITAL,
    initial_btc: float = BACKTEST_INITIAL_BTC,
    fee_rate: float = BACKTEST_FEE_RATE,
) -> dict[str, Any]:
    """
    Compute the passive buy-and-hold return over the same window as the
    backtest, using the first non-NaN close as the entry price.

    This benchmark answers:
        "Would simply holding BTC for the full window have been better
         or worse than the active strategy?"

    The benchmark is computed on the **same initial portfolio** as the active
    strategy (``initial_usdt`` USDT + ``initial_btc`` BTC already held), so
    the denominator for ``bnh_total_return_pct`` matches the denominator used
    by ``_compute_stats()`` for ``total_return_pct``.  Without this alignment
    the two percentages are not comparable when ``initial_btc > 0``.

    Assumptions
    -----------
    * The ``initial_usdt`` USDT is fully converted to BTC at bar 0 (first
      non-NaN close), paying a taker fee of ``fee_rate × gross_buy``.
    * Any BTC already held (``initial_btc``) is kept as-is — no extra fee.
    * Total BTC = converted USDT BTC + pre-existing BTC, held to session end.
    * No exit fee is applied — the position is held to session end.
    * Entry price = first non-NaN ``close`` (safe against HMM warm-up NaN rows).

    Parameters
    ----------
    signals : pd.DataFrame
        Must contain a ``close`` column.  Either the raw micro klines
        DataFrame or the output of ``run_signals()`` works — only ``close``
        is used.
    initial_usdt : float
        Starting USDT balance.  Defaults to ``BACKTEST_INITIAL_CAPITAL``.
    initial_btc : float
        Starting BTC already held (NOT converted — included at no additional
        fee).  Defaults to ``BACKTEST_INITIAL_BTC``.  Set to ``0.0`` for a
        USDT-only portfolio.
    fee_rate : float
        Taker fee fraction applied once when converting USDT → BTC at entry.
        Defaults to ``BACKTEST_FEE_RATE``.

    Returns
    -------
    dict with keys:
        ``bnh_entry_price``       — close at bar 0 (first non-NaN).
        ``bnh_exit_price``        — close at last bar.
        ``bnh_btc_held``          — total BTC held (converted USDT + initial_btc).
        ``bnh_final_equity_usdt`` — final value of all BTC in USDT.
        ``bnh_total_return_pct``  — net return (%) vs full initial portfolio.
    """
    close_series = signals["close"].dropna()
    if close_series.empty:
        log.warning("compute_buy_and_hold: no valid close prices — returning NaN.")
        return {
            "bnh_entry_price": float("nan"),
            "bnh_exit_price": float("nan"),
            "bnh_btc_held": float("nan"),
            "bnh_final_equity_usdt": float("nan"),
            "bnh_total_return_pct": float("nan"),
        }

    entry_price = float(close_series.iloc[0])
    exit_price = float(close_series.iloc[-1])

    # Convert USDT → BTC at entry (fee charged on the gross buy).
    entry_fee = initial_usdt * fee_rate  # USDT paid in fees
    net_usdt_deployed = initial_usdt - entry_fee  # USDT remaining after fee
    btc_from_usdt = net_usdt_deployed / entry_price

    # Total BTC = freshly bought BTC + pre-existing BTC (held at no extra cost).
    btc_held = btc_from_usdt + initial_btc

    final_equity = btc_held * exit_price

    # Initial portfolio value = USDT + pre-existing BTC valued at entry price.
    # This matches the denominator used by _compute_stats() so the two
    # percentages (strategy vs BnH) are directly comparable.
    initial_equity = initial_usdt + initial_btc * entry_price
    total_return_pct = (final_equity - initial_equity) / initial_equity * 100

    log.info(
        "Buy-and-hold: entry=%.2f  exit=%.2f  btc_total=%.6f  "
        "(from_usdt=%.6f + initial=%.6f)  final=%.2f USDT  return=%.2f%%",
        entry_price,
        exit_price,
        btc_held,
        btc_from_usdt,
        initial_btc,
        final_equity,
        total_return_pct,
    )

    return {
        "bnh_entry_price": entry_price,
        "bnh_exit_price": exit_price,
        "bnh_btc_held": btc_held,
        "bnh_final_equity_usdt": final_equity,
        "bnh_total_return_pct": total_return_pct,
    }


# Regimes that block a BUY / SELL signal (must stay in sync with analysis.py)
_BUY_BLOCKED_REGIMES = {"trending_down", "high_volatility"}
_SELL_BLOCKED_REGIMES = {"trending_up", "high_volatility"}


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

    Position model — pyramiding with a cash-reserve floor
    -----------------------------------------------------
    The strategy is mean-reversion: scale into dips, exit the whole book on a
    rally, repeat.  BUY legs may **stack (pyramid)** — each leg spends at most
    ``MAX_POSITION_PCT`` of the *available* USDT, and successive legs accumulate
    via ``open_strategy_qty`` until invested exposure reaches
    ``(1 − MIN_CASH_RESERVE_PCT)`` of mark-to-market equity:

    * **BUY** fires whenever cash sits above the reserve floor
      (``usdt − MIN_CASH_RESERVE_PCT × equity > 0``).  The leg is clamped so the
      trade never spends into the reserve; once the floor is reached the signal
      is skipped and counted in ``n_position_guard_skips``.  The entry price is
      tracked as a VWAP across all open legs.
    * **SELL** always closes the **full open position** in one shot
      (``qty = open_strategy_qty``), ignoring the synthetic book quantity
      for the exit leg.  This guarantees the strategy returns to flat on
      every SELL signal regardless of book depth, so the BUY gate reopens
      on the very next qualifying signal.

    The initial BTC balance (``initial_btc``) is excluded from position
    tracking.  Any SELL that fires before the first strategy BUY sells
    from the pre-existing BTC balance as an "orphan SELL" (a warning is
    logged; the equity curve and cash flows remain correct).

    Parameters
    ----------
    signals : pd.DataFrame
        Output of ``backtest.signals.run_signals()``.  Required columns:
        ``close``, ``signal`` (+1 / -1 / 0), ``buy_qty``, ``sell_qty``,
        ``half_spread``, ``regime``, ``best_buy_micro``, ``best_sell_micro``,
        ``ask_vwap``.  Optional (enables whipsaw guard): ``high``, ``low``.
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

    # Tracks BTC opened exclusively by strategy BUY signals (summed across all
    # pyramided legs).  Does NOT include initial_btc (pre-existing balance).
    # BUY: adds each leg's qty (stacking allowed up to the cash-reserve floor).
    # SELL: closes this entire amount in one shot, then resets to 0.
    open_strategy_qty: float = 0.0

    # Count of BUY signals suppressed because the cash-reserve floor was already
    # reached (no spendable USDT above MIN_CASH_RESERVE_PCT × equity).  Reported
    # in stats and the summary report so the user can see how many BUY signals
    # the reserve floor absorbed vs. how many actually executed.  (The stat key
    # name is retained for backward-compat with existing reports.)
    n_position_guard_skips: int = 0

    # Count of forced pessimistic exits triggered by the intra-candle whipsaw
    # guard (same 1-minute bar touched both BUY zone and SELL zone).
    n_whipsaw_exits: int = 0

    # Track the volume-weighted average entry price of the open position.
    # Updated on every BUY via VWAP formula; reset to 0.0 on every full close.
    # Used by the adaptive stop-loss to compute the unrealised loss threshold.
    avg_entry_price: float = 0.0

    # Counter: how many times the adaptive stop-loss forced a position close.
    n_stop_loss_fires: int = 0
    # Counter: how many 1m bars were skipped because the macro frame was in a
    # sustained trend (trend_pause == True from signals.py).
    n_trend_pause_skips: int = 0

    # Whipsaw guard requires the ``high`` and ``low`` columns added to the
    # signals DataFrame by signals.py Step 3.  If absent (legacy frames or
    # unit tests that construct a minimal DataFrame) the guard is silently
    # disabled so backward-compat is preserved.
    _has_whipsaw_cols = "high" in signals.columns and "low" in signals.columns

    trade_rows: list[dict] = []
    equity_rows: list[dict] = []

    # itertuples() is ~5× faster than iterrows() because it yields lightweight
    # namedtuples instead of constructing a full pd.Series per row.
    # The 'type: ignore' comments below suppress false-positive IDE warnings —
    # itertuples() attributes are resolved at runtime, not statically inferred.
    for row in signals.itertuples():
        ts = row.Index  # type: ignore[union-attr]
        close = float(row.close)  # type: ignore[union-attr]
        sig = int(row.signal)  # type: ignore[union-attr]
        # Pre-extract half_spread once for use in the whipsaw guard below.
        _half_spread_ws = (
            float(row.half_spread) if pd.notna(row.half_spread) else 0.0  # type: ignore[union-attr]
        )

        # ── Adaptive stop-loss (unconditional — fires even during trend_pause) ─
        # Checked FIRST so an open position is always protected regardless of
        # whether the trend-pause or whipsaw gate would fire on the same bar.
        _sl_pct = float(getattr(row, "stop_loss_pct", 0.0) or 0.0)
        if (
            open_strategy_qty > _POSITION_DUST_BTC
            and _sl_pct > 0.0
            and avg_entry_price > 0.0
            and close < avg_entry_price * (1.0 - _sl_pct)
        ):
            _sl_qty = min(open_strategy_qty, btc)
            if _sl_qty > 0:
                _sl_fill = close - _half_spread_ws
                _sl_gross = _sl_qty * _sl_fill
                _sl_fee = abs(_sl_gross) * fee_rate
                _sl_proceeds = _sl_gross - _sl_fee
                usdt += _sl_proceeds
                btc -= _sl_qty
                open_strategy_qty = 0.0  # full close — always flat after stop-loss
                _sl_loss_pct = (close - avg_entry_price) / avg_entry_price * 100
                avg_entry_price = 0.0  # reset AFTER computing loss for the log
                n_stop_loss_fires += 1
                trade_rows.append(
                    {
                        "timestamp": ts,
                        "side": "SELL_STOP_LOSS",
                        "fill_price": _sl_fill,
                        "quantity": _sl_qty,
                        "gross": _sl_gross,
                        "fee": _sl_fee,
                        "net_cost": None,
                        "net_proceeds": _sl_proceeds,
                        "regime": getattr(row, "regime", None),
                    }
                )
                log.warning(
                    "STOP-LOSS │ FORCED EXIT │ %s │ qty=%.6f BTC │ "
                    "exit=%.2f │ threshold=%.2f%% │ actual_loss=%.2f%%",
                    ts,
                    _sl_qty,
                    _sl_fill,
                    _sl_pct * 100,
                    _sl_loss_pct,
                )

        # ── Intra-candle whipsaw guard ─────────────────────────────────────────
        # Fires when ALL of these hold:
        #   • we hold an open strategy position (already long)
        #   • the same 1-minute bar had low  ≤ best_buy_micro  (BUY zone reached)
        #   • the same 1-minute bar had high ≥ best_sell_micro (SELL zone reached)
        # At 1-minute bar resolution we cannot determine which extreme filled
        # first, so we take the pessimistic assumption: force-close the position
        # immediately at  low − half_spread  and skip normal signal processing
        # for this bar.
        _skip_signals = False
        if _has_whipsaw_cols and open_strategy_qty > _POSITION_DUST_BTC:
            _high = float(getattr(row, "high", float("nan")))
            _low = float(getattr(row, "low", float("nan")))
            _bsm = getattr(row, "best_buy_micro", None)
            _ssm = getattr(row, "best_sell_micro", None)
            if (
                _bsm is not None
                and pd.notna(_bsm)
                and _ssm is not None
                and pd.notna(_ssm)
                and not np.isnan(_high)
                and not np.isnan(_low)
                and _low <= float(_bsm)  # low reached BUY zone
                and _high >= float(_ssm)  # high also reached SELL zone
            ):
                _ws_price = _low - _half_spread_ws  # pessimistic fill
                _ws_qty = min(open_strategy_qty, btc)
                if _ws_qty > 0:
                    _ws_gross = _ws_qty * _ws_price
                    _ws_fee = abs(_ws_gross) * fee_rate
                    _ws_proceeds = _ws_gross - _ws_fee
                    usdt += _ws_proceeds
                    btc -= _ws_qty
                    open_strategy_qty = max(0.0, open_strategy_qty - _ws_qty)
                    if open_strategy_qty <= _POSITION_DUST_BTC:
                        avg_entry_price = 0.0  # position fully closed
                    n_whipsaw_exits += 1
                    trade_rows.append(
                        {
                            "timestamp": ts,
                            "side": "SELL_WHIPSAW",
                            "fill_price": _ws_price,
                            "quantity": _ws_qty,
                            "gross": _ws_gross,
                            "fee": _ws_fee,
                            "net_cost": None,
                            "net_proceeds": _ws_proceeds,
                            "regime": getattr(row, "regime", None),
                        }
                    )
                    log.warning(
                        "WHIPSAW │ FORCED EXIT │ %s │ qty=%.6f BTC │ "
                        "exit=%.2f │ bar: low=%.2f(≤buy_micro=%.2f) "
                        "high=%.2f(≥sell_micro=%.2f)",
                        ts,
                        _ws_qty,
                        _ws_price,
                        _low,
                        float(_bsm),
                        _high,
                        float(_ssm),
                    )
                _skip_signals = True  # do not re-process BUY/SELL for this bar

        # ── Trend-pause gate (blocks new entries; equity mark-to-market runs) ──
        # Checked AFTER the stop-loss (which fires unconditionally) and whipsaw
        # guard.  Sets _skip_signals=True to suppress BUY and regular SELL for
        # this bar, but control falls through to the equity mark-to-market below
        # so the equity curve remains continuous during the pause.
        _trend_paused = bool(getattr(row, "trend_pause", False))
        if _trend_paused and not _skip_signals:
            n_trend_pause_skips += 1
            _skip_signals = True

        # BUY
        if sig == 1 and not _skip_signals:
            raw_qty = float(row.buy_qty) if pd.notna(row.buy_qty) else 0.0  # type: ignore[union-attr]
            half_spread = float(row.half_spread) if pd.notna(row.half_spread) else 0.0  # type: ignore[union-attr]

            # Fill at the synthetic ask: close + half_spread.
            # This is the natural taker cost — you cross the spread when buying.
            # half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20_000 (bps-based).
            eff_price = close + half_spread

            # Per-leg budget: MAX_POSITION_PCT of *available* USDT (each leg ≤ 20 %).
            usdt_budget = usdt * MAX_POSITION_PCT

            # Cash-reserve floor: pyramiding is allowed (successive BUY legs may
            # stack via open_strategy_qty), but the trade is clamped so at least
            # MIN_CASH_RESERVE_PCT of the portfolio's mark-to-market equity always
            # stays in USDT.  This caps invested exposure at (1 − reserve) = 80 %
            # instead of the old single-position rule that left ~90 % idle in cash.
            # Shared with execution/order_executor.py so live and backtest size
            # BUYs identically.
            equity = usdt + btc * close
            spendable = usdt - MIN_CASH_RESERVE_PCT * equity
            usdt_budget = min(usdt_budget, spendable)

            if usdt_budget <= 0:
                # Cash already at/through the reserve floor — suppress this leg.
                n_position_guard_skips += 1
                log.info(
                    "HOLD │ cash-reserve floor │ %s │ cash=%.2f ≤ %.0f%% of equity=%.2f │ BUY suppressed",
                    ts,
                    usdt,
                    MIN_CASH_RESERVE_PCT * 100,
                    equity,
                )
            else:
                # Balance guard: never spend more than the reserve-clamped budget.
                # Total debit per unit = eff_price × (1 + fee_rate).
                max_affordable = (
                    usdt_budget / (eff_price * (1.0 + fee_rate))
                    if eff_price > 0
                    else 0.0
                )
                qty = min(raw_qty, max_affordable)

                if qty > 0:
                    gross = qty * eff_price
                    fee = gross * fee_rate
                    net_cost = gross + fee  # total USDT debited
                    usdt -= net_cost
                    btc += qty
                    open_strategy_qty += qty  # track position opened by this BUY
                    # Update the VWAP average entry price across all pyramided legs.
                    # open_strategy_qty already includes this leg, so _btc_before is
                    # the size held before it: 0 for the first leg (→ avg = eff_price),
                    # non-zero for stacked legs (→ each fill blends into the running VWAP).
                    _btc_before = open_strategy_qty - qty
                    avg_entry_price = (
                        avg_entry_price * _btc_before + eff_price * qty
                    ) / open_strategy_qty
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
                            "regime": getattr(row, "regime", None),
                        }
                    )
                    log.info(
                        "BUY  │ ENTERED LONG  │ %s │ qty=%.6f BTC │ price=%.2f │ cost=%.2f USDT │ fee=%.4f",
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

        # SELL
        elif sig == -1 and not _skip_signals:
            half_spread = float(row.half_spread) if pd.notna(row.half_spread) else 0.0  # type: ignore[union-attr]

            # Fill at the synthetic bid: close - half_spread.
            # You receive less than mid when selling — the spread is the cost.
            # half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20_000 (bps-based).
            eff_price = close - half_spread

            if open_strategy_qty > _POSITION_DUST_BTC:
                # Close the FULL strategy-opened position in one shot.
                # Using the full open_strategy_qty (not the synthetic book qty)
                # ensures the strategy returns to flat immediately so the BUY
                # gate reopens on the very next qualifying signal.
                # Safety cap against rounding: cannot sell more BTC than held.
                qty = min(open_strategy_qty, btc)
            else:
                # No strategy-opened position — fall back to book-depth qty.
                # This handles the "orphan SELL" case where initial_btc > 0
                # and a SELL fires before the first strategy BUY.
                raw_qty = float(row.sell_qty) if pd.notna(row.sell_qty) else 0.0  # type: ignore[union-attr]
                qty = min(raw_qty, btc)

            if qty > 0:
                gross = qty * eff_price
                fee = gross * fee_rate
                net_proceeds = gross - fee  # USDT credited after fee
                usdt += net_proceeds
                btc -= qty
                open_strategy_qty = max(0.0, open_strategy_qty - qty)  # reset to flat
                if open_strategy_qty <= _POSITION_DUST_BTC:
                    avg_entry_price = 0.0  # position fully closed
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
                        "regime": getattr(row, "regime", None),
                    }
                )
                log.info(
                    "SELL │ EXITED  LONG  │ %s │ qty=%.6f BTC │ price=%.2f │ proceeds=%.2f USDT │ fee=%.4f",
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

        # Mark-to-market at every candle (BUY, SELL, and HOLD)
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

    # Post-loop bookkeeping
    log.info(
        "Position guard SKIP: %d BUY signal(s) suppressed — position was already open "
        "(single-position mean-reversion mode).",
        n_position_guard_skips,
    )
    final_close = float(signals["close"].iloc[-1])
    final_equity = usdt + btc * final_close

    if btc > 0:
        log.info(
            "Session ended with %.8f BTC open (≈ %.2f USDT at last close %.2f).",
            btc,
            btc * final_close,
            final_close,
        )

    # Build DataFrames
    trades_df = (
        pd.DataFrame(trade_rows).set_index("timestamp")
        if trade_rows
        else pd.DataFrame()
    )

    equity_df = pd.DataFrame(equity_rows).set_index("timestamp")
    # Annotate the equity curve with a drawdown column for convenience.
    peak = equity_df["equity"].cummax()
    equity_df["drawdown_pct"] = (equity_df["equity"] - peak) / peak * 100

    # Summary statistics
    round_trips = _pair_round_trips(trades_df, final_close, fee_rate=fee_rate)
    stats = _compute_stats(
        signals,
        equity_df,
        round_trips,
        initial_usdt,
        initial_btc,
        final_equity,
        n_position_guard_skips,
        n_whipsaw_exits,
    )
    # Inject guardrail counters (kept outside _compute_stats to avoid changing
    # its signature — consistent with how n_position_guard_skips is handled).
    stats["n_stop_loss_fires"] = n_stop_loss_fires
    stats["n_trend_pause_skips"] = n_trend_pause_skips

    log.info(
        "P&L: return=%.2f%%  trades=%d  win_rate=%.1f%%  max_dd=%.2f%%  sharpe=%.3f",
        stats["total_return_pct"],
        stats["n_round_trips"],
        stats["win_rate_pct"],
        stats["max_drawdown_pct"],
        stats["sharpe_ratio"],
    )

    return trades_df, equity_df, stats


def _pair_round_trips(
    trades_df: pd.DataFrame,
    last_close: float,
    fee_rate: float = 0.0,
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
    _first_winner_logged: bool = False  # debug flag — log once per call

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

        elif row["side"].startswith("SELL") and not open_buys:
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

        elif row["side"].startswith("SELL") and open_buys:
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

                # P&L in USDT: (exit − entry) × matched_qty, then deduct both
                # taker fees.  fill_price = close ± half_spread embeds only the
                # synthetic bid-ask spread cost; the explicit fee_rate is charged
                # separately by simulate_pnl and must be subtracted here too.
                gross_pnl: float = (exit_price - entry["price"]) * matched_qty
                entry_fee: float = entry["price"] * matched_qty * fee_rate
                exit_fee_: float = exit_price * matched_qty * fee_rate
                total_fees_: float = entry_fee + exit_fee_
                pnl_usdt: float = gross_pnl - total_fees_

                # ── One-time debug log (first winner) ─────────────────────
                # NOTE: this fires on EVERY simulate_pnl() call (runner.py,
                # sensitivity.py, unit tests, …).  Numbers will differ across
                # runs that use different data windows or parameters — that is
                # expected and is NOT a bug.  Compare only logs from the same
                # run (identical timestamp prefix).
                if not _first_winner_logged and gross_pnl > 0:
                    log.info(
                        "\n"
                        "  ╔════════════════════════════════════════════════════════════╗\n"
                        "  ║  [PNL.PY — _pair_round_trips] First winning round-trip    ║\n"
                        "  ║  (fires on every simulate_pnl call; values are run-specific)║\n"
                        "  ╠════════════════════════════════════════════════════════════╣\n"
                        "  ║  Entry Price : %10.4f USDT                              ║\n"
                        "  ║  Exit  Price : %10.4f USDT                              ║\n"
                        "  ║  Size        : %10.6f BTC                               ║\n"
                        "  ╠════════════════════════════════════════════════════════════╣\n"
                        "  ║  1) Gross Profit  = (%.4f - %.4f) × %.6f             ║\n"
                        "  ║                   = %+.4f USDT                            ║\n"
                        "  ║  2) Total Fees    = entry_fee %.4f + exit_fee %.4f     ║\n"
                        "  ║     (fee_rate used here = %.5f)                           ║\n"
                        "  ║                   = %.4f USDT                             ║\n"
                        "  ║  3) Net Profit    = %.4f - %.4f = %+.4f USDT          ║\n"
                        "  ╚════════════════════════════════════════════════════════════╝",
                        entry["price"],
                        exit_price,
                        matched_qty,
                        exit_price,
                        entry["price"],
                        matched_qty,
                        gross_pnl,
                        entry_fee,
                        exit_fee_,
                        fee_rate,
                        total_fees_,
                        gross_pnl,
                        total_fees_,
                        pnl_usdt,
                    )
                    _first_winner_logged = True
                # ─────────────────────────────────────────────────────────

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

    # Drain remaining open legs at session-end mark-to-market
    # Any BUY that never found a matching SELL is closed at the last close. So basically a Market Order.
    # price so that unrealized P&L is captured in the statistics.
    while open_buys:
        entry = open_buys.popleft()
        gross_pnl_mtm = (last_close - entry["price"]) * entry["qty"]
        # Only the entry fee was paid (no actual SELL → no exit fee charged).
        entry_fee_mtm = entry["price"] * entry["qty"] * fee_rate
        pnl_usdt = gross_pnl_mtm - entry_fee_mtm
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
    initial_btc: float,
    final_equity: float,
    n_position_guard_skips: int = 0,
    n_whipsaw_exits: int = 0,
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
        Starting USDT balance (cash leg of the initial portfolio).
    initial_btc : float
        Starting BTC balance.  Valued at the first candle close to produce
        ``initial_equity_total_usdt`` — the correct denominator for
        ``total_return_pct``.  If ``initial_btc == 0`` this equals
        ``initial_usdt`` and the result is identical to the old formula.
    final_equity : float
        Terminal portfolio value (USDT + BTC mark-to-market).

    Returns
    -------
    dict
        **Overall return:**
        ``initial_usdt``, ``initial_btc_as_usdt``,
        ``initial_equity_total_usdt``, ``final_equity_usdt``,
        ``total_return_pct``.

        **Trade-level:**
        ``n_round_trips``, ``win_rate_pct``, ``avg_trade_pnl_usdt``,
        ``profit_factor``, ``avg_holding_minutes``.

        **Risk:**
        ``max_drawdown_pct``, ``sharpe_ratio``, ``sortino_ratio``.

        **Signal counts:**
        ``n_buy_signals``, ``n_sell_signals``,
        ``n_raw_buy_candidates``, ``n_raw_sell_candidates``.

        **Filter hit rates:**
        ``confidence_filter_hit_rate_pct``,
        ``regime_filter_hit_rate_pct``,
        ``vwap_filter_hit_rate_pct``.
    """
    # Value the initial BTC leg at the first candle close so the denominator
    # reflects the true starting portfolio, not just the cash component.
    # Using equity_df["close"].iloc[0] keeps consistency with how final_equity
    # uses the last close — both anchored to actual market prices in the window.
    initial_close = float(equity_df["close"].iloc[0])
    initial_btc_as_usdt = initial_btc * initial_close
    initial_equity = initial_usdt + initial_btc_as_usdt
    total_return_pct = (final_equity - initial_equity) / initial_equity * 100

    # Per-trade metrics
    # A "round trip" is one complete BUY → SELL pair as produced by
    # _pair_round_trips().  Each entry in round_trips carries:
    #   entry_price / exit_price — fill prices of the opening BUY and closing SELL
    #   quantity                 — matched BTC quantity for this leg
    #   pnl_usdt                 — (exit_price - entry_price) × quantity  (fees embedded)
    #   holding_minutes          — calendar minutes between BUY fill and SELL fill
    #                              (NaN for legs closed at session-end mark-to-market)
    n_trades = len(round_trips)  # total completed round trips
    pnls = [rt["pnl_usdt"] for rt in round_trips]  # USDT P&L per round trip
    n_wins = sum(1 for p in pnls if p > 0)  # round trips with positive P&L
    win_rate = (
        n_wins / n_trades if n_trades > 0 else float("nan")
    )  # fraction of profitable round trips
    avg_pnl = (
        float(np.mean(pnls)) if pnls else float("nan")
    )  # mean P&L per round trip (USDT)

    gross_profit = sum(p for p in pnls if p > 0)  # sum of all winning round-trip P&Ls
    gross_loss = abs(
        sum(p for p in pnls if p < 0)
    )  # absolute sum of all losing round-trip P&Ls
    # profit_factor > 1 means gross wins exceed gross losses; ∞ means no losing trades at all
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # holding_minutes: time in minutes a position was open (BUY → SELL).
    # NaN entries (session-end mark-to-market closes) are excluded so the
    # average reflects only round trips that completed within the session.
    holdings = [
        rt["holding_minutes"]
        for rt in round_trips
        if not np.isnan(rt["holding_minutes"])
    ]
    avg_hold = (
        float(np.mean(holdings)) if holdings else float("nan")
    )  # mean holding period (minutes)

    # Drawdown
    max_drawdown = float(equity_df["drawdown_pct"].min())

    # Sharpe & Sortino — adaptive resampling for 24/7 crypto (no weekend gaps).
    #
    # Hardcoding "1D" collapses short debug windows (e.g. BACKTEST_MAX_ROWS=500
    # ≈ 8 h) to a single data point, making pct_change() return all NaN.
    # Instead, pick the coarsest period that still yields ≥ 2 observations:
    #
    #   ≥ 2 days of 1-min data  → daily buckets,  annualise × √365
    #   ≥ 2 hours of 1-min data → hourly buckets, annualise × √(365 × 24)
    #   shorter / debug windows → 5-min buckets,  annualise × √(365 × 24 × 12)
    #
    # All three conventions are self-consistent: (mean/std) × √(periods_per_year)
    # produces a comparable Sharpe regardless of bucket size.
    #
    # Risk-free rate (Rf):
    # BACKTEST_RISK_FREE_RATE is annualised (default 0.0 for crypto).
    # Exact compounding: Rf_per_period = (1 + annual_rf)^(1/n) − 1
    # This is mathematically correct for all rate levels and degenerates
    # to the linear approximation (rate/n) only at very small rates.
    # At the default of 0.0 both forms give exactly 0 — no behavioural change.
    n_candles = len(equity_df)
    if n_candles >= 2 * 1440:  # ≥ 2 full days
        resample_freq, periods_per_year = "1D", 365
    elif n_candles >= 2 * 60:  # ≥ 2 hours
        resample_freq, periods_per_year = "1h", 365 * 24
    else:  # short / debug window
        resample_freq, periods_per_year = "5min", 365 * 24 * 12

    # Exact per-period risk-free rate via compounding: (1 + r_annual)^(1/n) − 1
    rf_per_period = (1.0 + BACKTEST_RISK_FREE_RATE) ** (1.0 / periods_per_year) - 1.0

    sampled_equity = equity_df["equity"].resample(resample_freq).last().dropna()
    period_ret = sampled_equity.pct_change().dropna()
    excess_ret = period_ret - rf_per_period  # excess return over risk-free rate

    if len(excess_ret) > 1 and excess_ret.std() > 0:
        sharpe = float(excess_ret.mean() / excess_ret.std() * np.sqrt(periods_per_year))
    else:
        sharpe = float("nan")

    downside_ret = excess_ret[excess_ret < 0]
    if len(downside_ret) > 1 and downside_ret.std() > 0:
        sortino = float(
            excess_ret.mean() / downside_ret.std() * np.sqrt(periods_per_year)
        )
    else:
        sortino = float("nan")

    # Filter hit rates
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
    # For SELL, exclude bars where a position was already open — in that case
    # the regime gate is bypassed (exit close, not a new short entry), so the
    # signal is not truly "blocked" by regime.  Falls back to the old count if
    # sim_position_open is absent (legacy signal DataFrames / unit tests).
    regime_blocked_buy = int(
        (
            signals["best_buy_micro"].notna()
            & _conf_passed
            & signals["regime"].isin(_BUY_BLOCKED_REGIMES)
        ).sum()
    )
    # NOTE: sim_position_open is computed in signals.py BEFORE pnl.py runs the
    # stop-loss check, so it does not reflect SELL_STOP_LOSS exits.  Bars
    # immediately after a stop-loss are still marked sim_position_open=True
    # even though open_strategy_qty has been reset to 0.  This causes
    # regime_blocked_sell to be slightly understated post-stop-loss (the
    # affected bars are filtered out as "exit bypasses" when they are actually
    # legitimate regime blocks).  Magnitude: bounded by n_stop_loss_fires.
    # Cash flows and equity curve are NOT affected — diagnostic stat only.
    if "sim_position_open" in signals.columns:
        regime_blocked_sell = int(
            (
                signals["best_sell_micro"].notna()
                & _conf_passed
                & signals["regime"].isin(_SELL_BLOCKED_REGIMES)
                & ~signals["sim_position_open"]
            ).sum()
        )
    else:
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
        # initial_equity_total_usdt = initial_usdt + initial_btc × first_close
        # This is the correct denominator — using initial_usdt alone would
        # overstate the return when the portfolio starts with a BTC position.
        "initial_usdt": initial_usdt,
        "initial_btc_as_usdt": initial_btc_as_usdt,
        "initial_equity_total_usdt": initial_equity,
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
        # Signal counts — pre-trade activity before any gate is applied
        "n_buy_signals": int(
            exec_buy
        ),  # BUY  candidates that passed ALL three gates → executed
        "n_sell_signals": int(
            exec_sell
        ),  # SELL candidates that passed ALL three gates → executed
        "n_raw_buy_candidates": int(
            raw_buy
        ),  # BUY  opportunities scored by the pipeline (pre-gate)
        "n_raw_sell_candidates": int(
            raw_sell
        ),  # SELL opportunities scored by the pipeline (pre-gate)
        # Position guard — BUY signals that fired but were suppressed because
        # the strategy already held an open position (single-position MR mode).
        "n_position_guard_skips": n_position_guard_skips,
        # Whipsaw guard — forced pessimistic exits when same 1-minute bar touched
        # both BUY zone (low ≤ best_buy_micro) and SELL zone (high ≥ best_sell_micro).
        # Each event records a SELL_WHIPSAW trade at low − half_spread.
        "n_whipsaw_exits": n_whipsaw_exits,
        # Filter hit rates — % of raw candidates blocked by each gate (sequentially)
        # confidence_filter: model posterior < HMM_MIN_CONFIDENCE → regime too uncertain to trade
        # regime_filter:     passed confidence but regime direction unfavourable (e.g. trending_down blocks BUY)
        # vwap_filter:       passed confidence + regime but micro_price did not confirm momentum
        "confidence_filter_hit_rate_pct": confidence_hit_rate,  # % blocked by confidence gate (first gate)
        "regime_filter_hit_rate_pct": regime_hit_rate,  # % blocked by regime direction gate (second gate)
        "vwap_filter_hit_rate_pct": vwap_hit_rate,  # % blocked by VWAP momentum gate (third gate)
    }
