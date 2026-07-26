from core.order_book_state import OrderBookState
import threading
import time
import logging
from datetime import datetime, timezone
import numpy as np

from config_parameters import (
    HFT_INTERVAL,
    HIST_INTERVAL,
    MIN_SNAPSHOTS,
    N_LEVELS,
    CCY,
    CRYPTOCCY,
    HMM_REFIT_INTERVAL,
    HMM_MIN_CONFIDENCE,
    VWAP_THRESHOLD_MULTIPLIER,
    TREND_CONSECUTIVE_BARS,
    TREND_COOLDOWN_BARS,
    MIN_CASH_RESERVE_PCT,
    MAX_PYRAMID_LEGS,
)
from execution.order_executor import OrderExecutor
from strategy.indicators import (
    add_trend_pause_flag,
    volume_weighted_average_price,
)
from strategy.book_utils import (
    build_levels,
    collect_candidates,
    select_best_opportunity,
)
from strategy.regime_director import RegimeDirector


class AnalysisEngine:
    """
    Strategy engine that consumes shared ``OrderBookState`` to run both
    low-latency and historical analyses in separate background threads.

    The class is intentionally decoupled from ``MessageHandler``: the two
    communicate exclusively through the injected ``OrderBookState`` instance,
    which guarantees that both threads always operate on the same data under
    the same locks.

    Both loops respect a shared ``threading.Event`` (``stop_event``) so that
    ``websocket_main.py`` can terminate the session cleanly after the
    configured duration has elapsed.

        Attributes:
        state (OrderBookState): Shared order book state, injected at construction.
        stop_event (threading.Event): Shared event; when set, both loops exit
            at their next scheduled wake-up.
        order_executor (OrderExecutor): Injected order placement handler.
        regime_director (RegimeDirector): Pre-fitted HMM regime detector,
            injected at construction (initial fit done in ``websocket_main.py``
            before threads start).  Updated every ``HIST_INTERVAL`` seconds
            inside ``historical_analysis()`` using a **two-speed** scheme:
            cheap Viterbi prediction on most iterations, full model re-fit
            every ``HMM_REFIT_INTERVAL`` seconds (default 300 s = 5 min).
            Reads are protected by ``_regime_lock``.
        _vwap_lock (threading.Lock): Serialises access to ``_bid_vwap`` and
            ``_ask_vwap`` between ``historical_analysis`` (writer) and
            ``low_latency_analysis`` (reader).
        _regime_lock (threading.Lock): Serialises access to
            ``regime_director.regime_label`` between ``historical_analysis``
            (writer via ``assign_regime_labels()``) and ``low_latency_analysis``
            (reader).
        _bid_vwap (float | None): Latest bid VWAP published by
            ``historical_analysis``; ``None`` until the first iteration.
        _ask_vwap (float | None): Latest ask VWAP published by
            ``historical_analysis``; ``None`` until the first iteration.
        _position_open (bool): Position guard flag — mirrors the single-position
            mean-reversion logic in ``backtest/pnl.py``.  Set to ``True`` after
            any BUY dispatch; reset to ``False`` after any SELL dispatch.
            Guards against order stacking in REST-fallback mode (balance never
            updated) and the WS race window (balance update arrives late).
        _position_guard_skips (int): Counter of BUY signals suppressed by the
            position guard this session; logged at session end.

    Note:
        The static helpers ``_build_levels``, ``_collect_candidates``, and
        ``_select_best_opportunity`` are thin wrappers that delegate to the
        public functions in ``strategy.book_utils``.  The implementations were
        extracted so that the backtesting pipeline (``backtest/signals.py``)
        can consume them without importing ``AnalysisEngine``.  The private
        names are retained here to keep all existing internal call sites
        unchanged.
    """

    def __init__(
        self,
        state: OrderBookState,
        stop_event: threading.Event,
        executor: OrderExecutor,
        regime_director: RegimeDirector,
        stop_loss_state: dict | None = None,
        refresh_stop_loss_fn=None,
        macro_trend_state: dict | None = None,
        refresh_macro_trend_fn=None,
        initial_avg_entry_price: float = 0.0,
    ):
        """
        Args:
            state (OrderBookState): The shared order book state instance that
                provides ``local_book``, ``history_order_book``, ``thread_lock``,
                ``balance_status``, and ``thread_balance_lock``.
            stop_event (threading.Event): Set by the session driver
                (``websocket_main.py``) when the session duration has elapsed.
                Both analysis loops check this event on every iteration and
                exit gracefully when it is set.
            executor (OrderExecutor): Injected order execution handler.
                Called by ``low_latency_analysis`` when a trade opportunity
                passes all filters.
            regime_director (RegimeDirector): Pre-fitted HMM regime detector.
                Must already have ``regime_label`` populated (i.e. called via
                ``websocket_main.py`` before threads start) so the very first
                ``low_latency_analysis`` iteration has a valid regime to read.
                Re-fitted on every ``historical_analysis`` iteration.
            initial_avg_entry_price (float): Session-start BTC price passed from
                ``websocket_main.py`` (``btc_start_price``).  Used as the
                stop-loss anchor when the position guard is pre-armed due to an
                inherited BTC balance.  Defaults to 0.0 (stop-loss disabled for
                inherited positions when the price fetch fails at startup).
        """
        self.regime_director = regime_director
        self.state = state
        self.stop_event = stop_event
        self.order_executor = executor
        self._vwap_lock = threading.Lock()
        self._regime_lock = threading.Lock()
        self._bid_vwap: float | None = None
        self._ask_vwap: float | None = None

        # Position guard — single-open-position mean-reversion mode.
        # Mirrors the backtest logic in backtest/pnl.py:
        #   _position_open = True  → a strategy BUY has been dispatched and
        #                            not yet offset by a SELL; skip further BUYs.
        #   _position_open = False → flat; the next qualifying BUY signal fires.
        #
        # This prevents order stacking in two real failure modes:
        #   1. REST-fallback mode: state.balance_status is never updated after
        #      an order (no outboundAccountPosition push), so the balance guard
        #      alone cannot stop repeated full-size BUY dispatches.
        #   2. WS-mode race window: a second tick can fire before the
        #      outboundAccountPosition event arrives and reduces free balance.
        self._position_open: bool = False
        self._position_guard_skips: int = 0

        # Volume-weighted average entry price of the strategy-opened position.
        # Set on BUY dispatch; reset to 0.0 on SELL or stop-loss exit.  Read by
        # the stop-loss check inside low_latency_analysis() — anchor of the
        # "mid < entry × (1 − pct)" floor.
        # Declared here (before the pre-arm block) so the conditional assignment
        # below is not overwritten by a later unconditional initialisation.
        self._avg_entry_price: float = 0.0

        # ── Pyramiding position accounting (live parity with backtest/pnl.py) ─
        # BUY legs may STACK up to the cash-reserve floor (MIN_CASH_RESERVE_PCT).
        # The running position is tracked as a volume-weighted cost basis so the
        # stop-loss anchor (_avg_entry_price) is the AVERAGE entry across all
        # open legs, not just the last one.
        #   _position_qty_btc   — total BTC held by strategy legs (inherited BTC
        #                         is folded in via the pre-arm below).
        #   _position_cost_usdt — Σ(leg_price × leg_qty) across open legs.
        # How the basis is updated:
        #   • When a BUY leg is PLACED, we add it to the basis straight away
        #     (order price × order qty) — we do NOT wait for the fill to be
        #     confirmed.
        #   • If that order never fills and is cancelled later (see
        #     cancel_stale_buy()), we subtract it back out.
        # We use the order price, not the exact fill price, because this average
        # entry only feeds the stop-loss trigger (not P&L accounting), so a tiny
        # price difference is harmless.
        self._position_qty_btc: float = 0.0
        self._position_cost_usdt: float = 0.0
        # Single-slot record of the CURRENT in-flight leg's optimistic accrual,
        # so it can be reversed on an unfilled cancel.  Legs are serialized (a
        # new one is dispatched only when no BUY is in flight), so one slot
        # always suffices.
        self._pending_leg_qty: float = 0.0
        self._pending_leg_cost: float = 0.0
        # Open-leg count — enforces MAX_PYRAMID_LEGS, the hard ceiling on how
        # many BUY legs may stack (reset to 0 on a full close).
        self._pyramid_legs: int = 0

        # Startup position (FLATTEN_ON_START=False): the account may already
        # hold BTC left over from a previous session.  Treat it as our STARTING
        # POSITION — inventory we already hold, NOT a pending order — so the
        # strategy simply trades around it: it can BUY more or SELL it as signals
        # dictate, and the stop-loss protects it.  It is recorded as the current
        # position (one held chunk) anchored at initial_avg_entry_price (the
        # session-start price, since the true original entry price is unknown),
        # which feeds the stop-loss and the reserve / exposure maths.
        with self.state.thread_balance_lock:
            _initial_btc = self.state.balance_status.get(CRYPTOCCY, 0.0)
        if _initial_btc >= 0.0001:
            self._position_open = True
            self._avg_entry_price = initial_avg_entry_price
            self._position_qty_btc = _initial_btc
            self._position_cost_usdt = _initial_btc * initial_avg_entry_price
            self._pyramid_legs = 1
            logging.info(
                "Startup position: %.8f %s already held — trading around it "
                "(BUY or SELL per signals). Stop-loss anchor: %.2f.",
                _initial_btc,
                CRYPTOCCY,
                initial_avg_entry_price,
            )

        # Trend-pause flag — mirrors backtest/signals.py.
        # Written by historical_analysis() on every HMM pulse, read by
        # low_latency_analysis() on every tick.  Both reads/writes go through
        # _regime_lock since the flag is logically tied to the macro regime
        # state (same source: regime_director.klines_df, same cadence: HMM pulse).
        #   True  → suppress all new BUY/SELL entries this tick.
        #   False → normal signal logic applies.
        # Starts False so the first ~5 min (until the first HMM pulse) trade
        # normally — identical to backtest behaviour where df_exec is filled
        # with False for the warm-up rows preceding the first regime label.
        self._trend_paused: bool = False
        self._trend_pause_skips: int = 0  # session counter for end-of-session log

        # Adaptive stop-loss — mirrors backtest/pnl.py.
        # stop_loss_state is a SHARED MUTABLE CONTAINER built by
        # websocket_main.py (a dict with keys "pct" and "last_day_utc").  We
        # never call out to Binance REST from here — the refresh_stop_loss_fn
        # closure does that, and we read the resulting float.  Both default to
        # disabled fallbacks so unit tests and legacy callers that don't pass
        # these args still work.
        self._stop_loss_state: dict = (
            stop_loss_state
            if stop_loss_state is not None
            else {"pct": 0.0, "last_day_utc": -1}
        )
        self._refresh_stop_loss_fn = refresh_stop_loss_fn  # None ⇒ no daily refresh
        self._stop_loss_lock = threading.Lock()

        # Session counter — logged at session end.
        self._n_stop_loss_fires: int = 0

        # Macro-trend overlay — mirrors backtest/signals.py + pnl.py.
        # macro_trend_state is a SHARED MUTABLE CONTAINER built by
        # websocket_main.py (a dict with keys "state" and "last_day_utc").  As
        # with the stop-loss, this thread never touches Binance REST — the
        # refresh_macro_trend_fn closure fetches daily klines and returns the
        # new "down"/"neutral"/"up" state, which we read under a lock.  Defaults
        # to a permanently-"neutral" container so unit tests and callers that
        # don't pass these args behave exactly as before (overlay inert).
        self._macro_trend_state: dict = (
            macro_trend_state
            if macro_trend_state is not None
            else {"state": "neutral", "last_day_utc": -1}
        )
        self._refresh_macro_trend_fn = refresh_macro_trend_fn  # None ⇒ no refresh
        self._macro_trend_lock = threading.Lock()

        # Session counter — logged at session end.
        self._n_macro_downtrend_liquidations: int = 0

        # ── Equity snapshot history (consumed by visualization/session_chart.py) ──
        # One entry per HFT tick: (utc_now, usdt_total, btc_total, mid_price),
        # where *_total = free + locked so a position resting in a LIMIT order
        # still counts toward equity.
        # Written exclusively by the HFT thread; read by the main thread AFTER
        # hft_thread.join() in websocket_main.py's finally block, so no lock
        # is needed.  For a 10-minute session this is ~600 entries (~50 KB).
        self._equity_snapshots: list[tuple] = []

    # ── Pyramiding cost-basis helpers ─────────────────────────────────────
    def _recompute_avg_entry(self) -> None:
        """Refresh ``_avg_entry_price`` from the running cost basis (0.0 flat)."""
        self._avg_entry_price = (
            self._position_cost_usdt / self._position_qty_btc
            if self._position_qty_btc > 1e-9
            else 0.0
        )

    def _add_leg_to_basis(self, price: float, qty: float) -> None:
        """Add a just-placed BUY leg to the running cost basis.

        Also records this leg's size/cost in the single-slot ``_pending_leg_*``
        so it can be removed again if the leg's LIMIT order never fills and is
        cancelled (see ``_back_out_pending_leg``).
        """
        self._pending_leg_qty = qty
        self._pending_leg_cost = price * qty
        self._position_qty_btc += qty
        self._position_cost_usdt += price * qty
        self._pyramid_legs += 1
        self._position_open = True
        self._recompute_avg_entry()

    def _back_out_pending_leg(self) -> None:
        """Remove the last-added leg from the basis (its order was cancelled
        without filling)."""
        self._position_qty_btc = max(
            0.0, self._position_qty_btc - self._pending_leg_qty
        )
        self._position_cost_usdt = max(
            0.0, self._position_cost_usdt - self._pending_leg_cost
        )
        self._pyramid_legs = max(0, self._pyramid_legs - 1)
        self._pending_leg_qty = 0.0
        self._pending_leg_cost = 0.0
        if self._pyramid_legs <= 0 or self._position_qty_btc <= 1e-9:
            self._reset_position_flat()
        else:
            self._recompute_avg_entry()

    def _reset_position_flat(self) -> None:
        """Clear all pyramiding state after a full close (SELL / stop-loss)."""
        self._position_open = False
        self._position_qty_btc = 0.0
        self._position_cost_usdt = 0.0
        self._pyramid_legs = 0
        self._pending_leg_qty = 0.0
        self._pending_leg_cost = 0.0
        self._avg_entry_price = 0.0

    @staticmethod
    def _build_levels(snaps_bids: dict, snaps_asks: dict, n: int = N_LEVELS) -> tuple:
        """
        Thin wrapper around :func:`strategy.book_utils.build_levels`.

        The implementation lives in ``book_utils.py`` so it can be consumed by
        both the live ``AnalysisEngine`` and the backtesting pipeline without
        importing ``AnalysisEngine``.  This wrapper preserves all existing
        internal call sites unchanged.

        :param snaps_bids: dictionary of bids streamed via websocket.
        :param snaps_asks: dictionary of asks streamed via websocket.
        :param n: depth of order book levels. Defaults to N_LEVELS (50).
        :return: (levels, median_depth, level_0_depth) — see
                 :func:`strategy.book_utils.build_levels` for full details.
        """
        return build_levels(snaps_bids, snaps_asks, n)

    @staticmethod
    def _collect_candidates(
        levels: list, median_depth: float, level_0_depth: float
    ) -> tuple:
        """
        Thin wrapper around :func:`strategy.book_utils.collect_candidates`.

        The implementation lives in ``book_utils.py`` so it can be consumed by
        both the live ``AnalysisEngine`` and the backtesting pipeline without
        importing ``AnalysisEngine``.  This wrapper preserves all existing
        internal call sites unchanged.

        :param levels: list of tuples (total_depth, mid_price, micro_price, obi, bq, aq) for the top N levels.
        :param median_depth: median total depth across the levels, used to identify thin order book conditions.
        :param level_0_depth: total depth at the best bid/ask level, used to assess liquidity.
        :return: (buy_candidates, sell_candidates) — see
                 :func:`strategy.book_utils.collect_candidates` for full details.
        """
        return collect_candidates(levels, median_depth, level_0_depth)

    @staticmethod
    def _select_best_opportunity(
        candidates: list, strategy_name: str, iteration: int
    ) -> tuple | None:
        """
        Thin wrapper around :func:`strategy.book_utils.select_best_opportunity`.

        The implementation lives in ``book_utils.py`` so it can be consumed by
        both the live ``AnalysisEngine`` and the backtesting pipeline without
        importing ``AnalysisEngine``.  This wrapper preserves all existing
        internal call sites unchanged.

        :param candidates: list of tuples (level_idx, delta, total_depth, obi, micro_price, bq, aq).
        :param strategy_name: ``"buy"`` or ``"sell"`` — for logging.
        :param iteration: current loop iteration number — for logging.
        :return: (level_idx, score|None, delta, depth, obi, micro_price, bq, aq)
                 or ``None`` — see :func:`strategy.book_utils.select_best_opportunity`.
        """
        return select_best_opportunity(candidates, strategy_name, iteration)

    def low_latency_analysis(self):
        """
        Periodically inspect the live order book and apply the low-latency
        trading strategy, gated by both a VWAP momentum filter and an HMM
        regime filter.

        Runs every ``HFT_INTERVAL`` seconds (default 1 s) until ``stop_event``
        is set.  On each wake-up it:

        1. **Balance guard** — reads ``state.balance_status`` under
           ``thread_balance_lock``.  Skips the iteration if both
           ``usdt_balance < 10.0`` and ``btc_balance < 0.0001``.
        2. **Order book copy** — acquires ``state.thread_lock``, copies bids
           and asks, releases the lock immediately.  Skips if bids are empty.
        3. **Level construction** — builds the top ``N_LEVELS`` (50) bid/ask
           pairs; computes ``total_depth``, ``mid_price``, ``micro_price``,
           ``OBI``, ``bq``, ``aq`` per level.
        4. **Candidate selection** — filters levels by depth adequacy (≥ 50 %
           of level-0 depth) and micro-mid direction (``micro_price > mid_price``
           → BUY candidate; ``micro_price < mid_price`` → SELL candidate).
        5. **Scoring** — ranks candidates by
           ``0.70 × norm_depth + 0.30 × norm_delta``; picks the best for each
           side.  Returns an 8-element tuple
           ``(level_idx, score|None, delta, depth, obi, micro_price, bq, aq)``.
         6. **VWAP threshold gate** (dip / strength confirmation with dead zone) —
            reads ``_bid_vwap`` / ``_ask_vwap`` under ``_vwap_lock``:
            - BUY:  execute only when ``bid_vwap`` is ``None`` **or**
              ``micro_price < bid_vwap × (1 − VWAP_THRESHOLD_MULTIPLIER)``.
              Anchored to the **bid** VWAP — the dip must be deep enough below
              the volume-weighted bid pressure to cover round-trip fees and
              leave profit.  Shallow dips inside the dead zone are rejected.
            - SELL: execute only when ``ask_vwap`` is ``None`` **or**
              ``micro_price ≥ ask_vwap × (1 + VWAP_THRESHOLD_MULTIPLIER)``.
              Anchored to the **ask** VWAP — the rally must be strong enough
              above the volume-weighted ask pressure to cover fees and profit.
              Weak rallies inside the dead zone are rejected.
            Each side uses its own reference price (bid for BUY, ask for SELL)
            to avoid cross-side anchoring bias.
            Both VWAPs are ``None`` for the first ~1 min; filter is transparent
            until ``historical_analysis`` publishes the first values.
        7. **Regime confidence gate** — reads ``regime_director.regime_confidence``
           (posterior probability from ``predict_proba()``) under ``_regime_lock``:
           - Skip **both** sides if ``regime_confidence < HMM_MIN_CONFIDENCE``
             (default 0.60) **and** ``_position_open`` is ``False``.  A
             sub-threshold score means the model cannot distinguish the current
             state clearly enough to justify a new entry.
           - When ``_position_open`` is ``True`` the gate is bypassed so a
             closing SELL is never blocked by a low-confidence reading —
             risk management filters apply to new entries only.
           - When ``regime_confidence`` is ``None`` (before the first
             ``historical_analysis`` run) the gate is transparent.
        8. **Trend-pause gate** — skips new BUY/SELL entries when
           ``_trend_paused`` is ``True`` (written by ``historical_analysis``
           when the macro frame shows ``TREND_CONSECUTIVE_BARS`` consecutive
           same-direction closes followed by a ``TREND_COOLDOWN_BARS`` cooldown).
           Bypassed when ``_position_open`` is ``True`` so a closing SELL can
           still execute during a trend-pause period.
        9. **Regime direction filter** (HMM gate) — reads
           ``regime_director.regime_label`` under ``_regime_lock``:
           - BUY:  skip if regime is ``"trending_down"`` or ``"high_volatility"``.
           - SELL: skip if regime is ``"trending_up"`` or ``"high_volatility"``
             **and** ``_position_open`` is ``False`` (i.e. no existing long to
             close).  When a position is open the SELL is always an exit, not a
             new short — the regime gate must never block it or the position
             becomes permanently stranded in a trending market.
           ``None`` label (before first ``historical_analysis`` run) is treated
           as transparent — all orders are allowed through.
        10. **Order execution** — delegates to
            ``OrderExecutor.execute("BUY"|"SELL", opportunity)``.

        Exits cleanly when ``stop_event`` is set, logging how many iterations
        were completed.
        """
        iteration = 0
        logging.info("HFT analysis loop started (interval: %ds).", HFT_INTERVAL)

        while not self.stop_event.is_set():
            with self.state.thread_balance_lock:
                usdt_balance = self.state.balance_status.get(CCY, 0.0)
                btc_balance = self.state.balance_status.get(CRYPTOCCY, 0.0)
            if usdt_balance < 10.0 and btc_balance < 0.0001:
                # logic is that if we do not have funds either in USD or BTC to buy/sell we just exit.
                self.stop_event.wait(HFT_INTERVAL)
                continue

            with self.state.thread_lock:
                if not self.state.local_book["bids"]:
                    logging.info("HFT: no bids available yet, skipping iteration.")
                    self.stop_event.wait(HFT_INTERVAL)
                    continue
                snaps_bids = self.state.local_book["bids"].copy()
                snaps_asks = self.state.local_book["asks"].copy()
            iteration += 1

            # ── Stale GTC BUY cancel ─────────────────────────────────────────
            # If the outstanding BUY order has been on the book for >10 seconds
            # without filling, cancel it so the locked USDT is freed and the
            # strategy can re-enter on the next valid signal.
            # cancel_stale_buy() returns True  → order cancelled (or no orderId);
            #                               False → order may have filled; keep guard.
            if self._position_open and self.order_executor.cancel_stale_buy():
                # A resting BUY leg was cancelled unfilled — reverse the
                # optimistic cost basis accrued for THIS leg at dispatch.
                # Earlier filled legs are untouched; we only go flat if this was
                # the last open leg.
                self._back_out_pending_leg()
                if self._pyramid_legs <= 0:
                    logging.info(
                        "HFT #%d — stale BUY cancelled; now flat (no open legs).",
                        iteration,
                    )
                else:
                    logging.info(
                        "HFT #%d — stale BUY leg cancelled; %d leg(s) still open "
                        "(avg entry %.2f).",
                        iteration,
                        self._pyramid_legs,
                        self._avg_entry_price,
                    )
                self.stop_event.wait(HFT_INTERVAL)
                continue

            # ── Resolve a resting LIMIT SELL (planned exit) ──────────────────
            # A non-urgent exit rests on the book like the BUY.  After the 10 s
            # window cancel_stale_sell() reports the outcome:
            #   "closed"     → it filled; we are flat again — reset the guard.
            #   "still_long" → cancelled unfilled (or never got an orderId);
            #                  keep the position and retry on the next signal.
            #   None         → no resting SELL, or the window has not elapsed.
            if self._position_open:
                _sell_outcome = self.order_executor.cancel_stale_sell()
                if _sell_outcome == "closed":
                    self._reset_position_flat()  # full close clears all legs
                    logging.info(
                        "HFT #%d — LIMIT SELL filled; position closed, now flat.",
                        iteration,
                    )
                    self.stop_event.wait(HFT_INTERVAL)
                    continue

            # Mirrors backtest/signals.py: the SELL block uses `and signal == 0`
            # to ensure at most one signal per bar.  In the live system we
            # achieve the same exclusivity with a per-tick local flag set when
            # the BUY block dispatches, then checked by the SELL block below.
            _buy_dispatched = False
            levels, median_depth, level_0_depth = self._build_levels(
                snaps_bids, snaps_asks
            )
            buy_candidates, sell_candidates = self._collect_candidates(
                levels, median_depth, level_0_depth
            )
            best_buy = self._select_best_opportunity(buy_candidates, "buy", iteration)
            best_sell = self._select_best_opportunity(
                sell_candidates, "sell", iteration
            )

            # ── Equity snapshot (consumed by end-of-session P&L chart) ─────
            # Captured every tick regardless of which gate fires below so the
            # equity curve is dense (~1 Hz).  Re-reads balances under the lock
            # to reflect any updates from cancel_stale_buy() earlier this tick.
            # mid_price comes from levels[0][1] — already computed; if levels
            # is empty (rare: asks not yet streamed) the snapshot is skipped.
            if levels:
                _mid_snap = float(levels[0][1])  # level[1] = mid_price
                with self.state.thread_balance_lock:
                    # Mark the FULL position (free + locked) so BTC locked in a
                    # resting LIMIT SELL (or USDT locked in a resting BUY) still
                    # counts toward equity — otherwise the index collapses while
                    # an exit rests on the book.
                    _usdt_snap = self.state.balance_status.get(
                        CCY, 0.0
                    ) + self.state.balance_locked.get(CCY, 0.0)
                    _btc_snap = self.state.balance_status.get(
                        CRYPTOCCY, 0.0
                    ) + self.state.balance_locked.get(CRYPTOCCY, 0.0)
                self._equity_snapshots.append(
                    (datetime.now(timezone.utc), _usdt_snap, _btc_snap, _mid_snap)
                )

            # ── Adaptive stop-loss check (mirrors backtest/pnl.py) ─────────
            # Fires UNCONDITIONALLY when we hold an open position and the
            # mid_price has fallen below avg_entry_price × (1 − stop_loss_pct).
            # Bypasses the confidence / trend-pause / regime / VWAP gates so a
            # stranded position is always protected.  mid_price comes from
            # levels[0] (level[1] is the mid per strategy.book_utils.build_levels).
            with self._stop_loss_lock:
                _sl_pct = self._stop_loss_state.get("pct", 0.0)
            _avg_ep = self._avg_entry_price

            if self._position_open and _sl_pct > 0.0 and _avg_ep > 0.0 and levels:
                _mid_price = float(levels[0][1])  # level[1] = mid_price
                _sl_floor = _avg_ep * (1.0 - _sl_pct)
                if _mid_price < _sl_floor:
                    # Use best_sell when available; otherwise build a fallback
                    # 8-element tuple matching select_best_opportunity()'s
                    # return shape so OrderExecutor.execute() can still place
                    # a closing order.
                    # IMPORTANT: OrderExecutor.execute() reads `bq` (NOT `aq`)
                    # as the requested SELL quantity (see execution/order_executor.py:
                    # `quantity = min(bq, btc)` in the SELL branch).  So we put
                    # the full BTC balance in `bq`; the executor's own balance
                    # cap will clamp it to what's actually available.
                    _sell_target = (
                        best_sell
                        if best_sell
                        else (
                            0,  # level_idx
                            None,  # score
                            0.0,  # delta
                            level_0_depth,  # depth
                            0.0,  # obi
                            _mid_price,  # micro_price (used as reference / fill price)
                            float(btc_balance),  # bq — used by SELL branch as qty
                            0.0,  # aq — unused on SELL
                        )
                    )
                    logging.warning(
                        "HFT #%d — STOP-LOSS triggered: mid=%.2f < entry=%.2f × "
                        "(1 − %.4f) = %.2f",
                        iteration,
                        _mid_price,
                        _avg_ep,
                        _sl_pct,
                        _sl_floor,
                    )
                    self._n_stop_loss_fires += 1
                    self._reset_position_flat()  # full close clears all legs
                    # Cancel any resting LIMIT exit first so the BTC it locked is
                    # freed for the urgent MARKET close, then force the exit.
                    self.order_executor.cancel_stale_sell(timeout_sec=0.0)
                    self.order_executor.execute("SELL", _sell_target, urgent=True)
                    self.stop_event.wait(HFT_INTERVAL)
                    continue  # skip normal signal evaluation this tick

            # ── Macro-trend force-to-cash (mirrors backtest/pnl.py) ────────
            # Symmetric counterpart to the "up" hold-&-ride rule below: when the
            # slow daily macro-trend overlay flags a persistent downtrend,
            # liquidate the whole book to cash immediately with an urgent MARKET
            # exit — exactly like the backtest's force-to-cash block.  Checked
            # AFTER the stop-loss (so a stop-loss on the same tick already
            # flattened → this no-ops) and BEFORE the regime / VWAP gates.  New
            # BUYs are suppressed in the BUY gate below, so this only ever needs
            # to handle the exit side.  Read here every tick so the value is in
            # scope for the BUY/SELL gates further down.
            with self._macro_trend_lock:
                _macro_state = self._macro_trend_state.get("state", "neutral")
            if self._position_open and _macro_state == "down" and levels:
                _mid_price = float(levels[0][1])  # level[1] = mid_price
                _sell_target = (
                    best_sell
                    if best_sell
                    else (
                        0,  # level_idx
                        None,  # score
                        0.0,  # delta
                        level_0_depth,  # depth
                        0.0,  # obi
                        _mid_price,  # micro_price (reference / fill price)
                        float(btc_balance),  # bq — used by SELL branch as qty
                        0.0,  # aq — unused on SELL
                    )
                )
                logging.warning(
                    "HFT #%d — MACRO-DOWNTREND force-to-cash: state='down' → "
                    "liquidating open position at mid=%.2f",
                    iteration,
                    _mid_price,
                )
                self._n_macro_downtrend_liquidations += 1
                self._reset_position_flat()  # full close clears all legs
                # Free any BTC locked in a resting LIMIT exit, then MARKET close.
                self.order_executor.cancel_stale_sell(timeout_sec=0.0)
                self.order_executor.execute("SELL", _sell_target, urgent=True)
                self.stop_event.wait(HFT_INTERVAL)
                continue  # skip normal signal evaluation this tick

            with self._vwap_lock:
                bid_vwap = self._bid_vwap
                ask_vwap = self._ask_vwap

            with self._regime_lock:
                current_regime = (
                    self.regime_director.regime_label
                )  # reads the label assigned in the historical_analysis thread.
                regime_confidence = self.regime_director.regime_confidence
                trend_paused = self._trend_paused  # mirrors backtest/signals.py

            # --- regime confidence gate ---
            # predict_proba()[-1][current_regime] < HMM_MIN_CONFIDENCE means
            # the model is not sure enough about the current state (e.g. 55 %
            # trending_up vs 45 % neutral).  Skip both sides to avoid trading
            # on a coin-flip signal.
            if regime_confidence is not None and regime_confidence < HMM_MIN_CONFIDENCE:
                if not self._position_open:
                    logging.info(
                        "HFT #%d — skipped: regime '%s' confidence %.2f < %.2f",
                        iteration,
                        current_regime,
                        regime_confidence,
                        HMM_MIN_CONFIDENCE,
                    )
                    self.stop_event.wait(HFT_INTERVAL)
                    continue
                logging.info(
                    "HFT #%d — low regime confidence %.2f but position open — allowing closing SELL.",
                    iteration,
                    regime_confidence,
                )

            # --- trend-pause gate (mirrors backtest/signals.py + pnl.py) ---
            # When the macro frame shows a sustained directional streak the
            # mean-reversion strategy should not enter — a "dip" inside a
            # downtrend is more likely the start of a deeper move than a
            # reversion target.  Skip BOTH BUY and SELL for this tick.
            # Backtest equivalent: pnl.py sets _skip_signals=True on bars
            # where trend_pause is True (see backtest/pnl.py around line 425).
            # Note: an open position is not closed here — the adaptive stop-loss
            # (checked earlier in this method) is the safety net; without it a
            # position could be stranded by trend-pause until the cooldown ends.
            if trend_paused:
                self._trend_pause_skips += 1
                logging.info(
                    "HFT #%d — trend_pause active (consecutive=%d, cooldown=%d, skips so far: %d)",
                    iteration,
                    TREND_CONSECUTIVE_BARS,
                    TREND_COOLDOWN_BARS,
                    self._trend_pause_skips,
                )
                if not self._position_open:
                    self.stop_event.wait(HFT_INTERVAL)
                    continue

            # After the first historical_analysis iteration (~1 min), _bid_vwap
            # and _ask_vwap are populated.  They act as a threshold-gated
            # dip/strength confirmation filter (mean-reversion strategy):
            #
            #   BUY  → execute only if bid_vwap is None **or**
            #          micro_price < bid_vwap × (1 − VWAP_THRESHOLD_MULTIPLIER)
            #          The dip must be deep enough to cover the round-trip fee
            #          and leave profit.  Microscopic noise inside the dead zone
            #          (≥ threshold below VWAP) is rejected.
            #
            #   SELL → execute only if ask_vwap is None **or**
            #          micro_price ≥ ask_vwap × (1 + VWAP_THRESHOLD_MULTIPLIER)
            #          The rally must be strong enough above the ask-side VWAP
            #          to cover fees and leave profit.
            #
            # Using bid_vwap for BUY and ask_vwap for SELL correctly anchors
            # each side to its own relevant reference price:
            #   bid_vwap tracks the volume-weighted bid pressure → dip benchmark.
            #   ask_vwap tracks the volume-weighted ask pressure → rally benchmark.
            #
            # While VWAPs are still None (first ~1 min) the filter is transparent
            # and orders execute based on regime + score alone.
            if best_buy:
                micro_price = best_buy[5]  # index 5 of the tuple
                buy_threshold = (
                    bid_vwap * (1.0 - VWAP_THRESHOLD_MULTIPLIER)
                    if bid_vwap is not None
                    else None
                )
                if _macro_state == "down":
                    # Macro-trend overlay: never dip-buy into a persistent
                    # downtrend (mirrors backtest/signals.py BUY gate).
                    logging.info(
                        "HFT #%d [buy] — skipped: macro-trend is 'down' "
                        "(persistent downtrend — no dip-buying).",
                        iteration,
                    )
                elif (
                    current_regime == "trending_down"
                    or current_regime == "high_volatility"
                ):
                    logging.info(
                        "HFT #%d [buy] — skipped: regime is '%s'",
                        iteration,
                        current_regime,
                    )
                elif buy_threshold is not None and micro_price >= buy_threshold:
                    logging.info(
                        "HFT #%d [buy] — skipped: micro_price %.2f >= vwap_floor %.2f "
                        "(bid_vwap=%.2f, threshold=%.4f — dip too shallow)",
                        iteration,
                        micro_price,
                        buy_threshold,
                        bid_vwap,
                        VWAP_THRESHOLD_MULTIPLIER,
                    )
                else:
                    # ── Ghost-reset ────────────────────────────────────────
                    # If we believe we hold legs but the account is truly flat
                    # (a LIMIT BUY that never filled), clear the stale basis.
                    # All must hold: no resting SELL (its locked BTC reads as
                    # free≈0), no unresolved BUY (handled by cancel_stale_buy
                    # after its 10 s window — disarming early re-fired the same
                    # signal every tick: 8 duplicate BUYs in 14 s on 2026-07-08),
                    # and a FRESH REST balance confirming flat (never on a ≤60 s
                    # snapshot).
                    if (
                        self._pyramid_legs > 0
                        and btc_balance < 0.0001
                        and not self.order_executor.has_pending_sell()
                        and not self.order_executor.has_pending_buy()
                        and self.order_executor.refresh_and_check_flat()
                    ):
                        self._reset_position_flat()
                        logging.info(
                            "HFT #%d [buy] — position reset: armed but BTC balance "
                            "%.8f ≈ 0 and no resting exit (LIMIT BUY unfilled).",
                            iteration,
                            btc_balance,
                        )

                    # ── Exposure gate (pyramiding, live parity w/ pnl.py) ──
                    # Replaces the old single-position guard.  A new BUY leg is
                    # allowed only when ALL hold; otherwise the signal is a skip:
                    #   • serialization — no BUY/SELL in flight (each leg fully
                    #     resolves before the next; this alone bounds how fast
                    #     legs stack and kills the stale-balance stampede that
                    #     pyramided to ~91% equity);
                    #   • leg cap — fewer than MAX_PYRAMID_LEGS legs open;
                    #   • reserve floor — free USDT above MIN_CASH_RESERVE_PCT of
                    #     mark-to-market equity (mirrors backtest/pnl.py; the
                    #     executor clamps the size to the same budget as a
                    #     second line of defence).
                    _equity = usdt_balance + btc_balance * micro_price
                    _spendable = usdt_balance - MIN_CASH_RESERVE_PCT * _equity
                    if (
                        self.order_executor.has_pending_buy()
                        or self.order_executor.has_pending_sell()
                    ):
                        self._position_guard_skips += 1
                        logging.info(
                            "HFT #%d [buy] — skipped: an order is still in flight "
                            "(serialized legs).",
                            iteration,
                        )
                    elif self._pyramid_legs >= MAX_PYRAMID_LEGS:
                        self._position_guard_skips += 1
                        logging.info(
                            "HFT #%d [buy] — skipped: max pyramid legs reached "
                            "(%d/%d).",
                            iteration,
                            self._pyramid_legs,
                            MAX_PYRAMID_LEGS,
                        )
                    elif _spendable <= 0.0:
                        self._position_guard_skips += 1
                        logging.info(
                            "HFT #%d [buy] — skipped: cash-reserve floor reached "
                            "(free %.2f ≤ %.0f%% of equity %.2f).",
                            iteration,
                            usdt_balance,
                            MIN_CASH_RESERVE_PCT * 100,
                            _equity,
                        )
                    else:
                        # Dispatch a new leg.  The executor sizes it (≤ 20 % of
                        # free USDT, clamped by the same reserve floor) and
                        # exposes the dispatched qty/price for the cost basis.
                        _buy_dispatched = True  # mirror signals.py exclusivity
                        self.order_executor.execute("BUY", best_buy)
                        _leg_qty = self.order_executor.last_buy_qty
                        if _leg_qty > 0.0:
                            self._add_leg_to_basis(
                                self.order_executor.last_buy_price, _leg_qty
                            )
                            logging.info(
                                "HFT #%d [buy] — leg %d dispatched: qty=%.6f @ "
                                "%.2f (avg entry %.2f).",
                                iteration,
                                self._pyramid_legs,
                                _leg_qty,
                                self.order_executor.last_buy_price,
                                self._avg_entry_price,
                            )
            if best_sell and not _buy_dispatched:
                # _buy_dispatched guard mirrors backtest/signals.py's
                # `signal == 0` exclusivity — at most one trade per tick.
                # Without this a BUY that fires earlier in the tick can be
                # immediately offset by a SELL in the same second when both
                # the bid_vwap and ask_vwap thresholds happen to align.
                micro_price = best_sell[5]
                sell_threshold = (
                    ask_vwap * (1.0 + VWAP_THRESHOLD_MULTIPLIER)
                    if ask_vwap is not None
                    else None
                )
                # Regime gate applies only when flat (no open position).
                # When _position_open is True the SELL closes an existing long,
                # not a new short entry — blocking it would strand the position
                # for the rest of the session.
                if _macro_state == "up":
                    # Macro-trend overlay: never take a mean-reversion SELL in a
                    # persistent uptrend — hold & ride.  Unlike the regime gate
                    # this is NOT bypassed when a position is open: in "up" the
                    # only permitted close is the stop-loss (checked earlier).
                    # Mirrors backtest/signals.py SELL gate.
                    logging.info(
                        "HFT #%d [sell] — skipped: macro-trend is 'up' "
                        "(hold & ride — no mean-reversion selling).",
                        iteration,
                    )
                elif not self._position_open and (
                    current_regime == "trending_up"
                    or current_regime == "high_volatility"
                ):
                    logging.info(
                        "HFT #%d [sell] — skipped: regime is '%s' (flat, no position to close)",
                        iteration,
                        current_regime,
                    )
                elif sell_threshold is not None and micro_price < sell_threshold:
                    logging.info(
                        "HFT #%d [sell] — skipped: micro_price %.2f < vwap_ceil %.2f "
                        "(ask_vwap=%.2f, threshold=%.4f — rally too weak)",
                        iteration,
                        micro_price,
                        sell_threshold,
                        ask_vwap,
                        VWAP_THRESHOLD_MULTIPLIER,
                    )
                else:
                    # Planned exit: rest a LIMIT GTC on the book (maker), like
                    # the BUY.  Keep the position guard armed and the stop-loss
                    # anchor intact until the order actually fills — resolved by
                    # cancel_stale_sell() at the top of the loop.  Do NOT mark
                    # flat here, or a BUY could fire while we still hold the
                    # position behind a resting SELL.  Skip if an exit is already
                    # working.
                    if self.order_executor.has_pending_sell():
                        logging.info(
                            "HFT #%d [sell] — exit already resting on the book; "
                            "skipping duplicate.",
                            iteration,
                        )
                    else:
                        self.order_executor.execute("SELL", best_sell)

            self.stop_event.wait(HFT_INTERVAL)  # sleep AFTER work, not before

        logging.info(
            "HFT analysis loop stopped after %d iteration(s) "
            "(%d BUY signal(s) suppressed by position guard, "
            "%d tick(s) suppressed by trend-pause gate, "
            "%d emergency stop-loss exit(s), "
            "%d macro-downtrend liquidation(s)).",
            iteration,
            self._position_guard_skips,
            self._trend_pause_skips,
            self._n_stop_loss_fires,
            self._n_macro_downtrend_liquidations,
        )

    def historical_analysis(self):
        """
        Periodically analyse the rolling order book snapshot window to compute
        VWAPs and refresh the HMM market-regime label.

        Runs every ``HIST_INTERVAL`` seconds (default 60 s / 1 min) until
        ``stop_event`` is set.  If fewer than ``MIN_SNAPSHOTS`` (100) snapshots
        have accumulated the iteration is skipped.

        **On each iteration (every 60 s):**

        1. **VWAP computation** — copies ``state.history_order_book`` under
           ``thread_lock``, converts to numpy arrays, and computes:

           .. code-block:: text

               bid_vwap = Σ(best_bid_i × volume_best_bid_i) / Σ(volume_best_bid_i)
               ask_vwap = Σ(best_ask_i × volume_best_ask_i) / Σ(volume_best_ask_i)

           Both VWAPs are published under ``_vwap_lock`` so that
           ``low_latency_analysis`` can read them safely on the next iteration.

        2. **HMM regime update — clock-boundary pulse (every 5 min)**

           The HMM block fires **only when a new 5-minute candle has closed**,
           detected by comparing the current UTC epoch floored to 300 s against
           the last processed boundary (``_last_hmm_boundary``).

           **Why clock-boundary instead of every 60 s?**

           * ``get_klines_data()`` fetches ``HMM_INTERVAL = 5m`` candles.  A new
             candle only closes every 300 s, so calling it on 4 out of every 5
             HIST_INTERVAL wakeups downloads identical data and wastes a Binance
             API call and O(n×k) Viterbi compute.
           * Aligning to the 5-minute candle close guarantees the HMM always
             sees a fresh, **complete** candle on every call — the last bar in
             the response is never a partial candle.
           * VWAP is unaffected and continues updating every 60 s.

           **Two-speed refit within HMM pulses:**

           * **Every HMM pulse (every 5 min)** — cheap path:
             ``get_klines_data()`` → ``predict_current_regime()`` (Viterbi,
             O(n×k)) → ``assign_regime_labels()``.
           * **Every ``hmm_refit_every`` pulses** — full path:
             ``get_klines_data()`` → ``select_hmm_model()`` (BIC search,
             O(n×k×iters)) → ``assign_regime_labels()``.
             ``hmm_refit_every = max(1, HMM_REFIT_INTERVAL // 300)``
             (default 1 at ``HMM_REFIT_INTERVAL = 300 s`` → full re-fit on
             every 5-minute boundary; increase ``HMM_REFIT_INTERVAL`` in
             ``config_parameters.py`` to space re-fits further apart).

           The slow network + CPU work runs **outside** ``_regime_lock``; only
           ``assign_regime_labels()`` is called inside the lock to minimise
           contention with ``low_latency_analysis``.

        Exits cleanly when ``stop_event`` is set, logging iteration and pulse
        counts.
        """
        iteration = 0
        hmm_iteration = 0
        _last_hmm_boundary: int = 0  # epoch seconds of the last processed 5m candle

        # Full BIC re-fit cadence among HMM pulses.
        # HMM_REFIT_INTERVAL = 300 s; pulse fires every 300 s → hmm_refit_every = 1
        # meaning every 5-minute boundary triggers a full re-fit.
        # Increase HMM_REFIT_INTERVAL in config_parameters.py to space re-fits apart
        # (e.g. 600 s → full re-fit every second pulse = every 10 min).
        hmm_refit_every = max(1, HMM_REFIT_INTERVAL // 300)

        logging.info(
            "Historical analysis loop started "
            "(VWAP interval: %ds, HMM pulse: 5-min boundaries, "
            "full refit every %d pulse(s) = every %d min).",
            HIST_INTERVAL,
            hmm_refit_every,
            hmm_refit_every * 5,
        )
        while not self.stop_event.is_set():
            self.stop_event.wait(HIST_INTERVAL)  # interruptible sleep
            if self.stop_event.is_set():
                break

            # ── Daily refresh of the adaptive stop-loss threshold ────────────
            # Fires once per UTC day.  Uses the refresher closure injected by
            # websocket_main.py so this thread never touches Binance REST
            # directly (keeps AnalysisEngine decoupled from the REST client).
            # On failure, we keep the previous value silently — the refresher
            # already logged the error.
            if self._refresh_stop_loss_fn is not None:
                today_utc = int(time.time()) // 86400
                with self._stop_loss_lock:
                    last_day = self._stop_loss_state.get("last_day_utc", -1)
                if today_utc != last_day:
                    new_pct = self._refresh_stop_loss_fn()
                    if new_pct is not None:
                        with self._stop_loss_lock:
                            self._stop_loss_state["pct"] = new_pct
                            self._stop_loss_state["last_day_utc"] = today_utc
                        logging.info(
                            "Stop-loss threshold refreshed: %.4f%% (UTC day %d).",
                            new_pct * 100,
                            today_utc,
                        )

            # ── Daily refresh of the macro-trend overlay state ───────────────
            # Fires once per UTC day via the refresher closure injected by
            # websocket_main.py (same decoupling / cadence as the stop-loss).
            # Keeps the previous state on failure (the refresher returns None and
            # logs).  None closure ⇒ overlay disabled, state stays "neutral".
            if self._refresh_macro_trend_fn is not None:
                _today_utc_mt = int(time.time()) // 86400
                with self._macro_trend_lock:
                    _last_day_mt = self._macro_trend_state.get("last_day_utc", -1)
                if _today_utc_mt != _last_day_mt:
                    _new_state = self._refresh_macro_trend_fn()
                    if _new_state is not None:
                        with self._macro_trend_lock:
                            self._macro_trend_state["state"] = _new_state
                            self._macro_trend_state["last_day_utc"] = _today_utc_mt
                        logging.info(
                            "Macro-trend state refreshed: '%s' (UTC day %d).",
                            _new_state,
                            _today_utc_mt,
                        )

            with self.state.thread_lock:
                snap_count = len(self.state.history_order_book)
                snaps = list(
                    self.state.history_order_book
                )  # copy under lock, release before heavy work

            if snap_count < MIN_SNAPSHOTS:
                logging.info(
                    "Historical: only %d snapshots available, need ≥%d — skipping.",
                    snap_count,
                    MIN_SNAPSHOTS,
                )
                continue

            iteration += 1
            # snaps is a plain list — lock is already released, safe to convert.
            # Bid-VWAP
            bids = np.array([s["best_bid"] for s in snaps])
            vols_bids = np.array([s["volume_best_bid"] for s in snaps])
            bid_vwap = volume_weighted_average_price(bids, vols_bids)
            # Ask-VWAP
            asks = np.array([s["best_ask"] for s in snaps])
            vols_asks = np.array([s["volume_best_ask"] for s in snaps])
            ask_vwap = volume_weighted_average_price(asks, vols_asks)
            with self._vwap_lock:
                self._bid_vwap = bid_vwap
                self._ask_vwap = ask_vwap
            logging.info(
                "Historical Analysis #%d — %d snapshots | bid_vwap=%.2f | ask_vwap=%.2f",
                iteration,
                snap_count,
                bid_vwap,
                ask_vwap,
            )

            # ── HMM regime update — 5-minute candle-boundary pulse ────────────
            # Only fire when UTC time has crossed a new 5-minute candle boundary
            # since the last HMM update.  This prevents redundant API calls and
            # Viterbi passes when the kline data has not yet changed.
            now = int(time.time())
            current_5m_boundary = now - (now % 300)  # floor to nearest 5m candle

            if current_5m_boundary > _last_hmm_boundary:
                _last_hmm_boundary = current_5m_boundary
                hmm_iteration += 1

                # Fetch latest 10 h of 5-minute klines (~120 bars).
                self.regime_director.get_klines_data()

                if hmm_iteration % hmm_refit_every == 0:
                    self.regime_director.select_hmm_model()  # full BIC re-fit
                    logging.info(
                        "Historical #%d HMM pulse #%d — full re-fit completed.",
                        iteration,
                        hmm_iteration,
                    )
                else:
                    self.regime_director.predict_current_regime()  # cheap Viterbi

                # Compute trend-pause flag on the same 5-min frame the HMM uses.
                # Mirrors backtest/signals.py exactly: pause new entries when
                # TREND_CONSECUTIVE_BARS same-direction closes are detected,
                # then keep paused for TREND_COOLDOWN_BARS extra bars.  Only the
                # latest bar matters — it decides what the next tick will do.
                try:
                    _tp_series = add_trend_pause_flag(
                        self.regime_director.klines_df,
                        n=TREND_CONSECUTIVE_BARS,
                        cooldown=TREND_COOLDOWN_BARS,
                    )
                    _current_trend_paused = bool(_tp_series.iloc[-1])
                except Exception as exc:
                    # If the macro frame is malformed (e.g. empty after dropna),
                    # default to False so the live system trades normally rather
                    # than freezing on a transient data issue.
                    logging.warning(
                        "Historical #%d HMM pulse #%d — trend-pause computation "
                        "failed (%s); defaulting to False.",
                        iteration,
                        hmm_iteration,
                        exc,
                    )
                    _current_trend_paused = False

                with self._regime_lock:
                    self.regime_director.assign_regime_labels()  # write label
                    self._trend_paused = _current_trend_paused  # publish flag
                logging.info(
                    "Historical #%d HMM pulse #%d — regime → '%s' | trend_paused=%s",
                    iteration,
                    hmm_iteration,
                    self.regime_director.regime_label,
                    self._trend_paused,
                )
            else:
                logging.debug(
                    "Historical #%d — no new 5m candle boundary (boundary=%d, "
                    "last=%d); HMM pulse skipped.",
                    iteration,
                    current_5m_boundary,
                    _last_hmm_boundary,
                )

        logging.info(
            "Historical analysis loop stopped after %d iteration(s) "
            "(%d HMM pulse(s) fired).",
            iteration,
            hmm_iteration,
        )

    @property
    def n_stop_loss_fires(self):
        return self._n_stop_loss_fires
