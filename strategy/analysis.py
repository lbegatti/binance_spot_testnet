from core.order_book_state import OrderBookState
import threading
import logging
import numpy as np

from config_parameters import (
    HFT_INTERVAL,
    HIST_INTERVAL,
    MIN_SNAPSHOTS,
    N_LEVELS,
    CCY,
    CRYPTOCCY,
    HMM_REFIT_INTERVAL,
)
from execution.order_executor import OrderExecutor
from strategy.indicators import volume_weighted_average_price
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
    """

    def __init__(
        self,
        state: OrderBookState,
        stop_event: threading.Event,
        executor: OrderExecutor,
        regime_director: RegimeDirector,
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
        """
        self.regime_director = regime_director
        self.state = state
        self.stop_event = stop_event
        self.order_executor = executor
        self._vwap_lock = threading.Lock()
        self._regime_lock = threading.Lock()
        self._bid_vwap: float | None = None
        self._ask_vwap: float | None = None

    @staticmethod
    def _build_levels(snaps_bids: dict, snaps_asks: dict, n: int = N_LEVELS) -> tuple:
        """
        Helper method to construct order book levels for HFT analysis.
        Sorting must happen before any metric is computed.

        :param snaps_bids: dictionary of bids streamed via websocket.
        :param snaps_asks: dictionary of asks streamed via websocket.
        :param n: integer signaling the depth of the order book levels to be used in the HFT strategy.
                  Defaults to N_LEVELS (50).
        :return: levels -> list of (total_depth, mid_price, micro_price, obi, bq, aq).
                 median_depth -> median total depth across the levels.
                 level_0_depth -> total depth at the best bid/ask level.
        """
        sorted_bids = sorted(
            snaps_bids.items(), key=lambda x: float(x[0]), reverse=True
        )[:n]
        sorted_asks = sorted(
            snaps_asks.items(), key=lambda x: float(x[0]), reverse=False
        )[:n]
        levels = []
        for (bp, bq), (ap, aq) in zip(sorted_bids, sorted_asks):
            bp, bq, ap, aq = float(bp), float(bq), float(ap), float(aq)
            total_depth = bq + aq
            mid_price = (bp + ap) / 2
            micro_price = (bp * aq + ap * bq) / total_depth
            obi = (bq - aq) / (bq + aq)
            levels.append((total_depth, mid_price, micro_price, obi, bq, aq))

        all_depths = [lv[0] for lv in levels]
        median_depth = float(
            np.median(all_depths)
        )  # mirrors np.median() in indicators.py
        level_0_depth = all_depths[0]

        return levels, median_depth, level_0_depth

    @staticmethod
    def _collect_candidates(
        levels: list, median_depth: float, level_0_depth: float
    ) -> tuple:
        """
        Helper method to identify potential HFT opportunities based on the
        computed order book levels and depth metrics.

        :param levels: list of tuples (total_depth, mid_price, micro_price, obi, bq, aq) for the top N levels.
        :param median_depth: median total depth across the levels, used to identify thin order book conditions.
        :param level_0_depth: total depth at the best bid/ask level, used to assess liquidity.
        :return: candidates -> list of dictionaries with opportunity indicators for each level.
        """
        buy_candidates = []
        sell_candidates = []

        for i, (total_depth, mid_price, micro_price, obi, bq, aq) in enumerate(levels):
            if i == 0:
                continue
            is_thin = total_depth < median_depth
            depth_ok = total_depth >= 0.5 * level_0_depth
            # obi > 0.0  # "Buy Wall" is heavier than the sell side, so the price might be pushed up.
            # obi < 0.0  # Indicates that sellers are crowding the book, suggesting a downward move.
            # obi = 0    # balance.
            if not is_thin and depth_ok:
                if micro_price > mid_price:  # buy signal
                    delta = micro_price - mid_price
                    buy_candidates.append(
                        (i, delta, total_depth, obi, micro_price, bq, aq)
                    )
                elif micro_price < mid_price:  # sell signal
                    delta = mid_price - micro_price
                    sell_candidates.append(
                        (i, delta, total_depth, obi, micro_price, bq, aq)
                    )

        return buy_candidates, sell_candidates

    @staticmethod
    def _select_best_opportunity(
        candidates: list, strategy_name: str, iteration: int
    ) -> tuple | None:
        """
        Helper method to score the identified candidates based on a weighted combination of depth and micro-mid delta,
        and pick the best one for potential execution.

        :param candidates: list of tuples (level_idx, delta, total_depth, obi, micro_price, bq, aq) for the identified opportunities.
        :param strategy_name: string indicating the strategy type ("buy" or "sell") for logging purposes.
        :param iteration: integer indicating the current iteration of the HFT loop for logging purposes.
        :return: tuple of (level_idx, score, delta, total_depth, obi, micro_price, bq, aq) for the best candidate or None if no
                 candidates are available.
        """
        if not candidates:
            logging.info(
                "HFT #%d [%s] — no opportunities found.", iteration, strategy_name
            )
            return None
        if len(candidates) == 1:
            level_idx, delta, depth, obi, micro_price, bq, aq = candidates[0]
            logging.info(
                "HFT #%d [%s] — single candidate at level %d | delta=%.6f | depth=%.4f | order_imbalance=%.3f "
                "| micro price = %.3f",
                iteration,
                strategy_name,
                level_idx,
                delta,
                depth,
                obi,
                micro_price,
            )
            return level_idx, None, delta, depth, obi, micro_price, bq, aq
        max_depth = max(c[2] for c in candidates)
        max_delta = max(c[1] for c in candidates)
        scored = []
        for level_idx, delta, depth, obi, micro_price, bq, aq in candidates:
            norm_depth = depth / max_depth
            norm_delta = delta / max_delta
            score = (norm_depth * 0.70) + (norm_delta * 0.30)
            scored.append((level_idx, score, delta, depth, obi, micro_price, bq, aq))
        trade_opportunity = max(scored, key=lambda x: x[1])
        logging.info(
            "HFT #%d [%s] — level %d | score=%.4f | delta=%.6f | depth=%.4f | order_imbalance = %.3f "
            "| micro price = %.3f",
            iteration,
            strategy_name,
            trade_opportunity[0],
            trade_opportunity[1],
            trade_opportunity[2],
            trade_opportunity[3],
            trade_opportunity[4],
            trade_opportunity[5],
        )
        return trade_opportunity

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
        6. **VWAP filter** (momentum confirmation) — reads ``_bid_vwap`` /
           ``_ask_vwap`` under ``_vwap_lock``:
           - BUY: skip if ``ask_vwap`` is set **and** ``micro_price ≤ ask_vwap``.
           - SELL: skip if ``bid_vwap`` is set **and** ``micro_price ≥ bid_vwap``.
           Both are ``None`` for the first ~1 min; filter is transparent until
           ``historical_analysis`` publishes the first VWAP.
        7. **Regime filter** (HMM gate) — reads
           ``regime_director.regime_label`` under ``_regime_lock``:
           - BUY:  skip if regime is ``"trending_down"`` or ``"high_volatility"``.
           - SELL: skip if regime is ``"trending_up"``  or ``"high_volatility"``.
           ``None`` label (before first ``historical_analysis`` run) is treated
           as transparent — all orders are allowed through.
        8. **Order execution** — delegates to
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
            with self._vwap_lock:
                bid_vwap = self._bid_vwap
                ask_vwap = self._ask_vwap

            with self._regime_lock:
                current_regime = (
                    self.regime_director.regime_label
                )  # reads the label assigned in the historical_analysis thread.

            # After the first historical_analysis iteration (~1 min), _bid_vwap
            # and _ask_vwap are populated.  They act as a confirmation filter:
            #   BUY  → execute only if micro_price > ask_vwap (upward momentum:
            #          current price exceeds the historical avg cost to buy).
            #   SELL → execute only if micro_price < bid_vwap (downward momentum:
            #          current price is below the historical avg bid).
            # While VWAPs are still None (first ~1 min) the filter is transparent
            # and orders execute based on the score alone.
            if best_buy:
                micro_price = best_buy[5]  # index 5 of the tuple
                if (
                    current_regime == "trending_down"
                    or current_regime == "high_volatility"
                ):
                    logging.info(
                        "HFT #%d [buy] — skipped: regime is '%s'",
                        iteration,
                        current_regime,
                    )
                elif ask_vwap is not None and micro_price <= ask_vwap:
                    logging.info(
                        "HFT #%d [buy] — skipped: micro_price %.2f ≤ ask_vwap %.2f",
                        iteration,
                        micro_price,
                        ask_vwap,
                    )
                else:
                    self.order_executor.execute("BUY", best_buy)
            if best_sell:
                micro_price = best_sell[5]
                if (
                    current_regime == "trending_up"
                    or current_regime == "high_volatility"
                ):
                    logging.info(
                        "HFT #%d [sell] — skipped: regime is '%s'",
                        iteration,
                        current_regime,
                    )
                elif bid_vwap is not None and micro_price >= bid_vwap:
                    logging.info(
                        "HFT #%d [sell] — skipped: micro_price %.2f ≥ bid_vwap %.2f",
                        iteration,
                        micro_price,
                        bid_vwap,
                    )
                else:
                    self.order_executor.execute("SELL", best_sell)

            self.stop_event.wait(HFT_INTERVAL)  # sleep AFTER work, not before

        logging.info("HFT analysis loop stopped after %d iteration(s).", iteration)

    def historical_analysis(self):
        """
        Periodically analyse the rolling order book snapshot window to compute
        VWAPs and refresh the HMM market-regime label.

        Runs every ``HIST_INTERVAL`` seconds (default 60 s / 1 min) until
        ``stop_event`` is set.  If fewer than ``MIN_SNAPSHOTS`` (100) snapshots
        have accumulated the iteration is skipped and a warm-up log message is
        emitted.

        **On each iteration:**

        1. **VWAP computation** — copies ``state.history_order_book`` under
           ``thread_lock``, converts to numpy arrays, and computes:

           .. code-block:: text

               bid_vwap = Σ(best_bid_i × volume_best_bid_i) / Σ(volume_best_bid_i)
               ask_vwap = Σ(best_ask_i × volume_best_ask_i) / Σ(volume_best_ask_i)

           Both VWAPs are published under ``_vwap_lock`` so that
           ``low_latency_analysis`` can read them safely on the next iteration.

        2. **HMM regime update** — two-speed update to avoid re-training on
           nearly identical data every minute:

           * **Every ``HIST_INTERVAL`` (60 s)** — cheap path:

             - ``regime_director.get_klines_data()`` — fetches the latest
               ``HMM_LOOKBACK`` (2 h) of ``HMM_INTERVAL`` (1 m) klines.
             - ``regime_director.predict_current_regime()`` — runs the Viterbi
               algorithm on the already-fitted model (O(n × k)), extracts the
               state of the last candle.  No re-training.

           * **Every ``HMM_REFIT_INTERVAL`` (300 s, i.e. every 5th iteration)** — full path:

             - ``regime_director.get_klines_data()`` — same as above.
             - ``regime_director.select_hmm_model()`` — fits all candidate
               ``GaussianHMM`` models (2 … ``HMM_MAX_REGIMES``), selects best
               by BIC, and replaces the stored model.
               O(n × k × ``HMM_N_ITERATIONS``) per candidate — expensive.

           Then, inside ``_regime_lock`` on **every** iteration:

           - ``regime_director.assign_regime_labels()`` — maps the current
             state integer to a human-readable label using cross-state mean/std
             thresholds on ``model.means_``.  Writes
             ``regime_director.regime_label``.

           The slow network + CPU work (kline download + model fitting) runs
           **outside** ``_regime_lock``; only the fast label assignment is
           performed inside the lock to minimise contention with
           ``low_latency_analysis``.

        Exits cleanly when ``stop_event`` is set, logging how many iterations
        were completed.
        """
        iteration = 0
        refit_every = max(
            1, HMM_REFIT_INTERVAL // HIST_INTERVAL
        )  # constant for the session

        logging.info(
            "Historical analysis loop started (interval: %ds / %.0f min, refit every %d iteration(s)).",
            HIST_INTERVAL,
            HIST_INTERVAL / 60,
            refit_every,
        )
        while not self.stop_event.is_set():
            self.stop_event.wait(HIST_INTERVAL)  # interruptible sleep
            if self.stop_event.is_set():
                break

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
            # snaps is a plain list — lock is already released, safe to convert to numpy arrays or a pandas DataFrame.
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

            self.regime_director.get_klines_data()  # fetch latest 2h of 1m candles

            # Full re-fit every HMM_REFIT_INTERVAL seconds (default 5 min);
            # cheap Viterbi inference on every other iteration.
            if iteration % refit_every == 0:
                self.regime_director.select_hmm_model()  # O(n × k × iters) — slow
                logging.info("Historical #%d — full HMM re-fit completed.", iteration)
            else:
                self.regime_director.predict_current_regime()  # O(n × k) — fast

            with self._regime_lock:
                self.regime_director.assign_regime_labels()  # write regime label
            logging.info(
                "Historical #%d — regime updated → '%s'",
                iteration,
                self.regime_director.regime_label,
            )

        logging.info(
            "Historical analysis loop stopped after %d iteration(s).", iteration
        )
