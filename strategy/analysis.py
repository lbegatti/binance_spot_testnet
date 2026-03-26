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
)
from execution.order_executor import OrderExecutor
from strategy.indicators import volume_weighted_average_price


class AnalysisEngine:
    """
    Strategy engine that consumes shared ``OrderBookState`` to run both
    high-frequency and historical analyses in separate background threads.

    The class is intentionally decoupled from ``MessageHandler``: the two
    communicate exclusively through the injected ``OrderBookState`` instance,
    which guarantees that both threads always operate on the same data and
    the same lock.

    Both loops respect a shared ``threading.Event`` (``stop_event``) so that
    ``websocket_spot_main.py`` can terminate the session cleanly after the
    user-defined duration has elapsed.

    Attributes:
        state (OrderBookState): Shared order book state, injected at construction.
        stop_event (threading.Event): Shared event; when set, both loops exit
            at their next scheduled wake-up.
    """

    def __init__(
        self,
        state: OrderBookState,
        stop_event: threading.Event,
        executor: OrderExecutor,
    ):
        """
        Args:
            state (OrderBookState): The shared order book state instance that
                provides ``local_book``, ``history_order_book``, and
                ``thread_lock``.
            stop_event (threading.Event): Set by the session driver
                (``websocket_spot_main.py``) when the session duration has
                elapsed.  Both analysis loops check this event on every
                iteration and exit gracefully when it is set.
        """
        self.state = state
        self.stop_event = stop_event
        self.order_executor = executor
        self._vwap_lock = threading.Lock()
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
        Periodically inspect the live order book and apply a high-frequency
        trading strategy.

        Runs every 5 seconds until ``stop_event`` is set.  On each wake-up it
        acquires ``state.thread_lock``, takes a local copy of the current bids
        and asks, then releases the lock before executing any heavier strategy
        logic — keeping the critical section as short as possible.

        Logs a message and skips the iteration if no bids are present yet
        (e.g. the WebSocket snapshot has not arrived yet).

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

            # TODO: review momentum-check logic below.
            # After the first historical_analysis iteration (~5 min), _bid_vwap
            # and _ask_vwap are populated.  They act as a confirmation filter:
            #   BUY  → execute only if micro_price > ask_vwap (upward momentum:
            #          current price exceeds the historical avg cost to buy).
            #   SELL → execute only if micro_price < bid_vwap (downward momentum:
            #          current price is below the historical avg bid).
            # While VWAPs are still None (first ~5 min) the filter is transparent
            # and orders execute based on the score alone.
            if best_buy:
                micro_price = best_buy[5]  # index 5 of the tuple
                if ask_vwap is not None and micro_price <= ask_vwap:
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
                if bid_vwap is not None and micro_price >= bid_vwap:
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
        Periodically analyze the rolling window of order book snapshots stored
        in ``state.history_order_book`` to identify longer-term patterns.

        Runs every 10 minutes until ``stop_event`` is set.  If fewer than 100
        snapshots have accumulated the iteration is skipped and a log message
        is emitted so the operator knows the engine is still warming up.

        Intended use-cases include backtesting sub-strategies, computing
        spread distributions, or detecting regime changes over the last 10
        minutes of data.

        Exits cleanly when ``stop_event`` is set, logging how many iterations
        were completed.
        """
        iteration = 0

        logging.info(
            "Historical analysis loop started (interval: %ds / %.0f min).",
            HIST_INTERVAL,
            HIST_INTERVAL / 60,
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

        logging.info(
            "Historical analysis loop stopped after %d iteration(s).", iteration
        )
