"""
strategy/book_utils.py
----------------------
Shared utility functions for order-book level construction, candidate
filtering, and opportunity scoring.

These functions were extracted from ``AnalysisEngine`` (``strategy/analysis.py``)
so that both the **live trading pipeline** and the **offline backtesting
pipeline** (``backtest/signals.py``) can consume them without importing the
full ``AnalysisEngine`` class — avoiding circular dependencies and
encapsulation violations.

``AnalysisEngine`` retains thin private wrappers (``_build_levels``,
``_collect_candidates``, ``_select_best_opportunity``) that delegate to the
functions defined here, preserving all existing internal call sites unchanged.

Public API:
    - :func:`build_levels`              — sort + compute per-level metrics
    - :func:`collect_candidates`        — filter + classify buy / sell signals
    - :func:`select_best_opportunity`   — score + pick the best candidate
"""

import logging

import numpy as np

from config_parameters import N_LEVELS


def build_levels(snaps_bids: dict, snaps_asks: dict, n: int = N_LEVELS) -> tuple:
    """
    Construct order book levels from raw bid/ask dictionaries.

    Sorts bids descending and asks ascending, then computes per-level metrics
    used by both the live ``AnalysisEngine`` and the backtesting pipeline.
    Extracting this logic here keeps it reusable without requiring callers to
    import ``AnalysisEngine``.

    Args:
        snaps_bids (dict): Mapping of price (str) → quantity (str/float) for
            the bid side.
        snaps_asks (dict): Mapping of price (str) → quantity (str/float) for
            the ask side.
        n (int): Number of price levels to retain on each side.
            Defaults to ``N_LEVELS`` (50).

    Returns:
        tuple:
            - **levels** (list[tuple]): One entry per level,
              each ``(total_depth, mid_price, micro_price, obi, bq, aq)``.
            - **median_depth** (float): Median of ``total_depth`` across all
              retained levels.
            - **level_0_depth** (float): ``total_depth`` at the best bid/ask
              level (level 0).
    """
    sorted_bids = sorted(snaps_bids.items(), key=lambda x: float(x[0]), reverse=True)[
        :n
    ]
    sorted_asks = sorted(snaps_asks.items(), key=lambda x: float(x[0]), reverse=False)[
        :n
    ]

    levels = []
    for (bp, bq), (ap, aq) in zip(sorted_bids, sorted_asks):
        bp, bq, ap, aq = float(bp), float(bq), float(ap), float(aq)
        total_depth = bq + aq
        mid_price = (bp + ap) / 2
        micro_price = (bp * aq + ap * bq) / total_depth
        obi = (bq - aq) / (bq + aq)
        levels.append((total_depth, mid_price, micro_price, obi, bq, aq))

    all_depths = [lv[0] for lv in levels]
    median_depth = float(np.median(all_depths))
    level_0_depth = all_depths[0]

    return levels, median_depth, level_0_depth


def collect_candidates(
    levels: list, median_depth: float, level_0_depth: float
) -> tuple:
    """
    Identify potential trade opportunities from pre-computed order book levels.

    Iterates over every level (skipping level 0, the best bid/ask) and applies
    two liquidity filters before classifying each level as a buy or sell signal:

    - **Thin-book filter**: the level's total depth must be ≥ ``median_depth``
      (i.e. the level is *not* thin).
    - **Depth-at-level-0 filter**: total depth must be ≥ 50 % of
      ``level_0_depth`` (sufficient liquidity relative to the best level).

    A level passes as a **buy candidate** when ``micro_price > mid_price``
    (aggressive buy pressure) and as a **sell candidate** when
    ``micro_price < mid_price`` (aggressive sell pressure).

    Extracting this logic here keeps it reusable by both the live
    ``AnalysisEngine`` and the backtesting pipeline without requiring callers
    to import ``AnalysisEngine``.

    Args:
        levels (list[tuple]): Output of :func:`build_levels` — each entry is
            ``(total_depth, mid_price, micro_price, obi, bq, aq)``.
        median_depth (float): Median total depth across all levels, used as
            the thin-book threshold.
        level_0_depth (float): Total depth at the best bid/ask level (level 0),
            used as the reference for the 50 % depth filter.

    Returns:
        tuple:
            - **buy_candidates** (list[tuple]): Entries of
              ``(level_idx, delta, total_depth, obi, micro_price, bq, aq)``
              where ``delta = micro_price - mid_price``.
            - **sell_candidates** (list[tuple]): Same shape as
              ``buy_candidates`` but ``delta = mid_price - micro_price``.
    """
    buy_candidates = []
    sell_candidates = []

    for i, (total_depth, mid_price, micro_price, obi, bq, aq) in enumerate(levels):
        if i == 0:
            continue
        is_thin = total_depth < median_depth
        depth_ok = total_depth >= 0.5 * level_0_depth
        # obi > 0.0 → buy wall heavier than sell side; price may be pushed up.
        # obi < 0.0 → sellers crowding the book; suggests a downward move.
        # obi = 0   → balanced book.
        if not is_thin and depth_ok:
            if micro_price > mid_price:  # buy signal
                delta = micro_price - mid_price
                buy_candidates.append((i, delta, total_depth, obi, micro_price, bq, aq))
            elif micro_price < mid_price:  # sell signal
                delta = mid_price - micro_price
                sell_candidates.append(
                    (i, delta, total_depth, obi, micro_price, bq, aq)
                )

    return buy_candidates, sell_candidates


def select_best_opportunity(
    candidates: list, strategy_name: str, iteration: int
) -> tuple | None:
    """
    Score identified candidates and pick the best one for potential execution.

    Each candidate is scored by a weighted combination of normalised depth and
    normalised micro-mid delta:

        ``score = 0.70 × norm_depth + 0.30 × norm_delta``

    Extracting this logic here keeps it reusable by both the live
    ``AnalysisEngine`` and the backtesting pipeline without requiring callers
    to import ``AnalysisEngine``.

    Args:
        candidates (list[tuple]): Output of :func:`collect_candidates` — each
            entry is ``(level_idx, delta, total_depth, obi, micro_price, bq, aq)``.
        strategy_name (str): ``"buy"`` or ``"sell"`` — used only for logging.
        iteration (int): Current loop iteration number — used only for logging.

    Returns:
        tuple | None:
            ``(level_idx, score | None, delta, total_depth, obi, micro_price, bq, aq)``
            for the best candidate.  ``score`` is ``None`` when there is only
            one candidate (no normalisation possible).  Returns ``None`` when
            ``candidates`` is empty.
    """
    if not candidates:
        logging.info("HFT #%d [%s] — no opportunities found.", iteration, strategy_name)
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
