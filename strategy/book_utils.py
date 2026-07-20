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

from config_parameters import (
    N_LEVELS,
    CANDIDATE_MEDIAN_FRAC,
    CANDIDATE_DEPTH_FRAC,
    CANDIDATE_FILTER_DEBUG,
    CANDIDATE_FILTER_DEBUG_EVERY,
)

log = logging.getLogger(__name__)

# Cumulative per-filter reject counters for the collect_candidates() diagnostic
# (enabled by CANDIDATE_FILTER_DEBUG).  Accumulated across calls within one process
# so a live session's log shows which gate starves candidates most.  Never read or
# mutated when the flag is off — zero overhead on the backtest hot path.
_CAND_FILTER_STATS: dict[str, int] = {
    "calls": 0,  # collect_candidates() invocations
    "levels": 0,  # total candidate levels examined (level 0 excluded)
    "rej_thin": 0,  # failed the thin-book filter (depth < FRAC × median_depth)
    "rej_depth": 0,  # failed the relative-depth filter (depth < FRAC × level_0_depth)
    "rej_both": 0,  # failed BOTH liquidity filters
    "passed_liq": 0,  # passed both liquidity filters
    "rej_dir": 0,  # passed liquidity but had NO imbalance (micro_price == mid_price)
    "buys": 0,  # emitted buy candidates
    "sells": 0,  # emitted sell candidates
}


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

    # Guard: both sides must have the same length before zipping into arrays.
    # The synthetic book (and occasionally a live snapshot) can return fewer
    # than n levels on one side — truncate to the shorter side so numpy
    # element-wise ops don't raise a shape-mismatch ValueError.
    depth = min(len(sorted_bids), len(sorted_asks))
    sorted_bids = sorted_bids[:depth]
    sorted_asks = sorted_asks[:depth]

    if depth == 0:
        return [], 0.0, 0.0

    # --- vectorised level computation -----------------------------------------
    # Convert the sorted pairs to numpy arrays in one pass so all per-level
    # metrics (total_depth, mid_price, micro_price, obi) are computed with
    # element-wise numpy ops instead of a Python for-loop.
    bp = np.array([float(p) for p, _ in sorted_bids])
    bq = np.array([float(q) for _, q in sorted_bids])
    ap = np.array([float(p) for p, _ in sorted_asks])
    aq = np.array([float(q) for _, q in sorted_asks])

    total_depth_arr = bq + aq

    # Guard: drop any level where both sides have zero quantity (e.g. a
    # zero-volume candle in the synthetic book sets base_volume=0, making
    # every qty=0).  Keeping such levels would cause division by zero in
    # micro_price and obi, silently producing NaN/inf downstream.
    positive = total_depth_arr > 0
    if not positive.any():
        return [], 0.0, 0.0
    bp, bq, ap, aq = bp[positive], bq[positive], ap[positive], aq[positive]
    total_depth_arr = total_depth_arr[positive]

    mid_price_arr = (bp + ap) / 2.0
    micro_price_arr = (bp * aq + ap * bq) / total_depth_arr
    obi_arr = (bq - aq) / total_depth_arr

    # Re-pack into the same list-of-tuples format expected by callers so the
    # public contract of build_levels() is unchanged.
    levels = list(
        zip(
            total_depth_arr.tolist(),
            mid_price_arr.tolist(),
            micro_price_arr.tolist(),
            obi_arr.tolist(),
            bq.tolist(),
            aq.tolist(),
        )
    )

    median_depth = float(np.median(total_depth_arr))
    level_0_depth = float(total_depth_arr[0])

    return levels, median_depth, level_0_depth


def collect_candidates(
    levels: list, median_depth: float, level_0_depth: float
) -> tuple:
    """
    Identify potential trade opportunities from pre-computed order book levels.

    Iterates over every level (skipping level 0, the best bid/ask) and applies
    two liquidity filters before classifying each level as a buy or sell signal:

    - **Thin-book filter**: the level's total depth must be ≥
      ``CANDIDATE_MEDIAN_FRAC × median_depth`` (i.e. the level is *not* thin).
    - **Depth-at-level-0 filter**: total depth must be ≥
      ``CANDIDATE_DEPTH_FRAC × level_0_depth`` (sufficient liquidity relative to
      the best level).

    Both fractions are configurable in ``config_parameters.py`` — lower values
    make the filters more permissive (more candidates survive).

    A level passes as a **buy candidate** when ``micro_price < mid_price``
    (ask-side dominance — price dipping below mid, confirming a buy-the-dip
    signal) and as a **sell candidate** when ``micro_price > mid_price``
    (bid-side dominance — price elevated above mid, confirming
    sell-into-strength).

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
    # Guard: need at least 2 levels (level 0 is always skipped).
    if len(levels) < 2:
        return [], []

    # --- vectorised candidate filtering ---------------------------------------
    # Slice off level 0 (best bid/ask — never a candidate) and lay the
    # remaining rows into a 2-D numpy array so all filter masks and delta
    # computations run as element-wise numpy ops instead of a Python for-loop.
    #
    # Column layout mirrors the tuple produced by build_levels():
    #   0: total_depth  1: mid_price  2: micro_price  3: obi  4: bq  5: aq
    arr = np.array(levels[1:])  # shape (n_levels-1, 6)
    total_depths = arr[:, 0]
    mid_prices = arr[:, 1]
    micro_prices = arr[:, 2]
    obis = arr[:, 3]
    bqs = arr[:, 4]
    aqs = arr[:, 5]

    # Level indices in the *original* levels list (1-based because level 0
    # was dropped).
    level_indices = np.arange(1, len(levels))

    # obi > 0.0 → bid wall heavier than ask side → micro_price pushed ABOVE mid → SELL signal.
    # obi < 0.0 → ask wall heavier than bid side → micro_price pulled BELOW mid → BUY signal.
    # obi = 0   → balanced book → micro_price ≈ mid_price → no directional candidate.
    not_thin = total_depths >= CANDIDATE_MEDIAN_FRAC * median_depth  # thin-book filter
    depth_ok = (
        total_depths >= CANDIDATE_DEPTH_FRAC * level_0_depth
    )  # relative depth filter
    valid = not_thin & depth_ok

    buy_mask = valid & (
        micro_prices < mid_prices
    )  # mean-reversion BUY: price dipped below mid
    sell_mask = valid & (
        micro_prices > mid_prices
    )  # mean-reversion SELL: price elevated above mid

    buy_deltas = micro_prices - mid_prices
    sell_deltas = mid_prices - micro_prices

    # Helper: materialise a masked selection back to a list of tuples.
    # int() / float() casts strip numpy scalar wrappers so downstream code
    # (e.g. logging %d / %f formatters) behaves identically to before.
    def _mask_to_tuples(mask: np.ndarray, deltas: np.ndarray) -> list:
        idxs = level_indices[mask].tolist()
        ds = deltas[mask].tolist()
        tds = total_depths[mask].tolist()
        obs = obis[mask].tolist()
        mps = micro_prices[mask].tolist()
        bq_ = bqs[mask].tolist()
        aq_ = aqs[mask].tolist()
        return list(zip(idxs, ds, tds, obs, mps, bq_, aq_))

    buy_candidates = _mask_to_tuples(buy_mask, buy_deltas)
    sell_candidates = _mask_to_tuples(sell_mask, sell_deltas)

    # --- diagnostic: per-filter reject accounting (read-only) -----------------
    # Attributes each rejected level to the gate(s) it failed so a live session's
    # log reveals whether the thin-book / relative-depth filters or the "no
    # imbalance" condition (micro_price == mid_price) is what starves candidates.
    # Guarded by the flag so the backtest hot path (~130k calls) is unaffected.
    if CANDIDATE_FILTER_DEBUG:
        n_levels = arr.shape[0]
        # A level with no imbalance (micro == mid) passes liquidity but yields no
        # directional candidate — attribute it to the "direction" gate.
        no_dir = valid & (micro_prices == mid_prices)
        s = _CAND_FILTER_STATS
        s["calls"] += 1
        s["levels"] += int(n_levels)
        s["rej_thin"] += int((~not_thin).sum())
        s["rej_depth"] += int((~depth_ok).sum())
        s["rej_both"] += int((~not_thin & ~depth_ok).sum())
        s["passed_liq"] += int(valid.sum())
        s["rej_dir"] += int(no_dir.sum())
        s["buys"] += len(buy_candidates)
        s["sells"] += len(sell_candidates)

        if s["calls"] % CANDIDATE_FILTER_DEBUG_EVERY == 0 and s["levels"] > 0:
            lv = s["levels"]
            log.info(
                "[cand-filter] calls=%d levels=%d (cumulative) | rejected of all "
                "levels: thin=%.1f%% depth=%.1f%% both=%.1f%% no-imbalance=%.1f%% | "
                "passed-liquidity=%.1f%% | candidates: buy=%d sell=%d",
                s["calls"],
                lv,
                100.0 * s["rej_thin"] / lv,
                100.0 * s["rej_depth"] / lv,
                100.0 * s["rej_both"] / lv,
                100.0 * s["rej_dir"] / lv,
                100.0 * s["passed_liq"] / lv,
                s["buys"],
                s["sells"],
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
            ``(level_idx, score, delta, total_depth, obi, micro_price, bq, aq)``
            for the best candidate.  ``score`` is ``1.0`` when there is only
            one candidate (trivially the best).  Returns ``None`` when
            ``candidates`` is empty.
    """
    if not candidates:
        logging.info("HFT #%d [%s] — no opportunities found.", iteration, strategy_name)
        return None
    if len(candidates) == 1:
        level_idx, delta, depth, obi, micro_price, bq, aq = candidates[0]
        # score is set to 1.0 for a single candidate — there is nothing to
        # normalise against, so the candidate trivially achieves the maximum
        # possible score.  Using 1.0 (instead of None) keeps the tuple type
        # consistent and avoids None-guard issues in callers.
        score = 1.0
        logging.info(
            "HFT #%d [%s] — single candidate at level %d | score=%.4f | delta=%.6f | depth=%.4f "
            "| order_imbalance=%.3f | micro price = %.3f",
            iteration,
            strategy_name,
            level_idx,
            score,
            delta,
            depth,
            obi,
            micro_price,
        )
        return level_idx, score, delta, depth, obi, micro_price, bq, aq

    # --- vectorised scoring ---------------------------------------------------
    # Extract the two scoring components into numpy arrays and compute the
    # weighted score in one shot rather than looping over each candidate.
    #   score = 0.70 × norm_depth + 0.30 × norm_delta
    arr_d = np.array([c[2] for c in candidates])  # total_depth column
    arr_k = np.array([c[1] for c in candidates])  # delta column

    max_depth = arr_d.max()
    max_delta = arr_k.max()

    # Guard: max_depth == 0 or max_delta == 0 should never occur in normal
    # operation (collect_candidates filters by depth and delta > 0), but
    # defend against it to avoid a silent division-by-zero.
    if max_depth == 0 or max_delta == 0:
        level_idx, delta, depth, obi, micro_price, bq, aq = candidates[0]
        logging.warning(
            "HFT #%d [%s] — degenerate candidates (max_depth=%.6f, max_delta=%.6f); "
            "falling back to first candidate at level %d.",
            iteration,
            strategy_name,
            max_depth,
            max_delta,
            level_idx,
        )
        return level_idx, 1.0, delta, depth, obi, micro_price, bq, aq

    scores = (arr_d / max_depth) * 0.70 + (arr_k / max_delta) * 0.30
    best = int(np.argmax(scores))

    level_idx, delta, depth, obi, micro_price, bq, aq = candidates[best]
    score = float(scores[best])

    logging.info(
        "HFT #%d [%s] — level %d | score=%.4f | delta=%.6f | depth=%.4f | order_imbalance = %.3f "
        "| micro price = %.3f",
        iteration,
        strategy_name,
        level_idx,
        score,
        delta,
        depth,
        obi,
        micro_price,
    )
    return level_idx, score, delta, depth, obi, micro_price, bq, aq
