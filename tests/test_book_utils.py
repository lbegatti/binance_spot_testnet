"""Tier B tests for strategy/book_utils.py — level construction, candidate
filtering, and opportunity scoring."""

import pytest

from strategy.book_utils import (
    build_levels,
    collect_candidates,
    select_best_opportunity,
)
from tests.fixtures.fake_order_book import make_book


# ── build_levels ───────────────────────────────────────────────────────────


def test_build_levels_empty_returns_empty():
    """With no bids or asks there is nothing to pair, so build_levels returns
    an empty level list and zero depth metrics."""
    assert build_levels({}, {}) == ([], 0.0, 0.0)


def test_build_levels_two_levels_metrics():
    """For a two-deep book, each level's derived metrics — total depth, mid,
    micro-price, and order-book imbalance — are computed by hand and checked,
    along with the median and level-0 depth summaries."""
    bids, asks = make_book({100.0: 2.0, 99.0: 1.0}, {101.0: 1.0, 102.0: 3.0})
    levels, median_depth, level_0_depth = build_levels(bids, asks)
    assert len(levels) == 2

    # level 0: bp=100 ap=101 bq=2 aq=1
    td0, mid0, micro0, obi0, bq0, aq0 = levels[0]
    assert td0 == 3.0
    assert mid0 == pytest.approx(100.5)
    assert micro0 == pytest.approx((100 * 1 + 101 * 2) / 3)  # (bp*aq + ap*bq)/total
    assert obi0 == pytest.approx((2 - 1) / 3)
    assert (bq0, aq0) == (2.0, 1.0)

    # level 1: bid=99 ask=102 bq=1 aq=3
    td1, mid1, micro1, obi1, _bq1, _aq1 = levels[1]
    assert td1 == 4.0
    assert micro1 == pytest.approx((99 * 3 + 102 * 1) / 4)
    assert obi1 == pytest.approx((1 - 3) / 4)

    assert level_0_depth == 3.0
    assert median_depth == pytest.approx(3.5)  # median([3, 4])


def test_build_levels_truncates_to_shorter_side():
    """Levels are paired bid-to-ask, so the output length is capped by the
    shorter side of the book (here 1 ask level despite 3 bid levels)."""
    bids, asks = make_book({100.0: 1.0, 99.0: 1.0, 98.0: 1.0}, {101.0: 1.0})
    levels, _, _ = build_levels(bids, asks)
    assert len(levels) == 1  # asks side has only 1 level


def test_build_levels_drops_zero_total_depth():
    """A level whose bid and ask quantities are both zero carries no depth
    and is dropped, leaving an empty result."""
    bids, asks = make_book({100.0: 0.0}, {101.0: 0.0})
    assert build_levels(bids, asks) == ([], 0.0, 0.0)


def test_build_levels_respects_n():
    """The optional n argument caps how many levels deep the book is read
    (n=2 here, even though 3 levels are available on each side)."""
    bids, asks = make_book(
        {100.0: 1.0, 99.0: 1.0, 98.0: 1.0}, {101.0: 1.0, 102.0: 1.0, 103.0: 1.0}
    )
    levels, _, _ = build_levels(bids, asks, n=2)
    assert len(levels) == 2


# ── collect_candidates ─────────────────────────────────────────────────────
# Level tuple layout: (total_depth, mid, micro, obi, bq, aq)


def _two_candidate_levels():
    return [
        (10.0, 100.5, 100.6, 0.1, 1.0, 1.0),  # level 0 — always skipped
        (10.0, 100.5, 100.4, -0.1, 2.0, 3.0),  # micro < mid → BUY candidate
        (10.0, 100.5, 100.7, 0.2, 3.0, 2.0),  # micro > mid → SELL candidate
    ]


def test_collect_candidates_needs_two_levels():
    """Level 0 is always skipped (it is the touch), so a book with a single
    level yields no buy or sell candidates."""
    single = [(10.0, 100.5, 100.6, 0.1, 1.0, 1.0)]
    assert collect_candidates(single, 10.0, 10.0) == ([], [])


def test_collect_candidates_classifies_buy_and_sell():
    """A level whose micro-price sits below the mid is a BUY candidate and one
    above the mid is a SELL candidate; the resulting deltas and micro-prices
    are checked against hand math."""
    buys, sells = collect_candidates(
        _two_candidate_levels(), median_depth=10.0, level_0_depth=10.0
    )
    assert len(buys) == 1 and len(sells) == 1
    # candidate tuple: (level_idx, delta, total_depth, obi, micro, bq, aq)
    assert buys[0][0] == 1
    assert buys[0][1] == pytest.approx(100.4 - 100.5)  # buy delta = micro - mid
    assert buys[0][4] == pytest.approx(100.4)
    assert sells[0][0] == 2
    assert sells[0][1] == pytest.approx(100.5 - 100.7)  # sell delta = mid - micro


def test_collect_candidates_thin_filter_excludes():
    """A level thinner than the median depth is filtered out as illiquid, so
    it produces no candidate even though its micro-price would qualify."""
    levels = [
        (10.0, 100.5, 100.6, 0.1, 1.0, 1.0),  # level 0
        (1.0, 100.5, 100.4, -0.1, 0.5, 0.5),  # thin (1.0 < median 10.0) → excluded
    ]
    buys, sells = collect_candidates(levels, median_depth=10.0, level_0_depth=10.0)
    assert buys == [] and sells == []


# ── select_best_opportunity ────────────────────────────────────────────────
# Candidate tuple layout: (level_idx, delta, total_depth, obi, micro, bq, aq)


def test_select_best_opportunity_empty_returns_none():
    """With no candidates to score there is nothing to pick, so the selector
    returns None."""
    assert select_best_opportunity([], "buy", 0) is None


def test_select_best_opportunity_single_scores_one():
    """A lone candidate is trivially the best, scores 1.0, and is returned as
    the 8-tuple with that score inserted in position 1."""
    cand = (1, -0.1, 10.0, 0.0, 100.4, 2.0, 3.0)
    result = select_best_opportunity([cand], "buy", 0)
    assert result is not None
    # score is 1.0 for a lone candidate; returns the 8-tuple with score inserted
    assert result == (1, 1.0, -0.1, 10.0, 0.0, 100.4, 2.0, 3.0)


def test_select_best_opportunity_picks_highest_score():
    """Across multiple candidates the selector returns the one with the
    highest blended score (70% depth, 30% delta), verified by hand here."""
    c0 = (1, 2.0, 10.0, 0.0, 100.0, 1.0, 1.0)
    c1 = (2, 4.0, 20.0, 0.0, 101.0, 1.0, 1.0)
    # scores: c0 = (10/20)*0.7 + (2/4)*0.3 = 0.5 ; c1 = 1.0 → c1 wins
    result = select_best_opportunity([c0, c1], "buy", 0)
    assert result is not None
    assert result[0] == 2
    assert result[1] == pytest.approx(1.0)


def test_select_best_opportunity_degenerate_zero_delta_falls_back_to_first():
    """When every candidate has zero delta the score is undefined (max_delta
    == 0), so the selector falls back to the first candidate with score 1.0
    instead of dividing by zero."""
    c0 = (1, 0.0, 10.0, 0.0, 100.0, 1.0, 1.0)
    c1 = (2, 0.0, 20.0, 0.0, 101.0, 1.0, 1.0)
    # max_delta == 0 → degenerate branch → first candidate, score 1.0
    result = select_best_opportunity([c0, c1], "buy", 0)
    assert result is not None
    assert result[0] == 1
    assert result[1] == 1.0
