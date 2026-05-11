# ── REST PATH ONLY ────────────────────────────────────────────────────────────
# This module is used exclusively by the REST execution path (restapi_main.py).
# The WebSocket and backtesting paths use strategy/book_utils.py directly.
# Do not remove — restapi_main.py depends on this.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

from strategy.indicators import add_strategy_indicators
from strategy.scores import get_weighted_volume_micro_spread_score


def find_best_quote(order_book_df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Evaluate a single strategy (buy or sell) on an order book snapshot.

    Returns a single-row DataFrame with the best quote for the strategy,
    or an empty DataFrame if no opportunity is found.
    """
    df = add_strategy_indicators(order_book_df.copy(), strategy=position)

    # For buy-strategy we want micro > mid;
    # for sell-strategy we want micro <= mid
    micro_filter = df["micro_vs_mid"] if position == "buy" else ~df["micro_vs_mid"]

    is_opportunity = (
            (~df["is_thin_micro_effect"])
            & micro_filter
            & df["is_total_depth_50pct_l0"]
            & (df.index != 0)  # skip level 0 (best quote)
    )

    if "micro_mid_delta" not in df.columns or not is_opportunity.any():
        return pd.DataFrame()

    df["norm_vol_delta_score"] = get_weighted_volume_micro_spread_score(df)

    best = (
        df.loc[is_opportunity]
        .reset_index(drop=True)
        .nlargest(1, "norm_vol_delta_score")
    )
    best["strategy"] = position
    return best
