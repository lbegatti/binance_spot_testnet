# ── REST PATH ONLY ────────────────────────────────────────────────────────────
# This module is used exclusively by the REST execution path (restapi_main.py).
# The WebSocket and backtesting paths use strategy/book_utils.py directly.
# Do not remove — restapi_main.py depends on this.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd


def get_weighted_volume_micro_spread_score(order_book_df: pd.DataFrame):
    """Calculate a weighted score based on total depth and micro-mid delta to identify potential trading opportunities.
    The score is calculated as:
    - 70% weight on total depth (normalized by the maximum total depth).
    - 30% weight on micro-mid delta (normalized by the maximum micro-mid delta).
    Args:
        order_book_df (pd.DataFrame): DataFrame containing order book data with calculated metrics.
    Returns:
        pd.Series: A Series containing the weighted score for each level of the order book.
    """
    if (
            "total_depth" not in order_book_df.columns
            or "micro_mid_delta" not in order_book_df.columns
    ):
        raise ValueError(
            "Input DataFrame must contain 'total_depth' and 'micro_mid_delta' columns."
        )

    # Normalize total depth and micro-mid delta
    normalized_total_depth = (
            order_book_df["total_depth"] / order_book_df["total_depth"].max()
    )
    normalized_micro_mid_delta = (
            order_book_df["micro_mid_delta"] / order_book_df["micro_mid_delta"].max()
    )

    # Calculate the weighted score
    weighted_score = (normalized_total_depth * 0.7) + (normalized_micro_mid_delta * 0.3)

    return weighted_score
