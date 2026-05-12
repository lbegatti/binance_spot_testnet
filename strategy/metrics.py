# ── REST PATH ONLY ────────────────────────────────────────────────────────────
# This module is used exclusively by the REST execution path (restapi_main.py).
# The WebSocket and backtesting paths compute order-book metrics directly in
# strategy/book_utils.py (see AnalysisEngine.low_latency_analysis() and
# backtest/signals.py).  Do not remove — restapi_main.py depends on this.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd


def get_order_book_metrics(order_book_df: pd.DataFrame):
    """Calculate various order book metrics and add them as new columns to the DataFrame.
    Metrics calculated:
    - Total Depth: Sum of bid and ask quantities.
    - Mid-Price: Average of bid and ask prices.
    - Order Book Imbalance (OBI): (Bid Quantity - Ask Quantity) / (Bid Quantity + Ask Quantity).
    - Micro Price: (Bid Price * Ask Quantity + Ask Price * Bid Quantity) / (Bid Quantity + Ask Quantity).
    - Micro vs Mid: Boolean indicating if Micro Price is greater than Mid-Price.
    - Bid-Ask Spread: (Ask Price - Bid Price) / Mid-Price.
    - Is Large Spread: Boolean indicating if the bid-ask spread is greater than 0.10% of the mid-price.
    - Is Small Spread: Boolean indicating if the bid-ask spread is less than or equal to 0.02% of the mid-price.
    Args:
        order_book_df (pd.DataFrame): DataFrame containing 'bid_price', 'bid_quantity', 'ask_price', and 'ask_quantity' columns.
    Returns:
        pd.DataFrame: The input DataFrame with additional columns for the calculated metrics.
    """
    order_book_df["total_depth"] = (
        order_book_df["bid_quantity"] + order_book_df["ask_quantity"]
    )
    order_book_df["mid_price"] = (
        order_book_df["bid_price"] + order_book_df["ask_price"]
    ) / 2
    order_book_df["obi"] = (
        order_book_df["bid_quantity"] - order_book_df["ask_quantity"]
    ) / (order_book_df["bid_quantity"] + order_book_df["ask_quantity"])
    order_book_df["micro_price"] = (
        (order_book_df["bid_price"] * order_book_df["ask_quantity"])
        + (order_book_df["ask_price"] * order_book_df["bid_quantity"])
    ) / (order_book_df["bid_quantity"] + order_book_df["ask_quantity"])
    order_book_df["micro_vs_mid"] = (
        order_book_df["micro_price"] > order_book_df["mid_price"]
    ).astype(bool)
    order_book_df["bid_ask_spread"] = (
        order_book_df["ask_price"] - order_book_df["bid_price"]
    ) / order_book_df["mid_price"]
    order_book_df["is_large_spread"] = order_book_df["bid_ask_spread"] > 0.001
    order_book_df["is_small_spread"] = order_book_df["bid_ask_spread"] <= 0.0002

    return order_book_df
