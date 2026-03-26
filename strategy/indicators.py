import logging
import pandas as pd
import numpy as np


def add_strategy_indicators(order_book_df: pd.DataFrame, strategy: str = "buy"):
    """Add indicators to the order book DataFrame to identify potential opportunities based on the specified strategy.
    Indicators added:
    - micro_mid_delta: The difference between the micro price and the mid-price.
    - is_thin_micro_effect: Boolean indicating if the total depth is below the median depth, hinting a thin order book.
    - is_total_depth_50pct_l0: Boolean if the total depth is at least 50% of the level 0 depth, hinting enough liq.
    - w_volume_micro_spread_score: A weighted score with total depth and micro-mid delta to get potential opportunities.
    Args:
        order_book_df (pd.DataFrame): DataFrame containing order book data with calculated metrics.
        strategy (str): The trading strategy to apply. Currently, supports "b" for buy-side opportunities.
    Returns:
        pd.DataFrame: The input DataFrame with additional columns for the opportunity indicators.
    """
    if strategy == "buy":
        if not order_book_df["micro_vs_mid"].any():
            logging.info("No micro price above mid-price. No opportunities detected.")
            return order_book_df  # Return early if no opportunities
        order_book_df["micro_mid_delta"] = (
            order_book_df["micro_price"] - order_book_df["mid_price"]
        )
    elif strategy == "sell":
        if order_book_df["micro_vs_mid"].all():
            logging.info(
                "Micro price is above mid-price everywhere. No sell opportunities detected."
            )
            return order_book_df

        order_book_df["micro_mid_delta"] = (
            order_book_df["mid_price"] - order_book_df["micro_price"]
        )
    else:
        raise ValueError("Unsupported strategy. Use 'buy' or 'sell'.")

    order_book_df["is_thin_micro_effect"] = np.where(
        order_book_df["total_depth"] >= np.median(order_book_df["total_depth"]),
        False,
        True,
    ).astype(bool)
    order_book_df["is_total_depth_50pct_l0"] = (
        order_book_df["total_depth"] >= 0.5 * order_book_df["total_depth"].iloc[0]
    ).astype(bool)

    return order_book_df


def volume_weighted_average_price(
    price: pd.Series | int | float | list | np.ndarray,
    volume: pd.Series | int | float | list | np.ndarray,
) -> float:
    """Calculate the weighted average price for volume weighted average price.
    VWAP is calculated as the sum of (price * volume) divided by the sum of volume.
    Args:
        price (pd.Series | int | float): The price(s) to be weighted.
        volume (pd.Series | int | float): The corresponding volume(s) for the price(s).
    Returns:
        pd.Series: The calculated volume weighted average price.
    """

    return float(np.sum(price * volume) / np.sum(volume))
