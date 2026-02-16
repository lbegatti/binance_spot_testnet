import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from plot_helpers import plot_depth_bid_ask, plot_ohlc_with_volume, Client

# 1. Load environment variables from .env file
load_dotenv()

api_key = os.getenv("BINANCE_TESTNET_API_KEY")
api_secret = os.getenv("BINANCE_TESTNET_SECRET_KEY")

if not api_key or not api_secret:
    raise ValueError("API keys not found. Check your .env file.")

# 2. Initialize the Client
# The 'testnet=True' parameter tells the library to use the testnet.binance.vision URL
client = Client(api_key, api_secret, testnet=True)

# 3. Example: Get account information
balance = client.get_asset_balance(asset="BNB", recvWindow=5000)

# 4. Get different bids and asks for a symbol.
depths_limit = [5, 10, 20, 50]
quotes = []

for d in depths_limit:
    depth = client.get_order_book(symbol="BNBUSDT", limit=d)
    max_depth = min(len(depth["bids"]), len(depth["asks"]))
    depth_bids = pd.DataFrame(
        depth["bids"][:max_depth], columns=["bid_price", "bid_quantity"], dtype=float
    )
    depth_asks = pd.DataFrame(
        depth["asks"][:max_depth], columns=["ask_price", "ask_quantity"], dtype=float
    )
    depths_bid_ask = pd.concat([depth_bids, depth_asks], axis=1)
    # Total Depth = Bid Quantity + Ask Quantity
    depths_bid_ask["total_depth"] = (
            depths_bid_ask["bid_quantity"] + depths_bid_ask["ask_quantity"]
    )
    # Mid-Price = (Bid Price + Ask Price) / 2
    depths_bid_ask["mid_price"] = (
                                          depths_bid_ask["bid_price"] + depths_bid_ask["ask_price"]
                                  ) / 2
    # Order Book Imbalance (OBI) = (Bid Quantity - Ask Quantity) / (Bid Quantity + Ask Quantity)
    depths_bid_ask["obi"] = (
                                    depths_bid_ask["bid_quantity"] - depths_bid_ask["ask_quantity"]
                            ) / (depths_bid_ask["bid_quantity"] + depths_bid_ask["ask_quantity"])
    # Micro Price = (Bid Price * Ask Quantity + Ask Price * Bid Quantity) / (Bid Quantity + Ask Quantity)
    ## logic: If the ask volume is small, then , the price will likely stay closer to bid.
    ## If the bid volume is massive, then the price will move closer to the ask.
    ## If bid quantity > ask quantity (Bullish), OBI will be positive and micro-price will be higher than mid-price.
    depths_bid_ask["micro_price"] = (
                                            (depths_bid_ask["bid_price"] * depths_bid_ask["ask_quantity"])
                                            + (depths_bid_ask["ask_price"] * depths_bid_ask["bid_quantity"])
                                    ) / (depths_bid_ask["bid_quantity"] + depths_bid_ask["ask_quantity"])
    depths_bid_ask["micro_vs_mid"] = (
            depths_bid_ask["micro_price"] > depths_bid_ask["mid_price"]
    )
    depths_bid_ask["micro_vs_mid"] = (
            depths_bid_ask["micro_price"] > depths_bid_ask["mid_price"]
    )
    depths_bid_ask["bid_ask_spread"] = ((depths_bid_ask["ask_price"] - depths_bid_ask["bid_price"]) / depths_bid_ask[
        "mid_price"])
    # Identify if the spread is large (e.g., greater than 0.10% of the mid-price)
    depths_bid_ask["is_large_spread"] = depths_bid_ask["bid_ask_spread"] > 0.001
    depths_bid_ask["is_small_spread"] = depths_bid_ask["bid_ask_spread"] <= 0.0002
    if depths_bid_ask["micro_vs_mid"].any():
        print("Micro price is above mid-price for at least one level. Checking for opportunities...")
        depths_bid_ask["micro_mid_delta"] = (
                depths_bid_ask["micro_price"] - depths_bid_ask["mid_price"]
        )
        depths_bid_ask["is_thin_micro_effect"] = np.where(
            depths_bid_ask["total_depth"] >= np.median(depths_bid_ask["total_depth"]),
            False,
            True,
        )
        depths_bid_ask["is_total_depth_50pct_l0"] = (depths_bid_ask["total_depth"] >= 0.5 *
                                                         depths_bid_ask["total_depth"].iloc[0]).astype(bool)
        # ensure boolean dtype for clarity
        depths_bid_ask["is_thin_micro_effect"] = depths_bid_ask[
            "is_thin_micro_effect"
        ].astype(bool)
        depths_bid_ask["micro_vs_mid"] = depths_bid_ask["micro_vs_mid"].astype(bool)

        thin_micro_mid_80pct_lev0 = ((~depths_bid_ask["is_thin_micro_effect"]) & depths_bid_ask["micro_vs_mid"]
                                     & depths_bid_ask["is_total_depth_50pct_l0"])
        potential_buy_quote = (
            depths_bid_ask.loc[thin_micro_mid_80pct_lev0]
            .reset_index(drop=True)
            .nlargest(1, "micro_mid_delta")
        )
        quotes.append(potential_buy_quote)

    plot_depth_bid_ask(df=depths_bid_ask)

## 4.0 Get historical klines for a symbol
# plot_ohlc_with_volume(
#     client=client,
#     symbol="BNBUSDT",
#     interval=Client.KLINE_INTERVAL_1MINUTE,
#     lookback="5 day ago UTC",
# )
# 4.1 Analyze the order book depth for a symbol

# Total Depth = Bid Quantity + Ask Quantity
# depths_bid_ask["total_depth"] = (
#         depths_bid_ask["bid_quantity"] + depths_bid_ask["ask_quantity"]
# )
# Mid-Price = (Bid Price + Ask Price) / 2
# depths_bid_ask["mid_price"] = (
#                                       depths_bid_ask["bid_price"] + depths_bid_ask["ask_price"]
#                               ) / 2
# Order Book Imbalance (OBI) = (Bid Quantity - Ask Quantity) / (Bid Quantity + Ask Quantity)
# depths_bid_ask["obi"] = (
#                                 depths_bid_ask["bid_quantity"] - depths_bid_ask["ask_quantity"]
#                         ) / (depths_bid_ask["bid_quantity"] + depths_bid_ask["ask_quantity"])

# Micro Price = (Bid Price * Ask Quantity + Ask Price * Bid Quantity) / (Bid Quantity + Ask Quantity)
# logic: If the ask volume is small, then , the price will likely stay closer to bid.
# If the bid volume is massive, then the price will move closer to the ask.
# If bid quantity > ask quantity (Bullish), OBI will be positive and micro-price will be higher than mid-price.
# depths_bid_ask["micro_price"] = (
#                                         (depths_bid_ask["bid_price"] * depths_bid_ask["ask_quantity"])
#                                         + (depths_bid_ask["ask_price"] * depths_bid_ask["bid_quantity"])
#                                 ) / (depths_bid_ask["bid_quantity"] + depths_bid_ask["ask_quantity"])
# depths_bid_ask["micro_vs_mid"] = (
#         depths_bid_ask["micro_price"] > depths_bid_ask["mid_price"]
# )
# if depths_bid_ask["micro_vs_mid"].any():
#     print("Micro price is above mid-price for at least one level.")
#     depths_bid_ask["micro_mid_delta"] = (
#             depths_bid_ask["micro_price"] - depths_bid_ask["mid_price"]
#     )
#     depths_bid_ask["thin_micro_effect"] = np.where(
#         depths_bid_ask["total_depth"] >= np.median(depths_bid_ask["total_depth"]),
#         False,
#         True,
#     )
#     # ensure boolean dtype for clarity
#     depths_bid_ask["thin_micro_effect"] = depths_bid_ask["thin_micro_effect"].astype(bool)
#     depths_bid_ask["micro_vs_mid"] = depths_bid_ask["micro_vs_mid"].astype(bool)
#
#     thin_micro_mid = (~depths_bid_ask["thin_micro_effect"]) & depths_bid_ask["micro_vs_mid"]
#     potential_buy_quote = (
#         depths_bid_ask.loc[thin_micro_mid].reset_index(drop=True).nlargest(1, "micro_mid_delta")
#     )
# 4.2 Get recent trades for a symbol
# 4.3 Get historical trades for a symbol
# 4.4 Get aggregate trades for a symbol
