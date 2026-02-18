import os
import pandas as pd
from dotenv import load_dotenv

from metrics import get_order_book_metrics
from indicators import add_strategy_indicators
from scores import get_weighted_volume_micro_spread_score
from plot_helpers import plot_depth_bid_ask, Client

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

    order_depth_df = pd.concat([depth_bids, depth_asks], axis=1)
    order_depth_df = get_order_book_metrics(order_depth_df)
    order_depth_df_with_indicators = add_strategy_indicators(
        order_depth_df, strategy="buy"
    )

    is_potential_opportunity = (
        (~order_depth_df_with_indicators["is_thin_micro_effect"])
        & order_depth_df_with_indicators["micro_vs_mid"]
        & order_depth_df_with_indicators["is_total_depth_50pct_l0"]
        & (
            order_depth_df_with_indicators.index != 0
        )  # ensure we are not looking at the level 0 (first quote).
    )
    # Normalized Score: 70% Volume (Safety) + 30% Delta (Aggression)
    order_depth_df_with_indicators["norm_vol_delta_score"] = (
        get_weighted_volume_micro_spread_score(order_depth_df_with_indicators)
    )
    potential_buy_quote = (
        order_depth_df_with_indicators.loc[is_potential_opportunity]
        .reset_index(drop=True)
        .nlargest(1, "norm_vol_delta_score")
    )
    if not potential_buy_quote.empty:
        print(f"Depth: {d} - "
              f"Strategy: {potential_buy_quote['strategy'].iloc[0]} - "
              f"\nPotential Buy Quote:\n{potential_buy_quote}\n")
    quotes.append(potential_buy_quote)

    plot_depth_bid_ask(df=order_depth_df)

## 4.0 Get historical klines for a symbol
# plot_ohlc_with_volume(
#     client=client,
#     symbol="BNBUSDT",
#     interval=Client.KLINE_INTERVAL_1MINUTE,
#     lookback="5 day ago UTC",
# )
# 4.2 Get recent trades for a symbol
# 4.3 Get historical trades for a symbol
# 4.4 Get aggregate trades for a symbol
