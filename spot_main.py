import os
import time
import pandas as pd
from dotenv import load_dotenv
import logging

from metrics import get_order_book_metrics
from quotes import find_best_quote
from plot_helpers import plot_depth_bid_ask, Client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)

# ---------------------------------------------------------------------------
# 1. Load environment variables from .env file
# ---------------------------------------------------------------------------
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
depths_limit = [5, 10, 15, 20, 50]
best_quotes = []

initial_id = client.get_order_book(symbol="BNBUSDT")["lastUpdateId"]

for d in depths_limit:
    order_book = client.get_order_book(symbol="BNBUSDT", limit=d)
    current_id = order_book["lastUpdateId"]
    gap_id = current_id - initial_id
    while gap_id >= 100:
        logging.warning(
            f"Order book update gap is {gap_id} - EXTREME VOLATILE market conditions!"
        )
        logging.warning(
            f"Waiting for a better market condition... Current gap_id: {gap_id}"
        )
        time.sleep(1)
        current_id = order_book["lastUpdateId"]
        order_book = client.get_order_book(symbol="BNBUSDT", limit=d)
        new_id = order_book["lastUpdateId"]
        gap_id = new_id - current_id

    if gap_id <= 5:
        logging.info(f"Order book update gap is {gap_id} - QUIET market conditions!")
    elif 5 <= gap_id <= 50:
        logging.warning(
            f"Order book update gap is {gap_id} - NORMAL market conditions!"
        )
    elif 50 <= gap_id < 100:
        logging.warning(
            f"Order book update gap is {gap_id} - VOLATILE market conditions!"
        )
    bids = order_book["bids"]
    asks = order_book["asks"]
    max_depth = min(len(bids), len(asks))
    depth_bids = pd.DataFrame(
        bids[:max_depth], columns=["bid_price", "bid_quantity"], dtype=float
    )
    depth_asks = pd.DataFrame(
        asks[:max_depth], columns=["ask_price", "ask_quantity"], dtype=float
    )

    order_depth_df = pd.concat([depth_bids, depth_asks], axis=1)
    order_depth_df = get_order_book_metrics(order_depth_df)

    for strategy in ("buy", "sell"):
        best_quote = find_best_quote(order_depth_df, position=strategy)
        if not best_quote.empty:
            print(
                f"Depth: {d} - Strategy: {best_quote['strategy'].iloc[0]}\n"
                f"Best Quote:\n{best_quote}\n"
            )
            best_quotes.append(best_quote)

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
