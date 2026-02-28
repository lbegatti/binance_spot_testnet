from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from binance.spot import Spot as Client
import json
import logging
from dotenv import load_dotenv
import os

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

rest_client = Client(api_key, api_secret, testnet=True)

# 2. Fetch the "Starting Point" (Snapshot)
print("Fetching snapshot...")
snapshot = rest_client.depth(symbol="BTCUSDT", limit=100)
local_book = {
    "bids": {price: qty for price, qty in snapshot["bids"]},
    "asks": {price: qty for price, qty in snapshot["asks"]},
    "lastUpdateId": snapshot["lastUpdateId"],
}


def handle_depth_message(_, message):
    data = json.loads(message)

    # Skip the subscription confirmation message (only has 'id' and 'result')
    if "result" in data and "b" not in data:
        logging.info("WebSocket subscription confirmed.")
        return
    # SYNC LOGIC: Ignore updates that are older than our snapshot
    if data["u"] <= local_book["lastUpdateId"]:
        return
    # 1. Logic to sync with lastUpdateId goes here
    # 2. Update local_book dictionary — bids ('b') and asks ('a')
    for price, qty in data.get("b", []):
        if float(qty) == 0:
            local_book["bids"].pop(price, None)
        else:
            local_book["bids"][price] = qty

    for price, qty in data.get("a", []):
        if float(qty) == 0:
            local_book["asks"].pop(price, None)
        else:
            local_book["asks"][price] = qty

    # Now your strategy logic can always read from 'local_book'
    # which is updated in real-time (no 1-second lag!)
    calculate_best_quote()


def calculate_best_quote():
    # Get top of book without a network call
    if not local_book["bids"] or not local_book["asks"]:
        return

        # Get the highest Buy price and lowest Sell price
    best_bid = max(local_book["bids"].keys(), key=float)
    best_ask = min(local_book["asks"].keys(), key=float)
    print(f"Spread: {best_bid} | {best_ask}", end="\r")


ws_client = SpotWebsocketStreamClient(on_message=handle_depth_message)
ws_client.diff_book_depth(symbol="BTCUSDT", speed=100)  # 100ms updates
