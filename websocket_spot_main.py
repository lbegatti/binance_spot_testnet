from functools import partial
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from binance.spot import Spot as Client
from message_handler import handle_depth_message
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

rest_client = Client(
    api_key,
    api_secret,
    base_url="https://testnet.binance.vision",
)

# ---------------------------------------------------------------------------
# Symbol configuration
# ---------------------------------------------------------------------------
symbol = "BTCUSDT"
ccy = "USDT"
cryptoccy = "BTC"

# ---------------------------------------------------------------------------
# 2. Fetch account balance before any trading logic
# ---------------------------------------------------------------------------
# noinspection PyArgumentList
account_info = rest_client.account(recvWindow=5000)
balances = {
    item["asset"]: {"free": float(item["free"]), "locked": float(item["locked"])}
    for item in account_info["balances"]
    if float(item["free"]) > 0 or float(item["locked"]) > 0
}

if not balances:
    logging.warning("No non-zero balances found on this testnet account.")
else:
    logging.info("Account balances (non-zero):")
    for asset, amounts in balances.items():
        logging.info(f"  {asset}: free={amounts['free']}, locked={amounts['locked']}")

# Check that we have USDT (or the quote asset) available for trading
usdt_balance = balances.get(ccy, {}).get("free", 0.0)
btc_balance = balances.get(cryptoccy, {}).get("free", 0.0)
logging.info(f"Available {ccy}: {usdt_balance} | Available {cryptoccy}: {btc_balance}")

if usdt_balance == 0 and btc_balance == 0:
    raise ValueError(
        f"No {ccy} or {cryptoccy} balance available. Fund your testnet account before trading."
    )

# ---------------------------------------------------------------------------
# 3. Fetch the "Starting Point" (Snapshot)
logging.info("Fetching order book snapshot...")
# noinspection PyArgumentList
snapshot = rest_client.depth(symbol=symbol, limit=100)
local_book = {
    "bids": {price: qty for price, qty in snapshot["bids"]},
    "asks": {price: qty for price, qty in snapshot["asks"]},
    "lastUpdateId": snapshot["lastUpdateId"],
}

ws_client = SpotWebsocketStreamClient(
    stream_url="wss://testnet.binance.vision/ws",
    on_message=partial(handle_depth_message, local_book=local_book),
)
ws_client.diff_book_depth(symbol=symbol, speed=100)  # 100ms updates
