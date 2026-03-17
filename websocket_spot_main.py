import logging
import os
import threading
import time
from binance.spot import Spot as Client
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from dotenv import load_dotenv
from analysis import AnalysisEngine
from message_handler import MessageHandler
from order_book_state import OrderBookState
from config_parameters import (
    DEFAULT_SESSION_MINUTES,
    RECV_WINDOW,
    SNAPSHOT_DEPTH,
    WS_SPEED,
    HTF_JOIN_TIMEOUT,
    HIST_JOIN_TIMEOUT,
    SYMBOL,
    CCY,
    CRYPTOCCY,
)

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
# 2. Fetch account balance before any trading logic
# ---------------------------------------------------------------------------
# noinspection PyArgumentList
account_info = rest_client.account(recvWindow=RECV_WINDOW)
balances = {
    item["asset"]: {"free": float(item["free"]), "locked": float(item["locked"])}
    for item in account_info["balances"]
    if float(item["free"]) > 0 or float(item["locked"]) > 0
}

if not balances:
    logging.warning("No non-zero balances found on this testnet account.")
else:
    # Fetch valid USDT pairs so we only attempt sells that can succeed
    # noinspection PyArgumentList
    exchange_info = rest_client.exchange_info()
    tradable_bases = {
        s["baseAsset"]
        for s in exchange_info["symbols"]
        if s["quoteAsset"] == CCY and s["status"] == "TRADING"
    }

    sold_any = False
    sold = []
    failed = []
    logging.info("Account balances (non-zero):")
    for asset, amounts in balances.items():
        logging.info(f"  {asset}: free={amounts['free']}, locked={amounts['locked']}")
        if (
            amounts["locked"] == 0
            and amounts["free"] > 0
            and asset not in (CCY, CRYPTOCCY)
            and asset in tradable_bases
        ):
            sell_symbol = f"{asset}{CCY}"
            logging.warning(
                f"Selling {amounts['free']} {asset} ({sell_symbol}) to consolidate into {CCY}."
            )
            try:
                # noinspection PyArgumentList
                rest_client.new_order(
                    symbol=sell_symbol,
                    side="SELL",
                    type="MARKET",
                    quantity=amounts["free"],
                    recvWindow=RECV_WINDOW,
                )
                logging.info(f"  Sold {amounts['free']} {asset} via {sell_symbol}.")
                sold.append(asset)
                sold_any = True
            except Exception as e:
                logging.error(f"  Failed to sell {asset} via {sell_symbol}: {e}")
                failed.append(asset)

    logging.info(
        f"Consolidation summary: {len(sold)} sold, {len(failed)} failed "
        f"(kept in balance)."
    )
    if failed:
        logging.info(f"  Failed assets (kept): {failed}")

    # Re-fetch balances only if we actually sold something
    if sold_any:
        # noinspection PyArgumentList
        account_info = rest_client.account(recvWindow=RECV_WINDOW)
        balances = {
            item["asset"]: {
                "free": float(item["free"]),
                "locked": float(item["locked"]),
            }
            for item in account_info["balances"]
            if float(item["free"]) > 0 or float(item["locked"]) > 0
        }

# Check that we have USDT (or the quote asset) available for trading
usdt_balance = balances.get(CCY, {}).get("free", 0.0)
btc_balance = balances.get(CRYPTOCCY, {}).get("free", 0.0)
logging.info(f"Available {CCY}: {usdt_balance} | Available {CRYPTOCCY}: {btc_balance}")

if usdt_balance == 0 and btc_balance == 0:
    raise ValueError(
        f"No {CCY} or {CRYPTOCCY} balance available. Fund your testnet account before trading."
    )

# ---------------------------------------------------------------------------
# 3. Session duration
# ---------------------------------------------------------------------------
# At the default of 15 min the engine runs:
#   • low_latency_analysis → 15 × 60 / 5  =  180 iterations  (every 5 s)
#   • historical_analysis  → 15 / 5       =    3 iterations  (every 5 min)

session_minutes = DEFAULT_SESSION_MINUTES
session_seconds = session_minutes * 60
logging.info(
    "Session configured: %d minute(s) → ~%d HFT iterations, ~%d historical iterations.\n",
    session_minutes,
    session_seconds // 5,
    session_minutes // 5,
)

# ---------------------------------------------------------------------------
# 4. Fetch the starting snapshot and initialise shared state
# ---------------------------------------------------------------------------
logging.info("Fetching order book snapshot...")
# noinspection PyArgumentList
snapshot = rest_client.depth(symbol=SYMBOL, limit=SNAPSHOT_DEPTH)

# OrderBookState is the single source of truth shared by MessageHandler and
# AnalysisEngine.  Both classes receive the same instance so they operate on
# the same data and the same lock.
state = OrderBookState()
state.local_book = {
    "bids": {price: qty for price, qty in snapshot["bids"]},
    "asks": {price: qty for price, qty in snapshot["asks"]},
    "lastUpdateId": snapshot["lastUpdateId"],
}

# stop_event is set by this file when the session duration elapses; both
# analysis loops check it on every iteration and exit if stop_event is reached.
stop_event = threading.Event()

# ---------------------------------------------------------------------------
# 5. Instantiate engine and handler
# ---------------------------------------------------------------------------
handler = MessageHandler(state=state)
engine = AnalysisEngine(state=state, stop_event=stop_event)

low_latency_thread = threading.Thread(
    target=engine.low_latency_analysis, daemon=True, name="low-latency-analysis"
)
hist_thread = threading.Thread(
    target=engine.historical_analysis, daemon=True, name="hist-analysis"
)

# ---------------------------------------------------------------------------
# 6. Open WebSocket stream
# ---------------------------------------------------------------------------
# NOTE: The Binance Spot Testnet does not support WebSocket market-data streams.
# We use the production stream endpoint for real-time depth data (read-only, no auth).
# Trading orders are still routed through the testnet REST client.
ws_client = SpotWebsocketStreamClient(
    on_message=handler.handle_depth_message,
)
logging.info(
    "WebSocket stream opened. Waiting %ds for initial depth data...", WS_SPEED // 100
)
ws_client.diff_book_depth(symbol=SYMBOL, speed=WS_SPEED)

# Give the WebSocket a moment to deliver the first diff-depth messages so that
# local_book["bids"] is populated before the low-latency loop runs its first iteration.
time.sleep(1)

low_latency_thread.start()
hist_thread.start()
logging.info("Analysis threads started. Running for %d minute(s)...\n", session_minutes)

# ---------------------------------------------------------------------------
# 7. Block the main thread for the session duration, then shut down cleanly
# ---------------------------------------------------------------------------
try:
    threading.Event().wait(
        timeout=session_seconds
    )  # interruptible by KeyboardInterrupt
except KeyboardInterrupt:
    logging.info("KeyboardInterrupt received — shutting down early.")
finally:
    time.sleep(1)
    logging.info("Session complete. Stopping WebSocket and analysis threads...\n")
    stop_event.set()  # signal both analysis loops to exit
    ws_client.stop()  # close the WebSocket connection
    low_latency_thread.join(timeout=HTF_JOIN_TIMEOUT)
    hist_thread.join(timeout=HIST_JOIN_TIMEOUT)
    logging.info("All threads stopped. Exiting.")
