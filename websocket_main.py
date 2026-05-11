import logging
import os
import threading
import time
from binance.spot import Spot as Client
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from dotenv import load_dotenv
from strategy.analysis import AnalysisEngine
from core.message_handler import MessageHandler
from core.order_book_state import OrderBookState
from execution.order_executor import OrderExecutor
from strategy.regime_director import RegimeDirector
from strategy.param_loader import load_best_params
from config_parameters import (
    DEFAULT_SESSION_MINUTES,
    HFT_INTERVAL,
    HIST_INTERVAL,
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

# Apply best_params.json overrides to strategy.regime_director BEFORE
# RegimeDirector() is instantiated (see strategy/param_loader.py).
load_best_params()

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
# NOTE on balance tracking:
# The old listenKey / User Data Stream mechanism was discontinued by Binance
# for the Spot API (Feb 2026 — REST returns 410 Gone, WS returns 404).
# Real-time balance updates now flow through the OrderExecutor's WebSocket
# API connection:  session.logon → userDataStream.subscribe →
# outboundAccountPosition push events.  No listenKey is needed.
# ---------------------------------------------------------------------------

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

# Snapshot the BTC price at session start so the end-of-session report can
# separate trading P&L from price-appreciation P&L.
try:
    # noinspection PyArgumentList
    btc_start_price = float(rest_client.ticker_price(symbol=SYMBOL)["price"])
    start_total_usdt = usdt_balance + btc_balance * btc_start_price
    logging.info(
        "Portfolio snapshot at session start: %.2f %s  (BTC @ %.2f)",
        start_total_usdt,
        CCY,
        btc_start_price,
    )
except Exception as e:
    btc_start_price = None
    start_total_usdt = None
    logging.warning("Could not fetch BTC start price for P&L attribution: %s", e)

# ---------------------------------------------------------------------------
# 3. Session duration
# ---------------------------------------------------------------------------
# At the default of 10 min the engine runs:
#   • low_latency_analysis → 10 × 60 / 1  =  600 iterations  (every 1 s)
#   • historical_analysis  → 10 × 60 / 60 =   10 iterations  (every 60 s / 1 min)

session_minutes = DEFAULT_SESSION_MINUTES
session_seconds = session_minutes * 60
logging.info(
    "Session configured: %d minute(s) → ~%d low-latency iterations, ~%d historical iterations.\n",
    session_minutes,
    session_seconds // HFT_INTERVAL,
    session_seconds // HIST_INTERVAL,
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
state.balance_status[CCY] = usdt_balance
state.balance_status[CRYPTOCCY] = btc_balance
state.local_book = {
    "bids": {price: qty for price, qty in snapshot["bids"]},
    "asks": {price: qty for price, qty in snapshot["asks"]},
    "lastUpdateId": snapshot["lastUpdateId"],
}
# ---------------------------------------------------------------------------
# 4b. Pre-session regime detection (initial HMM fit on recent klines)
# ---------------------------------------------------------------------------
logging.info("Running initial regime detection — fetching klines and fitting HMM...")
regime_director = RegimeDirector()
regime_director.get_klines_data()
regime_director.select_hmm_model()
regime_director.assign_regime_labels()
logging.info("Initial market regime: '%s'", regime_director.regime_label)

# stop_event is set by this file when the session duration elapses; both
# analysis loops check it on every iteration and exit if stop_event is reached.
stop_event = threading.Event()

# ---------------------------------------------------------------------------
# 5. Instantiate engine and handler
# ---------------------------------------------------------------------------
handler = MessageHandler(state=state)

# OrderExecutor owns its own WebSocket API connection for lower-latency
# order placement.  On connection open it sends session.logon (HMAC-signed)
# followed by userDataStream.subscribe, so that outboundAccountPosition
# push events keep state.balance_status current in real time — no listenKey
# needed.  If the testnet WS API is unreachable, execution falls back to REST
# and balances rely on the startup REST snapshot.
executor = OrderExecutor(
    state=state,
    stream_url="wss://testnet.binance.vision/ws-api/v3",
    api_key=api_key,
    api_secret=api_secret,
    rest_client=rest_client,
)
logging.info(
    "OrderExecutor initialised (WS=%s, REST fallback=%s, user-data=%s).",
    "yes" if executor.ws_api_client is not None else "unavailable",
    "yes" if executor.rest_client is not None else "no",
    "pending" if executor.ws_api_client is not None else "REST-only",
)

engine = AnalysisEngine(
    state=state,
    stop_event=stop_event,
    executor=executor,
    regime_director=regime_director,
)

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
# Trading orders and balance updates are routed through the testnet WebSocket API
# connection owned by OrderExecutor.
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
    ws_client.stop()  # close the market-data WebSocket stream
    executor.stop()  # close the WebSocket API order + user-data connection
    low_latency_thread.join(timeout=HTF_JOIN_TIMEOUT)
    hist_thread.join(timeout=HIST_JOIN_TIMEOUT)
    logging.info("All threads stopped. Exiting.")

    # -----------------------------------------------------------------------
    # End-of-session order report
    # -----------------------------------------------------------------------
    executor.order_status_report()

    # -----------------------------------------------------------------------
    # End-of-session balance report
    # -----------------------------------------------------------------------
    try:
        final_info = rest_client.account(recvWindow=RECV_WINDOW)
        final_balances = {
            item["asset"]: float(item["free"])
            for item in final_info["balances"]
            if item["asset"] in (CCY, CRYPTOCCY)
        }
        final_usdt = final_balances.get(CCY, 0.0)
        final_btc  = final_balances.get(CRYPTOCCY, 0.0)
        d_usdt     = final_usdt - usdt_balance
        d_btc      = final_btc  - btc_balance

        # Fetch current BTC price for portfolio valuation.
        # noinspection PyArgumentList
        btc_end_price   = float(rest_client.ticker_price(symbol=SYMBOL)["price"])
        end_total_usdt  = final_usdt + final_btc * btc_end_price

        # ── P&L attribution ────────────────────────────────────────────────
        # trading_pnl  = what the STRATEGY contributed.
        #   Mark BOTH the starting and ending BTC holding at the END price so
        #   that BTC price moves cancel out. Only the change in mix (selling
        #   BTC for USDT at a certain price vs. buying USDT back later)
        #   remains — that is the strategy's contribution.
        #     trading_pnl = Δusdt + Δbtc × end_price
        #
        # price_pnl    = what BTC price appreciation/depreciation contributed.
        #   The starting BTC holding changes value solely because the price moved.
        #     price_pnl = btc_balance × (end_price − start_price)
        #
        # total_pnl    = end_total − start_total (includes both effects).
        #
        # All three are in USDT.

        if btc_start_price is not None and start_total_usdt is not None:
            trading_pnl  = d_usdt + d_btc * btc_end_price
            price_pnl    = btc_balance * (btc_end_price - btc_start_price)
            total_pnl    = end_total_usdt - start_total_usdt
            pct_return   = total_pnl / start_total_usdt * 100 if start_total_usdt else 0.0
            # Sanity-check: trading_pnl + price_pnl == total_pnl (algebraic identity).
            # Any residual is floating-point rounding noise — shown for transparency.
            residual = total_pnl - (trading_pnl + price_pnl)

            logging.info(
                "\n"
                "========== END-OF-SESSION BALANCE REPORT ==========\n"
                "  %-10s  start: %14.2f   end: %14.2f   Δ %+.2f\n"
                "  %-10s  start: %14.8f   end: %14.8f   Δ %+.8f\n"
                "  BTC price   start: %14.2f   end: %14.2f   Δ %+.2f\n"
                "\n"
                "  Portfolio value (USDT)\n"
                "    Start  : %14.2f   (USDT + BTC × start price)\n"
                "    End    : %14.2f   (USDT + BTC × end price)\n"
                "\n"
                "  P&L decomposition  [A + B = Total]\n"
                "    A  Trading alpha : %+.2f  (Δusdt + Δbtc × end_price)\n"
                "       Strategy bought/sold BTC; this is the net result\n"
                "       marked at the END price — isolated from market moves.\n"
                "    B  Price move    : %+.2f  (starting BTC × price change)\n"
                "       Your %.8f BTC start position gained/lost value\n"
                "       purely because BTC moved %+.2f.\n"
                "    ─────────────────────────────────────────────────────\n"
                "    A + B  Total P&L : %+.2f  (%+.3f %%)\n"
                "====================================================",
                CCY,       usdt_balance, final_usdt, d_usdt,
                CRYPTOCCY, btc_balance,  final_btc,  d_btc,
                btc_start_price, btc_end_price, btc_end_price - btc_start_price,
                start_total_usdt,
                end_total_usdt,
                trading_pnl,
                price_pnl, btc_balance, btc_end_price - btc_start_price,
                total_pnl, pct_return,
            )
            if abs(residual) > 0.01:
                logging.warning("P&L residual %.4f (floating-point rounding).", residual)
        else:
            # Fallback: no start price captured — show raw deltas only.
            logging.info(
                "\n"
                "========== END-OF-SESSION BALANCE REPORT ==========\n"
                "  %-10s  start: %14.2f   end: %14.2f   Δ %+.2f\n"
                "  %-10s  start: %14.8f   end: %14.8f   Δ %+.8f\n"
                "  Portfolio (USDT equiv, end price %.2f)\n"
                "    End total : %.2f %s\n"
                "====================================================",
                CCY,      usdt_balance, final_usdt, d_usdt,
                CRYPTOCCY, btc_balance, final_btc,  d_btc,
                btc_end_price,
                end_total_usdt, CCY,
            )
    except Exception as e:
        logging.error("Could not fetch final balance: %s", e)
