import logging
import os
import threading
import time
from datetime import datetime
from binance.spot import Spot as Client
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from dotenv import load_dotenv
from strategy.analysis import AnalysisEngine
from core.message_handler import MessageHandler
from core.order_book_state import OrderBookState
from execution.order_executor import OrderExecutor
from strategy.regime_director import RegimeDirector
from strategy.param_loader import load_best_params
from strategy.indicators import compute_live_stop_loss_pct, compute_live_macro_trend
from config_parameters import (
    DEFAULT_SESSION_MINUTES,
    HFT_INTERVAL,
    HIST_INTERVAL,
    RECV_WINDOW,
    SNAPSHOT_DEPTH,
    WS_SPEED,
    HTF_JOIN_TIMEOUT,
    HIST_JOIN_TIMEOUT,
    BALANCE_REFRESH_INTERVAL,
    SYMBOL,
    CCY,
    CRYPTOCCY,
    STOP_LOSS_ROLLING_DAYS,
    STOP_LOSS_STD_MULT,
    MACRO_TREND_ENABLED,
    MACRO_TREND_SMA_DAYS,
    MACRO_TREND_SLOPE_DAYS,
    MACRO_TREND_BAND_PCT,
    FLATTEN_ON_START,
    LIVE_POSITION_STATE_PATH,
)
from strategy.position_store import load_position, save_position

# Log to BOTH the console and a timestamped file so a full multi-hour run is
# preserved on disk — terminal scrollback rolls over and silently loses early
# lines (e.g. the order-placement logs from the start of a long session).
_log_dir = "logs"
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(
    _log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_path, encoding="utf-8"),
    ],
)
logging.info("Session log file: %s", _log_path)

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
# Market-data REST client — PRODUCTION, keyless (public endpoints only).
# The local book streams from the production diff-depth stream, so its seed
# snapshot and gap-recovery snapshots must come from the production REST book
# too: update IDs form one continuous sequence only within a single exchange.
# (Mixing the prod stream with testnet snapshots put every frame ~5,000× out
# of sequence — 7,297 resyncs on 2026-07-08, ALL stream data discarded and
# the VWAP gate never activated.)  Signals thus read the real market, like
# the HMM klines and the backtest already do; all account and trading calls
# stay on the authenticated testnet rest_client.
market_data_client = Client(base_url="https://api.binance.com")
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

# ---------------------------------------------------------------------------
# 2a. Startup inventory policy — flatten inherited BTC, or carry it (FLATTEN_ON_START)
# ---------------------------------------------------------------------------
# The backtest assumes BACKTEST_INITIAL_BTC = 0 — it always starts with no open
# position.  A live testnet account, however, often carries BTC left over from
# previous sessions (e.g. BUYs that filled but whose SELLs never did).
#
# FLATTEN_ON_START controls what we do with that inherited BTC:
#   True  → MARKET-sell it now so the session starts flat (matches the backtest;
#           per-session skill test).
#   False → keep it.  AnalysisEngine pre-arms the position guard on the inherited
#           BTC, treating it as an open position: the bot can then only exit it
#           (mean-reversion rally or stop-loss) before buying again.  This is the
#           realistic "carry inventory across restarts" mode; component B in the
#           end-of-session report attributes the carried bag's market drift.
inherited_btc = balances.get(CRYPTOCCY, {}).get("free", 0.0)
# Floor to the BTCUSDT LOT_SIZE grid (stepSize 1e-5 → 5 decimals) so the MARKET
# sell quantity is an exact multiple and not rejected with -1013.
inherited_btc = int(inherited_btc * 1e5) / 1e5
if FLATTEN_ON_START and inherited_btc >= 1e-5:
    logging.warning(
        "Flattening %s %s of inherited BTC at startup via MARKET sell so the "
        "session begins flat (matches BACKTEST_INITIAL_BTC = 0).",
        inherited_btc,
        CRYPTOCCY,
    )
    try:
        # noinspection PyArgumentList
        rest_client.new_order(
            symbol=SYMBOL,
            side="SELL",
            type="MARKET",
            quantity=inherited_btc,
            recvWindow=RECV_WINDOW,
        )
        logging.info("Inherited BTC flattened — re-fetching balances.")
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
    except Exception as e:
        logging.error(
            "Could not flatten inherited BTC (%s) — the position guard will "
            "pre-arm on it and the bot may not BUY this session.",
            e,
        )
elif inherited_btc >= 1e-5:
    # FLATTEN_ON_START is False: keep the inherited BTC.  The AnalysisEngine
    # position guard pre-arms on it (treated as an open position), and the
    # report's component B attributes its market drift separately.
    logging.info(
        "\nFLATTEN_ON_START is False — carrying %s %s of inherited BTC into the "
        "session; the position guard will pre-arm on it (stop-loss anchored at "
        "the session-start price until position persistence is added).\n",
        inherited_btc,
        CRYPTOCCY,
    )

# Check that we have USDT (or the quote asset) available for trading
usdt_balance = balances.get(CCY, {}).get("free", 0.0)
btc_balance = balances.get(CRYPTOCCY, {}).get("free", 0.0)
# Locked balances present at session start belong to PRE-EXISTING orders not
# placed by this strategy (shared testnet account).  Captured so the equity
# chart can subtract them: snapshots mark free+locked, but only the locked the
# STRATEGY itself adds during the session (e.g. BTC resting in its own LIMIT
# SELL) should count toward its equity.  Foreign locked would otherwise inflate
# the curve — and since start_total is free-only, it causes a spurious jump.
locked_usdt_at_start = balances.get(CCY, {}).get("locked", 0.0)
locked_btc_at_start = balances.get(CRYPTOCCY, {}).get("locked", 0.0)
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
# 2b. Adaptive stop-loss threshold (mirrors backtest/pnl.py)
# ---------------------------------------------------------------------------
# Computed once at startup from STOP_LOSS_ROLLING_DAYS + buffer days of daily
# klines, then refreshed once per UTC day inside historical_analysis() via the
# refresher closure defined below.
#
# The "stop_loss_state" dict is the SHARED MUTABLE CONTAINER between the
# REST-aware websocket_main.py (writer) and the REST-agnostic AnalysisEngine
# (reader).  This keeps AnalysisEngine decoupled from Binance — it never
# imports the REST client; it only ever reads a float.
try:
    # Production daily klines (market_data_client) — mirrors backtest/signals.py,
    # which computes the same threshold from production data.
    _initial_sl_pct = compute_live_stop_loss_pct(
        market_data_client, SYMBOL, STOP_LOSS_ROLLING_DAYS, STOP_LOSS_STD_MULT
    )
    logging.info(
        "Adaptive stop-loss threshold at session start: %.4f%% "
        "(%d-day rolling std × %.1f).",
        _initial_sl_pct * 100,
        STOP_LOSS_ROLLING_DAYS,
        STOP_LOSS_STD_MULT,
    )
except Exception as _sl_exc:
    _initial_sl_pct = 0.0
    logging.warning(
        "Could not compute initial stop-loss threshold (%s) — stop-loss disabled "
        "for this session until next daily refresh succeeds.",
        _sl_exc,
    )

stop_loss_state: dict = {
    "pct": _initial_sl_pct,  # float — current threshold (e.g. 0.038 = 3.8%)
    "last_day_utc": int(time.time()) // 86400,  # int — UTC day of last refresh
}


def refresh_stop_loss_pct() -> float | None:
    """
    Closure injected into AnalysisEngine so historical_analysis() can refresh
    the threshold once per UTC day without ever touching the Binance REST
    client directly (decoupling).

    Returns:
        float | None: The new threshold, or ``None`` on failure (caller keeps
            the previous value).
    """
    try:
        return compute_live_stop_loss_pct(
            market_data_client, SYMBOL, STOP_LOSS_ROLLING_DAYS, STOP_LOSS_STD_MULT
        )
    except Exception as exc:
        logging.warning("Stop-loss daily refresh failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Macro-trend overlay — SHARED MUTABLE CONTAINER (writer: this file; reader:
# AnalysisEngine).  Mirrors the stop_loss_state pattern.  Computed once at
# startup from production daily klines and refreshed once per UTC day inside
# historical_analysis() via refresh_macro_trend() below.  When the overlay is
# disabled (MACRO_TREND_ENABLED = False) the initial state is forced "neutral"
# and NO refresher is injected, so the live path is fully inert — the clean
# ablation baseline, matching backtest/signals.py.
# ---------------------------------------------------------------------------
if MACRO_TREND_ENABLED:
    try:
        _initial_macro_state = compute_live_macro_trend(
            market_data_client,
            SYMBOL,
            MACRO_TREND_SMA_DAYS,
            MACRO_TREND_SLOPE_DAYS,
            MACRO_TREND_BAND_PCT,
        )
        logging.info(
            "Macro-trend overlay state at session start: '%s' "
            "(SMA %dd, slope %dd, band %.1f%%).",
            _initial_macro_state,
            MACRO_TREND_SMA_DAYS,
            MACRO_TREND_SLOPE_DAYS,
            MACRO_TREND_BAND_PCT * 100,
        )
    except Exception as _mt_exc:
        _initial_macro_state = "neutral"
        logging.warning(
            "Could not compute initial macro-trend state (%s) — defaulting to "
            "'neutral' (overlay inert until next daily refresh succeeds).",
            _mt_exc,
        )
else:
    _initial_macro_state = "neutral"
    logging.info("Macro-trend overlay DISABLED (MACRO_TREND_ENABLED=False).")

macro_trend_state: dict = {
    "state": _initial_macro_state,  # "down" / "neutral" / "up"
    "last_day_utc": int(time.time()) // 86400,  # int — UTC day of last refresh
}


def refresh_macro_trend() -> str | None:
    """
    Closure injected into AnalysisEngine so historical_analysis() can refresh
    the macro-trend state once per UTC day without touching the Binance REST
    client directly (same decoupling as refresh_stop_loss_pct).

    Returns:
        str | None: The new "down"/"neutral"/"up" state, or ``None`` on failure
            (caller keeps the previous value).
    """
    try:
        return compute_live_macro_trend(
            market_data_client,
            SYMBOL,
            MACRO_TREND_SMA_DAYS,
            MACRO_TREND_SLOPE_DAYS,
            MACRO_TREND_BAND_PCT,
        )
    except Exception as exc:
        logging.warning("Macro-trend daily refresh failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 3. Session duration
# ---------------------------------------------------------------------------
# At the default of 60 min the engine runs:
#   • low_latency_analysis → 60 × 60 / 1  = 3600 iterations  (every 1 s)
#   • historical_analysis  → 60 × 60 / 60 =   60 iterations  (every 60 s / 1 min)

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
snapshot = market_data_client.depth(symbol=SYMBOL, limit=SNAPSHOT_DEPTH)

# OrderBookState is the single source of truth shared by MessageHandler and
# AnalysisEngine.  Both classes receive the same instance so they operate on
# the same data and the same lock.
state = OrderBookState()
state.balance_status[CCY] = usdt_balance
state.balance_status[CRYPTOCCY] = btc_balance
# Seed locked balances too, so the FIRST equity snapshot already includes any
# foreign-locked amount (otherwise snapshot #0 is free-only while later ones add
# locked once the first REST refresh runs — the chart's locked_*_at_start
# subtraction would then over-correct t0 and spike the index down).
state.balance_locked[CCY] = locked_usdt_at_start
state.balance_locked[CRYPTOCCY] = locked_btc_at_start
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
handler = MessageHandler(state=state, rest_client=market_data_client)

# OrderExecutor owns its own WebSocket API connection for lower-latency
# order placement.  On connection open it sends session.logon (HMAC-signed)
# followed by userDataStream.subscribe, so that outboundAccountPosition
# push events keep state.balance_status current in real time — no listenKey
# needed.  If the testnet WS API is unreachable, execution falls back to REST
# and balances rely on the startup REST snapshot.
executor = OrderExecutor(
    state=state,
    stream_url="wss://ws-api.testnet.binance.vision/ws-api/v3",
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

# Resolve the position guard's stop-loss anchor.  Default: the session-start
# price (Phase-1 behaviour).  When carrying inventory (FLATTEN_ON_START=False)
# and a persisted position matches the BTC actually on the account, restore the
# TRUE cost basis saved at the previous shutdown so the stop-loss anchors
# correctly.  Purely additive and fail-safe: any miss falls back to the
# session-start price, and the strategy/engine code is unchanged — only the
# value handed to the existing initial_avg_entry_price parameter differs.
initial_avg_entry_price = btc_start_price or 0.0
if not FLATTEN_ON_START and btc_balance >= 0.0001:
    _persisted = load_position(LIVE_POSITION_STATE_PATH, symbol=SYMBOL)
    if (
        _persisted
        and _persisted.get("position_open")
        and abs(_persisted.get("btc_qty", 0.0) - btc_balance)
        <= max(1e-5, 0.01 * btc_balance)
    ):
        initial_avg_entry_price = float(_persisted["avg_entry_price"])
        logging.info(
            "Restored carried position cost basis %.2f (%.8f BTC, saved %s) — "
            "stop-loss will anchor here instead of the session-start price.",
            initial_avg_entry_price,
            _persisted["btc_qty"],
            _persisted.get("saved_at", "?"),
        )
    elif _persisted:
        logging.info(
            "Persisted position state found but not restored (open=%s, saved "
            "qty=%.8f vs account %.8f) — anchoring stop-loss at session-start "
            "price %.2f.",
            _persisted.get("position_open"),
            _persisted.get("btc_qty", 0.0),
            btc_balance,
            initial_avg_entry_price,
        )

engine = AnalysisEngine(
    state=state,
    stop_event=stop_event,
    executor=executor,
    regime_director=regime_director,
    stop_loss_state=stop_loss_state,
    refresh_stop_loss_fn=refresh_stop_loss_pct,
    macro_trend_state=macro_trend_state,
    # Overlay disabled → no refresher, so the state stays "neutral" forever.
    refresh_macro_trend_fn=(refresh_macro_trend if MACRO_TREND_ENABLED else None),
    initial_avg_entry_price=initial_avg_entry_price,
)

low_latency_thread = threading.Thread(
    target=engine.low_latency_analysis, daemon=True, name="low-latency-analysis"
)
hist_thread = threading.Thread(
    target=engine.historical_analysis, daemon=True, name="hist-analysis"
)


def _balance_refresh_loop() -> None:
    """
    Defense-in-depth balance poller (driver-side; touches no strategy code).

    When the WS user-data push is live, ``outboundAccountPosition`` keeps
    balances current.  When it is NOT (REST-only fallback), balances are only
    refreshed as a side effect of placing/cancelling an order — so during a long
    idle stretch with no qualifying signal, ``balance_status`` freezes and the
    end-of-session equity snapshots freeze with it.  This loop polls a fresh REST
    snapshot every ``BALANCE_REFRESH_INTERVAL`` seconds while in REST-only mode
    so the equity curve (and any balance-dependent logic) stays current.  It is a
    no-op once the WS push is confirmed healthy.
    """
    while not stop_event.wait(timeout=BALANCE_REFRESH_INTERVAL):
        if not executor._user_data_active:
            try:
                executor._refresh_balance_rest()
            except Exception as _bal_exc:
                logging.warning(
                    "Periodic REST balance refresh failed (%s) — non-fatal.",
                    _bal_exc,
                )


balance_refresh_thread = threading.Thread(
    target=_balance_refresh_loop, daemon=True, name="balance-refresh"
)

# ---------------------------------------------------------------------------
# 6. Open WebSocket stream
# ---------------------------------------------------------------------------
# PRODUCTION diff-depth stream (the SpotWebsocketStreamClient default;
# read-only, no auth — account type is irrelevant for market data).  It MUST
# be paired with the production market_data_client above for the seed and
# resync snapshots: update IDs only form a continuous sequence within one
# exchange, and a mismatched pair silently discards every stream frame.
# Trading orders and balance updates are routed through the TESTNET WebSocket
# API connection owned by OrderExecutor.
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
balance_refresh_thread.start()
logging.info("Analysis threads started. Running for %d minute(s)...\n", session_minutes)

# ---------------------------------------------------------------------------
# 7. Block the main thread for the session duration, then shut down cleanly
# ---------------------------------------------------------------------------
try:
    stop_event.wait(
        timeout=session_seconds
    )  # interruptible by KeyboardInterrupt; also wakes if stop_event is set early
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
    balance_refresh_thread.join(timeout=HTF_JOIN_TIMEOUT)
    logging.info("All threads stopped. Exiting.")

    # -----------------------------------------------------------------------
    # Cancel the session's still-open orders (frees locked funds; prevents
    # unattended fills after shutdown — e.g. 38 resting BUYs holding ~305k
    # USDT locked on 2026-07-08, 13 of which filled after the session ended)
    # -----------------------------------------------------------------------
    executor.cancel_session_open_orders()

    # -----------------------------------------------------------------------
    # Stop-loss summary
    # -----------------------------------------------------------------------
    logging.info(
        "Adaptive stop-loss summary: %d emergency exit(s) this session "
        "(final threshold: %.4f%%).",
        engine.n_stop_loss_fires,
        stop_loss_state.get("pct", 0.0) * 100,
    )

    # -----------------------------------------------------------------------
    # End-of-session order report
    # -----------------------------------------------------------------------
    executor.order_status_report()

    # -----------------------------------------------------------------------
    # End-of-session P&L chart (writes HTML to backtest/reporting/)
    # -----------------------------------------------------------------------
    # Skipped silently if fewer than 2 equity snapshots were captured (very
    # short or interrupted session) or if start_total_usdt could not be
    # computed at session start (REST ticker_price failure).
    try:
        from datetime import datetime as _dt, timezone as _tz
        from visualization.session_chart import generate_session_pnl_chart
        from config_parameters import BACKTEST_REPORTING_DIR

        _chart_path = os.path.join(
            BACKTEST_REPORTING_DIR,
            f"session_pnl_{_dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S')}.html",
        )
        generate_session_pnl_chart(
            snapshots=engine._equity_snapshots,
            orders=executor.placed_orders,
            start_total_usdt=start_total_usdt or 0.0,
            btc_start_price=btc_start_price or 0.0,
            out_path=_chart_path,
            locked_usdt_at_start=locked_usdt_at_start,
            locked_btc_at_start=locked_btc_at_start,
        )
    except Exception as _chart_exc:
        logging.warning(
            "Session P&L chart generation failed (%s) — non-fatal.",
            _chart_exc,
        )

    # -----------------------------------------------------------------------
    # End-of-session balance report
    # -----------------------------------------------------------------------
    try:
        final_info = rest_client.account(recvWindow=RECV_WINDOW)
        # free + locked: funds resting in the strategy's own open LIMIT orders
        # still belong to the portfolio (free-only under-reported the session
        # by the full locked amount, e.g. -98.97% on 2026-07-08).  Foreign
        # locked captured at session start is subtracted, mirroring the equity
        # chart's locked_*_at_start correction.
        final_balances = {
            item["asset"]: float(item["free"]) + float(item["locked"])
            for item in final_info["balances"]
            if item["asset"] in (CCY, CRYPTOCCY)
        }
        final_usdt = final_balances.get(CCY, 0.0) - locked_usdt_at_start
        final_btc = final_balances.get(CRYPTOCCY, 0.0) - locked_btc_at_start
        d_usdt = final_usdt - usdt_balance
        d_btc = final_btc - btc_balance

        # Fetch current BTC price for portfolio valuation.
        # noinspection PyArgumentList
        btc_end_price = float(rest_client.ticker_price(symbol=SYMBOL)["price"])
        end_total_usdt = final_usdt + final_btc * btc_end_price

        # Persist the open-position state (cost basis) for the next restart.
        # Decorative + fail-safe: a failure here never affects the session, and
        # the engine attributes are only READ (strategy code untouched).  Uses
        # free + locked BTC so a position resting in a LIMIT order still counts;
        # honored on startup only when FLATTEN_ON_START is False.
        try:
            _btc_total = sum(
                float(it["free"]) + float(it["locked"])
                for it in final_info["balances"]
                if it["asset"] == CRYPTOCCY
            )
            save_position(
                LIVE_POSITION_STATE_PATH,
                position_open=engine._position_open,
                avg_entry_price=engine._avg_entry_price,
                btc_qty=_btc_total,
                symbol=SYMBOL,
            )
        except Exception as _ps_exc:
            logging.warning("Position state save failed (%s) — non-fatal.", _ps_exc)

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
            trading_pnl = d_usdt + d_btc * btc_end_price
            price_pnl = btc_balance * (btc_end_price - btc_start_price)
            total_pnl = end_total_usdt - start_total_usdt
            pct_return = total_pnl / start_total_usdt * 100 if start_total_usdt else 0.0
            # Buy & Hold benchmark — identical to the chart's bnh_index: what the
            # whole starting equity would have returned if held as BTC for the
            # entire session.  Reference only (NOT part of the strategy's P&L);
            # a positive "vs B&H" means the strategy beat simply holding BTC.
            bnh_return_pct = (btc_end_price / btc_start_price - 1.0) * 100
            strategy_vs_bnh = pct_return - bnh_return_pct
            # Sanity-check: trading_pnl + price_pnl == total_pnl (algebraic identity).
            # Any residual is floating-point rounding noise — shown for transparency.
            residual = total_pnl - (trading_pnl + price_pnl)
            # When the session carried inventory (FLATTEN_ON_START = False),
            # component B is non-zero and worth calling out — it is market drift
            # on the carried bag, NOT trading skill.  Empty when starting flat.
            b_note = (
                "  ← carried inventory drifted with the market (not trading P&L)"
                if abs(price_pnl) >= 0.01
                else ""
            )

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
                "    B  Price P&L     : %+.2f  (%.8f BTC × %+.2f BTC/USDT)%s\n"
                "       Starting BTC holding × price change this session.\n"
                "    ─────────────────────────────────────────────────────\n"
                "    A + B  Total P&L : %+.2f  (%+.3f %%)\n"
                "\n"
                "  Benchmark  [reference only — not part of P&L; matches chart]\n"
                "    Buy & Hold return : %+.3f %%  (all start equity held as BTC)\n"
                "    Strategy vs B&H   : %+.3f %%  (positive = beat holding BTC)\n"
                "====================================================",
                CCY,
                usdt_balance,
                final_usdt,
                d_usdt,
                CRYPTOCCY,
                btc_balance,
                final_btc,
                d_btc,
                btc_start_price,
                btc_end_price,
                btc_end_price - btc_start_price,
                start_total_usdt,
                end_total_usdt,
                trading_pnl,
                price_pnl,
                btc_balance,
                btc_end_price - btc_start_price,  # price change (BTC/USDT)
                b_note,
                total_pnl,
                pct_return,
                bnh_return_pct,
                strategy_vs_bnh,
            )
            if abs(residual) > 0.01:
                logging.warning(
                    "P&L residual %.4f (floating-point rounding).", residual
                )
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
                CCY,
                usdt_balance,
                final_usdt,
                d_usdt,
                CRYPTOCCY,
                btc_balance,
                final_btc,
                d_btc,
                btc_end_price,
                end_total_usdt,
                CCY,
            )
    except Exception as e:
        logging.error("Could not fetch final balance: %s", e)
