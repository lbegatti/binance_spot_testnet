# Binance Spot Testnet — Order Book Analysis

---

> ## ⚠️ DISCLAIMER — PLEASE READ BEFORE USING THIS PROJECT
>
> **This project is a personal side project and a technical learning exercise. It is provided strictly for educational and research purposes.**
>
> - 🚫 **Not financial advice.** Nothing in this repository constitutes financial advice, investment advice, trading advice, or any other form of advice. The strategies, signals, metrics, and outputs produced by this code should **not** be interpreted as recommendations to buy, sell, or hold any financial instrument.
>
> - 🚫 **Not a solicitation.** This project is not a solicitation or offer to trade any asset, cryptocurrency, or financial product — on Binance or any other platform.
>
> - 🚫 **Not legal advice.** Nothing in this repository constitutes legal advice of any kind. Trading cryptocurrency may be regulated, restricted, or prohibited in your jurisdiction. It is your sole responsibility to ensure compliance with all applicable laws and regulations before engaging in any trading activity.
>
> - 🧪 **Testnet / paper trading only.** All code in this repository is designed and tested exclusively against the **Binance Spot Testnet** (`testnet.binance.vision`), which uses **simulated funds with no real monetary value**. No real Binance account, real funds, or real orders are involved in any way.
>
> - 🏢 **No employer affiliation.** This is an independent personal project. It does not represent, involve, or reflect the views, work, products, or interests of my current or any previous employer in any way.
>
> - 🤖 **AI-assisted development.** Parts of this codebase, including implementation, documentation, and code review, were produced with the assistance of AI language models (e.g. GitHub Copilot / GPT-4 class models) to speed up development and act as a four-eyes check. AI models can be wrong, produce misleading output, introduce subtle bugs, or generate plausible-sounding but incorrect logic. **All AI-generated content should be treated as unverified until independently reviewed.** The author takes no responsibility for errors or omissions that originate from, or were not caught by, AI-assisted tooling.
>
> - ⚡ **Use at your own risk.** If you choose to adapt or extend this code to interact with a real Binance account or any live trading environment, you do so entirely **at your own risk**. The author accepts no liability for any financial loss, legal consequence, or damage arising from such use.
>
> **By cloning, forking, or using this repository you acknowledge that you have read and understood this disclaimer.**

---

# Binance Spot Testnet — Order Book Analysis

A Python toolkit for order book analysis on the **Binance Spot Testnet**.  
Two execution modes are available:

| Mode | Entry point | Data source | Symbol |
|------|-------------|-------------|--------|
| **REST** (polling) | `restapi_main.py` | `client.get_order_book()` snapshots | BTCUSDT |
| **WebSocket** (real-time) | `websocket_main.py` | `diff_book_depth` stream (100 ms) | BTCUSDT |

The REST path is fully wired — metrics → indicators → scores → best quote.  
The WebSocket path runs the full strategy pipeline (metrics → indicators → scores) in near real-time via the `AnalysisEngine`.

---

## Project Structure

```
binance_spot_testnet/
├── config_parameters.py               # Central configuration — all tunable constants in one place
├── restapi_main.py                    # REST orchestration — loops over depth limits
├── websocket_main.py                  # WebSocket — real-time local order book + session driver
├── README.md
├── BACKTESTING.md                     # Backtesting design, pseudo-code, and implementation roadmap
├── system_architecture.txt
│
├── core/                              # Shared state and data ingestion
│   ├── __init__.py
│   ├── order_book_state.py            # Shared state container (local book + history + balances + locks)
│   └── message_handler.py             # WebSocket callbacks — maintains local book and balances in real time
│
├── strategy/                          # Analysis and scoring pipeline
│   ├── __init__.py
│   ├── analysis.py                    # AnalysisEngine — low-latency (1 s) and historical (1 min) loops
│   ├── book_utils.py                  # Shared order-book utilities (build_levels, collect_candidates, select_best_opportunity)
│   ├── regime_director.py             # RegimeDirector — HMM regime detection (fitted pre-session, refreshed every 1 min)
│   ├── best_quote_calculator.py       # Live spread printer — prints best_bid | best_ask on every tick
│   ├── metrics.py                     # Order book metric calculations
│   ├── indicators.py                  # Strategy-specific indicator columns
│   ├── scores.py                      # Weighted opportunity scoring
│   └── quotes.py                      # Best quote selection logic
│
├── execution/                         # Order placement
│   ├── __init__.py
│   └── order_executor.py             # OrderExecutor — LIMIT GTC orders via WebSocket API
│
├── backtest/                          # Offline backtesting framework (see BACKTESTING.md)
│   ├── __init__.py
│   ├── data.py                        # Historical kline downloader (30 days, 1 m)
│   ├── synthetic_book.py              # Synthetic 50-level order book builder (per kline row)
│   ├── signals.py                     # Signal replay loop — full pipeline + regime & VWAP filters
│   ├── pnl.py                         # P&L simulation — balance guard, fill price, equity curve, metrics
│   ├── runner.py                      # Top-level runner — chains all modules; opt-in plot/CSV export
│   ├── visualization.py               # Step 7 — interactive Plotly chart (6-panel); run via run_backtest(plot=True)
│   ├── sensitivity.py                 # Step 8 — OAT / full-grid sensitivity sweep; writes best_params.json
│   ├── regime_validation.py           # Offline long-horizon HMM validation — python -m backtest.regime_validation
│   └── reporting/                     # Console report formatting and CSV export (AI-authored)
│       ├── __init__.py
│       └── formatters.py              # fmt(), print_report(), save_csv() — public helpers
│
└── visualization/                     # Plotting utilities
    ├── __init__.py
    └── plot_helpers.py                # Plotly visualisations (depth, OHLC)
```

---

## Configuration (`config_parameters.py`)

All tunable constants are centralised in `config_parameters.py`. Edit this file to change behaviour across the entire project without touching any logic files.

| Group | Constant | Default | Description |
|---|---|---|---|
| **Symbol** | `SYMBOL` | `"BTCUSDT"` | Trading pair used across all REST and WebSocket calls |
| **Symbol** | `CCY` | `"USDT"` | Quote currency |
| **Symbol** | `CRYPTOCCY` | `"BTC"` | Base / crypto currency |
| **Order book state** | `HISTORY_MAXLEN` | `3000` | Max snapshots in `history_order_book` — at 100 ms intervals this covers ~5 min.  Each entry: `{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}` (all numeric) |
| **Order book state** | `N_LEVELS` | `50` | Number of order book levels used in `low_latency_analysis` |
| **Analysis cadence** | `HFT_INTERVAL` | `1` s | Time between low-latency evaluations |
| **Analysis cadence** | `HIST_INTERVAL` | `60` s | Time between historical analyses (1 min) |
| **Analysis cadence** | `MIN_SNAPSHOTS` | `100` | Minimum snapshots required before historical analysis runs |
| **WebSocket session** | `DEFAULT_SESSION_MINUTES` | `10` min | Default session length (fixed — no startup prompt) |
| **WebSocket session** | `HTF_JOIN_TIMEOUT` | `10` s | Max wait for `low_latency_analysis` thread on shutdown |
| **WebSocket session** | `HIST_JOIN_TIMEOUT` | `15` s | Max wait for `historical_analysis` thread on shutdown |
| **Binance connection** | `RECV_WINDOW` | `5000` ms | Binance REST request validity window |
| **Binance connection** | `SNAPSHOT_DEPTH` | `100` | Order book levels in the seed snapshot |
| **Binance connection** | `WS_SPEED` | `100` ms | WebSocket diff-depth update interval |
| **Quote throttle** | `QUOTE_EVERY_N_TICKS` | `10` | Ticks between `calculate_best_quote()` calls.  At `WS_SPEED=100 ms`, 10 ticks ≈ 1 s |
| **HMM** | `HMM_FEATURE_COLS` | `["return", "volatility", "obi_proxy", "trade_density"]` | Feature columns fed to the `GaussianHMM` |
| **HMM** | `HMM_N_ITERATIONS` | `1000` | Max EM iterations per model fit |
| **HMM** | `HMM_MAX_REGIMES` | `4` | Upper bound on hidden states evaluated during BIC search (2 … 4) |
| **HMM** | `HMM_RANDOM_STATE` | `46` | Random seed for reproducible HMM initialisation |
| **HMM** | `HMM_INTERVAL` | `Client.KLINE_INTERVAL_1MINUTE` | Kline granularity for regime detection (1 m — intra-session resolution) |
| **HMM** | `HMM_LOOKBACK` | `"2 hours ago UTC"` | Kline history window (~120 rows at 1 m — captures recent intra-day regime without being stale) |
| **HMM** | `HMM_MIN_COVAR` | `1e-3` | Regularisation floor for covariance matrices — prevents positive-definite errors |
| **HMM** | `HMM_TRAIN_ROWS` | `80` | Number of kline rows used to **fit** the model (~⅔ of the 120-row window).  The remaining ~40 rows are genuinely out-of-sample during training — `predict()` / `predict_proba()` run on the full window |
| **HMM** | `HMM_MIN_CONFIDENCE` | `0.70` | Minimum posterior probability (`predict_proba()[-1][current_regime]`) required to allow an order.  Below this threshold the regime signal is treated as ambiguous and both BUY and SELL are skipped |
| **HMM** | `HMM_REFIT_INTERVAL` | `300` s | Cadence of **full** HMM re-fit inside `historical_analysis()`.  Between re-fits only a cheap Viterbi prediction runs.  Must be a multiple of `HIST_INTERVAL` |
| **Order report** | `ORDER_REPORT_LIMIT` | `100` | Max orders shown at head *and* tail of the end-of-session report.  Middle block collapsed when total > 2 × limit |
| **Backtesting** | `BACKTEST_LOOKBACK` | `"180 days ago UTC"` | How far back to fetch klines for the backtest dataset (~259,200 candles at 1 m).  A 180-day window captures a broad range of market regimes, making both signal replay and regime-validation results more statistically meaningful. |
| **Backtesting** | `VOLUME_DECAY_FACTOR` | `0.80` | Exponential decay factor for synthetic order-book depth — each level retains 80 % of the previous level's volume |
| **Backtesting** | `HMM_LOOKBACK_ROWS` | `120` | Number of kline rows used as the HMM warm-up window in the backtest (2 h at 1 m — matches `HMM_LOOKBACK`) |
| **Backtesting** | `VWAP_WINDOW` | `5` | Rolling window size (in candles) for the backtest VWAP computation (5 candles = 5 min at 1 m — matches live VWAP cadence) |
| **Backtesting** | `REFIT_EVERY` | `60` | Iterations between full HMM BIC re-fits during the signal loop.  At 1 m resolution this means one full re-fit per hour — frequent enough to track regime shifts, cheap enough to keep backtest runtime manageable (~720 refits over 30 days vs ~8,600 at the previous value of 5) |
| **Backtesting P&L** | `BACKTEST_INITIAL_CAPITAL` | `5_000.0` | Starting USDT balance for the simulation |
| **Backtesting P&L** | `BACKTEST_INITIAL_BTC` | `0.0735` | Starting BTC balance for the simulation (set > 0 to simulate an existing position) |
| **Backtesting P&L** | `BACKTEST_FEE_RATE` | `0.001` | Taker fee fraction per side (0.10 %) |
| **Backtesting P&L** | `BACKTEST_RISK_FREE_RATE` | `0.0` | Annualised risk-free rate for Sharpe / Sortino denominator (0.0 = no adjustment; set to e.g. 0.04 for a 4 % T-bill proxy) |
| **Backtesting P&L** | `BACKTEST_MAX_ROWS` | `500` | Max replay candles in debug mode (`None` for full run) |
| **Sensitivity** | `SENSITIVITY_REFIT_EVERY` | `480` | HMM refit cadence used **only** inside `sensitivity.py` (8 h at 1 m → ~90 refits/run vs ~360 at the default, ~4× speedup). `config_parameters.py` defaults and the live system are never affected. |
| **Sensitivity** | `SENSITIVITY_LOOKBACK` | `"30 days ago UTC"` | Data-fetch window used **only** by `sensitivity.py` (~43,200 rows). Using the full 180-day window per run would make each OAT run 6× slower than `run_backtest.py`. `run_backtest.py` always uses `BACKTEST_LOOKBACK`. |
| **Sensitivity** | `SENSITIVITY_PREDICT_EVERY` | `5` | Viterbi predict cadence used **only** by `sensitivity.py`. Between refit calls, `predict_current_regime()` is called only every 5 candles; the last known regime label is reused otherwise, cutting Viterbi overhead ~5×. `run_backtest.py` always predicts every candle. |

**Imported by:**
- `core/order_book_state.py` — `HISTORY_MAXLEN`, `CRYPTOCCY`, `CCY`
- `core/message_handler.py` — `CRYPTOCCY`, `CCY`, `QUOTE_EVERY_N_TICKS`
- `strategy/analysis.py` — `HFT_INTERVAL`, `HIST_INTERVAL`, `MIN_SNAPSHOTS`, `N_LEVELS`, `CCY`, `CRYPTOCCY`, `HMM_REFIT_INTERVAL`, `HMM_MIN_CONFIDENCE`
- `strategy/book_utils.py` — `N_LEVELS`
- `strategy/regime_director.py` — `HMM_FEATURE_COLS`, `HMM_N_ITERATIONS`, `HMM_RANDOM_STATE`, `HMM_MAX_REGIMES`, `HMM_INTERVAL`, `HMM_LOOKBACK`, `HMM_MIN_COVAR`, `HMM_TRAIN_ROWS`
- `execution/order_executor.py` — `SYMBOL`, `CRYPTOCCY`, `CCY`, `RECV_WINDOW`, `ORDER_REPORT_LIMIT`
- `backtest/signals.py` — `HMM_LOOKBACK_ROWS`, `VWAP_WINDOW`, `REFIT_EVERY`, `BACKTEST_MAX_ROWS`, `BACKTEST_LOOKBACK`, `HMM_MIN_CONFIDENCE`, `BACKTEST_FILL_SPREAD_BPS`, `HMM_MAX_REGIMES`
- `backtest/data.py` — `SYMBOL`, `BACKTEST_LOOKBACK`
- `backtest/synthetic_book.py` — `N_LEVELS`, `VOLUME_DECAY_FACTOR`
- `backtest/pnl.py` — `BACKTEST_FEE_RATE`, `BACKTEST_INITIAL_BTC`, `BACKTEST_INITIAL_CAPITAL`, `BACKTEST_RISK_FREE_RATE`, `BACKTEST_MAX_POSITION_PCT`, `HMM_MIN_CONFIDENCE`
- `backtest/runner.py` — `BACKTEST_FEE_RATE`, `BACKTEST_INITIAL_BTC`, `BACKTEST_INITIAL_CAPITAL`, `SYMBOL`
- `backtest/reporting/formatters.py` — `BACKTEST_FEE_RATE`, `BACKTEST_INITIAL_BTC`, `BACKTEST_INITIAL_CAPITAL`, `SYMBOL`
- `backtest/sensitivity.py` — `SENSITIVITY_REFIT_EVERY`, `SENSITIVITY_LOOKBACK`, `SENSITIVITY_PREDICT_EVERY`
- `websocket_main.py` — `SYMBOL`, `CCY`, `CRYPTOCCY`, and all session / connection constants

---

## Setup

1. **Install dependencies**

   ```bash
   pip install binance-connector python-dotenv pandas numpy plotly
   ```

2. **Create a `.env` file** in the project root:

   ```
   BINANCE_TESTNET_API_KEY=your_api_key
   BINANCE_TESTNET_SECRET_KEY=your_secret_key
   ```

   Keys are generated at <https://testnet.binance.vision/>.

3. **Run**

   ```bash
   # REST (static snapshots)
   python restapi_main.py

   # WebSocket (real-time)
   python websocket_main.py
   ```

---

## Pipeline Overview

### REST path (`restapi_main.py`)

```
Binance REST API
    │  get_order_book()
    ▼
┌──────────────────┐     ┌────────────────┐     ┌──────────────┐     ┌────────────┐
│ restapi_main.py  │────▶│  metrics.py    │────▶│ indicators.py│────▶│  scores.py │
│  (orchestrate)   │     │  (enrich df)   │     │ (strategy)   │     │  (score)   │
└──────────────────┘     └────────────────┘     └──────────────┘     └────────────┘
                                                                           │
                                                                           ▼
                                                                   ┌──────────────┐
                                                                   │  quotes.py   │
                                                                   │ (best quote) │
                                                                   └──────────────┘
```

### WebSocket path (`websocket_main.py`)

```
Binance REST API                   Binance WebSocket (production)
    │  depth() snapshot                │  diff_book_depth (100 ms)
    ▼                                  ▼
┌──────────────────┐       ┌──────────────────────┐
│  OrderBookState  │◀──────│   MessageHandler      │
│  (shared state)  │       │  .handle_depth_msg()  │
│                  │       └──────────────────────┘
│  • local_book    │
│  • history_book  │       ┌──────────────────────┐
│  • balance_status│◀──────│   AnalysisEngine      │  (read-only)
│  • thread_lock   │       │  .low_latency_analysis()  │
│  • thread_       │       │   (balance check +        │
│    balance_lock  │       │    order book strategy     │
│                  │       │    + VWAP filter)          │
│                  │       │  .historical_analysis()    │
│                  │       └──────────┬───────────┘
│                  │                  │  opportunity tuple
│                  │                  ▼
│                  │       ┌──────────────────────────────┐   Binance WS API (testnet)
│                  │◀──────│   OrderExecutor               │──▶ wss://testnet.binance.vision
│                  │       │  .execute()                   │    /ws-api/v3
│  (balance_status │       │   (balance guard + LIMIT GTC) │
│   updated by     │       │                               │◀── handle_order_response()
│   push events)   │       │  session.logon                │    (routes: orders + balance
│                  │       │  userDataStream.subscribe     │     push events)
│                  │       │  _handle_balance_update()     │
│                  │       └──────────────────────────────┘
│                  │                  ▲
│                  │       ┌──────────────────────┐
│                  │       │  websocket_main      │
│                  │       │  (session driver)     │
└──────────────────┘       └──────────────────────┘
```

**Component responsibilities:**

| Class / Module | File | Role |
|---|---|---|
| *(constants)* | `config_parameters.py` | Single source of truth for all tunable constants (`SYMBOL`, `CCY`, `CRYPTOCCY`, intervals, depths, timeouts, HMM parameters) — imported by every package |
| `OrderBookState` | `core/order_book_state.py` | Single source of truth — owns `local_book`, `history_order_book`, `balance_status`, `thread_lock`, and `thread_balance_lock` |
| `MessageHandler` | `core/message_handler.py` | One active WebSocket callback: `handle_depth_message` (merges diff-depth ticks into `local_book`, appends snapshots, calls `calculate_best_quote` every 10th tick).  `handle_balance_message` is preserved but superseded — see `OrderExecutor` |
| `RegimeDirector` | `strategy/regime_director.py` | Detects the current market regime via a `GaussianHMM` fitted on recent 1-minute klines.  Features are z-score scaled (`StandardScaler`, fitted on the first `HMM_TRAIN_ROWS` rows only).  Fitted once at pre-session startup; then updated every `HIST_INTERVAL` (60 s) via a **two-speed** scheme: cheap Viterbi prediction on most iterations, full model re-fit every `HMM_REFIT_INTERVAL` (300 s).  Exposes `regime_label` (`"trending_up"`, `"trending_down"`, `"high_volatility"`, `"neutral"`) and `regime_confidence` (posterior probability from `predict_proba()`), both protected by `_regime_lock` |
| `AnalysisEngine` | `strategy/analysis.py` | Runs two background loops (`low_latency_analysis` and `historical_analysis`) that read from `OrderBookState` via the shared locks; applies VWAP, regime-confidence, and regime-direction filters before delegating order placement to `OrderExecutor` |
| `OrderExecutor` | `execution/order_executor.py` | Places LIMIT GTC orders **and** maintains real-time balance updates via a single Binance WebSocket API connection.  On connect: `session.logon` (HMAC-signed) → `userDataStream.subscribe` → receives `outboundAccountPosition` push events on the **same** socket.  Falls back to REST for orders if WS is unavailable; balances fall back to startup REST snapshot |
| `websocket_main` | `websocket_main.py` | Session driver — instantiates all classes, seeds initial balances into `state`, runs pre-session regime detection, opens WebSocket streams, starts all threads, manages session lifetime and shutdown |

**How the components interact:**

1. `websocket_main.py` creates a single `OrderBookState` instance and injects it into both `MessageHandler` and `AnalysisEngine`.  After construction it immediately seeds `state.balance_status` with the REST-fetched balances before any thread starts.
2. Order-book data (`local_book`, `history_order_book`) is serialised through `state.thread_lock`.  `MessageHandler.handle_depth_message` acquires it to write; `AnalysisEngine.low_latency_analysis` acquires it to take a read-only copy and releases it before any heavy computation.
3. Balance data (`balance_status`) is serialised through the dedicated `state.thread_balance_lock`, completely independent of `thread_lock`.  This prevents the high-frequency WebSocket order-book path (every 100 ms) from blocking on the lower-frequency balance path.
4. `MessageHandler` is the **only writer** to order-book data in `OrderBookState`.  `OrderExecutor._handle_balance_update` is the **only writer** to `balance_status`.  `AnalysisEngine` is **read-only** — it copies data under the appropriate lock and immediately releases it.
5. `AnalysisEngine` delegates order placement to `OrderExecutor`.  When `_select_best_opportunity()` returns a non-`None` 8-element tuple `(level_idx, score, delta, total_depth, obi, micro_price, bq, aq)`, the engine calls `executor.execute("BUY", best_buy)` or `executor.execute("SELL", best_sell)`.  `OrderExecutor` validates the strategy, checks balances under `thread_balance_lock`, computes quantity (`aq` for BUY, `bq` for SELL), and sends a LIMIT GTC order via its own `SpotWebsocketAPIClient`.  The response arrives asynchronously in `handle_order_response`.
6. **VWAP momentum-confirmation filter** — `AnalysisEngine` owns a private `_vwap_lock` plus two attributes `_bid_vwap` and `_ask_vwap` (initially `None`).  `historical_analysis` computes both VWAPs from `history_order_book` every 1 min and publishes them under `_vwap_lock`.  `low_latency_analysis` reads them under the same lock on every iteration and gates order execution:
   - **BUY**: execute only if `_ask_vwap is None` (first ~1 min) **or** `micro_price > ask_vwap` (upward momentum confirmed).
   - **SELL**: execute only if `_bid_vwap is None` (first ~1 min) **or** `micro_price < bid_vwap` (downward momentum confirmed).
   - This logic may be inverted for a buy-the-dip / mean-reversion strategy — see the *Historical VWAP & Momentum Filter* section.
7. **Real-time balance tracking (no listenKey)** — `OrderExecutor` owns the single `SpotWebsocketAPIClient` connection (`wss://testnet.binance.vision/ws-api/v3`).  On socket open (`on_open` callback) it sends a signed `session.logon` frame.  On success it immediately sends `userDataStream.subscribe`.  Once confirmed, Binance pushes `outboundAccountPosition` events on the **same** connection whenever a balance changes (e.g. after an order fill).  `handle_order_response` routes these push events (frames with no `"id"` field) to `_handle_balance_update`, which writes to `state.balance_status` under `thread_balance_lock`.  If the testnet doesn't support `session.logon` or `userDataStream.subscribe`, the executor falls back silently to the REST snapshot taken at session startup.
8. **HMM regime filter** — `websocket_main.py` instantiates `RegimeDirector` and calls `get_klines_data()` → `select_hmm_model()` → `assign_regime_labels()` **before** the analysis threads start, so `regime_label` and `regime_confidence` are never `None` on the first `low_latency_analysis` iteration.  Every `historical_analysis` iteration applies a **two-speed** update: cheap Viterbi prediction (`predict_current_regime()`) on most iterations, full model re-fit (`select_hmm_model()`) every `HMM_REFIT_INTERVAL` (300 s, i.e. every 5th iteration).  Features are **z-score scaled** via `StandardScaler` before every `fit()` / `predict()` / `predict_proba()` call — the scaler is fitted on the first `HMM_TRAIN_ROWS` (80) rows only (in-sample) and applied to the full window (out-of-sample) to avoid data leakage.  `assign_regime_labels()` runs inside `_regime_lock` on every iteration.  `low_latency_analysis` reads both `regime_label` and `regime_confidence` under `_regime_lock` and applies **two sequential gates** before any order: (a) **confidence gate** — if `regime_confidence < HMM_MIN_CONFIDENCE` (0.70), both BUY and SELL are skipped regardless of label (the model is uncertain); (b) **direction gate** — BUY orders are suppressed in `"trending_down"` or `"high_volatility"` regimes; SELL orders are suppressed in `"trending_up"` or `"high_volatility"` regimes.

---

## Session Duration

### Rationale

Both analysis loops in `AnalysisEngine` are designed to run indefinitely:

| Loop | Cadence | Purpose |
|------|---------|---------|
| `low_latency_analysis` | every **1 s** | Near-real-time best bid/ask evaluation |
| `historical_analysis` | every **1 min** | VWAP computation over the rolling snapshot window |

Rather than running forever, `websocket_main.py` uses a fixed session duration set by `DEFAULT_SESSION_MINUTES` (no startup prompt).

The **default of 10 minutes** is chosen deliberately:

| Metric | Value at 10 min |
|--------|----------------|
| Low-latency iterations (`low_latency_analysis`) | $10 \times 60 / 1 = \mathbf{600}$ |
| Historical iterations (`historical_analysis`) | $10 \times 60 / 60 = \mathbf{10}$ |
| Order book snapshots in history | up to $10 \times 60 \times 10 = \mathbf{6{,}000}$ ticks (capped at `maxlen=3000` ≈ last 5 min) |

When the session duration elapses, `websocket_main.py` sets `stop_event`, calls `ws_client.stop()` to close the stream cleanly, and joins both analysis threads (with timeouts of 10 s and 15 s respectively). A `KeyboardInterrupt` (Ctrl-C) triggers the same shutdown path early.

### Thread Timeline (default 10-min session)

```
t=0s      Both threads start
          ├── low_latency: runs immediately, then every 1 s
          └── historical:  sleeps 1 min first (stop_event.wait(HIST_INTERVAL))

t=1s      low_latency iteration #1
t=2s      low_latency iteration #2
...
t=1min    low_latency iteration #60
          historical iteration #1 → computes bid_vwap / ask_vwap,
            publishes under _vwap_lock
          ↓ from this point, low_latency reads the VWAP and applies the filter

t=2min    low_latency iteration #120
          historical iteration #2 → refreshes VWAP

t=5min    low_latency iteration #300
          historical iteration #5 → deque now full (3 000 entries), true rolling window
...
t=10min   low_latency iteration #600
          historical iteration #10 → refreshes VWAP one last time
          stop_event set → both threads exit
```

### Deque Fill-Up (`history_order_book`)

The deque size is driven by the **WebSocket tick rate** (~10 entries/sec at 100 ms), not by the historical analysis interval.  `historical_analysis` only *reads* the deque — it never clears it.

| Time elapsed | WebSocket ticks | Deque size | Historical iterations |
|---|---|---|---|
| 1 min | ~600 | 600 | 1st runs (reads 600 entries) |
| 3 min | ~1 800 | 1 800 | 3rd runs (reads 1 800 entries) |
| 5 min | ~3 000 | **3 000 (full)** | 5th runs (reads 3 000 entries) |
| 10 min | ~6 000 sent | **3 000 (capped — oldest evicted)** | 10th runs (reads 3 000 entries) |
| 10 min | ~6 000 sent | **3 000 (capped)** | 10th runs (reads 3 000 entries) |

After ~5 minutes the deque hits `maxlen=3000` and becomes a true **rolling window** of the last ~5 minutes. Each `historical_analysis` iteration operates on whatever is currently in the window — not a fixed block.

---

## WebSocket Execution Flow (`websocket_main.py`)

1. Load API keys from `.env`.
2. Connect to Binance Testnet via `binance-connector` REST client.
3. Consolidate non-BTC/USDT balances into USDT via market sell orders.
4. Seed `state.balance_status` with the REST-fetched `usdt_balance` and `btc_balance` before any thread starts.
5. Session duration fixed at `DEFAULT_SESSION_MINUTES` (10 min) — no user prompt.
6. Fetch a depth snapshot for **BTCUSDT** (100 levels) to seed `OrderBookState.local_book`.
7. **Pre-session regime detection** — instantiate `RegimeDirector` and call `get_klines_data()` → `select_hmm_model()` → `assign_regime_labels()`.  This downloads ~120 rows of 1-minute klines (last 2 hours), fits `GaussianHMM` models for 2–4 states, selects the best by BIC, and assigns `regime_label` before any thread starts.
8. Instantiate `OrderBookState`, `MessageHandler`, `OrderExecutor`, and `AnalysisEngine` (with `regime_director` injected), wiring the shared state.  `OrderExecutor` creates its own `SpotWebsocketAPIClient` internally (connected to `wss://testnet.binance.vision/ws-api/v3`) with `on_open=self._on_ws_open` and `on_message=self.handle_order_response`.  On socket open it automatically sends `session.logon` → `userDataStream.subscribe` to enable real-time balance push events on the same connection.  Set `stop_event = threading.Event()`.
9. Open one `SpotWebsocketStreamClient` (`ws_client`) on the production stream endpoint; subscribe to `diff_book_depth` at 100 ms intervals; callback: `handle_depth_message`.
10. Wait 1 second for the first diff-depth messages to arrive and populate `local_book["bids"]`.
11. Start daemon threads:
    - `low_latency_thread` → `AnalysisEngine.low_latency_analysis`
    - `hist_thread` → `AnalysisEngine.historical_analysis`
12. On each incoming depth message (`MessageHandler.handle_depth_message`):
    - Skip the initial subscription confirmation (`{"id": 1, "result": null}`).
    - Drop stale updates where `data["u"] <= state.local_book["lastUpdateId"]`.
    - Acquire `state.thread_lock`, apply bid/ask deltas, update `lastUpdateId`, append snapshot (`{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}`) to `state.history_order_book`, release lock.
    - Call `calculate_best_quote()` every `QUOTE_EVERY_N_TICKS` ticks (~1 s) with the updated book (prints live spread to stdout).  The local book is still updated on every tick.
13. On each incoming `outboundAccountPosition` push event (`OrderExecutor.handle_order_response` → `_handle_balance_update`):
    - Under `state.thread_balance_lock`, update `state.balance_status` for each tracked asset (`CRYPTOCCY`, `CCY`) using the `"f"` (free) field.
14. After `session_seconds`, set `stop_event`, stop `ws_client`, stop `executor` (closes the WS API + user-data connection), and join both analysis threads (10 s low-latency, 15 s historical).  A `KeyboardInterrupt` (Ctrl-C) triggers the same shutdown path early.

## Notation

| Symbol | Column name | Description |
|--------|-------------|-------------|
| $P_b$, $Q_b$ | `bid_price`, `bid_quantity` | Best bid price and quantity at a level |
| $P_a$, $Q_a$ | `ask_price`, `ask_quantity` | Best ask price and quantity at a level |
| $bq$ | `bq` | Individual bid quantity at a level (carried through the opportunity pipeline for order sizing — SELL uses this) |
| $aq$ | `aq` | Individual ask quantity at a level (carried through the opportunity pipeline for order sizing — BUY uses this) |
| $D$ | `total_depth` | Sum of bid and ask quantities |
| $P_{\text{mid}}$ | `mid_price` | Arithmetic mid-price |
| $P_{\mu}$ | `micro_price` | Volume-weighted micro-price |
| $\text{OBI}$ | `obi` | Order book imbalance |
| $S$ | `bid_ask_spread` | Relative bid-ask spread |
| $\mathbb{1}_{\mu > \text{mid}}$ | `micro_vs_mid` | True when micro-price exceeds mid |
| $\Delta_{\mu}$ | `micro_mid_delta` | Directional micro–mid delta |
| $\mathbb{1}_{\text{thin}}$ | `is_thin_micro_effect` | True when depth is below median |
| $\mathbb{1}_{D \geq 50\%}$ | `is_total_depth_50pct_l0` | True when depth ≥ 50 % of level-0 |

---

## Metrics (`metrics.py`)

`get_order_book_metrics()` enriches a raw order book DataFrame with the columns below.

Let $P_b$, $Q_b$ denote the bid price and quantity and $P_a$, $Q_a$ the ask price and quantity at a given level.

### Total Depth

$$D = Q_b + Q_a$$

### Mid-Price

$$P_{\text{mid}} = \frac{P_b + P_a}{2}$$

### Order Book Imbalance (OBI)

$$\text{OBI} = \frac{Q_b - Q_a}{Q_b + Q_a}$$

Values range in $[-1, 1]$.  
$\text{OBI} > 0$ → excess bid-side liquidity; $\text{OBI} < 0$ → excess ask-side liquidity.

### Micro-Price

$$P_{\mu} = \frac{P_b \cdot Q_a + P_a \cdot Q_b}{Q_b + Q_a}$$

A volume-weighted fair price that shifts towards the side with **less** resting quantity (i.e. the side more likely to be consumed).

### Micro vs Mid (`micro_vs_mid`)

$$\mathbb{1}_{\mu > \text{mid}} = \begin{cases} \text{True}  & \text{if } P_{\mu} > P_{\text{mid}} \\ \text{False} & \text{otherwise} \end{cases}$$

When $P_{\mu} > P_{\text{mid}}$, buy-side pressure is implied (ask quantity is thinner than bid quantity).

### Bid-Ask Spread (relative)

$$S = \frac{P_a - P_b}{P_{\text{mid}}}$$

### Spread Flags

| Flag | Condition | Decimal threshold |
|------|-----------|-------------------|
| `is_large_spread` | $S > 0.10\%$ | $S > 0.001$ |
| `is_small_spread` | $S \leq 0.02\%$ | $S \leq 0.0002$ |

---

## Strategy Indicators (`indicators.py`)

`add_strategy_indicators(df, strategy)` adds directional columns depending on the chosen strategy.

### Micro–Mid Delta

$$\Delta_{\mu} = \begin{cases} P_{\mu} - P_{\text{mid}} & \text{if strategy = buy} \\ P_{\text{mid}} - P_{\mu} & \text{if strategy = sell} \end{cases}$$

A positive $\Delta_{\mu}$ signals that the micro-price diverges from mid in the direction favourable to the strategy.

### Thin Micro Effect (`is_thin_micro_effect`)

$$\mathbb{1}_{\text{thin}} = \begin{cases} \text{True}  & \text{if } D < \widetilde{D} \\ \text{False} & \text{otherwise} \end{cases}$$

where $\widetilde{D}$ is the **median** total depth across all levels in the snapshot.  
When `True`, the level's depth is below median, meaning the micro-price signal may be unreliable (thin book artefact).

### Depth Adequacy (`is_total_depth_50pct_l0`)

$$\mathbb{1}_{D \geq 50\%} = \begin{cases} \text{True}  & \text{if } D \geq 0.5 \cdot D_0 \\ \text{False} & \text{otherwise} \end{cases}$$

where $D_0$ is the total depth at level 0 (best bid/ask).  
Ensures the level carries at least 50 % of the top-of-book liquidity.

---

## Opportunity Scoring (`scores.py`)

`get_weighted_volume_micro_spread_score()` produces a composite score per level:

$$\text{Score} = 0.7 \cdot \frac{D}{D_{\max}} + 0.3 \cdot \frac{\Delta_{\mu}}{\Delta_{\mu,\max}}$$

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Normalised depth $\frac{D}{D_{\max}}$ | 70 % | **Safety** — prefer levels with substantial liquidity |
| Normalised delta $\frac{\Delta_{\mu}}{\Delta_{\mu,\max}}$ | 30 % | **Aggression** — prefer levels where the micro-price divergence is largest |

---

## Historical VWAP & Momentum Filter (`indicators.py` + `analysis.py`)

### Volume-Weighted Average Price (VWAP)

`historical_analysis()` computes two VWAPs from the rolling `history_order_book` deque (up to 3 000 snapshots ≈ last 5 min of ticks):

$$\text{VWAP}_{\text{bid}} = \frac{\displaystyle\sum_{i=1}^{N} P_{\text{bid},i} \cdot V_{\text{bid},i}}{\displaystyle\sum_{i=1}^{N} V_{\text{bid},i}}$$

$$\text{VWAP}_{\text{ask}} = \frac{\displaystyle\sum_{i=1}^{N} P_{\text{ask},i} \cdot V_{\text{ask},i}}{\displaystyle\sum_{i=1}^{N} V_{\text{ask},i}}$$

where $P_{\text{bid},i}$, $V_{\text{bid},i}$ (respectively $P_{\text{ask},i}$, $V_{\text{ask},i}$) are the best bid (ask) price and quantity recorded at tick $i$, and $N$ is the number of snapshots currently in the deque.

The helper `volume_weighted_average_price(price, volume)` in `indicators.py` implements this as:

```python
float(np.sum(price * volume) / np.sum(volume))
```

Both VWAPs are published under `_vwap_lock` so the low-latency thread can read them safely.

### Momentum-Confirmation Filter

After the first `historical_analysis` iteration (≈ 1 min into the session), `_bid_vwap` and `_ask_vwap` are populated.  `low_latency_analysis` reads them on every iteration and uses them as a **momentum-confirmation gate** before sending an order:

| Side | Condition to execute | Interpretation |
|------|---------------------|----------------|
| **BUY** | `_ask_vwap is None` **or** `micro_price > ask_vwap` | The current micro-price exceeds the recent volume-weighted average cost to buy → **upward momentum** is confirmed |
| **SELL** | `_bid_vwap is None` **or** `micro_price < bid_vwap` | The current micro-price is below the recent volume-weighted average bid → **downward momentum** is confirmed |

While VWAPs are still `None` (first ~1 min) the filter is transparent and orders execute based on the opportunity score alone.

### ⚠️ Note on alternative strategies

The current momentum filter is designed to **trade with the trend**: buy when the price is pushing above the historical average, sell when it is dropping below.

This logic may be **inverted** in the future to implement a **buy-the-dip** / **sell-the-rally** mean-reversion strategy instead — e.g. buy when `micro_price < ask_vwap` (price has dipped below average) or sell when `micro_price > bid_vwap` (price has rallied above average).  The gating condition in `low_latency_analysis` is the single point of change for switching between momentum and mean-reversion modes.  This decision depends on the market regime and is subject to further experimentation.

---

## HMM Regime Detection (`strategy/regime_director.py`)

`RegimeDirector` trains a `GaussianHMM` on recent Binance klines to classify the current market into one of several **hidden regimes** and exposes the result as a plain string label that gates order execution in `low_latency_analysis`.

### Theoretical Background

> 📖 **Reference:** The HMM theory is explained with examples in this paper by
> Jurafsky, D. & Martin, J.H. — *Speech and Language Processing*, 3rd ed.,
> Appendix A — *Hidden Markov Models*.
> [Read online (Stanford)](https://web.stanford.edu/~jurafsky/slp3/A.pdf) **[1]**

A **Hidden Markov Model (HMM)** **[1]** is a statistical model for systems that move through a finite set of states over time, where those states are **not directly observable** — they are _hidden_.  What you _can_ observe at each time step is a signal that depends probabilistically on whichever hidden state the system is currently in.

In this project:

- The **hidden states** are the market regimes (`trending_up`, `trending_down`, `high_volatility`, `neutral`).  You cannot observe the regime directly; you can only infer it from market data.
- The **observations** are the four feature vectors computed from 1-minute klines: `[return, volatility, obi_proxy, trade_density]`.

An HMM is fully described by three components:

#### 1. Transition Probability Matrix $A$

$$A_{ij} = P(q_{t+1} = s_j \mid q_t = s_i)$$

The probability of moving from regime $s_i$ at time $t$ to regime $s_j$ at time $t+1$.  Stored in `model.transmat_` (shape $n \times n$, rows sum to 1).

In this context: if the market is currently in `trending_up`, $A$ encodes how likely it is to stay there vs. transition to `neutral` or `high_volatility` on the next candle.  A high diagonal value means the regime is **persistent**; a low diagonal value means it is **short-lived**.

#### 2. Emission Probability $B_i(o)$

Because we use a **Gaussian HMM**, the emission probability for state $i$ is a multivariate Gaussian over the observation vector:

$$B_i(o) = \mathcal{N}(o\,;\,\mu_i,\,\Sigma_i)$$

where $\mu_i$ is the mean feature vector and $\Sigma_i$ is the covariance matrix for state $i$.  Stored in `model.means_` and `model.covars_`.

In this context: given that the market is in regime $i$, $B_i$ describes the distribution of `[return, volatility, obi_proxy, trade_density]` values we expect to observe.  For example, a `trending_up` state would have a $\mu_i$ with a positive `return` and a positive `obi_proxy` — this is exactly what `assign_regime_labels()` reads to assign human-readable names.

#### 3. Initial Probability $\pi$

$$\pi_i = P(q_1 = s_i)$$

The probability of starting in each regime at $t = 1$.  In practice the model learns this from data and it has minimal effect after a sufficiently long observation sequence.

---

#### Learning and Inference

| Step | Algorithm | hmmlearn call | What it does |
|---|---|---|---|
| **Training** | Baum-Welch (Expectation-Maximisation) | `model.fit(features)` | Learns $A$, $B$ ($\mu_i$, $\Sigma_i$), $\pi$ from the kline feature matrix |
| **Inference** | Viterbi | `model.predict(features)` | Finds the most likely hidden state sequence $q_1, q_2, \ldots, q_T$ given the observed features |
| **Model selection** | BIC | `model.bic(features)` | Penalises model complexity; used to choose the best $n$ (number of regimes) |

The last element of the predicted sequence, `regimes[-1]`, corresponds to the **most recent candle** — this becomes `current_regime` and is what `assign_regime_labels()` uses to set `regime_label`.

### Features

| Feature | Formula | Interpretation |
|---|---|---|
| `return` | `close.pct_change()` | Per-candle price momentum |
| `volatility` | `(high - low) / close` | Normalised intra-bar price swing |
| `obi_proxy` | `(taker_buy_base_vol / volume) × 2 − 1` | Taker-flow imbalance proxy ∈ `[-1, +1]`; approximates live OBI from kline data |
| `trade_density` | `num_trades / volume` | Trade fragmentation: high → many small trades (retail/HFT); low → large blocks (institutional) |

### Feature Scaling

Before any `fit()`, `predict()`, or `predict_proba()` call, the four HMM features are **z-score normalised** via `sklearn.preprocessing.StandardScaler`:

- The scaler is **`fit_transform`'d on the first `HMM_TRAIN_ROWS` (80) rows only** — the older in-sample portion of the 2-hour window.
- The same scaler is then **`transform`'d onto the full window** for prediction, so the most recent ~40 rows are genuinely out-of-sample.
- Without scaling, `trade_density` (`num_trades / volume`) can be orders of magnitude larger than `return` or `obi_proxy` (both near `[-1, +1]`), which would distort the covariance structure and bias BIC selection.

### Model Selection (BIC)

`GaussianHMM` models are fitted for `n = 2 … HMM_MAX_REGIMES` (default 4) hidden states with full covariance.  The best model is selected by **Bayesian Information Criterion**:

$$\text{BIC} = -2 \ln \hat{L} + k \ln N$$

where $\hat{L}$ is the model likelihood, $k$ the number of free parameters, and $N$ the number of **training** observations (`HMM_TRAIN_ROWS`).  The model with the **lowest BIC** is retained.

### Two-Speed Update

To avoid re-training the full model on nearly identical data every 60 s, `historical_analysis()` uses a two-speed scheme:

| Cadence | Method called | Cost | When |
|---|---|---|---|
| Every `HIST_INTERVAL` (60 s) | `predict_current_regime()` | O(n × k) — single Viterbi pass | Iterations 1, 2, 3, 4, 6, 7, … |
| Every `HMM_REFIT_INTERVAL` (300 s) | `select_hmm_model()` | O(n × k × `HMM_N_ITERATIONS`) per candidate — expensive | Iterations 5, 10, 15, … |

`assign_regime_labels()` runs inside `_regime_lock` on **every** iteration regardless of path.

### Regime Confidence

After each `predict()` call, `predict_proba(features_scaled)` is called to obtain the **posterior state probabilities** (Forward-Backward algorithm) for the full window.  The probability of the current (latest) candle's state is stored as `regime_confidence`:

$$\text{regime\_confidence} = P(\text{state} = \text{current\_regime} \mid \text{observations})$$

| `regime_confidence` | Interpretation |
|---|---|
| ≥ `HMM_MIN_CONFIDENCE` (0.70) | Model is confident — gates apply normally |
| < `HMM_MIN_CONFIDENCE` | Model is uncertain (e.g. 55 % vs 45 %) — **both BUY and SELL are skipped** |
| `None` (warm-up) | No model fitted yet — gate is transparent |

### State Labelling

State integers (0, 1, 2, …) carry no inherent meaning.  `assign_regime_labels()` assigns labels using a **rank-based** directional scheme and threshold-based secondary rules — no hard-coded price constants:

**Directional labels** — guaranteed to be unique (exactly one each):

A combined **direction score** is computed per state:

$$\text{direction\_score}_i = \text{rank}(\text{return}_i) + \text{rank}(\text{obi\_proxy}_i)$$

| Assignment | Rule |
|---|---|
| `"trending_up"` | State with the **highest** direction score (most bullish: highest return + strongest buy-side flow) |
| `"trending_down"` | State with the **lowest** direction score (most bearish: lowest return + weakest flow) |

Because `idxmax()` and `idxmin()` are exclusive by definition, duplicate directional labels are structurally impossible regardless of `n_components`.

**Secondary labels** — applied to remaining states via volatility and trade-density thresholds:

| Flag | Condition |
|---|---|
| `high_vol` | `volatility > cross-state mean + 1 × std` |
| `high_td` | `trade_density > cross-state mean + 0.5 × std` |

| Condition | Label | Rationale |
|---|---|---|
| `high_vol` **or** `high_td` | `"high_volatility"` | Either large price swings (unpredictable fills) **or** heavy trade fragmentation (noisy, no directional intent) makes the market **unreliable to trade in** |
| *(default)* | `"neutral"` | No dominant signal |

> **Why OR?** Both `high_vol` and `high_td` independently indicate an unreliable market — one from the price side (large swings), the other from the flow side (fragmented activity). Using OR ensures `trade_density` has a meaningful role in the label assignment rather than being used only for model training.

### Regime Gates in `low_latency_analysis`

Two gates are applied **sequentially**:

**Gate 1 — Confidence** (evaluated first, before any label check):

| `regime_confidence` | Result |
|---|---|
| `None` (before first historical run) | ✅ transparent — all orders allowed |
| ≥ `HMM_MIN_CONFIDENCE` (0.70) | ✅ proceed to direction gate |
| < `HMM_MIN_CONFIDENCE` | ❌ **both** BUY and SELL skipped |

**Gate 2 — Direction** (evaluated only if confidence gate passed):

| Regime | BUY | SELL |
|---|---|---|
| `"trending_up"` | ✅ allowed | ❌ suppressed |
| `"trending_down"` | ❌ suppressed | ✅ allowed |
| `"high_volatility"` | ❌ suppressed | ❌ suppressed |
| `"neutral"` | ✅ allowed | ✅ allowed |
| `None` (before first historical run) | ✅ transparent | ✅ transparent |

### Session Timeline with HMM

```
websocket_main.py startup
│
├── Pre-session: RegimeDirector.get_klines_data()     ← fetches last 2 h of 1-min klines (~120 rows)
│               StandardScaler.fit_transform(rows[:80]) ← scale training rows in-sample
│               RegimeDirector.select_hmm_model()     ← fits HMM n=2..4, selects best BIC
│               RegimeDirector.assign_regime_labels() ← sets regime_label + regime_confidence
│               Both are set BEFORE threads start
│
├── low_latency_analysis — every 1 s
│   └── reads regime_label + regime_confidence under _regime_lock  ← never None
│       ├── Gate 1: skip if regime_confidence < HMM_MIN_CONFIDENCE
│       └── Gate 2: skip buy/sell based on regime_label direction
│
└── historical_analysis — every 60 s
    ├── compute bid_vwap / ask_vwap                           ← existing VWAP logic
    ├── get_klines_data()                                     ← refresh features from Binance
    ├── if iteration % 5 == 0:
    │     select_hmm_model()                                  ← full re-fit + new scaler (every 5 min)
    │   else:
    │     predict_current_regime()                            ← cheap Viterbi + predict_proba
    └── assign_regime_labels() under _regime_lock             ← label + confidence write (every iteration)
```

---

## Backtesting

A backtesting framework has been built to replay the live strategy against
180 days of historical 1-minute klines (~259,200 candles).  Because Binance does
not expose historical Level-2 order book data, a **synthetic 50-level depth
ladder** is reconstructed from each kline's OHLCV data and taker volume split,
then fed through the **same production scoring pipeline** used by
`low_latency_analysis()`.

Three independent data flows — opportunity scoring (synthetic book), HMM regime
filter (kline features), and VWAP momentum filter (level-0 rolling window) —
are combined into a single signal per candle, exactly mirroring the live
architecture.

All design decisions, pseudo-code, data-flow diagrams, approximation caveats,
and a step-by-step implementation roadmap with progress tracking are documented
in **[`BACKTESTING.md`](BACKTESTING.md)**.

| Module | Status | Purpose |
|---|---|---|
| `backtest/data.py` | ✅ done | Download historical klines |
| `backtest/synthetic_book.py` | ✅ done | Build synthetic 50-level order book per candle |
| `backtest/signals.py` | ✅ done | Signal replay loop (full pipeline + filters) |
| `backtest/pnl.py` | ✅ done | Simulated P&L — balance guard, bps-based `half_spread` fill model (`BACKTEST_FILL_SPREAD_BPS`), per-trade position cap (`BACKTEST_MAX_POSITION_PCT`), equity curve, FIFO round-trip pairing, Step 5 metrics |
| `backtest/runner.py` | ✅ done | Top-level orchestration — chains all modules, delegates report/CSV to `reporting/`; exposes `plot` and `save_png` flags for Step 7 |
| `backtest/reporting/formatters.py` | ✅ done | Console report formatting (`print_report`, `print_regime_validation_report`) and CSV export (`save_csv`) — AI-authored |
 `backtest/regime_validation.py`  ✅ done  Offline long-horizon regime validation — **70/30 train-test split** on 1 year (~525,000 rows, `VALIDATION_LOOKBACK = "365 days ago UTC"`), self-contained (no `RegimeDirector`), fits HMM on full train set, **vectorised** single-pass Viterbi on ~157,500 test candles, six statistical checks, `python -m backtest.diagnostics.regime_validation`
| `backtest/visualization.py` | ✅ done | Interactive six-panel Plotly chart — equity curve, drawdown, BUY/SELL markers, regime timeline, VWAP vs micro-price, signal funnel, signals-by-regime |
| `backtest/sensitivity.py` | ✅ done (Use Case A) | OAT sweep (6 runs, ~36–108 min) and full-grid (24 combinations) over `HMM_LOOKBACK_ROWS`, `HMM_MAX_REGIMES`, `VWAP_WINDOW`, `BACKTEST_FEE_RATE`. Writes `best_params.json` loaded by `websocket_main.py` at startup. Use Case B (180-day window) deferred. |

**Running the backtest:**

```bash
# Console report only (default)
python -m backtest.runner

# Console report + open interactive Plotly chart in browser
python -c "from backtest.runner import run_backtest; run_backtest(plot=True)"

# Console report + chart + save PNG (requires kaleido) or HTML fallback
python -c "from backtest.runner import run_backtest; run_backtest(plot=True, save_png=True)"

# Console report + export trade/equity CSVs
python -c "from backtest.runner import run_backtest; run_backtest(export_csv=True)"
```

> ⚠️ **Important — re-run after any change to `strategy/regime_director.py`**  
> `regime_validation.py` is the **sanity check** for the HMM regime filter.  
> Whenever `RegimeDirector` logic is modified (feature columns, BIC search range,
> label-assignment rules, confidence threshold, etc.) the validation tool **must**
> be re-run to confirm that the frozen model still produces statistically meaningful
> labels on out-of-sample data.  A failing check (especially Check 1 — direction
> test, or Check 2 — Welch's t-test) is a strong signal that the change broke the
> regime filter's discriminative power and should be reviewed before deploying live.
>
> ```bash
> python -m backtest.diagnostics.regime_validation
> ```

---

## Appendix A — `RegimeDirector` Deep Dive: How It Fits Into the Whole Codebase

This appendix traces `RegimeDirector` from its configuration constants all the
way through to its effect on individual order decisions, file by file.

---

### A.1 Configuration — `config_parameters.py`

Every tunable parameter `RegimeDirector` needs is centralised here.
No magic numbers appear anywhere else in the strategy code.

| Constant | Value | Purpose |
|---|---|---|
| `HMM_FEATURE_COLS` | `["return", "volatility", "obi_proxy", "trade_density"]` | Features fed to `GaussianHMM` |
| `HMM_INTERVAL` | `Client.KLINE_INTERVAL_1MINUTE` | Kline granularity (1 min — intra-session resolution) |
| `HMM_LOOKBACK` | `"2 hours ago UTC"` | Rolling window (~120 rows); responsive to intra-day BTC shifts while keeping enough data for stable EM convergence |
| `HMM_MAX_REGIMES` | `4` | Upper bound on hidden states evaluated during BIC search (2 … 4) |
| `HMM_N_ITERATIONS` | `1000` | Max EM iterations per model fit |
| `HMM_RANDOM_STATE` | `46` | Seed for reproducible state numbering across fits |
| `HMM_MIN_COVAR` | `1e-3` | Regularisation floor for covariance matrices |
| `HMM_REFIT_INTERVAL` | `300` | Full re-fit cadence (s).  Between re-fits only Viterbi prediction runs |

> **Why 2 hours and not 4?**  
> A market regime persists for minutes to hours, but Bitcoin's direction can
> shift quickly.  A 4-hour window means a regime change 30 minutes ago barely
> moves the model (the session's own data is only ~8 % of the training set).
> 2 hours (~120 candles) is reactive enough to capture intra-day shifts while
> keeping enough data points for stable EM convergence with up to 4 states.

---

### A.2 Pre-session Startup — `websocket_main.py` (step 4b)

`RegimeDirector` is instantiated and fully fitted **before any thread starts**,
in the single-threaded startup block of `websocket_main.py`:

```python
# websocket_main.py — step 4b (after seeding OrderBookState, before engine)
from strategy.regime_director import RegimeDirector

regime_director = RegimeDirector()             # 1. instantiate — no data yet
regime_director.get_klines_data()              # 2. download ~120 rows, compute features
regime_director.select_hmm_model()             # 3. fit HMM n=2..4, pick best BIC
regime_director.assign_regime_labels()        # 4. map state int → label string
# → regime_director.regime_label == e.g. "trending_up"

engine = AnalysisEngine(
    state=state,
    stop_event=stop_event,
    executor=executor,
    regime_director=regime_director,           # 5. inject into AnalysisEngine
)
```

**Why fit before threads start?**  
`low_latency_analysis` reads `regime_label` on its very first iteration
(t = 1 s). If the fit were deferred to the first `historical_analysis` run
(t = 60 s), every order in the first minute would be made with `regime_label = None`
(transparent filter — no regime gating at all).  Fitting at startup guarantees
the regime filter is active from iteration #1.

---

### A.3 Inside `AnalysisEngine` — `strategy/analysis.py`

`AnalysisEngine.__init__` stores the injected instance and creates a dedicated
lock to protect the label between the two threads:

```python
self.regime_director = regime_director   # injected — NOT re-created internally
self._regime_lock    = threading.Lock()  # protects regime_label across threads
```

Two background threads then interact with it in opposite roles:

---

#### Thread A — `historical_analysis()` every 60 s → **writer**

After computing VWAPs from the live order book, this thread re-fits the model
on the latest 4 hours of klines.  The expensive work runs **outside** the lock;
only the instant label assignment is locked:

```python
# OUTSIDE _regime_lock — slow: network download + CPU model fit/predict
self.regime_director.get_klines_data()         # re-fetch latest 2 h of klines

# Two-speed update: full re-fit every HMM_REFIT_INTERVAL (300 s),
# cheap Viterbi prediction on all other iterations.
if iteration % refit_every == 0:
    self.regime_director.select_hmm_model()    # full re-fit — slow
else:
    self.regime_director.predict_current_regime()  # Viterbi only — fast

# INSIDE _regime_lock — fast: dict lookup + string assignment only
with self._regime_lock:
    self.regime_director.assign_regime_labels()   # write regime_label
```

This split means `low_latency_analysis` is **never blocked** waiting for a
model fit to finish — it only waits for the lock during the microsecond string
write.

---

#### Thread B — `low_latency_analysis()` every 1 s → **reader**

After scoring the order book candidates but before calling `execute()`, this
thread reads the current label under the same lock:

```python
with self._regime_lock:
    current_regime = self.regime_director.regime_label   # fast read
```

The label is then used as the **first gate** before the VWAP filter:

```python
if best_buy:
    if current_regime in ("trending_down", "high_volatility"):
        # regime blocks BUY — skip entirely, do not evaluate VWAP
    elif ask_vwap is not None and micro_price <= ask_vwap:
        # VWAP blocks BUY — skip
    else:
        executor.execute("BUY", best_buy)      # both filters passed

if best_sell:
    if current_regime in ("trending_up", "high_volatility"):
        # regime blocks SELL — skip entirely
    elif bid_vwap is not None and micro_price >= bid_vwap:
        # VWAP blocks SELL — skip
    else:
        executor.execute("SELL", best_sell)    # both filters passed
```

---

### A.4 Complete Data-Flow Diagram

```
config_parameters.py
  HMM_FEATURE_COLS, HMM_INTERVAL, HMM_LOOKBACK,
  HMM_MAX_REGIMES, HMM_N_ITERATIONS, HMM_RANDOM_STATE,
  HMM_REFIT_INTERVAL
        │
        ▼
websocket_main.py  ── step 4b, single-threaded, BEFORE threads start ──
  RegimeDirector()
    .get_klines_data()           ← Binance public REST (no auth needed)
    .select_hmm_model()          ← GaussianHMM BIC search (n = 2 … 4)
    .assign_regime_labels()     ← regime_label = "trending_up" / ...
        │
        │  injected as parameter
        ▼
AnalysisEngine.__init__
  self.regime_director = regime_director
  self._regime_lock    = threading.Lock()
        │
        ├────────────────────────────────────────────────┐
        │ every 60 s                                     │ every 1 s
        ▼                                                ▼
historical_analysis()                       low_latency_analysis()
  ① compute bid_vwap / ask_vwap               ① balance guard
    (from live order book deque)               ② copy order book under lock
  ② write VWAPs under _vwap_lock              ③ build levels, score candidates
  ③ get_klines_data()    ← outside lock       ④ read regime_label
     if iter % 5 == 0:                              under _regime_lock
       select_hmm_model()← full refit         ⑤ REGIME FILTER
     else:                                         BUY  blocked if "trending_down"
       predict_current_regime()                         or "high_volatility"
                          ← Viterbi only           SELL blocked if "trending_up"
  ④ assign_regime_labels()                          or "high_volatility"
        under _regime_lock                    ⑥ VWAP FILTER
        (fast write only)                          BUY  blocked if micro ≤ ask_vwap
                                                   SELL blocked if micro ≥ bid_vwap
                                              ⑦ OrderExecutor.execute()
                                                   LIMIT GTC via WebSocket API
```

---

### A.5 Regime Label Reference

| `regime_label` | BUY | SELL | Typical market condition |
|---|---|---|---|
| `"trending_up"` | ✅ | ❌ | Highest combined return + OBI rank (most bullish state) |
| `"trending_down"` | ❌ | ✅ | Lowest combined return + OBI rank (most bearish state) |
| `"high_volatility"` | ❌ | ❌ | Large intra-bar swings OR heavy trade fragmentation — unreliable market |
| `"neutral"` | ✅ | ✅ | No dominant signal in any feature |
| `None` *(impossible after step 4b)* | ✅ | ✅ | Transparent — all orders pass through |

---

### A.6 Threading Safety Summary

| Operation | Lock held | Duration |
|---|---|---|
| `get_klines_data()` | none | ~1–2 s (network I/O) |
| `select_hmm_model()` | none | ~2–5 s (CPU — EM iterations) — every 5 min |
| `predict_current_regime()` | none | < 50 ms (single Viterbi pass) — every 60 s |
| `assign_regime_labels()` | `_regime_lock` | < 1 ms (dict lookup + string write) |
| Read `regime_label` in `low_latency_analysis` | `_regime_lock` | < 1 ms |

`low_latency_analysis` is **never blocked** for more than a microsecond by the
regime machinery.  The only contention point is the label write/read, which is
effectively instantaneous.

---

## References

**[1]** Jurafsky, D. & Martin, J.H. (2024). *Speech and Language Processing*, 3rd edition, Appendix A — *Hidden Markov Models*. Stanford University. Available at: <https://web.stanford.edu/~jurafsky/slp3/A.pdf>

> This appendix provides the formal definitions of the Transition Probability
> Matrix $A$, the Emission Probability $B$, the Initial Probability $\pi$, the
> Baum-Welch (EM) training algorithm, and the Viterbi decoding algorithm that
> underpin the `RegimeDirector` implementation in `strategy/regime_director.py`.
> Recommended reading for anyone who wants to follow the HMM theory described
> in the *HMM Regime Detection* section and in *Appendix A* of this document.

---

*Document last updated: 2026-04-14*

