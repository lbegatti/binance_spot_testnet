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
| **REST** (polling) | `rest_spot_main.py` | `client.get_order_book()` snapshots | BTCUSDT |
| **WebSocket** (real-time) | `websocket_spot_main.py` | `diff_book_depth` stream (100 ms) | BTCUSDT |

The REST path is fully wired — metrics → indicators → scores → best quote.  
The WebSocket path runs the full strategy pipeline (metrics → indicators → scores) in near real-time via the `AnalysisEngine`.

---

## Project Structure

```
binance_spot_testnet/
├── config_parameters.py               # Central configuration — all tunable constants in one place
├── rest_spot_main.py                  # REST orchestration — loops over depth limits
├── websocket_spot_main.py             # WebSocket — real-time local order book + session driver
├── README.md
├── system_architecture.txt
│
├── core/                              # Shared state and data ingestion
│   ├── __init__.py
│   ├── order_book_state.py            # Shared state container (local book + history + balances + locks)
│   └── message_handler.py             # WebSocket callbacks — maintains local book and balances in real time
│
├── strategy/                          # Analysis and scoring pipeline
│   ├── __init__.py
│   ├── analysis.py                    # AnalysisEngine — low-latency (5 s) and historical (5 min) loops
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
| **Order book state** | `HISTORY_MAXLEN` | `6000` | Max snapshots in `history_order_book` — at 100 ms intervals this covers ~10 min.  Each entry: `{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}` (all numeric) |
| **Order book state** | `N_LEVELS` | `50` | Number of order book levels used in `low_latency_analysis` |
| **Analysis cadence** | `HFT_INTERVAL` | `5` s | Time between low-latency evaluations |
| **Analysis cadence** | `HIST_INTERVAL` | `300` s | Time between historical analyses (5 min) |
| **Analysis cadence** | `MIN_SNAPSHOTS` | `100` | Minimum snapshots required before historical analysis runs |
| **WebSocket session** | `DEFAULT_SESSION_MINUTES` | `15` min | Default session length (fixed — no startup prompt) |
| **WebSocket session** | `HTF_JOIN_TIMEOUT` | `10` s | Max wait for `low_latency_analysis` thread on shutdown |
| **WebSocket session** | `HIST_JOIN_TIMEOUT` | `15` s | Max wait for `historical_analysis` thread on shutdown |
| **Binance connection** | `RECV_WINDOW` | `5000` ms | Binance REST request validity window |
| **Binance connection** | `SNAPSHOT_DEPTH` | `100` | Order book levels in the seed snapshot |
| **Binance connection** | `WS_SPEED` | `100` ms | WebSocket diff-depth update interval |

**Imported by:**
- `core/order_book_state.py` — `HISTORY_MAXLEN`, `CRYPTOCCY`, `CCY`
- `strategy/analysis.py` — `HFT_INTERVAL`, `HIST_INTERVAL`, `MIN_SNAPSHOTS`, `N_LEVELS`, `CCY`, `CRYPTOCCY`
- `execution/order_executor.py` — `SYMBOL`, `CRYPTOCCY`, `CCY`, `RECV_WINDOW`
- `websocket_spot_main.py` — `SYMBOL`, `CCY`, `CRYPTOCCY`, and all session / connection constants

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
   python rest_spot_main.py

   # WebSocket (real-time)
   python websocket_spot_main.py
   ```

---

## Pipeline Overview

### REST path (`rest_spot_main.py`)

```
Binance REST API
    │  get_order_book()
    ▼
┌──────────────────┐     ┌────────────────┐     ┌──────────────┐     ┌────────────┐
│ rest_spot_main.py│────▶│  metrics.py    │────▶│ indicators.py│────▶│  scores.py │
│  (orchestrate)   │     │  (enrich df)   │     │ (strategy)   │     │  (score)   │
└──────────────────┘     └────────────────┘     └──────────────┘     └────────────┘
                                                                           │
                                                                           ▼
                                                                   ┌──────────────┐
                                                                   │  quotes.py   │
                                                                   │ (best quote) │
                                                                   └──────────────┘
```

### WebSocket path (`websocket_spot_main.py`)

```
Binance REST API                   Binance WebSocket (production)      Binance WebSocket (testnet)
    │  depth() snapshot                │  diff_book_depth (100 ms)          │  User Data Stream
    ▼                                  ▼                                    ▼
┌──────────────────┐       ┌──────────────────────┐          ┌──────────────────────────┐
│  OrderBookState  │◀──────│   MessageHandler      │          │   MessageHandler          │
│  (shared state)  │       │  .handle_depth_msg()  │          │  .handle_balance_msg()   │
│                  │       └──────────────────────┘          └──────────────────────────┘
│  • local_book    │                                                        │
│  • history_book  │◀──────────────────────────────────────────────────────┘
│  • balance_status│         (outboundAccountPosition → balance_status)
│  • thread_lock   │
│  • thread_       │       ┌──────────────────────┐
│    balance_lock  │◀──────│   AnalysisEngine      │
└──────────────────┘       │  .low_latency_analysis()  │
                           │   (balance check +        │
                           │    order book strategy)   │
                           │  .historical_analysis()   │
                           └──────────┬───────────┘
                                      │  opportunity tuple
                                      ▼
                           ┌──────────────────────┐     Binance WebSocket API (testnet)
                           │   OrderExecutor       │────▶  wss://testnet.binance.vision
                           │  .execute()           │       /ws-api/v3
                           │   (balance guard +    │◀──── handle_order_response()
                           │    LIMIT GTC order)   │       (async response callback)
                           └──────────────────────┘
                                      ▲
                           ┌──────────────────────┐
                           │  websocket_spot_main  │
                           │  (session driver)     │
                           │  + keepalive thread   │
                           └──────────────────────┘
```

**Component responsibilities:**

| Class / Module | File | Role |
|---|---|---|
| *(constants)* | `config_parameters.py` | Single source of truth for all tunable constants (`SYMBOL`, `CCY`, `CRYPTOCCY`, intervals, depths, timeouts) — imported by every package |
| `OrderBookState` | `core/order_book_state.py` | Single source of truth — owns `local_book`, `history_order_book`, `balance_status`, `thread_lock`, and `thread_balance_lock` |
| `MessageHandler` | `core/message_handler.py` | Two WebSocket callbacks: `handle_depth_message` (merges diff-depth ticks into `local_book`, appends snapshots, calls `calculate_best_quote`) and `handle_balance_message` (processes `outboundAccountPosition` events to keep `balance_status` current) |
| `AnalysisEngine` | `strategy/analysis.py` | Runs two background loops (`low_latency_analysis` and `historical_analysis`) that read from `OrderBookState` via the shared locks; delegates order placement to `OrderExecutor` |
| `OrderExecutor` | `execution/order_executor.py` | Places LIMIT GTC orders via the Binance WebSocket API (`SpotWebsocketAPIClient`); owns its own WS connection; validates strategy, checks balances, uses `aq` (BUY) or `bq` (SELL) as quantity and `micro_price` as limit price; handles responses asynchronously |
| `websocket_spot_main` | `websocket_spot_main.py` | Session driver — instantiates all classes, seeds initial balances into `state`, opens WebSocket streams, starts all threads, manages session lifetime and shutdown |

**How the components interact:**

1. `websocket_spot_main.py` creates a single `OrderBookState` instance and injects it into both `MessageHandler` and `AnalysisEngine`.  After construction it immediately seeds `state.balance_status` with the REST-fetched balances before any thread starts.
2. Order-book data (`local_book`, `history_order_book`) is serialised through `state.thread_lock`.  `MessageHandler.handle_depth_message` acquires it to write; `AnalysisEngine.low_latency_analysis` acquires it to take a read-only copy and releases it before any heavy computation.
3. Balance data (`balance_status`) is serialised through the dedicated `state.thread_balance_lock`, completely independent of `thread_lock`.  This prevents the high-frequency WebSocket order-book path (every 100 ms) from blocking on the lower-frequency balance path.
4. `MessageHandler` is the **only writer** to `OrderBookState`; `AnalysisEngine` is **read-only** — it copies data under the appropriate lock and immediately releases it.
5. `AnalysisEngine` delegates order placement to `OrderExecutor`.  When `_select_best_opportunity()` returns a non-`None` 8-element tuple `(level_idx, score, delta, total_depth, obi, micro_price, bq, aq)`, the engine calls `executor.execute("BUY", best_buy)` or `executor.execute("SELL", best_sell)`.  `OrderExecutor` validates the strategy, checks balances under `thread_balance_lock`, computes quantity (`aq` for BUY, `bq` for SELL), and sends a LIMIT GTC order via its own `SpotWebsocketAPIClient`.  The response arrives asynchronously in `handle_order_response`.
6. **VWAP cross-thread integration** — `AnalysisEngine` owns a private `_vwap_lock` plus two attributes `_bid_vwap` and `_ask_vwap` (initially `None`).  `historical_analysis` computes both VWAPs from `history_order_book` every 5 min and publishes them under `_vwap_lock`.  `low_latency_analysis` reads them under the same lock on every iteration and gates order execution:
   - **BUY**: execute only if `_ask_vwap is None` (first ~5 min) **or** `micro_price > ask_vwap`.
   - **SELL**: execute only if `_bid_vwap is None` (first ~5 min) **or** `micro_price < bid_vwap`.
7. A second `SpotWebsocketStreamClient` (`ws_user_client`) connects to the Binance Testnet User Data Stream (`wss://testnet.binance.vision`) and routes all messages to `handle_balance_message`.  A dedicated `listenKey` is obtained from the testnet REST API before the streams are opened and renewed every 30 minutes by a lightweight `keepalive_thread`.

---

## Session Duration

### Rationale

Both analysis loops in `AnalysisEngine` are designed to run indefinitely:

| Loop | Cadence | Purpose |
|------|---------|---------|
| `low_latency_analysis` | every **5 s** | Near-real-time best bid/ask evaluation |
| `historical_analysis` | every **5 min** | Pattern detection over the rolling snapshot window |

Rather than running forever, `websocket_spot_main.py` uses a fixed session duration set by `DEFAULT_SESSION_MINUTES` (no startup prompt).

The **default of 15 minutes** is chosen deliberately:

| Metric | Value at 15 min |
|--------|----------------|
| Low-latency iterations (`low_latency_analysis`) | $15 \times 60 / 5 = \mathbf{180}$ |
| Historical iterations (`historical_analysis`) | $15 / 5 = \mathbf{3}$ |
| Order book snapshots in history | up to $15 \times 60 \times 10 = \mathbf{9{,}000}$ ticks (capped at `maxlen=6000` ≈ last 10 min) |

When the session duration elapses, `websocket_spot_main.py` sets `stop_event`, calls `ws_client.stop()` to close the stream cleanly, and joins both analysis threads (with timeouts of 10 s and 15 s respectively). A `KeyboardInterrupt` (Ctrl-C) triggers the same shutdown path early.

### Thread Timeline (default 15-min session)

```
t=0s      Both threads start
          ├── low_latency: runs immediately, then every 5 s
          └── historical:  sleeps 5 min first (stop_event.wait(HIST_INTERVAL))

t=5s      low_latency iteration #1
t=10s     low_latency iteration #2
...
t=5min    low_latency iteration #60
          historical iteration #1 → computes bid_vwap / ask_vwap,
            publishes under _vwap_lock
          ↓ from this point, low_latency reads the VWAP and applies the filter

t=10min   low_latency iteration #120
          historical iteration #2 → refreshes VWAP with the latest ~10 min of data

t=15min   low_latency iteration #180
          historical iteration #3 → refreshes VWAP again
          stop_event set → both threads exit
```

### Deque Fill-Up (`history_order_book`)

The deque size is driven by the **WebSocket tick rate** (~10 entries/sec at 100 ms), not by the historical analysis interval.  `historical_analysis` only *reads* the deque — it never clears it.

| Time elapsed | WebSocket ticks | Deque size | Historical iterations |
|---|---|---|---|
| 1 min | ~600 | 600 | 0 |
| 5 min | ~3 000 | 3 000 | 1st runs (reads 3 000 entries) |
| 10 min | ~6 000 | **6 000 (full)** | 2nd runs (reads 6 000 entries) |
| 15 min | ~9 000 sent | **6 000 (capped — oldest evicted)** | 3rd runs (reads 6 000 entries) |

After ~10 minutes the deque hits `maxlen=6000` and becomes a true **rolling window** of the last ~10 minutes. Each `historical_analysis` iteration operates on whatever is currently in the window — not a fixed block.

---

## WebSocket Execution Flow (`websocket_spot_main.py`)

1. Load API keys from `.env`.
2. Connect to Binance Testnet via `binance-connector` REST client.  Obtain a `listenKey` for the User Data Stream from the testnet REST endpoint.
3. Consolidate non-BTC/USDT balances into USDT via market sell orders.
4. Seed `state.balance_status` with the REST-fetched `usdt_balance` and `btc_balance` before any thread starts.
5. Session duration fixed at `DEFAULT_SESSION_MINUTES` (15 min) — no user prompt.
6. Fetch a depth snapshot for **BTCUSDT** (100 levels) to seed `OrderBookState.local_book`.
7. Instantiate `OrderBookState`, `MessageHandler`, `OrderExecutor`, and `AnalysisEngine`, injecting the shared state into the latter three.  `OrderExecutor` creates its own `SpotWebsocketAPIClient` internally (connected to `wss://testnet.binance.vision/ws-api/v3`) so that `self.handle_order_response` can be passed directly as the callback.  Set `stop_event = threading.Event()`.
8. Open two `SpotWebsocketStreamClient` instances:
   - `ws_client` — production stream endpoint, subscribes to `diff_book_depth` at 100 ms intervals; callback: `handle_depth_message`.
   - `ws_user_client` — testnet endpoint (`wss://testnet.binance.vision`), subscribes to `user_data(listen_key=...)`; callback: `handle_balance_message`.
9. Wait 1 second for the first diff-depth messages to arrive and populate `local_book["bids"]`.
10. Start all daemon threads:
    - `low_latency_thread` → `AnalysisEngine.low_latency_analysis`
    - `hist_thread` → `AnalysisEngine.historical_analysis`
    - `keepalive_thread` → `_keepalive_listen_key` (renews `listenKey` every 30 min via REST; exits when `stop_event` is set)
11. On each incoming depth message (`MessageHandler.handle_depth_message`):
    - Skip the initial subscription confirmation (`{"id": 1, "result": null}`).
    - Drop stale updates where `data["u"] <= state.local_book["lastUpdateId"]`.
    - Acquire `state.thread_lock`, apply bid/ask deltas, update `lastUpdateId`, append snapshot (`{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}`) to `state.history_order_book`, release lock.
    - Call `calculate_best_quote()` with the updated book (prints live spread to stdout).
12. On each incoming User Data Stream message (`MessageHandler.handle_balance_message`):
    - Ignore any event that is not `outboundAccountPosition`.
    - Under `state.thread_balance_lock`, update `state.balance_status` for each tracked asset (`CRYPTOCCY`, `CCY`) using the `"f"` (free) field.
13. After `session_seconds`, set `stop_event`, stop both WebSocket stream clients (`ws_client.stop()`, `ws_user_client.stop()`), stop the order executor (`executor.stop()`), and join all threads (10 s HFT, 15 s historical, 5 s keepalive).  A `KeyboardInterrupt` (Ctrl-C) triggers the same shutdown path early.

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

## Best Quote Selection (`quotes.py`)

`find_best_quote(df, position)` applies the full pipeline and returns the single best level for a given strategy (`"buy"` or `"sell"`).

A level is considered an **opportunity** when **all** of the following hold:

| # | Condition | Purpose |
|---|-----------|---------|
| 1 | $\mathbb{1}_{\text{thin}} = \text{False}$ | Micro-price signal is backed by real depth |
| 2 | $\mathbb{1}_{\mu > \text{mid}}$ matches strategy direction | Directional alignment ($P_{\mu} > P_{\text{mid}}$ for buy, $\leq$ for sell) |
| 3 | $\mathbb{1}_{D \geq 50\%} = \text{True}$ | Sufficient liquidity relative to top-of-book |
| 4 | Level index $\neq 0$ | Skip the best quote itself (we target deeper levels) |

Among qualifying levels, the one with the **highest Score** is returned.

After iterating over all depth limits, `rest_spot_main.py` collects every best quote into `all_quotes` and selects the **latest buy** and **latest sell** results.

---

## Visualisation (`plot_helpers.py`)

| Function | Description |
|----------|-------------|
| `plot_depth_bid_ask(df)` | Two-panel Plotly chart — bid/ask prices (top) and bid/ask volumes (bottom) |
| `plot_ohlc_with_volume(client, symbol, interval, lookback)` | Fetches historical klines and plots OHLC candlesticks with a volume sub-chart |

---

## REST Execution Flow (`rest_spot_main.py`)

1. Load API keys from `.env`.
2. Connect to Binance Testnet (`testnet=True`).
3. Capture the initial `lastUpdateId` from the order book as a baseline.
4. For each depth limit in `[5, 10, 15, 20, 50]`:
   - Fetch the order book for **BTCUSDT**.
   - **Staleness guard** — compute the gap between the current and initial `lastUpdateId`. If the gap exceeds 100, the book has changed too much for the microstructure signals to be reliable; the loop re-fetches every second until conditions calm down.

     | Gap | Regime |
     |-----|--------|
     | $\leq 5$ | Quiet |
     | $6 – 50$ | Normal |
     | $51 – 99$ | Volatile |
     | $\geq 100$ | Extreme — **wait & retry** |

   - Compute metrics via `get_order_book_metrics()`.
   - For each strategy (`buy`, `sell`):
     - Run `find_best_quote()` → print the best level if found.
   - Plot the depth snapshot.
5. Select the latest buy and sell quotes from all collected results.

---


## Current Status & Roadmap

| Status | Item |
|--------|------|
| ✅ Done | REST path: metrics → indicators → scores → best quote (buy & sell) |
| ✅ Done | WebSocket: real-time local order book maintained from snapshot + diff stream |
| ✅ Done | Staleness guard in REST path (`gap_id` logic) |
| ✅ Done | Plotly visualisations (depth chart, OHLC with volume) |
| ✅ Done | Consistent symbol across REST and WebSocket paths (BTCUSDT) |
| ✅ Done | `config_parameters.py` — central constants file; `SYMBOL`, `CCY`, `CRYPTOCCY`, `N_LEVELS` added and used project-wide |
| ✅ Done | `OrderBookState` — shared state container (local book, history, balance, two locks) |
| ✅ Done | `MessageHandler` class — `handle_depth_message` (order book) and `handle_balance_message` (User Data Stream) |
| ✅ Done | `AnalysisEngine` class — `low_latency_analysis` (5 s, with balance guard) and `historical_analysis` (5 min) background loops |
| ✅ Done | Session duration fixed at `DEFAULT_SESSION_MINUTES = 15 min` (no user prompt); historical every 5 min → 3 iterations per session |
| ✅ Done | Thread startup order fixed — WebSocket opens first, 1 s warm-up, then threads start |
| ✅ Done | Port strategy logic (metrics → indicators → scores → best quote) into the WebSocket path via `low_latency_analysis` |
| ✅ Done | Balance check — `balance_status` seeded from REST on startup, kept live via `outboundAccountPosition` User Data Stream events; `low_latency_analysis` skips iterations when both balances are below threshold |
| ✅ Done | User Data Stream — dedicated `ws_user_client` on testnet endpoint; `listenKey` keepalive thread renews every 30 min |
| ✅ Done | `OrderExecutor` — LIMIT GTC orders via Binance WebSocket API (`SpotWebsocketAPIClient`); quantity = `aq` for BUY / `bq` for SELL; `micro_price` as limit price; balance guards; strategy validation; async response handling via `handle_order_response` callback |
| ✅ Done | Project reorganised into packages: `core/` (state + data ingestion), `strategy/` (analysis + scoring), `execution/` (order placement), `visualization/` (plotting) |
| ✅ Done | `history_order_book` snapshot enriched — now stores `best_bid`, `best_ask` (float), `volume_best_bid`, `volume_best_ask` (float) per tick; string→float conversion at append time |
| ✅ Done | `historical_analysis` — acquires `thread_lock`, copies deque to plain list, computes `bid_vwap` and `ask_vwap` via `volume_weighted_average_price()` (numpy), publishes under `_vwap_lock` |
| ✅ Done | VWAP cross-thread integration — `low_latency_analysis` reads `_bid_vwap` / `_ask_vwap` under `_vwap_lock` and gates execution: BUY only if `micro_price > ask_vwap` (or VWAP not yet available); SELL only if `micro_price < bid_vwap` (or VWAP not yet available) |
| 💡 Idea | Replace per-tick `calculate_best_quote()` calls with a periodic evaluation (every N updates) to avoid running the full pipeline on every 100 ms tick |
| 🔜 Todo | `historical_analysis` — implement further historical logic (spread distributions, regime detection, etc.) using `history_order_book` snapshots |
