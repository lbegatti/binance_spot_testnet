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
| **Order book state** | `HISTORY_MAXLEN` | `3000` | Max snapshots in `history_order_book` — at 100 ms intervals this covers ~5 min.  Each entry: `{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}` (all numeric) |
| **Order book state** | `N_LEVELS` | `50` | Number of order book levels used in `low_latency_analysis` |
| **Analysis cadence** | `HFT_INTERVAL` | `1` s | Time between low-latency evaluations |
| **Analysis cadence** | `HIST_INTERVAL` | `60` s | Time between historical analyses (1 min) |
| **Analysis cadence** | `MIN_SNAPSHOTS` | `100` | Minimum snapshots required before historical analysis runs |
| **WebSocket session** | `DEFAULT_SESSION_MINUTES` | `20` min | Default session length (fixed — no startup prompt) |
| **WebSocket session** | `HTF_JOIN_TIMEOUT` | `10` s | Max wait for `low_latency_analysis` thread on shutdown |
| **WebSocket session** | `HIST_JOIN_TIMEOUT` | `15` s | Max wait for `historical_analysis` thread on shutdown |
| **Binance connection** | `RECV_WINDOW` | `5000` ms | Binance REST request validity window |
| **Binance connection** | `SNAPSHOT_DEPTH` | `100` | Order book levels in the seed snapshot |
| **Binance connection** | `WS_SPEED` | `100` ms | WebSocket diff-depth update interval |
| **Quote throttle** | `QUOTE_EVERY_N_TICKS` | `10` | Ticks between `calculate_best_quote()` calls.  At `WS_SPEED=100 ms`, 10 ticks ≈ 1 s |

**Imported by:**
- `core/order_book_state.py` — `HISTORY_MAXLEN`, `CRYPTOCCY`, `CCY`
- `core/message_handler.py` — `CRYPTOCCY`, `CCY`, `QUOTE_EVERY_N_TICKS`
- `strategy/analysis.py` — `HFT_INTERVAL`, `HIST_INTERVAL`, `MIN_SNAPSHOTS`, `N_LEVELS`, `CCY`, `CRYPTOCCY`
- `execution/order_executor.py` — `SYMBOL`, `CRYPTOCCY`, `CCY`, `RECV_WINDOW`
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
| *(constants)* | `config_parameters.py` | Single source of truth for all tunable constants (`SYMBOL`, `CCY`, `CRYPTOCCY`, intervals, depths, timeouts) — imported by every package |
| `OrderBookState` | `core/order_book_state.py` | Single source of truth — owns `local_book`, `history_order_book`, `balance_status`, `thread_lock`, and `thread_balance_lock` |
| `MessageHandler` | `core/message_handler.py` | One active WebSocket callback: `handle_depth_message` (merges diff-depth ticks into `local_book`, appends snapshots, calls `calculate_best_quote` every 10th tick).  `handle_balance_message` is preserved but superseded — see `OrderExecutor` |
| `AnalysisEngine` | `strategy/analysis.py` | Runs two background loops (`low_latency_analysis` and `historical_analysis`) that read from `OrderBookState` via the shared locks; delegates order placement to `OrderExecutor` |
| `OrderExecutor` | `execution/order_executor.py` | Places LIMIT GTC orders **and** maintains real-time balance updates via a single Binance WebSocket API connection.  On connect: `session.logon` (HMAC-signed) → `userDataStream.subscribe` → receives `outboundAccountPosition` push events on the same socket.  Falls back to REST for orders if WS is unavailable; balances fall back to startup REST snapshot |
| `websocket_main` | `websocket_main.py` | Session driver — instantiates all classes, seeds initial balances into `state`, opens WebSocket streams, starts all threads, manages session lifetime and shutdown |

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

---

## Session Duration

### Rationale

Both analysis loops in `AnalysisEngine` are designed to run indefinitely:

| Loop | Cadence | Purpose |
|------|---------|---------|
| `low_latency_analysis` | every **1 s** | Near-real-time best bid/ask evaluation |
| `historical_analysis` | every **1 min** | VWAP computation over the rolling snapshot window |

Rather than running forever, `websocket_main.py` uses a fixed session duration set by `DEFAULT_SESSION_MINUTES` (no startup prompt).

The **default of 20 minutes** is chosen deliberately:

| Metric | Value at 20 min |
|--------|----------------|
| Low-latency iterations (`low_latency_analysis`) | $20 \times 60 / 1 = \mathbf{1{,}200}$ |
| Historical iterations (`historical_analysis`) | $20 \times 60 / 60 = \mathbf{20}$ |
| Order book snapshots in history | up to $20 \times 60 \times 10 = \mathbf{12{,}000}$ ticks (capped at `maxlen=3000` ≈ last 5 min) |

When the session duration elapses, `websocket_main.py` sets `stop_event`, calls `ws_client.stop()` to close the stream cleanly, and joins both analysis threads (with timeouts of 10 s and 15 s respectively). A `KeyboardInterrupt` (Ctrl-C) triggers the same shutdown path early.

### Thread Timeline (default 20-min session)

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
t=20min   low_latency iteration #1200
          historical iteration #20 → refreshes VWAP one last time
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
| 20 min | ~12 000 sent | **3 000 (capped)** | 20th runs (reads 3 000 entries) |

After ~5 minutes the deque hits `maxlen=3000` and becomes a true **rolling window** of the last ~5 minutes. Each `historical_analysis` iteration operates on whatever is currently in the window — not a fixed block.

---

## WebSocket Execution Flow (`websocket_main.py`)

1. Load API keys from `.env`.
2. Connect to Binance Testnet via `binance-connector` REST client.
3. Consolidate non-BTC/USDT balances into USDT via market sell orders.
4. Seed `state.balance_status` with the REST-fetched `usdt_balance` and `btc_balance` before any thread starts.
5. Session duration fixed at `DEFAULT_SESSION_MINUTES` (20 min) — no user prompt.
6. Fetch a depth snapshot for **BTCUSDT** (100 levels) to seed `OrderBookState.local_book`.
7. Instantiate `OrderBookState`, `MessageHandler`, `OrderExecutor`, and `AnalysisEngine`, injecting the shared state.  `OrderExecutor` creates its own `SpotWebsocketAPIClient` internally (connected to `wss://testnet.binance.vision/ws-api/v3`) with `on_open=self._on_ws_open` and `on_message=self.handle_order_response`.  On socket open it automatically sends `session.logon` → `userDataStream.subscribe` to enable real-time balance push events on the same connection.  Set `stop_event = threading.Event()`.
8. Open one `SpotWebsocketStreamClient` (`ws_client`) on the production stream endpoint; subscribe to `diff_book_depth` at 100 ms intervals; callback: `handle_depth_message`.
9. Wait 1 second for the first diff-depth messages to arrive and populate `local_book["bids"]`.
10. Start daemon threads:
    - `low_latency_thread` → `AnalysisEngine.low_latency_analysis`
    - `hist_thread` → `AnalysisEngine.historical_analysis`
11. On each incoming depth message (`MessageHandler.handle_depth_message`):
    - Skip the initial subscription confirmation (`{"id": 1, "result": null}`).
    - Drop stale updates where `data["u"] <= state.local_book["lastUpdateId"]`.
    - Acquire `state.thread_lock`, apply bid/ask deltas, update `lastUpdateId`, append snapshot (`{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}`) to `state.history_order_book`, release lock.
    - Call `calculate_best_quote()` every `QUOTE_EVERY_N_TICKS` ticks (~1 s) with the updated book (prints live spread to stdout).  The local book is still updated on every tick.
12. On each incoming `outboundAccountPosition` push event (`OrderExecutor.handle_order_response` → `_handle_balance_update`):
    - Under `state.thread_balance_lock`, update `state.balance_status` for each tracked asset (`CRYPTOCCY`, `CCY`) using the `"f"` (free) field.
13. After `session_seconds`, set `stop_event`, stop `ws_client`, stop `executor` (closes the WS API + user-data connection), and join both analysis threads (10 s low-latency, 15 s historical).  A `KeyboardInterrupt` (Ctrl-C) triggers the same shutdown path early.

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

After iterating over all depth limits, `restapi_main.py` collects every best quote into `all_quotes` and selects the **latest buy** and **latest sell** results.

---

## Visualisation (`plot_helpers.py`)

| Function | Description |
|----------|-------------|
| `plot_depth_bid_ask(df)` | Two-panel Plotly chart — bid/ask prices (top) and bid/ask volumes (bottom) |
| `plot_ohlc_with_volume(client, symbol, interval, lookback)` | Fetches historical klines and plots OHLC candlesticks with a volume sub-chart |

---

## REST Execution Flow (`restapi_main.py`)

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
| ✅ Done | `MessageHandler` class — `handle_depth_message` (order book); `handle_balance_message` preserved but superseded by `OrderExecutor._handle_balance_update` |
| ✅ Done | `AnalysisEngine` class — `low_latency_analysis` (1 s, with balance guard) and `historical_analysis` (1 min) background loops |
| ✅ Done | Session duration fixed at `DEFAULT_SESSION_MINUTES = 20 min` (no user prompt); historical every 1 min → 20 iterations per session |
| ✅ Done | Thread startup order fixed — WebSocket opens first, 1 s warm-up, then threads start |
| ✅ Done | Port strategy logic (metrics → indicators → scores → best quote) into the WebSocket path via `low_latency_analysis` |
| ✅ Done | Balance check — `balance_status` seeded from REST on startup, kept live via `session.logon` → `userDataStream.subscribe` → `outboundAccountPosition` push events on the OrderExecutor's WS API connection (no listenKey needed); `low_latency_analysis` skips iterations when both balances are below threshold |
| ✅ Done | `OrderExecutor` — LIMIT GTC orders **and** real-time balance tracking via a single Binance WebSocket API connection (`session.logon` + `userDataStream.subscribe`); REST fallback for orders when WS is unavailable; `handle_order_response` routes push events to `_handle_balance_update` |
| ✅ Done | Project reorganised into packages: `core/` (state + data ingestion), `strategy/` (analysis + scoring), `execution/` (order placement), `visualization/` (plotting) |
| ✅ Done | `history_order_book` snapshot enriched — now stores `best_bid`, `best_ask` (float), `volume_best_bid`, `volume_best_ask` (float) per tick; string→float conversion at append time |
| ✅ Done | `historical_analysis` — acquires `thread_lock`, copies deque to plain list, computes `bid_vwap` and `ask_vwap` via `volume_weighted_average_price()` (numpy), publishes under `_vwap_lock` |
| ✅ Done | VWAP momentum-confirmation filter — `low_latency_analysis` reads `_bid_vwap` / `_ask_vwap` under `_vwap_lock` and gates execution: BUY only if `micro_price > ask_vwap` (upward momentum) or VWAP not yet available; SELL only if `micro_price < bid_vwap` (downward momentum) or VWAP not yet available.  May be inverted for buy-the-dip strategy in the future |
| ✅ Done | `calculate_best_quote()` throttled to every 10th tick (~1 s) instead of every 100 ms tick — reduces CPU work and console noise while keeping the local book updated on every tick |
| 🔜 Todo | **HMM regime detection** — train a `GaussianHMM` (via `hmmlearn`) on kline-derived features (`return`, `volatility`, `obi_proxy = taker_buy_vol / vol × 2 − 1`) to classify the market into hidden states (e.g. trending up / ranging / trending down).  Use the predicted regime as a pre-filter in `low_latency_analysis`: execute BUY signals only in an upward-trending regime, SELL signals only in a downward-trending regime, and suppress both in a ranging regime — sitting on top of the existing VWAP momentum-confirmation filter |
| 🔜 Todo | **Backtesting** — once the HMM regime filter is in place, replay one month of Binance 1-minute OHLCV klines (downloaded via `Client.get_historical_klines`) through the full strategy logic (`_build_levels`, `_collect_candidates`, `_select_best_opportunity` + VWAP filter + HMM regime gate) using a synthetic N-level order book reconstructed from `taker_buy_volume` / `volume` imbalance.  Simulate LIMIT GTC fill logic (fill if kline low/high crosses order price), track P&L, and produce a report (equity curve vs buy-and-hold baseline, Sharpe ratio, max drawdown, win rate, fill rate) |
