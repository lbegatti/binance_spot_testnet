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
├── config_parameters.py       # Central configuration — all tunable constants in one place
├── rest_spot_main.py          # REST orchestration — loops over depth limits
├── websocket_spot_main.py     # WebSocket — real-time local order book + session driver
├── order_book_state.py        # Shared state container (local book + history + lock)
├── message_handler.py         # WebSocket callback — maintains local book in real time
├── analysis.py                # AnalysisEngine — HFT (5 s) and historical (5 min) loops
├── best_quote_calculator.py   # Live spread printer — prints best_bid | best_ask to stdout on every tick
├── metrics.py                 # Order book metric calculations
├── indicators.py              # Strategy-specific indicator columns
├── scores.py                  # Weighted opportunity scoring
├── quotes.py                  # Best quote selection logic
├── plot_helpers.py            # Plotly visualisations (depth, OHLC)
└── README.md
```

---

## Configuration (`config_parameters.py`)

All tunable constants are centralised in `config_parameters.py`. Edit this file to change behaviour across the entire project without touching any logic files.

| Group | Constant | Default | Description |
|---|---|---|---|
| **Symbol** | `SYMBOL` | `"BTCUSDT"` | Trading pair used across all REST and WebSocket calls |
| **Symbol** | `CCY` | `"USDT"` | Quote currency |
| **Symbol** | `CRYPTOCCY` | `"BTC"` | Base / crypto currency |
| **Order book state** | `HISTORY_MAXLEN` | `6000` | Max snapshots in `history_order_book` — at 100 ms intervals this covers ~10 min |
| **Analysis cadence** | `HFT_INTERVAL` | `5` s | Time between HFT evaluations |
| **Analysis cadence** | `HIST_INTERVAL` | `300` s | Time between historical analyses (5 min) |
| **Analysis cadence** | `MIN_SNAPSHOTS` | `100` | Minimum snapshots required before historical analysis runs |
| **WebSocket session** | `DEFAULT_SESSION_MINUTES` | `15` min | Default session length (fixed — no startup prompt) |
| **WebSocket session** | `HTF_JOIN_TIMEOUT` | `10` s | Max wait for `low_latency_analysis` thread on shutdown |
| **WebSocket session** | `HIST_JOIN_TIMEOUT` | `15` s | Max wait for `historical_analysis` thread on shutdown |
| **Binance connection** | `RECV_WINDOW` | `5000` ms | Binance REST request validity window |
| **Binance connection** | `SNAPSHOT_DEPTH` | `100` | Order book levels in the seed snapshot |
| **Binance connection** | `WS_SPEED` | `100` ms | WebSocket diff-depth update interval |

**Imported by:**
- `order_book_state.py` — `HISTORY_MAXLEN`
- `analysis.py` — `HFT_INTERVAL`, `HIST_INTERVAL`, `MIN_SNAPSHOTS`
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
Binance REST API                   Binance WebSocket
    │  depth() snapshot                │  diff_book_depth (100 ms)
    ▼                                  ▼
┌──────────────────┐       ┌──────────────────────┐
│  OrderBookState  │◀──────│   MessageHandler      │
│  (shared state)  │       │   .handle_depth_msg() │
│                  │       └──────────────────────┘
│  • local_book    │
│  • history_book  │       ┌──────────────────────┐
│  • thread_lock   │◀──────│   AnalysisEngine      │
└──────────────────┘       │   .low_latency_analysis() │
                           │    (runs strategy)        │
                           │   .historical_anal..()    │
                           └──────────────────────┘
                                      ▲
                           ┌──────────────────────┐
                           │  websocket_spot_main  │
                           │  (session driver)     │
                           └──────────────────────┘
```

**Component responsibilities:**

| Class / Module | File | Role |
|---|---|---|
| *(constants)* | `config_parameters.py` | Single source of truth for all tunable constants (`SYMBOL`, `CCY`, `CRYPTOCCY`, intervals, depths, timeouts) — imported by `order_book_state`, `analysis`, and `websocket_spot_main` |
| `OrderBookState` | `order_book_state.py` | Single source of truth — owns `local_book`, `history_order_book`, and `thread_lock` |
| `MessageHandler` | `message_handler.py` | WebSocket callback — merges every diff-depth tick into `OrderBookState` using `state.thread_lock`, appends snapshots to `state.history_order_book`, calls `calculate_best_quote()` |
| `AnalysisEngine` | `analysis.py` | Runs two background loops (HFT and historical) that read from `OrderBookState` via the shared lock |
| `websocket_spot_main` | `websocket_spot_main.py` | Session driver — instantiates all three classes, starts threads, manages session lifetime |

**How the components interact:**

1. `websocket_spot_main.py` creates a single `OrderBookState` instance and injects it into both `MessageHandler` and `AnalysisEngine`.
2. All concurrent access to `local_book` and `history_order_book` is serialised through `state.thread_lock` — the single `threading.Lock` that lives on `OrderBookState` and is shared by every consumer.  `MessageHandler.handle_depth_message` acquires it to write; `AnalysisEngine.low_latency_analysis` acquires it to take a read-only copy, then releases it before any heavy computation.
3. `MessageHandler` is the **only writer** to `OrderBookState`; `AnalysisEngine` is **read-only** — it copies the data under the lock and immediately releases it before doing heavier computation.

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

---

## WebSocket Execution Flow (`websocket_spot_main.py`)

1. Load API keys from `.env`.
2. Connect to Binance Testnet via `binance-connector` REST client.
3. Consolidate non-BTC/USDT balances into USDT via market sell orders.
4. Session duration fixed at `DEFAULT_SESSION_MINUTES` (15 min) — no user prompt.
5. Fetch a depth snapshot for **BTCUSDT** (100 levels) to seed `OrderBookState.local_book`.
6. Instantiate `OrderBookState`, `MessageHandler`, and `AnalysisEngine`, injecting the shared state into the latter two.  Set `stop_event = threading.Event()`.
7. Open a `SpotWebsocketStreamClient` subscribing to `diff_book_depth` at 100 ms intervals.
8. Wait 1 second for the first diff-depth messages to arrive and populate `local_book["bids"]`.
9. Start `AnalysisEngine.low_latency_analysis` and `AnalysisEngine.historical_analysis` as named daemon threads — **after** the WebSocket is open so bids are available on the first low-latency wake-up.
10. On each incoming message (`MessageHandler.handle_depth_message`):
    - Skip the initial subscription confirmation (`{"id": 1, "result": null}`).
    - Drop stale updates where `data["u"] <= state.local_book["lastUpdateId"]`.
    - Acquire `state.thread_lock`, apply bid/ask deltas, update `lastUpdateId`, append snapshot to `state.history_order_book`, release lock.
    - Call `calculate_best_quote()` with the updated book (prints live spread to stdout).
11. After `session_seconds`, set `stop_event`, call `ws_client.stop()`, and join both threads (timeouts: 10 s HFT, 15 s historical).  A `KeyboardInterrupt` triggers the same shutdown path early.

## Notation

| Symbol | Column name | Description |
|--------|-------------|-------------|
| $P_b$, $Q_b$ | `bid_price`, `bid_quantity` | Best bid price and quantity at a level |
| $P_a$, $Q_a$ | `ask_price`, `ask_quantity` | Best ask price and quantity at a level |
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
| ✅ Done | `config_parameters.py` — central constants file; `config.py` renamed; `SYMBOL`, `CCY`, `CRYPTOCCY` added and used project-wide |
| ✅ Done | `OrderBookState` — shared state container (local book, history, lock) |
| ✅ Done | `MessageHandler` class — WebSocket callback decoupled from analysis logic |
| ✅ Done | `AnalysisEngine` class — HFT (5 s) and historical (5 min) background loops |
| ✅ Done | Session duration fixed at `DEFAULT_SESSION_MINUTES = 15 min` (no user prompt); historical every 5 min → 3 iterations per session |
| ✅ Done | Thread startup order fixed — WebSocket opens first, 1 s warm-up, then threads start so `local_book["bids"]` is populated on the first HFT iteration |
| ✅ Done | Port strategy logic (metrics → indicators → scores → best quote) into the WebSocket path |
| 💡 Idea | Replace per-tick `calculate_best_quote()` calls with a periodic evaluation (every N updates) to avoid running the full pipeline on every 100 ms tick |
