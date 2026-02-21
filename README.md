# Binance Spot Testnet — Order Book Analysis

A Python toolkit for real-time order book analysis on the **Binance Spot Testnet**.  
It fetches multi-depth snapshots, computes microstructure metrics, scores each price level, and surfaces the single best quote for buy and sell strategies.

---

## Project Structure

```
binance_spot_testnet/
├── spot_main.py      # Orchestration — connects to Binance, loops over depths
├── metrics.py        # Order book metric calculations
├── indicators.py     # Strategy-specific indicator columns
├── scores.py         # Weighted opportunity scoring
├── quotes.py         # Best quote selection logic
├── plot_helpers.py   # Plotly visualisations (depth, OHLC)
└── README.md
```

---

## Setup

1. **Install dependencies**

   ```bash
   pip install python-binance python-dotenv pandas numpy plotly
   ```

2. **Create a `.env` file** in the project root:

   ```
   BINANCE_TESTNET_API_KEY=your_api_key
   BINANCE_TESTNET_SECRET_KEY=your_secret_key
   ```

   Keys are generated at <https://testnet.binance.vision/>.

3. **Run**

   ```bash
   python spot_main.py
   ```

---

## Pipeline Overview

```
Binance API
    │
    ▼
┌────────────────┐     ┌────────────────┐     ┌────────────┐     ┌────────────┐
│  spot_main.py  │────▶│  metrics.py    │────▶│ indicators │────▶│  scores.py │
│  (orchestrate) │     │  (enrich df)   │     │    .py     │     │  (score)   │
└────────────────┘     └────────────────┘     └────────────┘     └────────────┘
                                                                       │
                                                                       ▼
                                                               ┌──────────────┐
                                                               │  quotes.py   │
                                                               │ (best quote) │
                                                               └──────────────┘
```

---

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

---

## Visualisation (`plot_helpers.py`)

| Function | Description |
|----------|-------------|
| `plot_depth_bid_ask(df)` | Two-panel Plotly chart — bid/ask prices (top) and bid/ask volumes (bottom) |
| `plot_ohlc_with_volume(client, symbol, interval, lookback)` | Fetches historical klines and plots OHLC candlesticks with a volume sub-chart |

---

## Execution Flow (`spot_main.py`)

1. Load API keys from `.env`.
2. Connect to Binance Testnet (`testnet=True`).
3. Capture the initial `lastUpdateId` from the order book as a baseline.
4. For each depth limit in `[5, 10, 15, 20, 50]`:
   - Fetch the order book for **BNBUSDT**.
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
