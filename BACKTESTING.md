# Backtesting Plan — Binance Spot Testnet Strategy

> **Status:** Steps 1–7 implemented (``data.py``, ``synthetic_book.py``,
> ``signals.py``, ``pnl.py``, ``run_backtest.py``, ``regime_validation.py``,
> ``visualization.py``).  Step 8 (sensitivity analysis, Use Case A) is implemented; Use Case B (180-day window) is deferred.
> See the Implementation Roadmap at the bottom for a per-module progress tracker.

---

## Overview

The live strategy (defined in `strategy/analysis.py`) makes order decisions in
real time based on a 50-level order book snapshot, a VWAP momentum filter, and
an HMM regime filter.  The purpose of backtesting is to replay equivalent
logic against historical data and measure whether the strategy generates
positive risk-adjusted returns before committing further to live execution.

---

## Fundamental Limitation — Level-2 Data

The live strategy relies on **full depth-of-book data** (50 levels of bids and
asks at 100 ms resolution).  Binance **does not provide historical Level-2
order book data** via its public REST API.  The closest freely available
substitute is **OHLCV kline data**, which is available at any standard interval
(1 m, 5 m, 1 h, etc.) and includes taker volume, number of trades, and
high/low/close prices.

**Consequence:** A kline-based backtest is necessarily an approximation.
It tests whether the *signals* embedded in the strategy (momentum, OBI proxy,
volatility regime) have predictive power, but it cannot reproduce exact fill
prices or queue position.

---

## Step 1 — Data Acquisition

### Source
- Binance public REST endpoint: `GET /api/v3/klines`
- Available without authentication.
- Available intervals: `1m`, `3m`, `5m`, `15m`, `1h`, …

### Suggested window
- **Instrument:** BTCUSDT
- **Interval:** 1 minute  (matches the `HMM_INTERVAL` already used in production)
- **Lookback:** 180 days  (~259,200 candles — captures a broad range of market
  regimes; crypto trades 24/7 so all 30 days are fully populated with no weekend gaps)

### Columns returned per candle
```
open_time, open, high, low, close, volume,
close_time, quote_asset_volume, num_trades,
taker_buy_base_vol, taker_buy_quote_vol, ignore
```

---

## Step 2 — Feature Engineering

Two separate feature sets are needed:

### 2a — HMM Regime Features (from klines only)

These four columns are identical to the ones `RegimeDirector` uses in
production.  They are computed **directly from kline data** and do **not**
require a reconstructed order book:

| Feature | Formula | Used by |
|---|---|---|
| `return` | `close.pct_change()` | `RegimeDirector` |
| `volatility` | `(high − low) / close` | `RegimeDirector` |
| `obi_proxy` | `(taker_buy_base_vol / volume) × 2 − 1` ∈ `[-1, +1]` | `RegimeDirector` |
| `trade_density` | `num_trades / volume` | `RegimeDirector` |

Running `RegimeDirector` on these features produces the same regime labels
(`trending_up`, `trending_down`, `high_volatility`, `neutral`) that gate
orders in the live system.

### 2b — Synthetic Order Book (from klines → full pipeline)

See **Step 2c** below.  Once a synthetic 50-level order book is built for each
candle, it is passed directly into the **existing production code**:

```
synthetic book  →  metrics.py  →  indicators.py  →  scores.py  →  quotes.py
```

This means the backtest computes the **exact same metrics** as the live
strategy — `total_depth`, `mid_price`, `micro_price`, `OBI`, `bid_ask_spread`,
`micro_mid_delta`, `is_thin_micro_effect`, `is_total_depth_50pct_l0`, and the
weighted opportunity score — rather than simplified proxies.

> The only metric that requires special treatment is **OBI**: the symmetric
> synthetic book produces OBI = 0 at every level.  See the asymmetry
> injection described in Step 2c.

---

## Step 2c — Synthetic Order Book Reconstruction

The live strategy consumes **50 levels of bids and asks** per evaluation cycle.
Kline data provides only a single OHLCV bar per candle.  To feed the existing
scoring pipeline (`metrics.py` → `indicators.py` → `scores.py`) during a
backtest, a synthetic depth ladder must be constructed for each candle.

### Spread reconstruction — synthetic book vs fill model

The high-low range of a candle is used **only** to reconstruct the synthetic
order book depth (bid/ask level spacing).  It is **not** used as the fill-cost
model for P&L simulation:

```
synthetic_half_spread = (high − low) / 2
synthetic_best_bid    = close − synthetic_half_spread
synthetic_best_ask    = close + synthetic_half_spread
```

> **Why `close` and not `(high + low) / 2`?**  The `close` is the last
> traded price and is therefore the best single-point estimate of where the
> market was at the end of the candle.  Using `(high + low) / 2` as mid would
> over-smooth intra-bar direction.

#### Fill-cost model for P&L (Step 4)

For P&L simulation a **bps-based half-spread** is used instead of the candle
range, because a LIMIT order fills at or inside the real spread — not at the
candle extreme:

```
half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20 000
BUY  fill   = close + half_spread   (you pay the synthetic ask)
SELL fill   = close − half_spread   (you receive the synthetic bid)
```

Default `BACKTEST_FILL_SPREAD_BPS = 5` → `half_spread ≈ $20` at $80 k BTC,
matching the realistic Binance BTCUSDT spread of ~1–5 bps.

> **Why NOT `(high − low) / 2` for fills?**
> A 1-min candle range is typically $50–$300, giving `half_spread ≈ $25–$150`.
> Over 1,000+ round trips this is $25,000–$150,000 in friction on a $10,000
> portfolio — 10–100× the real exchange spread — and causes 100 % drawdown.
> The candle range is an upper bound on intra-bar volatility, not the cost of
> crossing the spread with a limit order.

### Generating N synthetic levels

Starting from the reconstructed best bid and best ask, deeper levels are
generated by stepping away from the mid at regular price intervals.  Volume
at each level decays exponentially to replicate the natural thinning of
liquidity away from the top of book:

```
tick_size      = synthetic_best_ask − synthetic_best_bid   (= high − low)
base_volume    = volume / N_LEVELS      (candle volume spread evenly as the base)
decay_factor   = 0.80                   (each level retains 80 % of the previous level's volume)

For level i in 0 … N_LEVELS − 1:
    bid_price(i)    = synthetic_best_bid  − i × tick_size
    ask_price(i)    = synthetic_best_ask  + i × tick_size
    bid_quantity(i) = base_volume × decay_factor ^ i
    ask_quantity(i) = base_volume × decay_factor ^ i
```

This produces a symmetric ladder of 50 bid levels and 50 ask levels.

However, a purely symmetric book makes **OBI = 0 at every level**, which
defeats the micro-price, OBI, and the 70/30 weighted score in `scores.py`.
The taker volume split available in the kline must be used to **inject
asymmetry** before the book is passed to the pipeline.

### OBI asymmetry injection

The kline provides `taker_buy_base_vol` and `volume`.  From these:

```
buy_ratio  = taker_buy_base_vol / volume          ∈ (0, 1)
sell_ratio = 1 − buy_ratio
```

Before passing the synthetic book to `metrics.py`, rescale each side:

```
For every level i:
    bid_quantity(i)  ×= 2 × buy_ratio
    ask_quantity(i)  ×= 2 × sell_ratio
```

| `buy_ratio` | Bid multiplier | Ask multiplier | Resulting OBI sign |
|---|---|---|---|
| 0.5 (neutral) | 1.0 | 1.0 | 0 |
| 0.7 (buy-heavy) | 1.4 | 0.6 | positive (more resting bids) |
| 0.3 (sell-heavy) | 0.6 | 1.4 | negative (more resting asks) |

This makes `micro_price` tilt towards the thinner side, OBI reflects taker
flow, and the downstream weighted score discriminates between levels — all
using the **real production code**, not proxy formulae.

### What this approximation captures and what it misses

| Aspect | Captured? | Note |
|---|---|---|
| Best-bid / best-ask prices | ✅ approximate | Derived from high-low spread |
| Relative depth thinning away from mid | ✅ | Exponential decay is a standard assumption |
| Directional OBI from taker volume | ✅ | Injected via buy/sell ratio scaling |
| `micro_price` tilt | ✅ | Follows from asymmetric depth (natural consequence) |
| Weighted opportunity score (70/30) | ✅ | Computed by the real `scores.py` code |
| Real resting order distribution | ❌ | True depth profiles are irregular and change tick-by-tick |
| Exact per-level asymmetry | ❌ | Asymmetry is uniform across all levels; real books have level-specific imbalances |
| Queue position within a level | ❌ | Irrelevant at kline resolution |

---

## Step 3 — Strategy Signal Replay (Full Pipeline)

Because a complete synthetic order book is available (Step 2c), the backtest
runs the **same production code path** used by `low_latency_analysis()` in the
live system.  No simplified proxy rules are needed.

### Per-candle procedure

The dataset contains **~259,200 rows** (180 days × 1,440 candles/day at 1 m).
For **each row** a fresh synthetic 50-level order book is constructed (Step 2c),
and the full production pipeline is run on it.  This produces one signal
decision per minute — the exact equivalent of one `low_latency_analysis()`
iteration in the live system (which runs every 1 s, but at 1 m kline resolution
the minimum granularity is 1 candle = 1 min).

> **Memory note:** 259,200 candles × 100 rows per book (50 bids + 50 asks) = 25,920,000
> order book rows in total — but they are **never all in memory at once**.  The loop
> constructs one 100-row DataFrame, runs the pipeline on it, records the signal,
> then discards it before moving to the next candle.  Peak memory at any moment is
> one synthetic book (100 rows) plus the 259,200-row klines DataFrame — well under
> 200 MB and entirely manageable.

There are **three independent data flows** running in parallel at every candle
`t`.  They use completely different input shapes and are then combined into a
single signal decision.

---

### Flow A — Opportunity Scoring (the 100-row synthetic book)

This is the **only** flow that uses the 100-row synthetic order book.  It
replicates what `low_latency_analysis()` does with the live depth snapshot:

```
synthetic_book(t)   ← 100 rows: 50 bid levels + 50 ask levels (Step 2c)
        │
        ▼
get_order_book_metrics(df)          ← metrics.py
        │                              total_depth, mid_price, micro_price,
        │                              OBI, bid_ask_spread, micro_vs_mid
        ▼
add_strategy_indicators(df, "buy")  ← indicators.py (run once per side)
add_strategy_indicators(df, "sell")    micro_mid_delta, is_thin_micro_effect,
        │                              is_total_depth_50pct_l0
        ▼
get_weighted_volume_micro_spread_score()  ← scores.py  (0.7 depth + 0.3 delta)
        │
        ▼
_select_best_opportunity()          ← quotes.py / analysis.py
                                       argmax score → candidate tuple or None
```

**Input:** 100 rows (one per synthetic order book level).
**Output:** 0, 1, or 2 candidate tuples `(level_idx, score, delta, total_depth,
obi, micro_price, bq, aq)` — one per side (BUY / SELL).

---

### Flow B — Regime Filter (rolling window of kline rows)

The HMM operates entirely on **kline-level features** — one scalar value per
candle.  It never sees the 100-row synthetic book.

```
klines[t−120 … t]   ← rolling window of 120 candle rows (≈ 2 h, matching HMM_LOOKBACK)
        │              each row contributes 4 scalar features:
        │              return, volatility, obi_proxy, trade_density
        ▼
RegimeDirector.get_klines_data()    ← computes the 4 features from the raw kline columns
        │
        ▼
RegimeDirector.select_hmm_model()   ← full refit every 5 min (every 5th candle)
  or
RegimeDirector.predict_current_regime()  ← Viterbi only on other candles
        │
        ▼
RegimeDirector.assign_regime_labels()
        │
        ├─► regime_label(t)         ← one string: "trending_up" / "trending_down" /
        │                                          "high_volatility" / "neutral"
        │
        └─► regime_confidence(t)    ← float in [0, 1]: posterior probability of
                                       regime_label from predict_proba()[-1].
                                       None before the first fit (warm-up).
```

**Input:** 120 × 4 feature matrix (120 kline rows, 4 scalar features each).
**Output:** one string label + one confidence float per candle.

Two gates are applied sequentially to Flow A output:

1. **Confidence gate** (applied first):
   - Both BUY and SELL are suppressed if `regime_confidence < HMM_MIN_CONFIDENCE` (default 0.70).
   - When `regime_confidence` is `None` (warm-up) the gate is transparent.

2. **Direction gate** (applied only to candidates that passed the confidence gate):
   - BUY candidate suppressed if `regime_label ∈ {"trending_down", "high_volatility"}`
   - SELL candidate suppressed if `regime_label ∈ {"trending_up", "high_volatility"}`

---

### Flow C — VWAP Momentum Filter (rolling window of level-0 values)

The VWAP operates on **one scalar pair per candle** — the best bid price +
volume and best ask price + volume extracted from level 0 of the synthetic
book.  It does not touch the remaining 49 levels.

```
synthetic_book(t)[level 0]   ← 2 rows only: best bid row + best ask row
        │                       extracted from Flow A's synthetic book
        │
        ▼
best_bid(t), volume_best_bid(t)
best_ask(t), volume_best_ask(t)
        │
        ▼  rolling window over preceding 5 candles (= 5 min at 1 m)
        │
bid_vwap(t) = Σ best_bid(i) × volume_best_bid(i) / Σ volume_best_bid(i)
ask_vwap(t) = Σ best_ask(i) × volume_best_ask(i) / Σ volume_best_ask(i)
                for i in t−5 … t−1
```

**Input:** 5 × 2 scalar pairs (5 candles, each contributing best_bid/ask price
and volume).
**Output:** two scalar values `bid_vwap(t)` and `ask_vwap(t)` per candle.

Gate applied to Flow A output (after regime filter):
- BUY: execute only if `micro_price(t) > ask_vwap(t)` (upward momentum)
- SELL: execute only if `micro_price(t) < bid_vwap(t)` (downward momentum)

---

### Combined Signal

All three flows converge into a single decision per candle:

```
Flow A: candidate tuple (or None)
Flow B: regime_label, regime_confidence
Flow C: bid_vwap, ask_vwap
        │
        ▼
signal(t) = +1 (BUY)   if Flow A produced a BUY candidate
                        AND regime_confidence ≥ HMM_MIN_CONFIDENCE (or None)
                        AND regime_label not in {"trending_down", "high_volatility"}
                        AND micro_price > ask_vwap (or ask_vwap is None)
          = −1 (SELL)  if Flow A produced a SELL candidate
                        AND regime_confidence ≥ HMM_MIN_CONFIDENCE (or None)
                        AND regime_label not in {"trending_up", "high_volatility"}
                        AND micro_price < bid_vwap (or bid_vwap is None)
          =  0 (flat)  otherwise
```

> The **confidence gate** is applied **before** the direction gate.  If the
> model's posterior probability for the predicted regime is below
> `HMM_MIN_CONFIDENCE`, both sides are skipped regardless of regime label or
> VWAP — identical to the behaviour of `low_latency_analysis()` in the live
> system.

---

## Step 4 — Simulated P&L

### Fill assumption
- Fill prices use a **bps-based half-spread** (`BACKTEST_FILL_SPREAD_BPS`), not
  the candle high-low range.  A LIMIT order fills at or inside the real spread
  — not at the candle extreme:
  ```
  half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20 000
  BUY  fill   = close + half_spread   (you pay the synthetic ask)
  SELL fill   = close − half_spread   (you receive the synthetic bid)
  ```
  Default `BACKTEST_FILL_SPREAD_BPS = 5` bps → `half_spread ≈ $20` at $80 k
  BTC, matching the realistic Binance BTCUSDT spread.

  > **Why NOT `(high − low) / 2`?**  A 1-min candle range gives
  > `half_spread ≈ $25–$150` per trade — 10–100× the real exchange spread.
  > Over 1,000+ round trips this produces 100 % drawdown on a $10,000 portfolio.

### Trade sizing
- The candidate tuple from the production pipeline contains `bq` (bid quantity)
  and `aq` (ask quantity) — the same values `OrderExecutor.execute()` uses to
  size LIMIT orders in the live system.
- BUY quantity = `aq` from the best-buy candidate; SELL quantity = `bq` from
  the best-sell candidate.
- A balance guard caps quantity at
  `min(quantity, usdt_budget / (eff_price × (1 + fee_rate)))` for BUY and
  `min(quantity, btc_balance)` for SELL.
- `usdt_budget = usdt × BACKTEST_MAX_POSITION_PCT` (default 10 %) so each BUY
  risks at most 10 % of available USDT, preventing all-in fee compounding.

### Costs
- Taker fee: 0.10 % per side (`BACKTEST_FEE_RATE = 0.001`; Binance Spot standard).
- Slippage estimate: add half the high-low spread as a conservative proxy.

### P&L per trade and round-trip pairing

A **round trip** is one complete BUY → SELL cycle and is the unit used to
compute win rate, profit factor, and average holding time.  Round-trip pairing
is handled by `_pair_round_trips()` in `backtest/pnl.py` **after** the
simulation loop has already settled all trades.  This is purely an accounting
step — every trade in `trades_df` is already executed; the pairing cursor
(`open_buys` deque) is not a live order tracker.

The pairing algorithm uses an **exhaustive FIFO `collections.deque`** to
support three real-world multi-leg entry strategies:

| Strategy | Behaviour |
|---|---|
| **Scaling in** | Multiple BUYs at descending prices → one SELL closes the oldest leg first |
| **Layering** | BUYs placed at regular intervals (grid-style) → each SELL pops the front of the queue |
| **Pyramiding** | Adding to a winning position → FIFO ensures the cheapest entry is closed first |

Two sub-cases are handled within each SELL iteration:

- **Partial close** — SELL qty < oldest leg qty: `matched_qty = sell_qty`,
  the leftover qty is pushed back to the **front** of the deque via
  `appendleft()` so the next SELL continues closing the same leg.
- **Over-sell** — SELL qty > oldest leg qty: the loop consumes as many legs
  as the remaining SELL qty allows, generating one round-trip record per
  consumed leg.

**Orphan SELL** (SELL with no open BUY leg — only possible when
`initial_btc > 0`): the trade is correctly reflected in the equity curve and
balance, but is excluded from round-trip stats.  A clearly visible WARNING box
is printed to the console.

```
BUY  P&L entry  = fill_price = close + half_spread    (half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20 000)
SELL P&L exit   = fill_price = close - half_spread
round_trip PnL  = (exit_fill - entry_fill) × matched_qty
```

Fees are already embedded in both fill prices via `fee_rate`; no separate
deduction is needed inside the pairing function.

---

## Step 5 — Performance Metrics

All metrics below are computed by `_compute_stats()` in `backtest/pnl.py`.
Exact formulae match the code; variable names are the same as the dict keys
returned by the function.

### Initial equity

The total return denominator accounts for **both** the USDT cash balance and
any pre-existing BTC position, valued at the first candle's close:

```
initial_btc_as_usdt     = initial_btc × close[0]
initial_equity          = initial_usdt + initial_btc_as_usdt
```

When `initial_btc = 0` this reduces to `initial_equity = initial_usdt`.

### Metric table

| Metric | Key in `stats` dict | Formula / Description |
|---|---|---|
| **Total return** | `total_return_pct` | `(final_equity − initial_equity) / initial_equity × 100` |
| **Win rate** | `win_rate_pct` | `n_wins / n_round_trips × 100`  where `n_wins` = round trips with `pnl_usdt > 0` |
| **Average trade PnL** | `avg_trade_pnl_usdt` | `mean(pnl_usdt)` over all round trips |
| **Max drawdown** | `max_drawdown_pct` | `min( (equity − peak) / peak × 100 )`  where `peak = equity.cummax()` — the running all-time high of the equity curve.  Drawdown is always ≤ 0; the most negative value is the worst peak-to-trough decline |
| **Sharpe ratio** | `sharpe_ratio` | `mean(Rp − Rf) / std(Rp − Rf) × √N` where `Rp` = period portfolio return, `Rf = BACKTEST_RISK_FREE_RATE / N` (default 0.0), and `N` = periods per year.  Bucket size is chosen **adaptively**: ≥ 2 days → daily (N = 365), ≥ 2 h → hourly (N = 8 760), otherwise 5-min (N = 105 120).  Crypto trades 24/7 so √365 (not √252) is the correct annualiser for daily buckets |
| **Sortino ratio** | `sortino_ratio` | Same as Sharpe but `std` is computed on **downside excess returns only** (`excess_ret[excess_ret < 0]`).  Penalises only negative volatility |
| **Profit factor** | `profit_factor` | `Σ(pnl > 0) / |Σ(pnl < 0)|`  — ratio of gross winning P&L to gross losing P&L.  `∞` when there are no losing round trips |
| **Avg holding period** | `avg_holding_minutes` | `mean(holding_minutes)` excluding NaN entries (session-end mark-to-market closes have no real holding period) |
| **Confidence filter hit rate** | `confidence_filter_hit_rate_pct` | `(confidence_blocked_buy + confidence_blocked_sell) / total_raw_candidates × 100` — % of raw candidates blocked because `regime_confidence < HMM_MIN_CONFIDENCE` (model too uncertain) |
| **Regime filter hit rate** | `regime_filter_hit_rate_pct` | `(regime_blocked_buy + regime_blocked_sell) / total_raw_candidates × 100` — % of raw candidates (that passed the confidence gate) blocked by the regime direction gate |
| **VWAP filter hit rate** | `vwap_filter_hit_rate_pct` | Residual: `raw − executed − confidence_blocked − regime_blocked`, divided by `total_raw_candidates × 100` — % blocked by the VWAP momentum gate (third and final gate) |

---

## Step 6 — Regime Validation

Two levels of regime validation are implemented:

### 6a — Inline train / test split (implemented in `strategy/regime_director.py`)

Every time `select_hmm_model()` is called (initial fit + every 5-minute refit
during the live session and in the backtesting loop), the following split is
applied to the 2-hour lookback window (`HMM_LOOKBACK = "2 hours ago UTC"`,
≈ 120 rows at 1-minute resolution):

```
features[:HMM_TRAIN_ROWS]    → fit()  +  bic()   (80 rows, ~67 % of window)
features[HMM_TRAIN_ROWS:]    → predict()  +  predict_proba()   (≈ 40 rows, ~33 %)
```

* The scaler (`StandardScaler`) is `fit_transform`'d on the **training rows
  only** — mean and standard deviation from held-out candles cannot leak into
  the model.
* `model.predict()` and `model.predict_proba()` run on `features[HMM_TRAIN_ROWS:]`
  only — the rows the model has **never seen** during `fit()`.
* `self.current_regime` and `self.regime_confidence` therefore always reflect a
  candle that was genuinely out-of-sample.  This is a direct application of the
  Step 6 philosophy within the live 2-hour window.

`predict_current_regime()` (cheap Viterbi path, used between full refits)
applies the **same** split: it transforms `features[HMM_TRAIN_ROWS:]` with the
already-fitted scaler and predicts only on those rows.

### 6b — Offline long-horizon validation (`backtest/diagnostics/regime_validation.py`) ✅

#### Purpose

The inline 80/40 split inside `select_hmm_model()` (Step 6a) validates regime
labels within a **2-hour rolling window**.  Step 6b asks a harder question:
*are the HMM labels still meaningful over a much longer, fully out-of-sample
horizon?*

Specifically it answers four questions:
- Does `trending_up` produce statistically higher forward returns than `trending_down`?
- Does `high_volatility` produce higher realised volatility than `neutral`?
- Are the labels stable in frequency (no regime collapse)?
- Is `regime_confidence` consistently above `HMM_MIN_CONFIDENCE` for each label?

#### Dataset

`VALIDATION_LOOKBACK = "365 days ago UTC"` fetches **~525,000 rows** at 1 m
(365 × 24 × 60 = 525,600 candles).  One year of BTC data captures multiple full
market cycle turns (trending, ranging, volatile), giving the HMM the broadest
possible basis for regime learning.  Crypto trades 24/7 so all days are fully
populated with no weekend gaps.

A **70 / 30 train-test split** is used.  `split_idx` is derived at runtime
from `int(len(features_df) * 0.70)` so the ratio stays exact regardless of
the actual number of rows Binance returns:

```
full dataset:  ~525,000 rows   (365 days × 1,440 candles/day at 1 m)
  ├── train set:  first 70 %  →  ~367,500 rows   (~255 days, ~8.5 months)
  └── test  set:  last  30 %  →  ~157,500 rows   (~109 days, ~3.6 months)

split_idx = int(len(features_df) * 0.70)
train_df  = features_df.iloc[:split_idx]
test_df   = features_df.iloc[split_idx:]
```

> **`BACKTEST_MAX_ROWS` is intentionally bypassed.**  `regime_validation.py`
> calls `fetch_klines()` and `_add_hmm_features()` directly.
> Use `VALIDATION_LOOKBACK = "90 days ago UTC"` for a faster smoke-test run.

#### Phase 1 — Training Fit on Full Train Set (~367,500 rows)

This module is **self-contained** — it does **not** use `RegimeDirector`.
It replicates the BIC search and label-assignment logic directly using raw
`GaussianHMM` + `StandardScaler` so that the training window is not
artificially capped at `HMM_TRAIN_ROWS` (80 rows):

1. `StandardScaler.fit_transform(train_features)` — learn mean/std from
   train rows only.  Test rows are never seen by the scaler during fitting.
2. `_fit_best_hmm(train_scaled)` — BIC search over `n = 2 … HMM_MAX_REGIMES`.
   Same retry logic as `RegimeDirector` (`HMM_N_INIT` seeds per n).
3. `_assign_labels(model)` — rank-based directional assignment identical to
   `RegimeDirector.assign_regime_labels()`.

#### Phase 2 — Vectorised Label Assignment on Test Set

The entire test set is scored in **four bulk operations** (no per-candle loop):

1. `scaler.transform(test_features)` — scale all ~157,500 test rows at once.
2. `model.predict(test_scaled)` — single Viterbi pass (hmmlearn C extension).
3. `model.predict_proba(test_scaled)` — Forward–Backward pass;
   `proba[np.arange(n), states]` extracts per-candle confidence via NumPy
   advanced indexing.
4. `label_array[states]` — maps state indices to labels with a NumPy lookup
   array (no Python loop).

| Approach | Viterbi calls | Estimated Phase 2 time |
|---|---|---|
| Per-candle loop | ~157,500 | hours |
| Vectorised (current) | **1** | **seconds** |

Total runtime is dominated by the **Binance paginated fetch** (~8–10 min for
1 year at 1 m resolution).

#### Phase 3 — Validation Checks

For every labelled candle `t`, the 1-candle forward return is computed:

```
forward_return(t) = close(t+1) / close(t) − 1
```

Six checks are then run on the ~4,200 labelled test candles:

**Check 1 — Direction Test** *(primary)*

The mean forward return must follow the expected ordering across all three
directional regimes:

```
mean(forward_return | trending_up)  >  mean(forward_return | neutral)
                                    >  mean(forward_return | trending_down)
```

**Check 2 — Statistical Significance**

Two-sample Welch's t-test (does not assume equal variance) between the
`trending_up` and `trending_down` forward return distributions:

```
t, p = scipy.stats.ttest_ind(
    forward_returns[regime == "trending_up"],
    forward_returns[regime == "trending_down"],
    equal_var=False,
)
```

**Pass condition:** p-value < 0.05

**Check 3 — Volatility Check**

```
mean(volatility_feature | regime == "high_volatility")
    >
mean(volatility_feature | regime == "neutral")
```

This should hold by construction (the label is assigned on volatility rank),
but verifying it on out-of-sample data confirms label stability outside the
training window.

**Check 4 — Confidence Distribution**

**Pass condition:** No label has a median `regime_confidence` below
`HMM_MIN_CONFIDENCE` (0.70).  A label that is consistently uncertain suggests
the model cannot distinguish that regime reliably on new data.

**Check 5 — Label Frequency**

**Pass condition:** No regime has < 1 % frequency over the ~4,200 test
candles.  Near-zero frequency means the model is effectively collapsing to
fewer states than `n_components` — a sign of over-parameterisation.

**Check 6 — Hit-rate Alignment** *(informational, no auto PASS/FAIL)*

Compare the regime-blocked percentage observed on the test set with the
`regime_filter_hit_rate_pct` reported by `runner.py` (Step 5).  A divergence
of more than ~5 percentage points indicates that the live 2 h rolling window
produces systematically different label distributions compared to the longer
out-of-sample horizon.

#### Phase 4 — Output Format

Results are printed by
`backtest/reporting/formatters.print_regime_validation_report()` to stdout
(no matplotlib required):

```
══════════════════════════════════════════════════════════════════
 REGIME VALIDATION REPORT — 8-day test set (~11,520 candles)
══════════════════════════════════════════════════════════════════

 Train:  22 days  (~31,680 rows)   model frozen after initial fit
 Test:    8 days  (~11,520 rows)   vectorised single Viterbi pass

──────────────────────────────────────────────────────────────────
 PER-REGIME STATISTICS
──────────────────────────────────────────────────────────────────
 Regime           Count   Freq%   Mean fwd-ret   Std fwd-ret   Med confidence
 trending_up      x,xxx   xx.x    +x.xxxxx %     x.xxxxx %     x.xx
 trending_down    x,xxx   xx.x    −x.xxxxx %     x.xxxxx %     x.xx
 high_volatility  x,xxx   xx.x    ±x.xxxxx %     x.xxxxx %     x.xx
 neutral          x,xxx   xx.x    ±x.xxxxx %     x.xxxxx %     x.xx

──────────────────────────────────────────────────────────────────
 STATISTICAL TESTS
──────────────────────────────────────────────────────────────────
 Direction test (trending_up > neutral > trending_down):
   trending_up mean:    +x.xxxxx %
   neutral mean:        ±x.xxxxx %
   trending_down mean:  −x.xxxxx %   →  [PASS / FAIL]

 Welch's t-test (trending_up vs trending_down):
   t = x.xx,  p = x.xxxxxx             →  [PASS / FAIL]

 Volatility check (high_vol mean vol > neutral mean vol):
   x.xxxxxx > x.xxxxxx                 →  [PASS / FAIL]

 Label frequency (all regimes > 1 %):  →  [PASS / FAIL]

 Confidence floor (all median conf ≥ 0.70): → [PASS / FAIL]

 Hit-rate alignment (compare with runner.py regime_filter_hit_rate_pct):
   BUY  blocked (trending_down|high_vol): xx.x %  (x,xxx/x,xxx)
   SELL blocked (trending_up|high_vol):   xx.x %  (x,xxx/x,xxx)
   Both sides blocked (high_vol only):    xx.x %  (x,xxx/x,xxx)
   [informational — no auto PASS/FAIL]
══════════════════════════════════════════════════════════════════
```

#### Key Design Decisions

| Decision | Rationale |
|---|---|
| Single 7/3 split, not rolling | Rolling refits on the test set would re-use test candles for fitting; a single split is the cleanest long-horizon OOS test |
| Only `predict_current_regime()` on test set | No refit → model never touches test candles during training |
| `BACKTEST_MAX_ROWS` must be bypassed | 500 rows collapses the split to a few hours of train data — statistically worthless |
| Welch's t-test (not Student's) | Regime return distributions are unlikely to have equal variance |
| Forward return = 1 candle | Consistent with the 1 m resolution; matches the minimum observable effect of a regime signal |
| Standalone script, not in `runner.py` | This is a one-off diagnostic, not part of the core P&L pipeline |

> Run with: `python -m backtest.regime_validation`
> Re-run this script whenever `strategy/regime_director.py` is modified to
> verify that the regime labels remain statistically meaningful.

---

## Step 8 — Sensitivity Analysis

### Objective

Run the full backtest pipeline (`run_signals` → `simulate_pnl` → `_compute_stats`)
across a grid of parameter values to answer one question:

> *"Does the strategy's performance degrade gracefully as each tunable parameter
> shifts away from its current default value?"*

A strategy is **robust** if total return stays positive, max drawdown does not
spike, and win rate / profit factor follow a consistent trend as parameters
move toward their tuned values.  A strategy is **fragile** if a small shift in
one parameter causes a large non-monotonic swing — suggesting the default is
over-fitted to the historical sample rather than capturing a genuine edge.

---

### Two Use Cases

#### Use Case A — Live parameter tuning ✅ *Implemented*

| Attribute | Value |
|---|---|
| Window | `SENSITIVITY_LOOKBACK = "30 days ago UTC"` (~43,200 rows at 1 m) |
| Rationale | The live HMM refits every 5 min on the latest 2 h of data. Tuning on 30 days of **recent** data is more appropriate than tuning on 6-month-old conditions. |
| Refit cadence | `SENSITIVITY_REFIT_EVERY = 480` (8 h at 1 m) — ~90 refits per run vs ~360 at the default, giving a ~4× speedup while preserving relative rankings. |
| Viterbi cadence | `SENSITIVITY_PREDICT_EVERY = 5` — Viterbi prediction called every 5 candles; last known regime reused otherwise (~5× fewer calls). `run_backtest.py` always predicts every candle. |
| Runtime (OAT, 6 runs) | ~12–30 min on a laptop (after optimisations) |
| Output | `backtest/results/best_params.json` — loaded by `strategy.param_loader` (shared by `websocket_main.py` and `run_backtest.py`) |
| Run | `python -m backtest.sensitivity` |

#### Use Case B — Backtest robustness validation ⬜ *Deferred*

| Attribute | Value |
|---|---|
| Window | Must match the main backtest (`"180 days ago UTC"`, ~259,200 rows) to avoid window-mismatch bias. A shorter window here would optimise for conditions not representative of the 180-day evaluation horizon. |
| Runtime (OAT, 6 runs) | ~4–12 hours on a laptop — impractical without dedicated compute |
| Status | Revisit if compute resources allow |

---

### Parameters

All four parameters live in `config_parameters.py`.  The first value in each
range is the **default** used by the baseline run.

| # | Parameter | Constant | Default | Values tested |
|---|---|---|---|---|
| 1 | HMM lookback window | `HMM_LOOKBACK_ROWS` | 120 (2 h) | 60, **120**, 240 |
| 2 | Max HMM regimes (BIC upper bound) | `HMM_MAX_REGIMES` | 3 | **3**, 2  *(4+ risks under-populated states with 4 features)* |
| 3 | VWAP momentum window | `VWAP_WINDOW` | 5 (5 min) | **5**, 2  *(10 min excluded — too slow for 1 s cadence)* |
| 4 | Taker fee rate | `BACKTEST_FEE_RATE` | 0.001 (0.10 %) | **0.001**, 0.0005  *(0.05 % achievable at VIP 3+)* |

**Excluded from scope:**
- Depth:delta score weights (0.70 / 0.30) — not a config constant.
- OBI threshold — adding one would be a new feature, not a config swap.

---

### Execution Phases

#### Phase 1 — One-At-a-Time (OAT) sweep *(default)*

Hold all parameters at defaults; vary one at a time.
- 1 baseline + 2 (lookback) + 1 (regimes) + 1 (vwap) + 1 (fee) = **6 runs**
- Shows which parameter individually drives the most sensitivity.
- Does **not** capture interaction effects.

Run: `python -m backtest.sensitivity`

#### Phase 2 — Full factorial grid *(triggered on demand)*

All `3 × 2 × 2 × 2 = 24 combinations`.
- Trigger rule: run Phase 2 only if any single OAT parameter causes `|ΔSharpe| > 0.5`.
- Captures pairwise and higher-order interactions.

Run: `python -m backtest.sensitivity --full-grid`

---

### Implementation (Approach A — keyword overrides)

`run_signals()` accepts six optional keyword arguments that default to `None`
(falling back to `config_parameters.py` constants when `None`).  Existing
callers (`run_backtest.py`) require zero changes.

```python
signals = run_signals(
    hmm_lookback_rows=60,                       # overrides HMM_LOOKBACK_ROWS
    hmm_max_regimes=2,                          # overrides HMM_MAX_REGIMES
    vwap_window=2,                              # overrides VWAP_WINDOW
    refit_every=SENSITIVITY_REFIT_EVERY,        # 480 — 4× fewer refits
    predict_every=SENSITIVITY_PREDICT_EVERY,    # 5 — 5× fewer Viterbi passes
    lookback=SENSITIVITY_LOOKBACK,              # "30 days ago UTC" — 6× fewer rows
)
_, _, stats = simulate_pnl(signals, fee_rate=0.0005)  # overrides BACKTEST_FEE_RATE
```

`simulate_pnl()` already accepted `fee_rate` as a keyword argument before this
change — no modification was needed there.

**Combined speedup per OAT run (30-day window vs naïve 180-day run):**

| Optimisation | Constant | Factor |
|---|---|---|
| Shorter fetch window | `SENSITIVITY_LOOKBACK = "30 days ago UTC"` | ~6× fewer rows |
| Less frequent HMM refit | `SENSITIVITY_REFIT_EVERY = 480` | ~4× fewer full BIC fits |
| Less frequent Viterbi | `SENSITIVITY_PREDICT_EVERY = 5` | ~5× fewer predict calls |
| `itertuples()` in P&L loop | — (code change in `pnl.py`) | ~5× faster P&L walk |
| Numpy-vectorised book build | — (code change in `synthetic_book.py`) | ~5–10× faster per candle |

Total wall time: **~12–30 min** for the full OAT (6 runs) on a laptop, vs ~66–180 min without these optimisations.

---

### Output

**Console** — sorted summary table (one row per combination):

```
════════════════════════════════════════════════════════════════════════════════════════
  SENSITIVITY ANALYSIS — BTCUSDT  (OAT, 6 combinations)
════════════════════════════════════════════════════════════════════════════════════════
 lookback  max_reg  vwap   fee      total_ret%  drawdown%  sharpe  sortino  n_trips
────────────────────────────────────────────────────────────────────────────────────────
  120        3       5    0.001      +x.xx       -x.xx      x.xx    x.xx     xxx   ← baseline
   60        3       5    0.001      +x.xx       -x.xx      x.xx    x.xx     xxx
  240        3       5    0.001      +x.xx       -x.xx      x.xx    x.xx     xxx
  120        2       5    0.001      +x.xx       -x.xx      x.xx    x.xx     xxx
  120        3       2    0.001      +x.xx       -x.xx      x.xx    x.xx     xxx
  120        3       5    0.0005     +x.xx       -x.xx      x.xx    x.xx     xxx
════════════════════════════════════════════════════════════════════════════════════════
```

**Files written:**
- `backtest/results/sensitivity_<mode>_<timestamp>.csv` — all metrics, all runs.
- `backtest/results/best_params.json` — winning parameter set (ranked by Sharpe).

---

### Persisting Best Parameters (`best_params.json`)

After the sweep, `sensitivity.py` writes the winning row to
`backtest/results/best_params.json`:

```json
{
  "hmm_lookback_rows": 60,
  "hmm_max_regimes": 3,
  "vwap_window": 2,
  "fee_rate": 0.0005,
  "generated_at": "2026-04-19T10:00:00+00:00",
  "source_metric": "sharpe_ratio",
  "source_value": 1.42
}
```

#### Who loads it and how

Both consumers delegate to **`strategy/param_loader.py`** — the single module
that owns all loading logic, keeping `websocket_main.py` and `run_backtest.py`
free of JSON / file-handling code.

| Consumer | Function called | When | What is overridden | Mechanism |
|---|---|---|---|---|
| `websocket_main.py` | `param_loader.load_best_params()` | At startup, before `RegimeDirector()` is instantiated | `HMM_MAX_REGIMES`, `HMM_LOOKBACK` | Patches `strategy.regime_director` module namespace directly (not `config_parameters`) — necessary because `regime_director.py` binds constants at import time via `from config_parameters import`. |
| `run_backtest.py` | `param_loader.load_best_params_for_backtest()` | At the top of `run_backtest()`, before `run_signals()` | `hmm_lookback_rows`, `hmm_max_regimes`, `vwap_window`, `fee_rate` | Returns a plain `dict`; values passed as keyword arguments to `run_signals()` and `simulate_pnl()`. Keys absent from `best_params.json` are simply omitted — both functions fall back to `config_parameters.py` defaults automatically. |

#### Parameter Flow

```
sensitivity.py  ──►  best_params.json  ──►  strategy/param_loader.py
                                                  ├── load_best_params()             → websocket_main.py  (live)
                                                  └── load_best_params_for_backtest() → run_backtest.py    (backtest)
```

Both consumers fall back silently to `config_parameters.py` defaults when
`best_params.json` is absent or unreadable.

#### Notes on `VWAP_WINDOW` and `fee_rate`

- `vwap_window` — backtest-only (`backtest/signals.py`).  The live system does
  not use `VWAP_WINDOW`, so `websocket_main.py` ignores this field.
- `fee_rate` — informational for the live system (Binance charges its own fees
  regardless).  `run_backtest.py` does pass it to `simulate_pnl()`.

#### Notes on `HMM_LOOKBACK` (live) vs `HMM_LOOKBACK_ROWS` (backtest)

`best_params.json` stores `hmm_lookback_rows` (an integer, used by the
backtest).  The live system uses `HMM_LOOKBACK` (a dateutil string such as
`"2 hours ago UTC"`).  The live loader converts via a fixed mapping:

| `hmm_lookback_rows` | `HMM_LOOKBACK` |
|---|---|
| 60  | `"1 hour ago UTC"` |
| 120 | `"2 hours ago UTC"` (default) |
| 240 | `"4 hours ago UTC"` |

If the JSON contains a value not in this table, `HMM_LOOKBACK` is left at
the `config_parameters.py` default and a `WARNING` is logged.

> **Important:** do **not** commit `best_params.json` to git — it is
> sample-specific.  Add `backtest/results/best_params.json` to `.gitignore`.
> Re-run `sensitivity.py` whenever the lookback period, feature set, or market
> conditions change significantly.

---

### Robustness Criteria

| Verdict | Condition |
|---|---|
| **Robust** ✅ | `total_return_pct` positive across all tested values; Sharpe degradation < 0.5 units; max drawdown worsens < 5 pp vs baseline |
| **Borderline** ⚠️ | Return flips sign for one extreme value — acceptable if the flip is at the edges (e.g. very long lookback) |
| **Fragile** ❌ | Return positive only at the exact current default and negative for all others — likely over-fitted |

---

## Known Approximations & Caveats

1. **Synthetic vs real Level-2 data** — the synthetic order book (Step 2c) lets
   the backtest run the exact same scoring pipeline as the live strategy, but
   the depth profile is artificial (exponential decay, uniform asymmetry).
   Real order books have irregular, level-specific imbalances that the
   synthetic reconstruction cannot capture.
2. **Fill price** — using `close ± half_spread` (synthetic bid/ask) is more
   realistic than a naive `close` fill, but still overstates fill quality for
   a LIMIT GTC strategy; real fills depend on queue position and latency.
3. **No partial fills** — the backtest assumes full fill on every signal.
4. **No latency** — the 100 ms WS delay and the 1-second analysis cadence are
   not modelled in a candle-level simulation.
5. **Survivorship bias** — the BTCUSDT pair has been continuously liquid during
   the test window.  Results should not be extrapolated to other pairs without
   re-running the full pipeline.
6. **Regime label instability** — HMM state indices are not stable across
   re-fits; `assign_regime_labels()` resolves this by rank-based labelling, but
   the ranks themselves can shift if the market's statistical properties change
   significantly during the lookback window.
7. **Not financial advice** — see the project disclaimer in `README.md`.

---

## Implementation Roadmap

Progress tracker for the modules described above.  Each step maps to a
concrete file in `backtest/`.

- ✅ **Step 1 — `backtest/data.py`** — Downloads 180 days of 1 m BTCUSDT
  klines via `binance.client.Client.get_historical_klines()` and returns a
  clean `pandas.DataFrame` (~259,200 rows).  *(Implemented.)*
- ✅ **Step 2 — `backtest/synthetic_book.py`** — Given a single kline row
  (`pd.Series`), constructs a 50-level synthetic order book with exponential
  volume decay and OBI asymmetry injection.  Returns a
  `{"bids": {…}, "asks": {…}}` dict matching the `state.local_book` format
  consumed by `AnalysisEngine._build_levels()`.  *(Implemented.)*
- ✅ **Step 3 — `backtest/signals.py`** — Loops over all ~43 200 candles,
  calls `build_synthetic_book()` per row, runs the **production pipeline**
  (`build_levels` → `collect_candidates` → `select_best_opportunity` via
  `strategy/book_utils.py`), applies three sequential gates — **confidence
  gate** (`regime_confidence ≥ HMM_MIN_CONFIDENCE`), **regime direction gate**
  (Flow B), and **VWAP momentum gate** (Flow C) — and returns a time-indexed
  signal DataFrame (`+1` BUY, `−1` SELL, `0` flat) plus regime label,
  `regime_confidence`, VWAP, and micro-price details.
  *(Implemented.)*
- ✅ **Step 4 — `backtest/pnl.py`** — Converts the signal Series into
  simulated trades (using ``bq``/``aq`` quantities and the balance guard)
  and computes the Step 5 performance metrics: total return, win rate,
  max drawdown, Sharpe, Sortino, profit factor, average holding period,
  and regime / VWAP filter hit rates.  Round-trip pairing uses an
  exhaustive FIFO `collections.deque` supporting scaling-in, layering, and
  pyramiding.  Orphan SELLs emit a visible WARNING box.  *(Implemented.)*
- ✅ **Step 5 — `backtest/runner.py` + `backtest/reporting/formatters.py`** —
  Top-level script that chains all four modules.  Report formatting
  (`print_report`) and CSV export (`save_csv`) are isolated in
  `backtest/reporting/formatters.py` (AI-authored; all symbols public).
  Prints a formatted console report: SESSION info, SIGNALS breakdown, P&L
  SUMMARY, RISK METRICS, TRADE LOG PREVIEW.  Optionally saves timestamped
  ``trades_*.csv`` and ``equity_*.csv`` to ``backtest/results/``.
  Run with ``python -m backtest.runner``.  *(Implemented.)*
- ✅ **Step 6a — Inline regime validation** (`strategy/regime_director.py`) —
  Train/test split is applied inside every `select_hmm_model()` and
  `predict_current_regime()` call: the model fits on `features[:HMM_TRAIN_ROWS]`
  (first 80 rows, ~67 % of the 2-hour window) and predicts only on
  `features[HMM_TRAIN_ROWS:]` (most-recent ~40 rows, never seen during fit).
  `self.current_regime` and `self.regime_confidence` always reflect a genuinely
  out-of-sample candle.  *(Implemented.)*
- ✅ **Step 6b — Offline long-horizon regime validation** (`backtest/diagnostics/regime_validation.py`) —
  Standalone diagnostic script (run with `python -m backtest.diagnostics.regime_validation`).
  Fetches **1 year** of 1-minute klines (~525,000 rows, `VALIDATION_LOOKBACK = "365 days ago UTC"`),
  applies a **70/30 train-test split** at runtime (`split_idx = int(len(df) * 0.70)`).
  **Self-contained** — does not use `RegimeDirector`; replicates BIC search and label
  assignment directly with raw `GaussianHMM` + `StandardScaler` so the training window
  is not capped at 80 rows.  Phase 2 scores all ~157,500 test candles in a
  **single vectorised Viterbi pass** with NumPy advanced indexing for confidence
  extraction (`proba[np.arange(n), states]`) and a lookup-array for label mapping
  (`label_array[states]`).  `BACKTEST_MAX_ROWS` is intentionally bypassed.
  Six checks (Phase 3) are computed on the labelled test candles:
  1. **Direction test** — `trending_up` mean > `neutral` mean > `trending_down`
     mean forward 1-candle return.
  2. **Welch's t-test** — p < 0.05 between `trending_up` and `trending_down`
     forward return distributions (does not assume equal variance).
  3. **Volatility check** — mean volatility feature higher for `high_volatility`
     than `neutral`.
  4. **Confidence floor** — median `regime_confidence` ≥ `HMM_MIN_CONFIDENCE`
     per label.
  5. **Label frequency** — no regime < 1 % (regime collapse guard).
  6. **Hit-rate alignment** *(informational, no auto PASS/FAIL)* — % of
     candles blocked per side (BUY / SELL / both) vs. `runner.py` hit rate.
  The formatted report (Phase 4) is printed by
  `backtest/reporting/formatters.print_regime_validation_report()`.
  Re-run whenever `strategy/regime_director.py` is modified.  *(Implemented.)*
- ✅ **Step 7 — `backtest/visualization.py`** — Interactive six-row Plotly
  figure generated from the four artefacts returned by ``run_backtest()``.
  Uses ``plotly`` (already in ``requirements.txt``); opt-in via
  ``run_backtest(plot=True, save_png=True)``; no effect on headless runs
  when ``plot=False`` (default).  Panels:
  1. **Equity curve** — continuous portfolio value with initial-equity dashed
     reference.
  2. **Drawdown (%)** — red ``tozeroy`` fill (shared x-axis with equity).
  3. **BTC close + BUY ▲ / SELL ▼** — fill-price markers and rolling
     ``bid_vwap`` / ``ask_vwap`` dashed lines.
   4. **Regime timeline + confidence** — three-layer panel: (a) colour-coded
      ``vrect`` background bands per HMM label; (b) a dark-slate ``shape="hv"``
      step-line that maps each label to an integer position via
      ``_REGIME_NUMERIC`` (0 = ``trending_down`` → 3 = ``trending_up``) so
      regime transitions appear as immediate vertical jumps; (c) a dotted navy
      confidence overlay scaled ×3 to fill the [0, 3] range (hover shows the
      real 0–1 value) with a dashed ``HMM_MIN_CONFIDENCE × 3`` threshold line.
      Y-axis tick labels show regime names, not raw integers.
      *Bug fixed 2026-04-25: previously only ``regime_confidence`` (0–1) was
      plotted, which appeared as a nearly flat line — the regime step-line
      and ×3 scaling were added to make the timeline readable.*
  5. **VWAP vs micro-price** — four series with grey dot (●) near-miss markers
     where the VWAP gate specifically blocked a candidate.
  6a. **Signal funnel** (stacked horizontal bar) — per-side breakdown:
      executed / confidence-blocked / regime-blocked / VWAP-blocked.
  6b. **Signals by regime** (stacked vertical bar) — BUY / SELL / HOLD per HMM
      label.
  All rows 1–5 share a synchronised datetime x-axis (zoom on one → all move).
  PNG export requires ``pip install kaleido``; without it an interactive HTML
  file is saved instead.  *(Implemented.)*

- ✅ **Step 8 — `backtest/sensitivity.py`** — OAT and full-grid sensitivity
  sweep over `HMM_LOOKBACK_ROWS`, `HMM_MAX_REGIMES`, `VWAP_WINDOW`, and
  `BACKTEST_FEE_RATE`.  **Use Case A (30-day live-tuning window) implemented.**
  Run with `python -m backtest.sensitivity` (OAT, default) or
  `python -m backtest.sensitivity --full-grid` (Phase 2, 24 combinations).
  Writes `best_params.json` — loaded via `strategy.param_loader`
  (`load_best_params()` by `websocket_main.py` at startup;
  `load_best_params_for_backtest()` by `run_backtest.py` before `run_signals()`).

  > **Why patches from `param_loader` take effect on `RegimeDirector`:**
  > `RegimeDirector.__init__` uses `None` sentinels for `lookback` and
  > `max_regimes` instead of default parameter values.  Python evaluates default
  > parameter expressions *once at `def` time* (import time), which would freeze
  > the constants before `load_best_params()` can patch the module namespace.
  > The `None` sentinel forces Python to re-read `HMM_LOOKBACK` / `HMM_MAX_REGIMES`
  > from `strategy.regime_director`'s own namespace on every `__init__` call —
  > after `load_best_params()` has already patched that namespace.
  > Full explanation in `strategy/param_loader.py` docstring.

  **Use Case B (180-day backtest validation) deferred** — runtime ~4–12 h for
  OAT alone on a laptop.  *(Use Case A implemented; Use Case B deferred.)*

---

*Document last updated: 2026-04-25*

