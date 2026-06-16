# Backtesting Plan — Binance Spot Testnet Strategy

> **Status:** Steps 1–9 implemented.
>
> * Steps 1–8: `data.py`, `synthetic_book.py`, `signals.py`, `pnl.py`, `runner.py`,
>   `regime_validation.py`, `visualization.py`, `sensitivity.py`.
> * **Step 9 — Multi-timeframe resolution decoupling** (2026-05-18):
>   * **Two-resolution architecture** introduced: `BACKTEST_MACRO_INTERVAL = "5m"` for
>     the `GaussianHMM` (regime classification) and `BACKTEST_MICRO_INTERVAL = "1m"` for
>     signal generation and PnL simulation.  The HMM never sees 1-minute bars; execution
>     never triggers a BIC re-fit.
>   * **Parquet cache** (`cache/klines/`) — 24-hour TTL; `--flush-cache` flag on all CLIs.
>   * **Intra-candle whipsaw guard** in `pnl.py` — same-bar pessimistic exit when the
>     1-minute bar's Low ≤ best‑buy micro-price AND High ≥ best‑sell micro-price.
>   * **Live `RegimeDirector` aligned**: `HMM_INTERVAL = "5m"`, `HMM_LOOKBACK = "10 hours ago UTC"` (120 bars).
>   * Step 8 (sensitivity, Use Case A) retained on the IS window.  Use Case B deferred.
>
> See the Implementation Roadmap at the bottom for a per-module progress tracker.

---

## Overview

The live strategy (defined in `strategy/analysis.py`) makes order decisions in
real time based on a 50-level order book snapshot, a VWAP dip/strength filter
(mean-reversion), and an HMM regime filter.  The purpose of backtesting is to
replay equivalent logic against historical data and measure whether the strategy
generates positive risk-adjusted returns before committing further to live
execution.

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

### Two-resolution architecture (Step 9)

The backtest now fetches **two separate OHLCV frames** through the typed wrappers in
`backtest/data.py`:

| Frame | Constant | Interval | Purpose |
|---|---|---|---|
| **Macro** | `BACKTEST_MACRO_INTERVAL = "5m"` | 5-minute | `GaussianHMM` regime classification |
| **Micro** | `BACKTEST_MICRO_INTERVAL = "1m"` | 1-minute | Rolling VWAP, signal generation, PnL simulation |

Decoupling the HMM cadence from the execution cadence yields two improvements:
1. **Faster EM convergence** — the HMM sees 5× fewer rows with richer structural information.
2. **Cleaner signal gating** — the execution loop still runs at full 1-minute resolution while the regime filter advances only on meaningful structural changes.

### Window sizes

| Window | Frame | Rows |
|---|---|---|
| IS (360 → 90 days ago, 270 d) | 5 m | ~77,760 |
| IS | 1 m | ~388,800 |
| OOS (90 days ago → today, 90 d) | 5 m | ~25,920 |
| OOS | 1 m | ~129,600 |

`sensitivity.py` fetches the IS window (`end_str=BACKTEST_OOS_START`); `runner.py`
fetches the OOS window (`lookback=BACKTEST_OOS_START`).

### Columns returned per candle
```
open_time, open, high, low, close, volume,
close_time, quote_asset_volume, num_trades,
taker_buy_base_vol, taker_buy_quote_vol, ignore
```

---

## Step 1b — Parquet Cache (`cache/klines/`)

Both `fetch_macro_klines()` and `fetch_micro_klines()` route through a parquet cache
before touching the Binance API:

| Attribute | Value |
|---|---|
| Cache directory | `cache/klines/` |
| Filename | `<SYMBOL>_<interval>_<MD5-12>.parquet` (deterministic from symbol + interval + start + end) |
| TTL | **24 hours** — IS data only changes when `BACKTEST_LOOKBACK` / `BACKTEST_OOS_START` change; OOS data refreshes automatically each day |
| Cache hit | Load from parquet (no API call) |
| Cache miss | Fetch from Binance → save to parquet → return |
| Explicit invalidation | `--flush-cache` flag on `sensitivity.py` and `runner.py` CLIs, or call `backtest.data.flush_kline_cache()` |

> Run `python -m backtest.sensitivity --flush-cache` or `python -m backtest.runner --flush-cache`
> whenever `BACKTEST_LOOKBACK` or `BACKTEST_OOS_START` change to force a fresh download.

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

For each candle, 50 synthetic bid/ask levels are built from OHLCV data with OBI asymmetry injected from taker-buy flow.  
Fill prices use `close × BACKTEST_FILL_SPREAD_BPS / 20 000` (default 5 bps).  
See `backtest/synthetic_book.py` docstring for implementation details.

---

## Step 3 — Strategy Signal Replay (Two-Frame Pipeline)

The backtest uses a **two-resolution architecture** to decouple the HMM regime classifier
(statistically stable, low-frequency) from the execution signal loop (granular,
execution-realistic).

### Three phases inside `run_signals()`

```
Phase 1 — HMM walk-forward on df_macro (5 m bars)
──────────────────────────────────────────────────
df_macro_raw (5 m) → _add_hmm_features() → features_macro
  └── rolling window [i−_lookback : i], step 1 per 5 m bar
        ├── every _refit_every bars : select_hmm_model()    ← full BIC re-fit
        └── every _predict_every bars: predict_current_regime()  ← cheap Viterbi
              └── assign_regime_labels()
                    └── regime_df {timestamp_5m → regime, regime_confidence}


Phase 2 — Temporal stitch (zero look-ahead)
────────────────────────────────────────────
pd.merge_asof(df_micro.sort_index(), regime_df.sort_index(), direction='backward')
  → df_exec (1 m bars).
Each 1-minute bar receives only the LAST completed 5-minute regime label.
1-minute bars that predate the first regime label are discarded.


Phase 3 — Execution loop on df_exec (1 m bars)
───────────────────────────────────────────────
For each 1 m candle:
  Flow A: build_synthetic_book() → build_levels → collect_candidates
          → select_best_opportunity   (same production code)
  Flow B: read regime / confidence from stitched column (no HMM calls here)
  Flow C: rolling bid_vwap / ask_vwap on 1 m top-of-book prices
  Gates:  confidence gate → regime gate → VWAP dip/strength gate
  Output: signal +1 / −1 / 0  per 1-minute bar
```

### Per-candle dataset sizes

| Phase                | Frame | Candles (IS 270 days) |
|----------------------|-------|-----------------------|
| 1 — HMM walk-forward | 5 m   | ~77,760               |
| 3 — Execution loop   | 1 m   | ~388,800              |

> **Memory:** `df_macro` ~6 MB, `df_micro` ~30 MB, one synthetic order book at a time
> (100 rows, discarded after each candle) — total well under 50 MB.

### Combined Signal (unchanged contract)

```
signal(t) = +1 (BUY)   if Flow A produced a BUY candidate
                         AND regime_confidence ≥ HMM_MIN_CONFIDENCE (or None)
                         AND regime_label not in {"trending_down", "high_volatility"}
                         AND (bid_vwap is None
                              OR micro_price < bid_vwap × (1 − VWAP_THRESHOLD_MULTIPLIER))
          = −1 (SELL)  if Flow A produced a SELL candidate
                         AND regime_confidence ≥ HMM_MIN_CONFIDENCE (or None)
                         AND regime_label not in {"trending_up", "high_volatility"}
                         AND (ask_vwap is None
                              OR micro_price ≥ ask_vwap × (1 + VWAP_THRESHOLD_MULTIPLIER))
          =  0 (flat)  otherwise
```

See **Step 2 / 2c / Flows A–C** above for the individual pipeline descriptions
(synthetic order book, HMM features, VWAP mechanics) — these are unchanged.

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

### Intra-candle whipsaw guard (Step 9)

At 1-minute bar resolution it is possible for a single candle's **Low** to touch
the BUY micro-price threshold AND its **High** to touch the SELL micro-price
threshold in the same bar.  In the live 1-second WS loop these events would
occur on different ticks; in the backtest we cannot determine which extreme was
reached first.  The guard resolves the ambiguity pessimistically:

```
if open_strategy_qty > _POSITION_DUST_BTC
   AND candle_low  ≤ best_buy_micro   # bar crossed into BUY zone
   AND candle_high ≥ best_sell_micro  # bar ALSO crossed into SELL zone:
       Force-close: SELL_WHIPSAW at (candle_low − half_spread)
       Record a "SELL_WHIPSAW" trade in trades_df
       Reset open_strategy_qty = 0          # strategy is now flat
       _skip_signals = True                 # skip normal BUY/SELL for this bar
```

- **Fill price:** `candle_low − half_spread` — the pessimistic exit price within
  the bar's observed range.
- **Counter:** `n_whipsaw_exits` is returned in the `stats` dict.
- **Backward-compat:** if the `high` / `low` columns are absent (legacy frames,
  unit tests) the guard is silently disabled.

### Position model — single open position at a time

`simulate_pnl()` enforces a **one-position-at-a-time** rule via `open_strategy_qty` (BTC held exclusively by strategy signals, excluding `initial_btc`). BUY fires only when `open_strategy_qty ≤ 1e-6` (threshold below which position is treated as flat). SELL always closes the full position in one shot, guaranteeing the strategy returns to flat. The position guard counts suppressed BUY signals in `n_position_guard_skips` (returned in the `stats` dict).

### Trade sizing and costs

- BUY quantity = `aq` from best-buy candidate, capped at `min(aq, usdt / (price × (1 + fee_rate)))` for BUY to reserve taker fee.
- SELL quantity = `bq` from best-sell candidate, capped at `min(bq, btc_balance)`.
- `usdt_budget = usdt × BACKTEST_MAX_POSITION_PCT` (10 %) limits each BUY to ~10 % of available USDT.
- Taker fee: 0.10 % per side (`BACKTEST_FEE_RATE = 0.001`).
- Fill prices: `close ± half_spread` where `half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20 000` (default 5 bps, ~$20 at $80k BTC).

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

**Orphan SELL** (SELL with no open BUY leg): the trade is correctly reflected
in the equity curve and balance, but is excluded from round-trip stats.  A
clearly visible WARNING box is printed to the console.  This scenario is
avoided entirely by setting `BACKTEST_INITIAL_BTC = 0.0`.

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

With `BACKTEST_INITIAL_BTC = 0.0` (always), initial equity equals the USDT
balance directly.  The formula in `_compute_stats()` is general — it supports
a non-zero BTC start — but the project always uses the simplified form:

```
initial_btc_as_usdt     = initial_btc × close[0]   # = 0 when BACKTEST_INITIAL_BTC = 0
initial_equity          = initial_usdt + initial_btc_as_usdt  # = initial_usdt
```

`BACKTEST_INITIAL_CAPITAL = 315_000.0` already incorporates the value of the
live account's 1 BTC holding (~65k at ~65k/BTC) so total starting equity
matches the live paper-trading account (~250k USDT + 1 BTC ≈ 315k).

### Metric table

| Metric                         | Key in `stats` dict              | Formula / Description                                                                                                                                                                                                                                                                                                                                                                                              |
|--------------------------------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Total return**               | `total_return_pct`               | `(final_equity − initial_equity) / initial_equity × 100`                                                                                                                                                                                                                                                                                                                                                           |
| **Buy-and-Hold return**        | `bnh_total_return_pct`           | Passive benchmark: buy at the first close, hold for the full window.  Computed by `compute_buy_and_hold()` on the same initial portfolio so the two percentages share the same denominator and are directly comparable.  Shown as a dashed orange line in the equity panel and in the figure title                                                                                                                 |
| **Win rate**                   | `win_rate_pct`                   | `n_wins / n_round_trips × 100`  where `n_wins` = round trips with `pnl_usdt > 0`                                                                                                                                                                                                                                                                                                                                   |
| **Average trade PnL**          | `avg_trade_pnl_usdt`             | `mean(pnl_usdt)` over all round trips                                                                                                                                                                                                                                                                                                                                                                              |
| **Max drawdown**               | `max_drawdown_pct`               | `min( (equity − peak) / peak × 100 )`  where `peak = equity.cummax()` — the running all-time high of the equity curve.  Drawdown is always ≤ 0; the most negative value is the worst peak-to-trough decline                                                                                                                                                                                                        |
| **Sharpe ratio**               | `sharpe_ratio`                   | `mean(Rp − Rf) / std(Rp − Rf) × √N` where `Rp` = period portfolio return, `Rf = (1 + BACKTEST_RISK_FREE_RATE)^(1/N) − 1` (exact per-period compounding; default 0.0), and `N` = periods per year.  Bucket size is chosen **adaptively**: ≥ 2 days → daily (N = 365), ≥ 2 h → hourly (N = 8 760), otherwise 5-min (N = 105 120).  Crypto trades 24/7 so √365 (not √252) is the correct annualiser for daily buckets |
| **Sortino ratio**              | `sortino_ratio`                  | Same as Sharpe but `std` is computed on **downside excess returns only** (`excess_ret[excess_ret < 0]`).  Penalises only negative volatility                                                                                                                                                                                                                                                                       |
| **Profit factor**              | `profit_factor`                  | Σ(pnl > 0) / Σ(pnl < 0) ratio of gross winning P&L to gross losing P&L.  INF when there are no losing round trips                                                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Avg holding period**         | `avg_holding_minutes`            | `mean(holding_minutes)` excluding NaN entries (session-end mark-to-market closes have no real holding period)                                                                                                                                                                                                                                                                                                      |
| **Confidence filter hit rate** | `confidence_filter_hit_rate_pct` | `(confidence_blocked_buy + confidence_blocked_sell) / total_raw_candidates × 100` — % of raw candidates blocked because `regime_confidence < HMM_MIN_CONFIDENCE` (model too uncertain)                                                                                                                                                                                                                             |
| **Regime filter hit rate**     | `regime_filter_hit_rate_pct`     | `(regime_blocked_buy + regime_blocked_sell) / total_raw_candidates × 100` — % of raw candidates (that passed the confidence gate) blocked by the regime direction gate                                                                                                                                                                                                                                             |
| **VWAP filter hit rate**       | `vwap_filter_hit_rate_pct`       | Residual: `raw − executed − confidence_blocked − regime_blocked`, divided by `total_raw_candidates × 100` — % blocked by the VWAP dip/strength gate (third and final gate)                                                                                                                                                                                                                                         |
| **Position guard skips**       | `n_position_guard_skips`         | Count of BUY signals that passed all three gates but were suppressed because the strategy already held an open position (`open_strategy_qty > 0`) — single-position mean-reversion mode                                                                                                                                                                                                                            |
| **Whipsaw exits**              | `n_whipsaw_exits`                | Count of forced pessimistic exits triggered by the intra-candle whipsaw guard (same 1-minute bar's Low ≤ best-buy micro AND High ≥ best-sell micro).  Each event records a `SELL_WHIPSAW` trade at `candle_low − half_spread`                                                                                                                                                                                      |
| **Stop-loss fires**            | `n_stop_loss_fires`              | Count of positions force-closed by the adaptive stop-loss (`close < avg_entry_price × (1 − stop_loss_pct)`).  Each event records a `SELL_STOP_LOSS` trade.  See §Stop-loss vs B&H above for calibration guidance                                                                                                                                                                                                   |
| **Trend-pause skips**          | `n_trend_pause_skips`            | Count of 1-minute bars where a BUY or SELL signal was suppressed because `trend_pause == True` (the macro frame was in a sustained directional run of ≥ `TREND_CONSECUTIVE_BARS` bars).  The stop-loss check still fires unconditionally on paused bars                                                                                                                                                            |

---

## Step 6 — Regime Validation

Two levels of regime validation are implemented:

### 6a — Inline train / test split (implemented in `strategy/regime_director.py`)

Every time `select_hmm_model()` is called (initial fit + every 5-minute refit
during the live session and in the backtesting loop), an **adaptive 2/3 split**
is applied to the current lookback window:

```
train_end = max(2, int(n_rows × 2/3))
features[:train_end]    → fit()  +  bic()   (~⅔ of window, oldest rows)
features[train_end:]    → predict()  +  predict_proba()   (~⅓ of window, most-recent rows)
```

At the default `HMM_LOOKBACK_ROWS = 120` this gives `train_end = 80` (~⅔ in-sample,
~⅓ out-of-sample) — identical to the previous `HMM_TRAIN_ROWS = 80` hard-cap.
Shorter windows now scale proportionally (e.g. 60 rows → 40/20, 30 rows → 20/10)
instead of collapsing to a single test row.

* The scaler (`StandardScaler`) is `fit_transform`'d on the **training rows
  only** — mean and standard deviation from held-out candles cannot leak into
  the model.
* `model.predict()` and `model.predict_proba()` run on `features[train_end:]`
  only — the rows the model has **never seen** during `fit()`.
* `self.current_regime` and `self.regime_confidence` therefore always reflect a
  candle that was genuinely out-of-sample.  This is a direct application of the
  Step 6 philosophy within the live 10-hour rolling window.

`predict_current_regime()` (cheap Viterbi path, used between full refits)
applies the **same** split: it transforms `features[train_end:]` with the
already-fitted scaler and predicts only on those rows.

### 6b — Offline long-horizon validation (`backtest/diagnostics/regime_validation.py`) ✅

#### Purpose

The inline adaptive 2/3 split inside `select_hmm_model()` (Step 6a) validates regime
labels within a **2-hour rolling window**.  Step 6b asks a harder question:
*are the HMM labels still meaningful over a much longer, fully out-of-sample
horizon?*

Specifically it answers four questions:
- Does `trending_up` produce statistically higher forward returns than `trending_down`?
- Does `high_volatility` produce higher realised volatility than `neutral`?
- Are the labels stable in frequency (no regime collapse)?
- Is `regime_confidence` consistently above `HMM_MIN_CONFIDENCE` for each label?

#### Dataset

`VALIDATION_LOOKBACK = "730 days ago UTC"` fetches **~210,000 rows** at 5m
(730 × 24 × 12 = 210,240 candles).  Two years of BTC data capture multiple full
market cycle turns (trending, ranging, volatile), giving the HMM the broadest
possible basis for regime learning.  Crypto trades 24/7 so all days are fully
populated with no weekend gaps.

A **70 / 30 train-test split** is used.  `split_idx` is derived at runtime
from `int(len(features_df) * 0.70)` so the ratio stays exact regardless of
the actual number of rows Binance returns:

```
full dataset:  ~210,000 rows   (730 days × 288 candles/day at 5m)
  ├── train set:  first 70 %  →  ~147,000 rows   (~511 days, ~17 months)
  └── test  set:  last  30 %  →   ~63,000 rows   (~219 days, ~7.3 months)

split_idx = int(len(features_df) * 0.70)
train_df  = features_df.iloc[:split_idx]
test_df   = features_df.iloc[split_idx:]
```

> **`BACKTEST_MAX_ROWS` is intentionally bypassed.**  `regime_validation.py`
> calls `fetch_klines()` and `_add_hmm_features()` directly.
> Use `VALIDATION_LOOKBACK = "90 days ago UTC"` for a faster smoke-test run.

#### Phase 1 — Training Fit on Full Train Set (~147,000 rows)

This module is **self-contained** — it does **not** use `RegimeDirector`.
It replicates the BIC search and label-assignment logic directly using raw
`GaussianHMM` + `StandardScaler` so that the training window is not
capped by the live adaptive split (`train_end = max(2, int(n_rows × 2/3))`);
instead it fits on the full 70 % train set (~147,000 rows) to give the HMM
the broadest possible regime-coverage check:

1. `StandardScaler.fit_transform(train_features)` — learn mean/std from
   train rows only.  Test rows are never seen by the scaler during fitting.
2. `_fit_best_hmm(train_scaled)` — BIC search over `n = 2 … HMM_MAX_REGIMES`.
   Same retry logic as `RegimeDirector` (`HMM_N_INIT` seeds per n).
3. `_assign_labels(model)` — rank-based directional assignment identical to
   `RegimeDirector.assign_regime_labels()`.

#### Phase 2 — Vectorised Label Assignment on Test Set

The entire test set is scored in **four bulk operations** (no per-candle loop):

1. `scaler.transform(test_features)` — scale all ~63,000 test rows at once.
2. `model.predict(test_scaled)` — single Viterbi pass (hmmlearn C extension).
3. `model.predict_proba(test_scaled)` — Forward–Backward pass;
   `proba[np.arange(n), states]` extracts per-candle confidence via NumPy
   advanced indexing.
4. `label_array[states]` — maps state indices to labels with a NumPy lookup
   array (no Python loop).

| Approach | Viterbi calls | Estimated Phase 2 time |
|---|---|---|
| Per-candle loop | ~63,000 | hours |
| Vectorised (current) | **1** | **seconds** |

Total runtime is dominated by the **Binance paginated fetch** (~3–5 min for
2 years at 5m resolution).

#### Phase 3 — Validation Checks

For every labelled candle `t`, the 1-hour cumulative forward log-return is computed
as the rolling sum of 12 single-period log-returns shifted forward by 12 candles
(12 × 5 min = 1 hour):

```
fwd_return(t) = log(close(t+12) / close(t))   [1-hour cumulative log-return]
```

Six checks are then run on the ~63,000 labelled test candles (~219 days):

**Check 1 — Direction Test** *(primary)*

The mean forward return must follow the expected ordering across all three
directional regimes:

```
mean(forward_return | trending_up)  >  mean(forward_return | neutral)
                                    >  mean(forward_return | trending_down)
```

**Check 2 — Statistical Significance**

Kruskal-Wallis H-test across **all** BIC-selected regime states (k ≥ 2).
H₀: all state forward-return distributions are identical (rank-based).

```
H, p = scipy.stats.kruskal(*[
    forward_returns[regime == lbl]
    for lbl in sorted(unique_labels)
])
```

Preferred over Welch's t-test because (a) BTC returns are fat-tailed,
violating the Gaussian assumption; (b) HMM states have unequal emission
variances by construction; (c) it handles k > 2 states without multiple
comparison inflation.  For k = 2 it reduces to Mann-Whitney U.

**Pass condition:** p-value < 0.10

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
`HMM_MIN_CONFIDENCE` (0.60).  A label that is consistently uncertain suggests
the model cannot distinguish that regime reliably on new data.

**Check 5 — Label Frequency**

**Pass condition:** No regime has < 1 % frequency over the ~63,000 test
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

 Kruskal-Wallis H-test (all k states):
   H = x.xx,  p = x.xxxxxx  k=x groups →  [PASS / FAIL]

 Volatility check (high_vol mean vol > neutral mean vol):
   x.xxxxxx > x.xxxxxx                 →  [PASS / FAIL]

 Label frequency (all regimes > 1 %):  →  [PASS / FAIL]

 Confidence floor (all median conf ≥ 0.60): → [PASS / FAIL]

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
| Kruskal-Wallis H-test (not Welch's t-test) | BTC returns are fat-tailed (leptokurtic); HMM states have unequal emission variances by construction; K-W handles any k ≥ 2 states without multiple-comparison inflation |
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
| IS window | `BACKTEST_LOOKBACK = "360 days ago UTC"` → `BACKTEST_OOS_START = "90 days ago UTC"` (270 days, ~77,760 rows at 5 m) |
| OOS window | `BACKTEST_OOS_START = "90 days ago UTC"` → today (90 days, ~25,920 rows at 5 m) — used by `runner.py`; never fetched by `sensitivity.py` |
| Rationale | Tuning on 270 days of **recent IS data** at 5 m resolution gives broad regime coverage (~3 market cycles) while ensuring `runner.py` validation is genuinely out-of-sample. `SENSITIVITY_LOOKBACK` has been removed — the IS window is now fully defined by `BACKTEST_LOOKBACK` + `BACKTEST_OOS_START`. |
| Refit cadence | `REFIT_EVERY = 480` (40 h at 5 m) — shared by `sensitivity.py` (~162 refits over 270-day IS window) and `runner.py` (~54 refits over 90-day OOS window).  Full-BIC refits use identical cadence in IS and OOS. |
| Viterbi cadence | `SENSITIVITY_PREDICT_EVERY = 5` — IS sweep re-predicts the regime every 25 min (5 macro candles, last known label reused between calls); `runner.py` re-predicts every 5 min (every macro candle).  **IS / OOS Sharpe are NOT pure equivalents because of this gap** — see caveat below. |
| ⚠ IS↔OOS Sharpe comparability | The full-BIC refit cadence matches in IS and OOS, but the Viterbi cadence does not (`SENSITIVITY_PREDICT_EVERY = 5` vs `runner.py`'s implicit `1`).  This means the IS sweep responds to intra-refit regime transitions 20 minutes later than the OOS run, producing a slightly different signal mix on identical data.  A non-zero IS-vs-OOS Sharpe gap is therefore NOT a pure overfitting signal — part of it is the cadence gap.  Treat the IS Sharpe as a **parameter ranking score**, not as an absolute predictor of OOS performance.  Lowering `SENSITIVITY_PREDICT_EVERY` to 1 would close the gap but slow the IS sweep ~5× without any guarantee the cadence component is the dominant one. |
| Runtime (Bayes, 40 trials) | ~3–6 h on a laptop (~5–8 min/trial; klines pre-fetched once, shared across all trials) |
| Runtime (OAT, 8 runs) | ~1–2 h on a laptop |
| Output | `backtest/results/best_params.json` — loaded by `strategy.param_loader` (shared by `websocket_main.py` and `runner.py`) |
| Run (Bayes, default) | `python -m backtest.sensitivity` or `python -m backtest.sensitivity --bayes` |
| Run (OAT) | `python -m backtest.sensitivity --oat` |

#### Use Case B — Backtest robustness validation ⬜ *Deferred*

| Attribute | Value |
|---|---|
| Window | Must match the OOS validation window used by `runner.py` (`BACKTEST_OOS_START → today`) to avoid window-mismatch bias. |
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
| 3 | VWAP dip/strength window | `VWAP_WINDOW` | 5 (5 min) | **5**, 2, 3  *(10 min excluded — too slow for 1 s cadence)* |
| 4 | Taker fee rate | `BACKTEST_FEE_RATE` | 0.0005 (0.05 %) | **0.0005**, 0.00025  *(0.025 % achievable at VIP 4+; note: Bayesian mode always uses `0.001`)* |

**Excluded from scope:**
- Depth:delta score weights (0.70 / 0.30) — not a config constant.
- OBI threshold — adding one would be a new feature, not a config swap.

---

### Execution Modes

Three modes optimize the strategy over the 270-day in-sample window:

- **Bayesian (Optuna)** — Default: Uses Optuna's TPE for intelligent hyperparameter search. ~40 trials (~3–6 h on laptop). `python -m backtest.sensitivity --bayes`
- **One-At-a-Time (OAT)** — Quick sanity check: baseline + 7 variations (~1–2 h). `python -m backtest.sensitivity --oat`
- **Full-grid** — Exhaustive 30-run sweep (deprecated). `python -m backtest.sensitivity --full-grid`

For full CLI options: `python -m backtest.sensitivity --help`

---

### Persisting Best Parameters (`best_params.json`)

After the sweep, `sensitivity.py` writes the winning row to
`backtest/results/best_params.json`:

```json
{
  "hmm_lookback_rows": 60,
  "hmm_max_regimes": 3,
  "vwap_window": 20,
  "fee_rate": 0.0005,
  "generated_at": "2026-04-19T10:00:00+00:00",
  "source_metric": "sharpe_ratio",
  "source_value": 1.42
}
```

#### Who loads it and how

Both consumers delegate to **`strategy/param_loader.py`** — the single module
that owns all loading logic, keeping `websocket_main.py` and `runner.py`
free of JSON / file-handling code.

| Consumer | Function called | When | What is overridden | Mechanism |
|---|---|---|---|---|
| `websocket_main.py` | `param_loader.load_best_params()` | At startup, before `RegimeDirector()` is instantiated | `HMM_MAX_REGIMES`, `HMM_LOOKBACK` | Patches `strategy.regime_director` module namespace directly (not `config_parameters`) — necessary because `regime_director.py` binds constants at import time via `from config_parameters import`. |
| `runner.py` | `param_loader.load_best_params_for_backtest()` | At the top of `run_backtest()`, before `run_signals()` | `hmm_lookback_rows`, `hmm_max_regimes`, `vwap_window`, `vwap_threshold`, `fee_rate` | Returns a plain `dict`; values passed as keyword arguments to `run_signals()` and `simulate_pnl()`. Keys absent from `best_params.json` are simply omitted — both functions fall back to `config_parameters.py` defaults automatically. |

#### Parameter Flow

```
sensitivity.py  ──►  best_params.json  ──►  strategy/param_loader.py
                                                  ├── load_best_params()             → websocket_main.py  (live)
                                                  └── load_best_params_for_backtest() → runner.py    (backtest)
```

Both consumers fall back silently to `config_parameters.py` defaults when
`best_params.json` is absent or unreadable.

#### Notes on `VWAP_WINDOW` and `fee_rate`

- `vwap_window` — backtest-only (`backtest/signals.py`).  The live system does
  not use `VWAP_WINDOW`, so `websocket_main.py` ignores this field.
- `fee_rate` — `runner.py` loads it from `best_params.json` and passes it
  directly to `simulate_pnl(fee_rate=...)`.  `fee_rate` is NOT a tunable Optuna
  parameter — it is always written as `SENSITIVITY_FEE_RATE = 0.001` (standard
  Binance Spot taker fee), so reading it from `best_params.json` is always safe.
  Falls back to `BACKTEST_FEE_RATE` from `config_parameters.py` when absent.

#### Notes on `HMM_LOOKBACK` (live) vs `HMM_LOOKBACK_ROWS` (backtest)

`best_params.json` stores `hmm_lookback_rows` (an integer, used by the
backtest).  The live system uses `HMM_LOOKBACK` (a dateutil string such as
`"10 hours ago UTC"`).  Since the live system uses 5-minute klines, the
conversion multiplies rows × 5 min before formatting as a time string.
The conversion is handled by `rows_to_lookback(n, interval_minutes=5)`
in `strategy/param_loader.py`:

- `total_minutes = rows × 5`
- `total_minutes < 60` → `"N minutes ago UTC"`
- Exact multiples of 60 → `"H hour(s) ago UTC"` (e.g. 120 rows → 600 min → `"10 hours ago UTC"`)
- All other values → `"N minutes ago UTC"` (handles any Optuna-discovered value like 40 rows → `"200 minutes ago UTC"`)

This replaces the previous static lookup table, which broke whenever Optuna
discovered a value outside the hand-coded set (e.g. 40 rows → `"40 minutes ago UTC"`).
If `hmm_lookback_rows` is absent from `best_params.json`, `HMM_LOOKBACK` is
left at the `config_parameters.py` default.

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

- ✅ **Step 1 — `backtest/data.py`** — Downloads klines from Binance via
  `binance.client.Client.get_historical_klines()`.  Two typed wrappers:
  `fetch_macro_klines()` (`BACKTEST_MACRO_INTERVAL = "5m"`, HMM input) and
  `fetch_micro_klines()` (`BACKTEST_MICRO_INTERVAL = "1m"`, PnL input).
  Both route through a **Parquet cache** (`cache/klines/`, 24-hour TTL,
  `flush_kline_cache()` / `--flush-cache` for explicit invalidation).
  `end_str=BACKTEST_OOS_START` enforces the IS boundary in `sensitivity.py`;
  `lookback=BACKTEST_OOS_START` fetches the OOS window in `runner.py`.
  Row counts: IS ~77,760 macro (5 m) / ~388,800 micro (1 m);
  OOS ~25,920 macro / ~129,600 micro.  *(Implemented.)*
- ✅ **Step 2 — `backtest/synthetic_book.py`** — Given a single kline row
  (`pd.Series`), constructs a 50-level synthetic order book with exponential
  volume decay and OBI asymmetry injection.  Returns a
  `{"bids": {…}, "asks": {…}}` dict matching the `state.local_book` format
  consumed by `AnalysisEngine._build_levels()`.  *(Implemented.)*
- ✅ **Step 3 — `backtest/signals.py`** — Two-frame orchestrator.
  **Phase 1:** HMM walk-forward on `df_macro` (5 m, ~77,760 IS rows) — rolling
  `[i-_lookback : i]` slices → `select_hmm_model` (full BIC re-fit) /
  `predict_current_regime` (cheap Viterbi) → `assign_regime_labels` →
  `regime_df`.  **Phase 2:** temporal stitch via `pd.merge_asof(direction='backward')`
  (zero look-ahead) → `df_exec` (1 m bars).  **Phase 3:** execution loop on
  `df_exec` (~388,800 IS rows) — `build_synthetic_book` → production pipeline
  (`build_levels → collect_candidates → select_best_opportunity`) → three
  sequential gates (confidence, regime, VWAP dip/strength) → signal `+1`/`−1`/`0`.
  `high` and `low` columns retained in output for the whipsaw guard.
  *(Implemented.)*
- ✅ **Step 4 — `backtest/pnl.py`** — Converts the signal Series into
  simulated trades (using ``bq``/``aq`` quantities and the balance guard)
  and computes the Step 5 performance metrics: total return, win rate,
- ✅ **Step 4 — `backtest/pnl.py`** — Converts the signal DataFrame into
  simulated trades (using ``bq``/``aq`` quantities and the balance guard)
  and computes the Step 5 performance metrics: total return, win rate,
  max drawdown, Sharpe, Sortino, profit factor, average holding period,
  and regime / VWAP filter hit rates.  **Single-position mean-reversion
  guard** (``open_strategy_qty`` / ``_POSITION_DUST_BTC = 1e-6``): BUY
  fires only when flat; SELL closes the full strategy-opened position in
  one shot and resets to flat; suppressed BUY count returned as
  ``n_position_guard_skips`` in the stats dict and reported in the console
  summary.  **Intra-candle whipsaw guard** (Step 9): fires when same 1-minute
  bar's `low ≤ best_buy_micro` AND `high ≥ best_sell_micro` — force-closes at
  `low − half_spread`, records `SELL_WHIPSAW` trade, returns `n_whipsaw_exits`
  in stats dict.  Round-trip pairing uses an exhaustive FIFO ``collections.deque``
  supporting scaling-in, layering, and pyramiding.  Orphan SELLs emit a
  visible WARNING box.  **Fee fix** — ``_pair_round_trips(fee_rate=...)``
  now explicitly deducts both taker fees from per-trade P&L stats:
  ``pnl_usdt = gross_pnl − entry_fee − exit_fee``.  *(Implemented.)*
- ✅ **Step 5 — `backtest/runner.py` + `backtest/reporting/formatters.py`** —
  Top-level script that chains all modules.  Fetches the **OOS window**
  (`BACKTEST_OOS_START = "90 days ago UTC"` → today, ~25,920 rows at 5 m /
  ~129,600 rows at 1 m) via `fetch_macro_klines` + `fetch_micro_klines`
  (both cached).  Report formatting (`print_report`) and CSV export
  (`save_csv`) are isolated in `backtest/reporting/formatters.py`.
  Prints a formatted console report: SESSION info, SIGNALS breakdown
  (including ``HOLD (position open) : N ← BUY suppressed by position guard``),
  P&L SUMMARY, RISK METRICS, TRADE LOG PREVIEW.  `runner.py` loads
  ``fee_rate`` from ``best_params.json`` and passes it to ``simulate_pnl()``
  (falls back to ``BACKTEST_FEE_RATE``).  Optionally saves timestamped
  ``trades_*.csv`` and ``equity_*.csv`` to ``backtest/results/``.
  `--flush-cache` flag clears the Parquet cache before fetching.
  Run with ``python -m backtest.runner``.  *(Implemented.)*
- ✅ **Step 6a — Inline regime validation** (`strategy/regime_director.py`) —
  An **adaptive 2/3 split** is applied inside every `select_hmm_model()` and
  `predict_current_regime()` call: `train_end = max(2, int(n_rows × 2/3))`;
  the model fits on `features[:train_end]` (oldest ~⅔ of the window) and
  predicts only on `features[train_end:]` (most-recent ~⅓, never seen during fit).
  At the default 120-row window `train_end = 80`; shorter windows scale
  proportionally.  `self.current_regime` and `self.regime_confidence` always
  reflect a genuinely out-of-sample candle.  This is a direct application of the
  Step 6 philosophy within the live 10-hour rolling window.

  *(Implemented.)*
- ✅ **Step 6b — Offline long-horizon regime validation** (`backtest/diagnostics/regime_validation.py`) —
  Standalone diagnostic script (run with `python -m backtest.diagnostics.regime_validation`).
  Fetches **2 years** of 5-minute klines (~210,000 rows, `VALIDATION_LOOKBACK = "730 days ago UTC"`),
  applies a **70/30 train-test split** at runtime (`split_idx = int(len(df) * 0.70)`).
  **Self-contained** — does not use `RegimeDirector`; replicates BIC search and label
  assignment directly with raw `GaussianHMM` + `StandardScaler` so the training window
  is not capped by the live adaptive split.  Unlike the live per-window split which
  uses ~⅔ of a 120-row rolling window, this diagnostic fits on the full 70 % train set
  (~147,000 rows) to give the HMM the broadest possible regime-coverage check.  Phase 2 scores all ~63,000 test candles in a
  **single vectorised Viterbi pass** with NumPy advanced indexing for confidence
  extraction (`proba[np.arange(n), states]`) and a lookup-array for label mapping
  (`label_array[states]`).  `BACKTEST_MAX_ROWS` is intentionally bypassed.
  Six checks (Phase 3) are computed on the labelled test candles:
  1. **Direction test** — `trending_up` mean > `neutral` mean > `trending_down`
     mean forward 1-candle return.
  2. **Kruskal-Wallis H-test** — p < 0.10 across all k BIC-selected regime
     states (non-parametric; no normality or equal-variance assumption).
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
  Uses ``plotly`` (already in ``requirements.txt``).  The chart is **shown by
  default** when running via ``python -m backtest.runner``; use ``--no-plot``
  to suppress in headless / CI environments.  Programmatic calls default to
  ``plot=False``; pass ``run_backtest(plot=True, save_png=True)`` explicitly.  Panels:
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

- ✅ **Step 8 — `backtest/sensitivity.py`** — Three modes: Bayesian (Optuna TPE, 40 trials default), OAT (`--oat`, 1 baseline + 7 variants), and deprecated full-grid.  Tunes `HMM_LOOKBACK_ROWS`, `HMM_MAX_REGIMES`, `VWAP_WINDOW`, `VWAP_THRESHOLD_MULTIPLIER` on the 270-day IS window; `fee_rate` fixed at `0.001` (not a strategy knob).  Data pre-fetched once via `_make_objective` factory closure (`prefetched_macro` + `prefetched_micro`); Optuna study persisted to `backtest/results/optuna.db` (`load_if_exists=True`).  Three HTML charts (optimisation history, fANOVA parameter importance, `hmm_lookback_rows × vwap_window` contour) auto-saved to `backtest/reporting/`.  `_check_existing_best_params()` guard prompts `[y/N]` before overwriting.  Sensitivity CSVs → `backtest/reporting/`; `best_params.json` → `backtest/results/` (loaded by `strategy.param_loader`).  Console formatting via `print_sensitivity_table`, `print_oat_sensitivity_report`, `print_bnh_comparison` in `backtest/reporting/formatters.py`.  Use Case B (OOS robustness) deferred — see **Step 8** above for full details.  *(Implemented.)*

- ✅ **Step 9 — Multi-timeframe resolution decoupling + documentation** (2026-05-18)

  **Why two resolutions?**
  The HMM uses 5-minute klines because microstructure noise in 1-minute bars
  degrades EM convergence and destabilises BIC selection. Regime shifts are
  structural (hours, not seconds), so 5m granularity is sufficient and produces
  richer per-bar signals with 5× fewer rows.

  **How `merge_asof` prevents look-ahead bias:**
  A naïve timestamp join would align a 5m regime label with 1m bars that formed
  *during* that candle's build. `merge_asof(direction='backward')` propagates
  only the last *completed* 5m label forward — each 1m bar sees only information
  that was available when it opened.

  **Why multiple 1m bars share one 5m label:**
  Each 5m candle close produces one regime update; the five 1m bars within that
  window intentionally carry the same label. This mirrors how the live system
  operates (HMM fires at candle close, order logic runs every second).

  **Live alignment:** `HMM_INTERVAL = "5m"` in the live system now matches the
  backtest macro frame, making IS↔OOS Sharpe comparisons reliable.

  **Phase 0 — `config_parameters.py`:**  `BACKTEST_MACRO_INTERVAL = "5m"`,
  `BACKTEST_MICRO_INTERVAL = "1m"` added.  `HMM_INTERVAL = "5m"`,
  `HMM_LOOKBACK = "10 hours ago UTC"` (live system aligned).
  `SENSITIVITY_PREDICT_EVERY = 5` retained.  `SENSITIVITY_REFIT_EVERY` removed — refit cadence unified under `REFIT_EVERY = 480` shared by both pipelines.  `HMM_N_INIT` reduced 10 → 5.

  **Phase 1 — `backtest/data.py`:** `fetch_macro_klines()` and `fetch_micro_klines()`
  typed wrappers added.  Parquet cache (`_cached_fetch`, `_cache_path`,
  `_is_cache_fresh`, `flush_kline_cache`) with 24-hour TTL implemented.
  `--flush-cache` CLI flag wired in `sensitivity.py` and `runner.py`.

  **Phase 2 — `backtest/signals.py`:** `run_signals()` refactored into three phases:
  (1) HMM walk-forward on `df_macro` (5 m); (2) temporal stitch via `merge_asof`
  (direction=`'backward'`, zero look-ahead); (3) execution loop on `df_exec` (1 m).
  `high` and `low` columns retained in output for the whipsaw guard.

  **Phase 3 — `backtest/pnl.py`:** Intra-candle whipsaw guard implemented — fires when
  the 1-minute bar's `low ≤ best_buy_micro` AND `high ≥ best_sell_micro`; force-closes
  at `low − half_spread`; records `SELL_WHIPSAW` trade; `n_whipsaw_exits` returned in
  `stats` dict.  `ask_vwap` read extracted to top of loop.

  **Phase 4 — `backtest/sensitivity.py`:** Pre-fetch block updated to
  `fetch_macro_klines` + `fetch_micro_klines` (IS window; both cached).
  Objective function passes `prefetched_macro` + `prefetched_micro` to
  `run_signals()`.  `--flush-cache` flag added.

  **Phase 5 — `backtest/runner.py`:** OOS fetch updated to
  `fetch_macro_klines` + `fetch_micro_klines` (no `end_str`).
  `run_signals()` called with both frames.  `--flush-cache` flag added.

  **Phase 6 — `strategy/regime_director.py` + `strategy/analysis.py`:**
  `get_klines_data()` now fetches 5-minute klines (`HMM_INTERVAL = "5m"`,
  120-bar window = 10 h).  `historical_analysis()` HMM re-fit trigger aligned
  to exact 5-minute clock boundaries (`timestamp % 300 == 0`).

  **Phase 7 — Docs:**  `BACKTESTING.md`, `SYSTEM_ARCHITECTURE.md`,
  `REGIME_DIRECTOR.md`, `README.md` updated to reflect the new
  two-resolution design, parquet cache, whipsaw guard, and live constants.
  *(Implemented.)*

---

*Document last updated: 2026-05-28*

