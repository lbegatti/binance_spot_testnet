# `RegimeDirector` Deep Dive: How It Fits Into the Whole Codebase

> This document was extracted from `README.md` (Appendix A) for readability.
> It traces `RegimeDirector` from its configuration constants all the way through
> to its effect on individual order decisions, file by file.

---

## A.1 Configuration — `config_parameters.py`

Every tunable parameter `RegimeDirector` needs is centralised here.
No magic numbers appear anywhere else in the strategy code.

| Constant | Value | Purpose |
|---|---|---|
| `HMM_FEATURE_COLS` | `["return", "volatility", "obi_proxy", "trade_density"]` | Features fed to `GaussianHMM` |
| `HMM_INTERVAL` | `Client.KLINE_INTERVAL_5MINUTE` | Kline granularity (5 min — consistent with `BACKTEST_MACRO_INTERVAL`; 12 bars/h) |
| `HMM_LOOKBACK` | `"10 hours ago UTC"` | Rolling window (~120 rows at 5 m); keeps enough data for stable EM convergence while tracking intra-day regime shifts |
| `HMM_MAX_REGIMES` | `3` | Upper bound on hidden states evaluated during BIC search (2 … 3, = `len(HMM_FEATURE_COLS) − 1`) |
| `HMM_N_ITERATIONS` | `1000` | Max EM iterations per model fit |
| `HMM_RANDOM_STATE` | `46` | Seed for reproducible state numbering across fits |
| `HMM_MIN_COVAR` | `1e-1` | Regularisation floor for covariance matrices (1e-1 recommended safe default for z-scored financial features) |
| `HMM_N_INIT` | `10` | Random-seed restarts per candidate `n_components`; reduces degenerate EM solutions on flat windows |
| `HMM_REFIT_INTERVAL` | `300` | Full re-fit cadence (s).  Between re-fits only Viterbi prediction runs |
| `HMM_TRAIN_ROWS` | `80` | Legacy constant — **no longer used** to cap the train/test split inside `regime_director.py`.  The split is now computed adaptively as `train_end = max(2, int(n_rows × 2/3))` per window.  Retained in `config_parameters.py` for reference and diagnostic use |

> **Why 10 hours and not 2?**  
> Switching from 1-minute to 5-minute klines requires a proportionally longer
> lookback to keep the same number of bars (~120) that HMM needs for stable EM
> convergence: 2 h × 5 = 10 h.  The 5-minute resolution is less noisy than 1-minute
> — each bar aggregates 5 minutes of tick data — while the 10-hour window still
> captures sharp intra-day regime shifts.  A 4-hour window at 5 m would give only
> 48 rows, which is insufficient for reliable multi-state EM with 4 features.

---

## A.2 Pre-session Startup — `websocket_main.py` (step 4b)

`RegimeDirector` is instantiated and fully fitted **before any thread starts**,
in the single-threaded startup block of `websocket_main.py`:

```python
# websocket_main.py — step 4b (after seeding OrderBookState, before engine)
from strategy.regime_director import RegimeDirector

regime_director = RegimeDirector()             # 1. instantiate — no data yet
regime_director.get_klines_data()              # 2. download ~120 rows of 5-min klines
                                               #    (last 10 h, HMM_INTERVAL="5m", public endpoint)
regime_director.select_hmm_model()             # 3. fit HMM n=2..4, pick best BIC
regime_director.assign_regime_labels()         # 4. map state int → label string
# → regime_director.regime_label == e.g. "trending_up"

engine = AnalysisEngine(
    state=state,
    stop_event=stop_event,
    executor=executor,
    regime_director=regime_director,           # 5. inject into AnalysisEngine
)
```

> **`None` sentinel for `lookback` and `max_regimes`**  
> `RegimeDirector.__init__` accepts `lookback: str | None = None` and
> `max_regimes: int | None = None`.  When `None`, each resolves to the
> module-level constant (`HMM_LOOKBACK` / `HMM_MAX_REGIMES`) **at call time**.  
> This matters because `load_best_params()` (called in `websocket_main.py`
> *before* `RegimeDirector()`) patches `strategy.regime_director.HMM_LOOKBACK`
> and `strategy.regime_director.HMM_MAX_REGIMES` directly.  If the parameters
> had default values baked in (e.g. `lookback: str = HMM_LOOKBACK`), Python
> would freeze those values at import time and the patch would have no effect.
> The `None` sentinel forces a fresh module-namespace lookup on every
> instantiation.  Full explanation in `strategy/param_loader.py` docstring.

**Why fit before threads start?**  
`low_latency_analysis` reads `regime_label` on its very first iteration
(t = 1 s). If the fit were deferred to the first `historical_analysis` run
(t = 60 s), every order in the first minute would be made with `regime_label = None`
(transparent filter — no regime gating at all).  Fitting at startup guarantees
the regime filter is active from iteration #1.

---

## A.3 Inside `AnalysisEngine` — `strategy/analysis.py`

`AnalysisEngine.__init__` stores the injected instance and creates a dedicated
lock to protect the label between the two threads:

```python
self.regime_director = regime_director   # injected — NOT re-created internally
self._regime_lock    = threading.Lock()  # protects regime_label across threads
```

Two background threads then interact with it in opposite roles:

---

### Thread A — `historical_analysis()` every 60 s → **writer**

After computing VWAPs from the live order book, this thread updates the HMM
regime **only when a new 5-minute clock boundary has elapsed** (i.e. the
current Unix timestamp crossed a multiple of 300 s since the last update).
The expensive work runs **outside** the lock; only the instant label assignment
is locked:

```python
import time

# VWAP update — runs every HIST_INTERVAL (60 s), unchanged
...

# HMM update — gated on 5-minute clock boundary (aligns to backtest merge_asof cadence)
now = int(time.time())
current_5m_boundary = now - (now % 300)          # round down to nearest 5-min mark

if current_5m_boundary > _last_hmm_boundary:     # a new 5-min bar has closed
    _last_hmm_boundary = current_5m_boundary
    hmm_iteration += 1

    # OUTSIDE _regime_lock — slow: network download + CPU model fit/predict
    self.regime_director.get_klines_data()       # re-fetch latest 10 h of 5-min klines

    # Two-speed update: full re-fit every hmm_refit_every boundaries (default 1 = every 5 min),
    # cheap Viterbi prediction on all other boundaries.
    if hmm_iteration % hmm_refit_every == 0:
        self.regime_director.select_hmm_model()          # full re-fit — slow
    else:
        self.regime_director.predict_current_regime()    # Viterbi only — fast

    # INSIDE _regime_lock — fast: dict lookup + string assignment only
    with self._regime_lock:
        self.regime_director.assign_regime_labels()      # write regime_label
```

This split means `low_latency_analysis` is **never blocked** waiting for a
model fit to finish — it only waits for the lock during the microsecond string
write.  The clock-boundary gate ensures the backtest and live system fire the
HMM pulse at exactly the same 5-minute cadence.

---

### Thread B — `low_latency_analysis()` every 1 s → **reader**

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
        pass
    elif bid_vwap is not None and micro_price >= bid_vwap * (1.0 - VWAP_THRESHOLD_MULTIPLIER):
        # VWAP dead-zone blocks BUY — dip too shallow to cover fees (inside ±δ of bid_vwap)
        pass
    elif self._position_open:
        # Position guard — already holding a strategy-opened position.
        # Prevents order stacking (grid behaviour) in both REST-fallback mode
        # (balance never updated) and the WS race window (balance update arrives late).
        # Mirrors the identical guard in backtest/pnl.py.
        self._position_guard_skips += 1
        logging.info("HFT #%d [buy] — skipped: position already open (guard skips: %d)",
                     iteration, self._position_guard_skips)
    else:
        self._position_open = True        # mark position as open BEFORE calling execute
        executor.execute("BUY", best_buy)      # all filters passed

if best_sell:
    if current_regime in ("trending_up", "high_volatility"):
        # regime blocks SELL — skip entirely
        pass
    elif ask_vwap is not None and micro_price < ask_vwap * (1.0 + VWAP_THRESHOLD_MULTIPLIER):
        # VWAP dead-zone blocks SELL — rally too weak to cover fees (inside ±δ of ask_vwap)
        pass
    else:
        self._position_open = False           # reset guard — strategy is now flat
        executor.execute("SELL", best_sell)    # both filters passed
```

BUY is anchored to `bid_vwap` (volume-weighted bid pressure); SELL is anchored to `ask_vwap` (volume-weighted ask pressure). Using separate anchors avoids cross-side VWAP bias.

---

## A.4 Complete Data-Flow Diagram

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
  ③ 5-min clock-boundary check:              ④ read regime_label
       now = int(time.time())                      under _regime_lock
       boundary = now - (now % 300)           ⑤ REGIME FILTER
       if boundary > _last_hmm_boundary:            BUY  blocked if "trending_down"
         get_klines_data()  ← outside lock               or "high_volatility"
         if hmm_iter%refit_every==0:               SELL blocked if "trending_up"
           select_hmm_model()    ← slow                   or "high_volatility"
         else:                                 ⑥ VWAP FILTER (mean-reversion + dead zone)
           predict_current_regime() ← fast          BUY  blocked if micro ≥ bid_vwap × (1−δ)
         assign_regime_labels()                     SELL blocked if micro < ask_vwap × (1+δ)
           under _regime_lock                  ⑦ POSITION GUARD (single-open-position MR)
                                                     BUY  blocked if _position_open == True
                                                          → _position_guard_skips++
                                                     SELL resets _position_open = False
                                               ⑧ OrderExecutor.execute()
                                                    LIMIT GTC via WebSocket API
```

---

## A.5 Regime Label Reference

| `regime_label` | BUY | SELL | Typical market condition |
|---|---|---|---|
| `"trending_up"` | ✅ | ❌ | Highest combined return + OBI rank (most bullish state) |
| `"trending_down"` | ❌ | ✅ | Lowest combined return + OBI rank (most bearish state) |
| `"high_volatility"` | ❌ | ❌ | Large intra-bar swings OR heavy trade fragmentation — unreliable market |
| `"neutral"` | ✅ | ✅ | No dominant signal in any feature |
| `None` *(impossible after step 4b)* | ✅ | ✅ | Transparent — all orders pass through |

---

## A.5a Confidence Gating — `HMM_MIN_CONFIDENCE`

Even with a valid regime label, the HMM's posterior probability for the current state may be ambiguous (e.g., 55% trending_up vs 45% neutral).
The `predict_proba()` method returns this posterior probability for the assigned regime state.

**Confidence gate**: If `regime_confidence < HMM_MIN_CONFIDENCE` (default 0.70 = 70% posterior probability threshold),
both BUY and SELL orders are skipped on that iteration.
This prevents trading on weak signals when the model is uncertain.

```python
# In low_latency_analysis():
regime_confidence = self.regime_director.regime_confidence  # posterior prob ∈ [0.0, 1.0]

if regime_confidence is not None and regime_confidence < HMM_MIN_CONFIDENCE:
    # Skip both sides — signal is ambiguous (coin-flip probability)
    logging.info("Regime '%s' confidence %.2f < %.2f — skipped",
                 current_regime, regime_confidence, HMM_MIN_CONFIDENCE)
    continue
```

`HMM_MIN_CONFIDENCE` is configured in `config_parameters.py` and can be tuned independently of the `sensitivity.py` Bayesian search
(it is not in the Optuna search space — fixed based on risk tolerance for ambiguous signals).

---

## A.6 Threading Safety Summary

| Operation | Lock held | Duration |
|---|---|---|
| `get_klines_data()` | none | ~1–2 s (network I/O) — fires on each 5-min clock boundary |
| `select_hmm_model()` | none | ~2–5 s (CPU — EM iterations) — every 5 min (default `hmm_refit_every = 1`) |
| `predict_current_regime()` | none | < 50 ms (single Viterbi pass) — between full refits |
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

