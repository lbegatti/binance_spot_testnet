# SYSTEM ARCHITECTURE: MULTI-TIMEFRAME ORDER BOOK STRATEGY

-----------------------------------------------------------------------------
SYMBOL    : BTCUSDT  (production WebSocket stream / testnet REST client)
CURRENCY  : USDT  |  CRYPTO CCY : BTC

FILES
─────
  config_parameters.py                — All tunable constants (symbol, intervals, depths, timeouts, HMM params …)
  websocket_main.py                   — Session driver: startup, pre-session regime detection, threads, shutdown
  restapi_main.py                     — REST-only static snapshot (reference / dev tool)

  core/
    order_book_state.py               — OrderBookState: shared state + two threading.Locks
    message_handler.py                — MessageHandler: one active callback (depth) + one superseded (balance)

  strategy/
    analysis.py                       — AnalysisEngine: low-latency loop + historical loop (VWAP + HMM regime filter)
    book_utils.py                     — Shared order-book utilities (build_levels, collect_candidates, select_best_opportunity) — NumPy-vectorised; shape-mismatch guard
    regime_director.py                — RegimeDirector: GaussianHMM regime detection on 5-min klines (HMM_INTERVAL="5m", HMM_LOOKBACK="10 hours ago UTC", 120 bars)
    best_quote_calculator.py          — Live spread printer (best bid | best ask on every tick)
    metrics.py                        — Order book depth metrics
    indicators.py                     — Technical indicators (trend confirmation, VWAP helper, etc.)
    scores.py                         — Normalised opportunity scores
    quotes.py                         — find_best_quote(): best bid/ask selection helpers

  execution/
    order_executor.py                 — OrderExecutor: LIMIT GTC orders via WebSocket API + balance guards

  visualization/
    plot_helpers.py                   — Charting utilities for the REST snapshot path

  backtest/                           — Offline backtesting framework (see BACKTESTING.md)
    data.py                           — Historical kline downloader: fetch_macro_klines() (5m, HMM) + fetch_micro_klines() (1m, PnL); Parquet cache (cache/klines/, 24h TTL); --flush-cache flag
    synthetic_book.py                 — Synthetic 50-level order book builder (per kline row)
    signals.py                        — Two-frame signal pipeline: Phase 1 HMM walk-forward on 5m + trend_pause flag + adaptive stop_loss_pct; Phase 2 merge_asof stitch; Phase 3 1m execution loop + regime/VWAP gates
    pnl.py                            — P&L simulation: adaptive stop-loss, trend-pause gate, balance guard, bps-based fill, intra-candle whipsaw guard, position cap, FIFO round-trip pairing, equity curve, Step 5 metrics
    runner.py                         — Top-level backtest runner: chains all modules, delegates report/CSV to reporting/
    regime_validation.py              — Offline long-horizon regime diagnostic (Step 6b): 70/30 train-test split on 2 years (~210,000 rows at 5m), self-contained BIC search + label assignment, vectorised single Viterbi pass on ~63,000 test candles, six checks
    visualization.py                  — 7-panel Plotly chart: equity curve + B&H overlay, drawdown, price+signals, regime timeline, VWAP vs micro-price, signal funnel, signals-by-regime
    reporting/                        — Console report formatting and CSV export (AI-authored)
      formatters.py                   — fmt(), print_report(), save_csv(), print_regime_validation_report(), print_bnh_comparison()

## 1. CONFIGURATION  (config_parameters.py)
  See `config_parameters.py` directly — all constants are inline-documented.

## 2. SHARED STATE  (order_book_state.py — OrderBookState)
  Single source of truth injected into both MessageHandler and AnalysisEngine.
  Two dedicated locks prevent contention between the 100 ms order-book path
  and the lower-frequency balance-update path:

    local_book           : dict   — {"bids": {price: qty}, "asks": {price: qty},
                                      "lastUpdateId": int}
    history_order_book   : deque  — rolling window of snapshots (maxlen=3000,
                                      ~5 min at 100 ms); each entry:
                                      {timestamp, lastUpdateId, best_bid,
                                       best_ask, volume_best_bid, volume_best_ask}
    balance_status       : dict   — {"BTC": float, "USDT": float}
                                      free balances; seeded from REST; updated in
                                      real time via outboundAccountPosition push events.
    thread_lock          : Lock   — serialises local_book + history_order_book
                                      (MessageHandler writes; AnalysisEngine reads)
    thread_balance_lock  : Lock   — serialises balance_status independently of
                                      thread_lock to avoid unnecessary contention

## 3. DATA INGESTION  (message_handler.py — MessageHandler)
  Merges incoming diff-depth WebSocket ticks into `local_book` under
  `thread_lock` (zero-qty levels are deleted; positive-qty levels are upserted),
  appends a best-bid/ask snapshot to `history_order_book` on every tick, and
  calls `calculate_best_quote()` every `QUOTE_EVERY_N_TICKS` (~1 s) to print
  the live spread.  A stale-update guard discards any tick whose update ID is
  ≤ `local_book["lastUpdateId"]`.

## 4. ANALYSIS ENGINE  (analysis.py — AnalysisEngine)
  Owns two background threads (started by `websocket_main.py`), both of which
  check `stop_event` on every iteration and exit gracefully when it is set.

  Private cross-thread attributes:
    _vwap_lock       : Lock          — serialises _bid_vwap / _ask_vwap
    _bid_vwap        : float | None  — latest bid VWAP from historical_analysis
    _ask_vwap        : float | None  — latest ask VWAP from historical_analysis
    _regime_lock     : Lock          — serialises regime_director.regime_label
    _position_open   : bool          — single-open-position guard; mirrors
                                        open_strategy_qty in backtest/pnl.py
    _position_guard_skips : int      — BUY signals suppressed this session

  ▸ low_latency_analysis()  [every HFT_INTERVAL = 1 s]
      Reads balance, copies order book under thread_lock, builds the top
      N_LEVELS (50) bid/ask pairs, scores candidates (70 % depth / 30 % delta),
      applies the VWAP gate (mean-reversion dead zone, δ = 0.002) and the
      two-gate HMM regime filter (confidence ≥ 0.60, direction check), then
      delegates to OrderExecutor.execute().

  ▸ historical_analysis()  [every HIST_INTERVAL = 60 s]
      Always recomputes bid_vwap / ask_vwap from history_order_book and
      publishes under _vwap_lock.  HMM block fires only when a new 5-minute
      UTC clock boundary is crossed: fetches fresh 5m klines, then either runs
      a full BIC re-fit (select_hmm_model) or a cheap Viterbi pass
      (predict_current_regime), and writes the new label under _regime_lock.

  ▸ Deque fill-up (at WS_SPEED = 100 ms → ~10 entries/sec):

      │ Time elapsed │ WS ticks  │ Deque size          │ Historical iterations │
      │──────────────│───────────│─────────────────────│───────────────────────│
      │ 1 min        │ ~600      │ 600                 │ 1st runs              │
      │ 3 min        │ ~1 800    │ 1 800               │ 3rd runs              │
      │ 5 min        │ ~3 000    │ 3 000 (full)        │ 5th runs              │
      │ 10 min       │ ~6 000    │ 3 000 (capped)      │ 10th runs             │
      │ 20 min       │ ~12 000   │ 3 000 (capped)      │ 20th runs             │

      After ~5 min the deque is full and becomes a true rolling window.

  ▸ Thread timeline (default 10-min session):

      t=0s      Both threads start
                ├── low_latency: runs immediately, then every 1 s
                └── historical:  sleeps 1 min first (stop_event.wait(HIST_INTERVAL))

      t=1s      low_latency iteration #1
      ...
      t=1min    low_latency iteration #60
                historical iteration #1 → computes bid_vwap / ask_vwap,
                  publishes under _vwap_lock
                ↓ from this point, low_latency reads the VWAP and applies filter

      t=5min    low_latency iteration #300
                historical iteration #5 → deque now full, true rolling window
      ...
      t=10min   low_latency iteration #600
                historical iteration #10 → refreshes VWAP one last time
                stop_event set → both threads exit

## 4b. REGIME DETECTION  (strategy/regime_director.py — RegimeDirector)
  Trains a GaussianHMM on recent 5-minute klines to classify the current
  market into a hidden regime.  Exposed as `regime_label` (string) and
  `regime_confidence` (float) consumed by AnalysisEngine under `_regime_lock`.

  HMM features (computed from OHLCV kline data):

    return        = close.pct_change()
    volatility    = (high - low) / close
    obi_proxy     = (taker_buy_base_vol / volume) × 2 - 1   ∈ [-1, +1]
    trade_density = num_trades / volume

  Feature scaling: StandardScaler fitted on `features[:train_end]` (oldest ~⅔
  of the 10-h window) and applied to the full window.  At 120 rows,
  `train_end = 80`.  `regime_confidence = predict_proba(full_scaled)[-1,
  current_regime]`.

  Label assignment (rank-based, no hard-coded price constants):
    direction_score = return.rank() + obi_proxy.rank()
    idxmax → "trending_up"    idxmin → "trending_down"
    remaining states: high_vol OR high_td → "high_volatility"; else "neutral"

  Regime gates in low_latency_analysis (sequential):
    Gate 1 — confidence < 0.60 → skip both BUY and SELL
    Gate 2 — BUY suppressed if regime ∈ {trending_down, high_volatility}
              SELL suppressed if regime ∈ {trending_up,  high_volatility}

  Lifecycle: initial full fit in websocket_main.py before threads start.
  Every 5-minute UTC boundary: predict_current_regime() (cheap) or
  select_hmm_model() (full BIC re-fit, every HMM_REFIT_INTERVAL seconds).

## 5. SESSION DRIVER  (websocket_main.py)
  Orchestrates the full lifecycle: loads API keys, consolidates non-BTC/USDT
  balances, seeds `OrderBookState`, runs the pre-session HMM fit (so
  `regime_label` is never `None` on the first low-latency tick), opens the
  WebSocket stream, starts both analysis threads, blocks for
  `DEFAULT_SESSION_MINUTES`, then shuts down cleanly (stop_event → ws_client.stop()
  → executor.stop() → thread joins → order report → balance P&L decomposition).

## 6. EXECUTION  (execution/order_executor.py — OrderExecutor)
  Places LIMIT GTC orders via the Binance WebSocket API and maintains real-time
  balance via `outboundAccountPosition` push events on the same connection
  (session.logon → userDataStream.subscribe), eliminating the need for a
  listenKey.  Falls back to REST if the WebSocket handshake fails.  Dynamic
  quantity cap: BUY = `min(aq, usdt / (price × (1 + fee)))` to reserve the
  taker fee; SELL = `min(bq, btc)`.

## 7. BACKTESTING  (backtest/)
  Full design, pseudocode, data-flow diagrams, approximation caveats, and
  step-by-step implementation notes → **BACKTESTING.md**.
