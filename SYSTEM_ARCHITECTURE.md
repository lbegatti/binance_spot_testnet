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
    regime_director.py                — RegimeDirector: GaussianHMM regime detection on 1-min klines
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
    data.py                           — Historical kline downloader (180 days, 1 m BTCUSDT; ~259,200 candles)
    synthetic_book.py                 — Synthetic 50-level order book builder (per kline row)
    signals.py                        — Signal replay loop: full pipeline + regime & VWAP filters → signal DataFrame
    pnl.py                            — P&L simulation: balance guard, bps-based half_spread fill (BACKTEST_FILL_SPREAD_BPS), per-trade position cap (BACKTEST_MAX_POSITION_PCT), FIFO round-trip pairing, equity curve, Step 5 metrics
    runner.py                         — Top-level backtest runner: chains all modules, delegates report/CSV to reporting/
    regime_validation.py              — Offline long-horizon regime diagnostic (Step 6b): 70/30 train-test split on 1 year (~525k rows), self-contained BIC search + label assignment (no RegimeDirector), vectorised single Viterbi pass on ~157k test candles, six checks
    reporting/                        — Console report formatting and CSV export (AI-authored)
      __init__.py                     — Re-exports fmt, print_report, save_csv, print_regime_validation_report
      formatters.py                   — fmt(), print_report(), save_csv(), print_regime_validation_report(), HEAVY/LIGHT/PREVIEW constants

## 1. CONFIGURATION  (config_parameters.py)
  All constants are defined in one place and imported by every other module.

    SYMBOL               = "BTCUSDT" # trading pair used across all calls
    CCY                  = "USDT"    # quote currency
    CRYPTOCCY            = "BTC"     # base / cryptocurrency
    HISTORY_MAXLEN       = 3000    # deque cap (~5 min at 100 ms updates)
    N_LEVELS             = 50      # order book levels used in low_latency_analysis
    HFT_INTERVAL         = 1       # seconds between low-latency evaluations
    HIST_INTERVAL        = 60      # seconds between historical analyses (1 min)
    MIN_SNAPSHOTS        = 100     # warm-up guard for historical loop
    DEFAULT_SESSION_MINUTES = 10   # default session length (fixed, no prompt)
    HTF_JOIN_TIMEOUT     = 10      # shutdown wait for low-latency thread (s)
    HIST_JOIN_TIMEOUT    = 15      # shutdown wait for historical thread (s)
    RECV_WINDOW          = 5000    # Binance REST validity window (ms)
    SNAPSHOT_DEPTH       = 100     # levels in the seed REST snapshot
    WS_SPEED             = 100     # WebSocket diff-depth update interval (ms)
    # HMM regime detection
    HMM_FEATURE_COLS     = ["return","volatility","obi_proxy","trade_density"]
    HMM_N_ITERATIONS     = 1000   # max EM iterations per model fit
    HMM_MAX_REGIMES      = len(HMM_FEATURE_COLS) - 1  # = 3; upper bound on hidden states
                                  # (BIC search: 2…3 states); capped at n_features-1 to
                                  # avoid under-populated states with only 4 features
    HMM_RANDOM_STATE     = 46     # seed for reproducible HMM initialisation
    HMM_INTERVAL         = KLINE_INTERVAL_1MINUTE  # kline granularity (1 m)
    HMM_LOOKBACK         = "2 hours ago UTC"        # ~120 rows at 1 m
    HMM_MIN_COVAR        = 1e-1   # regularisation floor for covariance matrices
                                  # (1e-1 recommended safe default for z-scored financial features)
    HMM_N_INIT           = 10     # random-seed restarts per candidate n_components value
                                  # inside select_hmm_model(); reduces degenerate EM solutions
                                  # (e.g. transmat row summing to 0) on flat/low-variance windows
    HMM_TRAIN_ROWS       = 80     # in-sample rows for fit(); remaining ~40 rows are out-of-sample
                                  # scaler is fit_transform'd on [:HMM_TRAIN_ROWS], transform'd on full window
    HMM_MIN_CONFIDENCE   = 0.70  # min posterior probability (predict_proba) to allow an order;
                                  # below this threshold both BUY and SELL are skipped
    HMM_REFIT_INTERVAL   = 300    # full re-fit cadence (s); cheap Viterbi prediction between refits
    ORDER_REPORT_LIMIT   = 100    # max orders shown at head/tail of end-of-session report
    # Backtesting
    BACKTEST_LOOKBACK    = "180 days ago UTC"  # kline fetch window for backtest dataset (~259,200 candles;
                                               # 180-day window for broad regime coverage)
    VOLUME_DECAY_FACTOR  = 0.80   # exponential decay per synthetic order-book level
    HMM_LOOKBACK_ROWS    = 120    # warm-up window for backtest HMM (2 h at 1 m — matches HMM_LOOKBACK)
    VWAP_WINDOW          = 5      # rolling candle window for backtest VWAP (5 min at 1 m)
    REFIT_EVERY          = 120    # iterations between full HMM BIC re-fits (live: every 2 h; backtest: ~1,080 refits over 180 days)
    BACKTEST_MAX_ROWS    = None   # max replay candles (None = full run; set to e.g. 500 for fast debug)
    # Backtesting P&L (Step 4)
    BACKTEST_INITIAL_CAPITAL = 5_000.0   # starting USDT balance for the P&L simulation
    BACKTEST_INITIAL_BTC     = 0.0735    # starting BTC balance (set > 0 to simulate existing position)
    BACKTEST_FEE_RATE        = 0.001     # 0.10 % taker fee per side (Binance Spot standard)
    BACKTEST_RISK_FREE_RATE  = 0.0       # annualised risk-free rate for Sharpe/Sortino
                                         # 0.0 = no adjustment (standard in crypto);
                                         # set to e.g. 0.04 for a 4% T-bill proxy.
                                         # Converted to per-period rate automatically.
    BACKTEST_FILL_SPREAD_BPS = 5.0       # bps-based half-spread for fill simulation.
                                         # half_spread = close × BPS / 20 000.
                                         # Default 5 bps → ~$20 at $80 k BTC.
    BACKTEST_MAX_POSITION_PCT = 0.10     # max fraction of USDT risked per BUY (10 %).
    SENSITIVITY_REFIT_EVERY = 480        # HMM refit cadence for sensitivity.py ONLY.
                                         # 480 iterations = 8 h at 1 m → ~90 refits
                                         # per 90-day run (~4× faster than REFIT_EVERY=120).
                                         # config_parameters defaults unchanged.
    # NOTE: BACKTEST_FEE_RATE is also imported by execution/order_executor.py
    # for the live BUY quantity cap: usdt / (micro_price × (1 + BACKTEST_FEE_RATE)).
    # Binance charges the taker fee on top of the order notional at fill time;
    # the fee-adjusted divisor ensures the total debit never exceeds usdt,
    # preventing "insufficient balance" rejections on the testnet.

## 2. SHARED STATE  (order_book_state.py — OrderBookState)
  Single source of truth injected into both MessageHandler and AnalysisEngine.
  Two dedicated locks prevent contention between the high-frequency order-book
  path (100 ms ticks) and the lower-frequency balance-update path:

    local_book           : dict   — {"bids": {price: qty}, "asks": {price: qty},
                                      "lastUpdateId": int}
    history_order_book   : deque  — rolling window of snapshots (maxlen=3000)
                                    each entry (all numeric, no strings):
                                      {timestamp        : int   — Binance event time (Unix ms),
                                       lastUpdateId     : int   — monotonic update ID,
                                       best_bid         : float — highest bid price,
                                       best_ask         : float — lowest ask price,
                                       volume_best_bid  : float — quantity at best bid,
                                       volume_best_ask  : float — quantity at best ask}
    balance_status       : dict   — {"BTC": float, "USDT": float}
                                    free balances for the traded pair.
                                    Seeded from REST on startup; updated in
                                    real time by OrderExecutor._handle_balance_update
                                    via outboundAccountPosition push events on
                                    the WS API connection (session.logon →
                                    userDataStream.subscribe — no listenKey).
                                    Falls back to REST snapshot if the testnet
                                    WS does not support session.logon.
                                    Only the free (non-locked) quantity is stored.
    thread_lock          : Lock   — serializes local_book and history_order_book
                                    (MessageHandler writes; AnalysisEngine reads)
    thread_balance_lock  : Lock   — serializes balance_status independently of
                                    thread_lock to avoid unnecessary contention

## 3. DATA INGESTION & SYNCING  (message_handler.py — MessageHandler)
  Exposes one active WebSocket callback:

  ── handle_depth_message(_, message)  [ws_client — production diff-depth] ──

  a) SUBSCRIPTION CONFIRMATION — if the message has only "id"/"result" (no
     bid/ask fields) log and return immediately.
  b) STALE-UPDATE GUARD — discard any update whose "u" (final update ID) is
     ≤ local_book["lastUpdateId"]; ensures monotonic consistency.
  c) BID/ASK MERGE (inside thread_lock) —
       qty == 0 → pop the price level
       qty  > 0 → upsert the price level
     local_book["lastUpdateId"] is advanced to data["u"].
  d) SNAPSHOT APPEND — best bid price and best ask price (converted from
     string dict key to float), together with their respective quantities
     (float), event timestamp, and update ID are recorded into
     history_order_book as {timestamp, lastUpdateId, best_bid, best_ask,
     volume_best_bid, volume_best_ask}.
  e) QUOTE CALCULATION (throttled) — calculate_best_quote(local_book) is
     called every QUOTE_EVERY_N_TICKS ticks (default 10 ≈ once per second)
     rather than on every 100 ms tick.  Reduces CPU work and console noise
     while keeping the local book updated in real time.

  ── handle_balance_message(_, message)  [SUPERSEDED — preserved for ref.] ──

  Balance updates now flow through OrderExecutor._handle_balance_update on
  the same WebSocket API connection used for order placement (session.logon →
  userDataStream.subscribe → outboundAccountPosition push events).  The old
  listenKey / User Data Stream was discontinued by Binance in Feb 2026.
  This method is preserved for reference but is not wired to any stream.

  NOTE: Production WebSocket stream is used for market data (read-only, no
  auth) because the Binance Spot Testnet does not expose WS market-data
  streams.  Order execution and real-time balance events are both routed
  through the testnet WebSocket API (wss://testnet.binance.vision/ws-api/v3)
  via OrderExecutor.

## 4. ANALYSIS ENGINE  (analysis.py — AnalysisEngine)
  Receives the same OrderBookState instance as MessageHandler.
  Checks stop_event on every iteration; exits gracefully when it is set.

  Owns five private attributes for cross-thread sharing:
    _vwap_lock      : Lock          — serializes _bid_vwap and _ask_vwap
    _bid_vwap       : float | None  — latest bid VWAP from historical_analysis
    _ask_vwap       : float | None  — latest ask VWAP from historical_analysis
    _regime_lock    : Lock          — serializes regime_director.regime_label
    regime_director : RegimeDirector — injected pre-fitted HMM detector;
                                       updated every HIST_INTERVAL seconds via
                                       a two-speed scheme: cheap Viterbi prediction
                                       on most iterations, full re-fit every
                                       HMM_REFIT_INTERVAL (300 s).
  _bid_vwap and _ask_vwap start as None (no historical data yet).
  regime_director.regime_label is set BEFORE threads start (initial fit in
  websocket_main.py step 4b) so the very first low_latency iteration has a
  valid regime to read.

  ▸ low_latency_analysis()  [Low-latency thread — every HFT_INTERVAL seconds = 1 s]
      • Reads balance_status under thread_balance_lock at the top of every
        iteration.  If both usdt_balance < 10.0 AND btc_balance < 0.0001,
        sleeps HFT_INTERVAL and continues (no spin; cadence is preserved).
      • Acquires thread_lock → copies bids & asks → releases lock.
      • Builds order book levels (top N_LEVELS = 50) and computes per-level
        metrics: total_depth, mid_price, micro_price, OBI, bq, aq.
        Level tuple: (total_depth, mid_price, micro_price, obi, bq, aq).
      • Identifies candidates using strategy logic (thin book check, micro-mid
        delta direction).
        Candidate tuple: (level_idx, delta, total_depth, obi, micro_price, bq, aq).
      • Scores candidates (70 % depth / 30 % delta) and picks the best
        opportunity for buy and sell sides.
        Opportunity tuple: (level_idx, score|None, delta, depth, obi, micro_price, bq, aq).
      • VWAP FILTER (mean-reversion / dip-and-strength confirmation) — reads _bid_vwap
        (and _ask_vwap, retained for diagnostics) under _vwap_lock:
          BUY:  execute only if _bid_vwap is None OR micro_price < bid_vwap.
                Interpretation: genuine dip — current price has fallen BELOW
                the historical avg bid, confirming a buy-the-dip signal.
                Skip with log if micro_price >= bid_vwap (price at/above average
                — not a genuine dip).
          SELL: execute only if _bid_vwap is None OR micro_price >= bid_vwap.
                Interpretation: genuine strength — current price is AT or ABOVE
                the historical avg bid, confirming a sell-into-strength signal.
                Skip with log if micro_price < bid_vwap (price below average
                — not genuine strength to sell into).
        Both comparisons use _bid_vwap; _ask_vwap is still computed and stored
        in state but is not used by the gate logic (retained for diagnostics).
        During the first ~1 min (before historical_analysis runs) both VWAPs
        are None, so the filter is transparent and orders execute solely on
        regime + score.
        NOTE: To switch back to a momentum strategy, reverse the conditions:
          BUY if micro_price > ask_vwap; SELL if micro_price < bid_vwap.
      • REGIME FILTER (two sequential gates) — reads regime_label AND
        regime_confidence under _regime_lock:

        GATE 1 — CONFIDENCE (evaluated first):
          If regime_confidence < HMM_MIN_CONFIDENCE (0.70):
            Skip BOTH BUY and SELL — model is uncertain (e.g. 55/45 split).
          If regime_confidence is None (before first historical run): transparent.

        GATE 2 — DIRECTION (evaluated only if Gate 1 passed):
          BUY:  suppressed if regime == "trending_down" or "high_volatility".
          SELL: suppressed if regime == "trending_up"   or "high_volatility".
          None (before first historical run) → transparent, all orders allowed.

        Gate 1 is evaluated BEFORE Gate 2; if confidence is too low, the
        direction label is irrelevant.  VWAP check is evaluated AFTER both
        regime gates pass.
      • Delegates to OrderExecutor.execute("BUY"|"SELL", opportunity).
        The executor validates strategy, checks balances, computes quantity
        (aq for BUY, bq for SELL), and sends a LIMIT GTC order via its own
        SpotWebsocketAPIClient.  Response handled async in handle_order_response.

  ▸ historical_analysis()  [HIST thread — every HIST_INTERVAL seconds = 60 s / 1 min]
      • Sleeps HIST_INTERVAL first (self.stop_event.wait(HIST_INTERVAL)).
      • Acquires thread_lock → copies history_order_book to a plain list →
        releases lock.  All computation happens outside the critical section.
      • Skips iteration if fewer than MIN_SNAPSHOTS (100) have accumulated.
      • VWAP computation — converts snapshot list to numpy arrays and computes:
          bid_vwap = volume_weighted_average_price(bids, vols_bids)
          ask_vwap = volume_weighted_average_price(asks, vols_asks)
        using the helper from indicators.py.
        Publishes bid_vwap and ask_vwap under _vwap_lock.
      • HMM REGIME UPDATE (two-speed — outside _regime_lock for the slow work):
          Every iteration (60 s):
            regime_director.get_klines_data()            — fetches last 2 h of 1-min klines,
                                                           computes return, volatility,
                                                           obi_proxy, trade_density.
          If iteration % refit_every == 0 (every 5 min / 300 s):
            regime_director.select_hmm_model()           — fits GaussianHMM for n=2..4 on
                                                           first HMM_TRAIN_ROWS (scaled) rows,
                                                           selects the best BIC model; runs
                                                           predict() + predict_proba() on the
                                                           full scaled window; sets
                                                           current_regime + regime_confidence.
          Otherwise (cheap path):
            regime_director.predict_current_regime()     — reuses existing model + scaler;
                                                           single Viterbi pass (predict) and
                                                           predict_proba() on full window;
                                                           updates current_regime + regime_confidence.
                                                           ~1000× cheaper than select_hmm_model.
        (inside _regime_lock — fast label assignment only):
          regime_director.assign_regime_labels() — maps state int → label string,
                                                    writes regime_label.

  ▸ Deque fill-up math (at WS_SPEED = 100 ms → ~10 entries/sec):

      │ Time elapsed │ WS ticks  │ Deque size          │ Historical iterations │
      │──────────────│───────────│─────────────────────│───────────────────────│
      │ 1 min        │ ~600      │ 600                 │ 1st runs              │
      │ 3 min        │ ~1 800    │ 1 800               │ 3rd runs              │
      │ 5 min        │ ~3 000    │ 3 000 (full)        │ 5th runs              │
      │ 10 min       │ ~6 000    │ 3 000 (capped)      │ 10th runs             │
      │ 20 min       │ ~12 000   │ 3 000 (capped)      │ 20th runs             │

      After ~5 min the deque is full and becomes a true rolling window.

  ▸ Thread timeline (default 20-min session):

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
                historical iteration #5 → deque now full, true rolling window
      ...
      t=20min   low_latency iteration #1200
                historical iteration #20 → refreshes VWAP one last time
                stop_event set → both threads exit

4b. REGIME DETECTION  (strategy/regime_director.py — RegimeDirector)
  Trains a Gaussian Hidden Markov Model on recent Binance 1-minute klines
  to classify the current market into a hidden regime.  The result is exposed
  as a plain string label AND a posterior confidence float consumed by
  AnalysisEngine.low_latency_analysis.

  FEATURES (computed from OHLCV kline data):
    return        = close.pct_change()
    volatility    = (high - low) / close
    obi_proxy     = (taker_buy_base_vol / volume) × 2 - 1   ∈ [-1, +1]
                    approximates live OBI from kline taker flow
    trade_density = num_trades / volume
                    high → many small trades (retail / HFT fragmentation)
                    low  → few large trades  (institutional blocks)

  FEATURE SCALING (StandardScaler):
    Before any fit() / predict() / predict_proba() call:
      scaler = StandardScaler()
      train_features = features[:HMM_TRAIN_ROWS]      (oldest ~⅔ of the 2-h window)
      train_scaled   = scaler.fit_transform(train_features)   (fit on in-sample only)
      full_scaled    = scaler.transform(features)             (applied to entire window)
    The scaler is stored as self.scaler and reused in predict_current_regime()
    so the distribution seen at training time is exactly reproduced at inference.
    Without scaling, trade_density can dominate the covariance structure.

  TRAIN / PREDICT SPLIT:
    Model is fitted on train_scaled (first HMM_TRAIN_ROWS rows).
    Regime is inferred on full_scaled (all rows including held-out recent ones).
    self.current_regime = regimes[-1]  (last candle — genuinely out-of-sample)

  MODEL SELECTION (BIC):
    GaussianHMM fitted for n = 2 … HMM_MAX_REGIMES (3) states.
    Best model selected by lowest BIC = -2 ln L̂ + k ln N  (N = HMM_TRAIN_ROWS).
    current_regime set to state index of the last candle.

  REGIME CONFIDENCE (predict_proba):
    After predict(), predict_proba(full_scaled) is also called:
      proba             = model.predict_proba(full_scaled)   # shape (n_rows, n_states)
      regime_confidence = float(proba[-1, current_regime])   # ∈ [0, 1]
    Stored as self.regime_confidence; published under _regime_lock alongside
    regime_label so low_latency_analysis can read both atomically.

  LABEL ASSIGNMENT (assign_regime_labels — rank-based + threshold secondary):
    Directional labels (guaranteed unique via rank):
      direction_score = return.rank() + obi_proxy.rank()
      best_state  (idxmax) → "trending_up"   — most bullish
      worst_state (idxmin) → "trending_down"  — most bearish

    Secondary labels for remaining states:
      high_vol : vol > mean_vol + 1.0 × std_vol        (large price swings)
      high_td  : td  > mean_td  + 0.5 × std_td         (trade fragmentation)
      high_vol OR high_td → "high_volatility"
        Both independently indicate an unreliable market: large swings
        (unpredictable fills) or fragmented activity (noisy, no directional
        intent).
      (default) → "neutral"

    After labelling:
      self.state_labels : dict[int, str]   — full {state_index → label} mapping
        stored as a new attribute on RegimeDirector (populated here, never in
        __init__).  Exposed so external tools (e.g. regime_validation.py
        vectorised Phase 2) can map state sequences to labels without
        re-running the full label-assignment logic.
  REGIME GATES in low_latency_analysis (applied sequentially):
    GATE 1 — CONFIDENCE:
      Skip both BUY and SELL if regime_confidence < HMM_MIN_CONFIDENCE (0.70).
      None (warm-up) → transparent.
    GATE 2 — DIRECTION (only if Gate 1 passed):
      BUY  suppressed if regime ∈ {"trending_down", "high_volatility"}
      SELL suppressed if regime ∈ {"trending_up",   "high_volatility"}
      None (before first historical run) → transparent

  LIFECYCLE (two-speed update):
    Instantiated in websocket_main.py (step 4b) before threads start.
    __init__ uses None sentinels for lookback and max_regimes (not default
    parameter values) so that load_best_params() patches applied to the
    strategy.regime_director module namespace are read at call time, not
    frozen at import time.  Full explanation: strategy/param_loader.py.
    Initial fit: get_klines_data() + select_hmm_model() + assign_regime_labels().

    Every HIST_INTERVAL (60 s) in historical_analysis():
      get_klines_data()           — refresh features (outside _regime_lock)
      predict_current_regime()    — cheap Viterbi + predict_proba (outside _regime_lock)
      assign_regime_labels()      — fast label write (inside _regime_lock)

    Every HMM_REFIT_INTERVAL (300 s) — replaces predict with full refit:
      get_klines_data()           — refresh features (outside _regime_lock)
      select_hmm_model()          — full EM refit + new scaler (outside _regime_lock)
      assign_regime_labels()      — fast label write (inside _regime_lock)

## 5. SESSION DRIVER  (websocket_main.py)
  Orchestrates the full lifecycle:

  Step 1 — Load .env (API key / secret), instantiate testnet REST client.
             NOTE: No listenKey is requested.  Balance tracking uses the
             session.logon + userDataStream.subscribe flow on the OrderExecutor
             WS API connection instead.
  Step 2 — Fetch & consolidate account balance:
               • Sell every non-USDT / non-BTC asset that has a live USDT pair
                 (single pass, each symbol attempted once).
               • Re-fetch balances if any sale succeeded.
               • Guard: raise if both USDT and BTC balances are zero.
               • Snapshot btc_start_price = ticker_price(SYMBOL) and compute
                 start_total_usdt = usdt_balance + btc_balance × btc_start_price
                 — stored for the end-of-session P&L decomposition.
  Step 3 — Session duration fixed at DEFAULT_SESSION_MINUTES (10 min).
               • 10 min → ~600 low-latency iterations, ~10 historical analyses.
  Step 4 — Seed local_book from REST snapshot (depth=100).
            Seed state.balance_status from the REST-fetched balances before
            any thread starts (single-threaded at this point; no lock needed).
  Step 4b— Pre-session HMM regime detection (RegimeDirector):
              • regime_director.get_klines_data()     — downloads ~120 rows of
                1-min klines (last 2 h, public endpoint, no auth).
              • regime_director.select_hmm_model()    — fits GaussianHMM for
                n=2..4, selects best BIC model, sets current_regime.
              • regime_director.assign_regime_labels()— assigns regime_label.
            regime_label is populated BEFORE threads start so the first
            low_latency_analysis iteration never reads None.
  Step 5 — Instantiate OrderBookState, MessageHandler, OrderExecutor,
            AnalysisEngine (with regime_director injected).
            OrderExecutor creates its own SpotWebsocketAPIClient internally
            (wss://testnet.binance.vision/ws-api/v3) with:
              on_message = self.handle_order_response
              on_open    = self._on_ws_open
            On socket open: _on_ws_open sends session.logon (HMAC-signed).
            On logon success: sends userDataStream.subscribe.
            On subscribe success: outboundAccountPosition push events arrive
            on the same connection → _handle_balance_update keeps
            state.balance_status current.
  Step 6 — Open one SpotWebsocketStreamClient (ws_client, production endpoint)
            and subscribe: diff_book_depth(SYMBOL, speed=WS_SPEED).
            Callback: handle_depth_message.
            Sleep 1 s so the first diff-depth messages populate
            local_book["bids"] before the low-latency thread wakes up.
  Step 7 — Start threads AFTER WebSocket is open and warmed up:
              • start low_latency_thread  (daemon=True, name="low-latency-analysis")
              • start hist_thread         (daemon=True, name="hist-analysis")
  Step 8 — Main thread blocks for session_seconds (interruptible by Ctrl-C).
  Step 9 — Shutdown:
               • stop_event.set()
               • ws_client.stop()
               • executor.stop()  — closes WS API connection (orders + user data)
               • low_latency_thread.join(timeout=HTF_JOIN_TIMEOUT)
               • hist_thread.join(timeout=HIST_JOIN_TIMEOUT)
  Step 10— End-of-session reports:
               Order report — queries every placed order via REST (GET /api/v3/order);
               logs fill status (FILLED / PARTIAL / OPEN / other).
               Balance report — fetches final balances + btc_end_price, then prints
               a three-line P&L decomposition:
                 A  Trading alpha  = Δusdt + Δbtc × end_price
                    (strategy contribution — isolated from BTC price movement)
                 B  Price move     = btc_start × (end_price − start_price)
                    (gain/loss on the starting BTC position due to market movement)
                 A + B  Total P&L  = end_total − start_total  (%  return on start portfolio)

## 6. EXECUTION  (execution/order_executor.py — OrderExecutor)
  Places LIMIT GTC orders AND maintains real-time balance updates via a
  single Binance WebSocket API connection, avoiding the per-request HTTP
  overhead of the REST API and eliminating the need for a listenKey.

  ▸ __init__(state, stream_url, api_key, api_secret, rest_client=None)
      • Creates SpotWebsocketAPIClient internally with:
          on_message = self.handle_order_response
          on_open    = self._on_ws_open
      • If the WS handshake fails, degrades to REST-only mode silently.
      • Tracks _logon_id and _subscribe_id for response routing.

  ▸ _on_ws_open(_ws)  [on_open callback]
      • Fires when socket connects → calls _send_session_logon().

  ▸ _send_session_logon()
      • Builds a signed session.logon frame using websocket_api_signature
        (HMAC-SHA256 on empty params + timestamp + apiKey).
      • Stores id in _logon_id for response matching.

  ▸ _send_user_data_subscribe()
      • Dispatches userDataStream.subscribe after successful logon.
      • Stores id in _subscribe_id for response matching.

  ▸ _handle_balance_update(data)
      • Called when a push event with "e" == "outboundAccountPosition"
        arrives (no "id" field — not a request/response pair).
      • Under thread_balance_lock, updates balance_status for tracked
        assets (CRYPTOCCY, CCY) using the "f" (free) field.

  ▸ handle_order_response(_, message)  [single on_message callback]
      Routes ALL incoming WS frames:
        • No "id"             → push event → _handle_balance_update
                                (or log debug for other event types)
        • id == _logon_id     → session.logon response → on success,
                                call _send_user_data_subscribe()
        • id == _subscribe_id → userDataStream.subscribe response →
                                set _user_data_active = True
        • any other id        → order placement response:
                                error → log; success → store in last_order
                                and append to placed_orders.

  ▸ execute(strategy, opportunity) → None  [fire-and-forget]
      1) STRATEGY VALIDATION — rejects anything other than "BUY" or "SELL".
      2) TUPLE UNPACKING — 8-element opportunity tuple:
           (level_idx, score|None, delta, total_depth, obi, micro_price, bq, aq)
      3) BALANCE READ — acquires thread_balance_lock, reads usdt and btc.
      4) QUANTITY — aq for BUY (ask-side liquidity), bq for SELL (bid-side).
       5) DYNAMIC QUANTITY CAP —
            BUY:  quantity = min(aq, usdt / (micro_price × (1 + BACKTEST_FEE_RATE))).
                  Dividing by price × (1 + fee) reserves the taker fee so the total
                  debit (notional + fee) never exceeds the available USDT balance.
                  Without the fee adjustment, Binance charges the fee on top of the
                  notional and rejects the order with "insufficient balance" when the
                  full balance is committed.
            SELL: quantity = min(bq, btc).
           If the capped quantity is > 0 but less than the requested amount,
           a logging.info message is emitted and the order proceeds at reduced
           size.  Only skipped (returns None) when quantity == 0 — i.e. the
           balance is fully depleted for that direction.
      6) ORDER SUBMISSION —
           WS path (preferred): ws_api_client.new_order(...)
           REST path (fallback): rest_client.new_order(...)

  ▸ stop()
      • Delegates to ws_api_client.stop(); server auto-unsubscribes
        userDataStream on disconnect.

  ▸ order_status_report()
      • Queries final status of placed_orders via REST.
      • To avoid flooding the console during long sessions, only the first
        and last ORDER_REPORT_LIMIT (100) orders are printed individually.
        When total > 2 × ORDER_REPORT_LIMIT the middle block is collapsed
        to a single summary line.
      • Each order is queried and logged by _query_and_log_order() (private
        class method).
      • Logs formatted summary (FILLED / PARTIAL / OPEN / other).

## 7. BACKTESTING  (backtest/)
  Offline replay of the live strategy on 180 days of historical 1-min klines
  (~259,200 candles).  Full design, pseudo-code, data-flow diagrams, and caveats → BACKTESTING.md.

  backtest/data.py            — fetch_klines(): downloads ~259,200 candles via
                                Client.get_historical_klines() (public, no auth).
  backtest/synthetic_book.py  — build_synthetic_book(row): per-kline synthetic
                                50-level order book (spread reconstruction +
                                exponential volume decay + OBI asymmetry injection).
                                VECTORISED: price/qty arrays computed with numpy
                                (_DECAY_FACTORS, _LEVEL_IDX pre-computed at import);
                                no Python for-loop over 50 levels.
                                Returns {"bids": {…}, "asks": {…},
                                         "_best_bid": float, "_best_ask": float,
                                         "_vol_best_bid": float, "_vol_best_ask": float}
                                The private _best_* keys allow callers to skip the
                                O(N_LEVELS) max/min key-scan, matching state.local_book
                                format → feeds _build_levels() directly.

  Three data flows per candle (mirroring the live architecture):
    Flow A — Opportunity scoring  : synthetic book → build_levels →
             collect_candidates → select_best_opportunity
             (all three via strategy/book_utils.py — the same public
             functions that AnalysisEngine._build_levels / _collect_candidates
             / _select_best_opportunity delegate to)
    Flow B — Regime filter        : rolling 120-candle kline window →
             RegimeDirector (HMM)
    Flow C — VWAP dip/strength filter : rolling 5-candle best-bid/ask → bid_vwap

  backtest/signals.py             — run_signals(): orchestrates all three flows
                                    per candle, applies the combined gate
                                    (regime + VWAP), and returns a time-indexed
                                    DataFrame with signal (+1/−1/0), regime
                                    label, bid/ask VWAP, micro-price, half_spread,
                                    and buy_qty/sell_qty details for P&L sizing.
                                    PERFORMANCE: feature columns pre-extracted as
                                    numpy arrays before the loop (avoids 43 k
                                    pandas iloc calls); _best_bid/_best_ask read
                                    directly from synthetic_book dict (no O(50)
                                    key scan).
                                    Optional keyword overrides (all default None →
                                    fall back to config constants):
                                      hmm_lookback_rows, hmm_max_regimes,
                                      vwap_window, refit_every, lookback,
                                      predict_every.
                                    predict_every: between refit calls, Viterbi is
                                    run only every N candles; last regime reused
                                    otherwise (sensitivity uses N=5 for ~5× speedup;
                                    run_backtest uses N=1, every candle).

  backtest/pnl.py                — simulate_pnl(): walks the signal DataFrame
                                    candle by candle using itertuples() (~5× faster
                                    than iterrows()), executing BUY/SELL trades
                                    with a balance guard identical to the live
                                    OrderExecutor.  Fill price per trade:
                                      BUY  = close + half_spread  (synthetic ask)
                                      SELL = close - half_spread  (synthetic bid)
                                    half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20 000
                                    (default 5 bps → ~$20 at $80 k BTC).
                                    NOT (high-low)/2 — the candle range is
                                    10-100× the real spread and causes 100% drawdown.
                                    Per-trade position cap: BACKTEST_MAX_POSITION_PCT
                                    (default 10 % of available USDT per BUY).
                                    Marks the portfolio to market at every candle
                                    (including HOLDs) to produce a continuous
                                    equity curve.  Outputs: trades_df, equity_df,
                                    and a stats dict with Step 5 metrics (total
                                    return, win rate, max drawdown, Sharpe,
                                    Sortino, profit factor, avg holding period,
                                    regime/VWAP filter hit rates).
                                    Sharpe and Sortino use excess returns over
                                    BACKTEST_RISK_FREE_RATE (default 0.0) and
                                    adaptive annualisation: √365 daily, √8760
                                    hourly, or √105120 5-min depending on window
                                    length.  Total return denominator includes
                                    initial BTC valued at first-candle close.
                                    _pair_round_trips(): exhaustive FIFO BUY→SELL
                                    matching via collections.deque — supports
                                    scaling-in, layering, and pyramiding (multiple
                                    concurrent open legs).  Partial closes push
                                    the remaining qty back to the front of the
                                    queue (appendleft); over-sells consume as many
                                    legs as the SELL qty allows.  Orphan SELLs
                                    (no open BUY leg — only possible when
                                    initial_btc > 0) emit a visible WARNING box.
                                    NOTE: open_buys deque is a pure accounting
                                    cursor — all trades in trades_df are already
                                    settled by simulate_pnl() before pairing runs.

  backtest/runner.py             — run_backtest(): top-level orchestrator.
                                    Chains run_signals() → simulate_pnl() →
                                    print_report() → (opt-in) plot_backtest().
                                    Parameters:
                                      export_csv (bool, default False) — saves
                                        trades_*.csv + equity_*.csv.
                                      plot (bool, default False) — generates the
                                        Step 7 Plotly figure; imported lazily so
                                        headless runs carry zero overhead.
                                      save_png (bool, default False) — persists
                                        figure as PNG (kaleido) or HTML fallback.
                                    Report formatting and CSV export delegated to
                                    backtest/reporting/formatters.py (public
                                    helpers: fmt, print_report, save_csv).
                                    Entry point: python -m backtest.runner.

  backtest/reporting/            — Console report formatting and CSV export.
    __init__.py                  — Re-exports fmt, print_report, save_csv,
                                   print_regime_validation_report.
    formatters.py                — print_report(): structured console report
                                    (SESSION, SIGNALS, P&L SUMMARY, RISK
                                    METRICS, TRADE LOG PREVIEW).
                                    print_regime_validation_report(): formats
                                    the Step 6b regime validation output
                                    (per-regime stats, direction test, Welch's
                                    t-test, volatility check, confidence
                                    floor, label frequency, hit-rate
                                    alignment).
                                    print_sensitivity_table(): per-run summary
                                    table for all sensitivity modes.
                                    print_oat_sensitivity_report(): OAT
                                    delta-Sharpe report per parameter.
                                    print_bnh_comparison(): strategy vs
                                    buy-and-hold comparison box.
                                    save_csv(): timestamped trades_*.csv and
                                    equity_*.csv to backtest/results/.
                                    fmt(): safe float formatter (NaN / inf safe).
                                    NOTE: this module was written entirely by AI
                                    (GitHub Copilot / GPT-4o) — see README.md
                                    disclaimer.

  backtest/diagnostics/
    regime_validation.py           — Standalone offline long-horizon regime
                                     diagnostic.  Entry point:
                                     python -m backtest.diagnostics.regime_validation.
                                     Fetches 1 year of 1 m klines (~525,000 rows,
                                     VALIDATION_LOOKBACK = "365 days ago UTC"),
                                     applies a 70/30 train-test split at runtime
                                     (split_idx = int(len(df) * 0.70), ~367,500 /
                                     ~157,500 rows).  SELF-CONTAINED: does not use
                                     RegimeDirector — replicates BIC search and
                                     label-assignment directly with raw GaussianHMM
                                     + StandardScaler so the training window is not
                                     capped at HMM_TRAIN_ROWS (80 rows).
                                     Phase 2 scores ~157,500 test candles in a
                                     SINGLE VECTORISED VITERBI PASS:
                                       • scaler.transform(test_features) — 1 call
                                       • model.predict(test_scaled) — 1 Viterbi
                                       • proba[np.arange(n), states] — confidence
                                       • label_array[states] — NumPy lookup
                                     Six checks (Phase 3):
                                       1. Direction test (return ordering)
                                       2. Welch's t-test (p < 0.05)
                                       3. Volatility check
                                       4. Confidence floor (≥ HMM_MIN_CONFIDENCE)
                                       5. Label frequency (no regime < 1 %)
                                       6. Hit-rate alignment (informational)
                                     BACKTEST_MAX_ROWS intentionally bypassed.
                                     Report printed by
                                     reporting/formatters.print_regime_validation_report().
                                     Use VALIDATION_LOOKBACK = "90 days ago UTC"
                                     for a faster smoke-test (~2 min fetch).

  backtest/visualization.py     — Step 7: plot_backtest() generates an
                                    interactive six-panel Plotly figure from the
                                    four artefacts returned by run_backtest().
                                    Panels (all rows 1-5 share a synchronised
                                    datetime x-axis):
                                      Row 1a: equity curve + initial reference.
                                      Row 1b: drawdown % (red tozeroy fill).
                                      Row 2:  BTC close + BUY ▲ / SELL ▼ markers
                                              at fill price + bid/ask VWAP lines.
                                       Row 3:  HMM regime timeline — three layers:
                                               (a) colour-coded vrect bands per
                                                   contiguous regime run; (b) dark-
                                                   slate shape="hv" step-line via
                                                   _REGIME_NUMERIC mapping
                                                   (0=trending_down → 3=trending_up)
                                                   so transitions are vertical jumps;
                                               (c) dotted navy confidence overlay
                                                   scaled ×3 (fills [0,3] range);
                                                   hover shows real 0–1 value.
                                                   HMM_MIN_CONFIDENCE×3 dashed line.
                                               Y-axis ticks show regime names.
                                               Bug fixed 2026-04-25: previously only
                                               regime_confidence (0–1) was plotted,
                                               appearing as a flat line.
                                      Row 4:  VWAP vs micro-price + grey dot
                                              near-miss markers where VWAP gate
                                              specifically blocked a candidate.
                                      Row 5a: stacked horizontal funnel bar
                                              (executed / conf-blocked /
                                              regime-blocked / VWAP-blocked).
                                      Row 5b: stacked vertical bar — BUY/SELL/HOLD
                                              count per HMM regime label.
                                    Opt-in: run_backtest(plot=True, save_png=True).
                                    save_png uses kaleido; falls back to HTML.
                                    Regime bands drawn with add_vrect per
                                    transition only — O(transitions), not O(rows).

  backtest/sensitivity.py       — Step 8: Sensitivity analysis (Use Case A).
                                     DEFAULT mode: Bayesian optimisation via
                                     Optuna TPE sampler (30 trials).  Also
                                     supports --oat (7 runs) and deprecated
                                     --full-grid (18 combos).
                                     Tunes HMM_LOOKBACK_ROWS, HMM_MAX_REGIMES,
                                     VWAP_WINDOW over a continuous search space
                                     (_OPTUNA_SPACE).  fee_rate fixed at 0.001
                                     in ALL modes (BACKTEST_FEE_RATE — not a
                                     strategy knob, not in the param grid).
                                     vwap_threshold fixed at
                                     VWAP_THRESHOLD_MULTIPLIER in OAT/full-grid
                                     (also not in the param grid).
                                     Overrides passed as keyword args to
                                     run_signals() / simulate_pnl() — no changes
                                     to config_parameters.py or the live system.
                                    Three sensitivity-only speed-up constants
                                    (all in config_parameters.py, no effect on
                                    runner.py or the live system):
                                      SENSITIVITY_REFIT_EVERY = 480  (~4× fewer refits)
                                      SENSITIVITY_LOOKBACK = "90 days ago UTC"
                                                             (~2× fewer rows than 180d)
                                      SENSITIVITY_PREDICT_EVERY = 5  (~5× fewer Viterbi)
                                    --lookback flag overrides SENSITIVITY_LOOKBACK
                                    per run (no config change needed).
                                    _check_existing_best_params() guard: reads
                                    best_params.json age/Sharpe/params and
                                    prompts [y/N] before every run (all modes).
                                     Bayesian mode:
                                       Data pre-fetched once; _make_objective()
                                       factory closure shares it across all trials.
                                       Study persisted to results/optuna.db
                                       (load_if_exists=True → auto-resume).
                                       HTML charts (always saved after every Bayesian
                                       study → written to reporting/ not results/):
                                          reporting/optuna_history_<ts>.html
                                          reporting/optuna_importance_<ts>.html
                                          reporting/optuna_contour_<ts>.html
                                       ~50–100 min for 30 trials (90-day window).
                                     Console report formatting delegated to
                                     reporting/formatters.py:
                                       print_sensitivity_table()    ← all modes
                                       print_oat_sensitivity_report() ← OAT only
                                       print_bnh_comparison()       ← all modes
                                      Default (Bayes): python -m backtest.sensitivity
                                      OAT (Phase 1):  python -m backtest.sensitivity --oat
                                      Full-grid (dep): python -m backtest.sensitivity --full-grid
                                      Writes:
                                        reporting/sensitivity_<mode>_<ts>.csv
                                            ← human-readable (alongside HTML charts)
                                        results/best_params.json
                                            ← machine-readable; loaded by live system
                                            └── consumed by strategy/param_loader.py
                                                   ├── load_best_params() ← websocket_main.py
                                                   │     called at startup BEFORE RegimeDirector();
                                                   │     patches strategy.regime_director namespace
                                                   │     (HMM_MAX_REGIMES, HMM_LOOKBACK).
                                                   │     Patches are visible because RegimeDirector
                                                   │     __init__ uses None sentinels — see 4b.
                                                   │     Falls back to defaults if absent.
                                                  └── load_best_params_for_backtest() ← runner.py
                                                        called at top of run_backtest() before
                                                        run_signals(); returns dict of cast values
                                                        passed as kwargs (hmm_lookback_rows,
                                                        hmm_max_regimes, vwap_window, fee_rate);
                                                        falls back to defaults if absent.
                                        results/optuna.db  ← Bayes mode only (SQLite study)
                                     Use Case B (180-day window) DEFERRED —
                                     ~4–12 h OAT runtime on a laptop.

  Implementation status: data.py ✅, synthetic_book.py ✅, signals.py ✅,
                         pnl.py ✅, runner.py ✅, reporting/ ✅,
                         regime_validation.py ✅, visualization.py ✅,
                         sensitivity.py ✅ (Use Case A; Use Case B deferred),
                         param_loader.py ✅ (load_best_params + load_best_params_for_backtest).