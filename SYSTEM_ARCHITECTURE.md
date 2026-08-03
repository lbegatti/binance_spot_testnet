# SYSTEM ARCHITECTURE: MULTI-TIMEFRAME ORDER BOOK STRATEGY

-----------------------------------------------------------------------------
SYMBOL    : BTCUSDT  (market data: production stream + production REST snapshots,
                      keyless — trading/account: authenticated testnet REST/WS API)
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
    analysis.py                       — AnalysisEngine: low-latency loop + historical loop (VWAP + HMM regime filter + trend-pause gate + adaptive stop-loss + macro-trend overlay)
    book_utils.py                     — Shared order-book utilities (build_levels, collect_candidates, select_best_opportunity) — NumPy-vectorised; shape-mismatch guard
    regime_director.py                — RegimeDirector: GaussianHMM regime detection on 5-min klines (HMM_INTERVAL="5m", HMM_LOOKBACK="10 hours ago UTC", 120 bars)
    best_quote_calculator.py          — Live spread printer (best bid | best ask on every tick)
    metrics.py                        — Order book depth metrics
    indicators.py                     — Technical indicators (trend confirmation, VWAP helper, etc.)
    scores.py                         — Normalised opportunity scores
    quotes.py                         — find_best_quote(): best bid/ask selection helpers

  execution/
    order_executor.py                 — OrderExecutor: LIMIT GTC (BUY) / IOC (SELL) orders via WebSocket API + balance refresh + 10-second stale-BUY cancel

  visualization/
    plot_helpers.py                   — Charting utilities for the REST snapshot path
    session_chart.py                  — End-of-session P&L chart (Plotly HTML): Strategy vs B&H equity index + filled/unfilled BUY/SELL markers (solid = traded, hollow green ▲ / red ▼ = cancelled / never matched) + USDT/BTC component panel; written to backtest/reporting/session_pnl_<ts>.html by websocket_main.py

  backtest/                           — Offline backtesting framework (see BACKTESTING.md)
    data.py                           — Historical kline downloader: fetch_macro_klines() (5m, HMM) + fetch_micro_klines() (1m, PnL); Parquet cache (cache/klines/, 24h TTL); --flush-cache flag
    synthetic_book.py                 — Synthetic 50-level order book builder (per kline row)
    signals.py                        — Two-frame signal pipeline: Phase 1 HMM walk-forward on 5m + trend_pause flag + adaptive stop_loss_pct + macro_trend state; Phase 2 merge_asof stitch; Phase 3 1m execution loop + regime/VWAP/macro-trend gates
    pnl.py                            — P&L simulation: adaptive stop-loss, macro-trend force-to-cash, trend-pause gate, balance guard, bps-based fill, intra-candle whipsaw guard, position cap, FIFO round-trip pairing, equity curve, Step 5 metrics
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
    balance_locked       : dict   — {"BTC": float, "USDT": float}
                                      locked balances (tied up in resting LIMIT
                                      orders); updated by the same two paths as
                                      balance_status. Read ONLY by the equity
                                      snapshot so a position resting in a LIMIT
                                      order still counts; trading reads free only.
    thread_lock          : Lock   — serialises local_book + history_order_book
                                      (MessageHandler writes; AnalysisEngine reads)
    thread_balance_lock  : Lock   — serialises balance_status independently of
                                      thread_lock to avoid unnecessary contention

## 3. DATA INGESTION  (message_handler.py — MessageHandler)
  Merges incoming diff-depth WebSocket ticks into `local_book` under
  `thread_lock` (zero-qty levels are deleted; positive-qty levels are upserted),
  appends a best-bid/ask snapshot to `history_order_book` on every tick, and
  calls `calculate_best_quote()` every `QUOTE_EVERY_N_TICKS` (~1 s) to print
  the live spread.  A stale-update guard discards any tick whose final update
  ID `u` is ≤ `local_book["lastUpdateId"]`.

  **Gap recovery / resync.**  Each event also carries a *first*
  update ID `U`.  A healthy stream is contiguous (`U == lastUpdateId + 1`); if
  `U > lastUpdateId + 1` the handler has missed one or more events (network
  drop or reconnect) and `local_book` is no longer trustworthy.  It then
  re-pulls a fresh `depth(limit=SNAPSHOT_DEPTH)` snapshot via the injected REST
  client — the keyless PRODUCTION `market_data_client`, which MUST come from
  the same exchange as the diff stream: update IDs form one continuous
  sequence only within a single exchange, and a mismatched pair (e.g. prod stream +
  testnet snapshots) silently discards every
  stream frame, degrades the book to a 2 s REST poll, and starves the VWAP
  deque so the VWAP gate never activates —
  and rebuilds `bids/asks/lastUpdateId` under `thread_lock`, dropping the
  gapped event (the next straddling event re-establishes the chain).  The resync
  is **fail-safe** (a REST error keeps the current book and retries on the next
  gap) and rate-limited by `DEPTH_RESYNC_MIN_INTERVAL_SEC` (default 2 s) so a
  burst of gapped events cannot storm the REST endpoint.  The WS callback is
  single-threaded, so no event buffering is required.

## 4. ANALYSIS ENGINE  (analysis.py — AnalysisEngine)
  Owns two background threads (started by `websocket_main.py`), both of which
  check `stop_event` on every iteration and exit gracefully when it is set.

  Private cross-thread attributes, grouped by concern:

**Locks** — each serialises the state named:

| Attribute | Type | Serialises |
|-----------|------|------------|
| `_vwap_lock` | Lock | `_bid_vwap` / `_ask_vwap` |
| `_regime_lock` | Lock | `regime_director.regime_label` **and** `_trend_paused` (same source frame, same write cadence) |
| `_stop_loss_lock` | Lock | `_stop_loss_state` |
| `_macro_trend_lock` | Lock | `_macro_trend_state` |

**VWAP state** — written by `historical_analysis`, read by `low_latency_analysis` under `_vwap_lock`:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_bid_vwap` | float \| None | Latest bid VWAP from `historical_analysis` |
| `_ask_vwap` | float \| None | Latest ask VWAP from `historical_analysis` |

**Position / order-guard state:**

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_position_open` | bool | True while any pyramiding BUY leg is open; mirrors `open_strategy_qty` in `backtest/pnl.py`. BUY legs stack via the exposure gate (`MAX_PYRAMID_LEGS` + `MIN_CASH_RESERVE_PCT` reserve floor); a full SELL / stop-loss closes the position. Startup carry behaviour → see the note below the tables. |
| `_pending_buy_placed_at` | float \| None | Wall-clock time the last GTC BUY was dispatched; used by `cancel_stale_buy()` to detect the 10-second timeout |
| `_pending_buy_id` | int \| None | Binance orderId of the outstanding GTC BUY order (set synchronously on REST, async via `handle_order_response` on WS) |
| `_trend_paused` | bool | Mirrors the `trend_pause` column in `backtest/signals.py`; True ⇒ skip BOTH BUY and SELL this tick |
| `_avg_entry_price` | float | VWAP entry price of the open strategy position (0.0 when flat); stop-loss anchor |

**Stop-loss & macro-trend state** — shared mutable containers owned by `websocket_main.py`; `AnalysisEngine` reads them, the injected refresher closure writes them (pure REST decoupling — no REST client touches this class):

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_stop_loss_state` | dict | Keys `"pct"`, `"last_day_utc"` |
| `_refresh_stop_loss_fn` | Callable \| None | Closure injected by `websocket_main.py` that re-computes the threshold from PRODUCTION daily klines (keyless `market_data_client` — mirrors `backtest/signals.py`, which uses production data). Called once per UTC day from `historical_analysis()`. |
| `_macro_trend_state` | dict | Keys `"state"` (`"down"`/`"neutral"`/`"up"`), `"last_day_utc"` — same REST decoupling as `_stop_loss_state` |
| `_refresh_macro_trend_fn` | Callable \| None | Closure injected by `websocket_main.py` that re-computes the state from PRODUCTION daily klines once per UTC day (None ⇒ overlay disabled, state stays `"neutral"` forever). |

**Session counters:**

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_position_guard_skips` | int | BUY signals suppressed this session |
| `_trend_pause_skips` | int | Ticks suppressed by the trend-pause gate this session |
| `_n_stop_loss_fires` | int | Emergency exits this session |
| `_n_macro_downtrend_liquidations` | int | Force-to-cash exits this session |

**Startup inventory carry** (`_position_open` at session start): when `FLATTEN_ON_START=False` (carry inventory, the default), `_position_open` is set True at session start if `balance_status[BTC] ≥ 0.0001`, so inherited BTC is carried as the session's STARTING POSITION and traded around normally (BUY more or SELL per signals). With `FLATTEN_ON_START=True` the account is flattened to USDT-only first and this does not fire.

  ▸ low_latency_analysis()  [every HFT_INTERVAL = 1 s]
      Reads balance, copies order book under thread_lock, builds the top
      N_LEVELS (50) bid/ask pairs, scores candidates (70 % depth / 30 % delta).
      On each iteration (in order):
      1. **Stale-BUY cancel** — if _position_open is True, calls
         OrderExecutor.cancel_stale_buy().  If the GTC BUY has been on the book
         ≥ 10 s and the cancel succeeds, resets _position_open = False so the
         strategy can re-enter.  If the cancel fails (order likely already filled),
         keeps the guard armed and waits for a SELL signal.
      2. **Resting-SELL resolve** — if _position_open is True, calls
         OrderExecutor.cancel_stale_sell().  A planned LIMIT GTC exit that has
         rested ≥ 10 s is cancelled-and-retried ("still_long") or, if the cancel
         fails because it already filled ("closed"), resets the guard to flat.
      3. **Equity snapshot** — appends (utc, usdt_total, btc_total, mid) for the
         end-of-session chart, where *_total = balance_status (free) +
         balance_locked.  Counting locked keeps a position resting in a LIMIT
         order on the equity curve instead of collapsing the index to ~0.  The
         chart then subtracts the locked balances that existed at session start
         (foreign resting orders not placed by this strategy — see
         `session_chart.py`'s `locked_*_at_start`) so only the strategy's OWN
         locks count; otherwise foreign locked USDT/BTC would inflate the curve
         against the free-only `start_total` and make the index jump.
      4. **Adaptive stop-loss check** (unconditional): if mid_price < avg_entry ×
         (1 − pct), fires an emergency SELL and skips the rest of the tick.
      4b. **Macro-trend force-to-cash** (unconditional, mirrors backtest/pnl.py):
         if the daily macro-trend state (read from _macro_trend_state under
         _macro_trend_lock) is "down" and a position is open, fires an urgent
         MARKET SELL to flatten the book to cash and skips the rest of the tick —
         the symmetric counterpart to the "up" hold-&-ride rule in step 5.
      5. **HMM confidence gate** (≥ 0.60), **trend-pause gate** (skip BOTH sides
         if _trend_paused is True), **macro-trend gate** (skip new BUYs when the
         state is "down"; skip ALL mean-reversion SELLs — even exits — when "up",
         so a position held in an uptrend rides until the state leaves "up" or
         the stop-loss fires), **VWAP gate** (mean-reversion dead zone,
         δ = 0.002), **HMM regime direction filter**, then delegates to
         OrderExecutor.execute().  The BUY-side ghost-position reset (guard armed
         but free BTC ≈ 0 ⇒ assume the LIMIT BUY never filled) fires ONLY when
         ALL three conditions hold: (1) no LIMIT SELL is resting — the BTC would
         merely be locked in that exit, and disarming would orphan it
         (cancel_stale_sell is gated on the guard); (2) no BUY is unresolved — a
         resting or just-filled LIMIT BUY is resolved by cancel_stale_buy after
         its 10 s window, NOT here (disarming earlier re-fired the same signal
         every tick and stacked duplicate BUYs); and
         (3) a FRESH REST balance read (OrderExecutor.refresh_and_check_flat)
         confirms the account is truly flat — the guard is never disarmed on a
         balance snapshot up to 60 s old.

  ▸ historical_analysis()  [every HIST_INTERVAL = 60 s]
      Always recomputes bid_vwap / ask_vwap from history_order_book and
      publishes under _vwap_lock.  Once per UTC day calls the injected
      refresh_stop_loss_fn() closure (updates _stop_loss_state under
      _stop_loss_lock) AND the refresh_macro_trend_fn() closure (updates the
      "down"/"neutral"/"up" _macro_trend_state under _macro_trend_lock) —
      never touches Binance REST directly.  HMM block
      fires only when a new 5-minute UTC clock boundary is crossed: fetches
      fresh 5m klines, runs either a full BIC re-fit (select_hmm_model) or a
      cheap Viterbi pass (predict_current_regime), then computes the
      trend-pause flag on the same klines_df via
      strategy.indicators.add_trend_pause_flag() and writes the new regime
      label AND the trend-pause flag under _regime_lock.

  ▸ Deque fill-up (at WS_SPEED = 100 ms → ~10 entries/sec):

      │ Time elapsed │ WS ticks  │ Deque size          │ Historical iterations │
      │──────────────│───────────│─────────────────────│───────────────────────│
      │ 1 min        │ ~600      │ 600                 │ 1st runs              │
      │ 3 min        │ ~1 800    │ 1 800               │ 3rd runs              │
      │ 5 min        │ ~3 000    │ 3 000 (full)        │ 5th runs              │
      │ 10 min       │ ~6 000    │ 3 000 (capped)      │ 10th runs             │
      │ 20 min       │ ~12 000   │ 3 000 (capped)      │ 20th runs             │

      After ~5 min the deque is full and becomes a true rolling window.

  ▸ Thread timeline (default 120-min session):

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
      t=20min   low_latency iteration #1200
                historical iteration #20 → refreshes VWAP one last time
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
  balances, **applies the startup inventory policy** (step 2a — flatten or carry
  inherited BTC per `FLATTEN_ON_START`; see below), seeds
  `OrderBookState`, runs the pre-session HMM fit (so `regime_label` is never
  `None` on the first low-latency tick), opens the WebSocket stream, starts the
  two analysis threads **plus a REST balance-refresh daemon**, blocks for
  `DEFAULT_SESSION_MINUTES`, then shuts down cleanly (stop_event →
  ws_client.stop() → executor.stop() → thread joins → cancel of the session's
  still-open orders → order report → session P&L chart → balance P&L
  decomposition → position-state save).

  **Shutdown order cancel (`OrderExecutor.cancel_session_open_orders`):** after
  the threads join, every order THIS session placed that is still open on the
  book is cancelled via REST, so no funds stay locked (and no unattended fills
  happen) once the session ends — otherwise resting BUYs can stay locked at
  shutdown and fill unattended afterwards.  Only
  the session's own orderIds are touched; foreign open orders on the shared
  testnet account are left alone.  The order report that follows shows these
  orders as CANCELED rather than OPEN.

  **REST balance-refresh daemon (`BALANCE_REFRESH_INTERVAL`, default 60 s):**
  the WS user-data push (`outboundAccountPosition` over
  `wss://ws-api.testnet.binance.vision/ws-api/v3`) keeps balances live, but if
  that connection is down the executor runs REST-only and balances are refreshed
  only as a side effect of placing/cancelling an order.  During a long idle
  stretch (no qualifying signal) `balance_status` would then freeze — and with
  it the end-of-session equity snapshots (the chart's Strategy line goes flat).
  The daemon polls a fresh REST `account()` snapshot every interval while in
  REST-only mode (`not executor._user_data_active`) so balances and the equity
  curve stay current; it is a no-op once the WS push is confirmed healthy.

  **Step 2a — startup inventory policy (`FLATTEN_ON_START`):** The backtest
  assumes `BACKTEST_INITIAL_BTC = 0` (always starts with no open position). A
  live testnet account often carries BTC left over from previous sessions (BUYs
  that filled but whose SELLs never did). The `FLATTEN_ON_START` flag chooses
  what to do with it:

  - **`True` — flatten (start flat, matches backtest):** before the portfolio
    snapshot, any free BTC ≥ `1e-5` is sold with a one-time REST **MARKET** order
    (floored to the LOT_SIZE 5-decimal grid) and balances are re-fetched, so the
    account begins flat and USDT-only. This zeroes price-P&L component **B**, so
    all session P&L is attributable to trading alpha, and makes each session an
    isolated skill test directly comparable to the backtest.
  - **`False` — carry inventory (default):** the inherited BTC is kept and
    `AnalysisEngine` pre-arms the position guard on it (treated as an open
    position): the bot can then only exit it (mean-reversion rally or stop-loss)
    before buying again. This is the realistic "carry across restarts" mode; the
    end-of-session report's component **B** becomes non-zero and attributes the
    carried bag's market drift separately from trading alpha (flagged inline in
    the report). The carried position's stop-loss anchor (cost basis) is
    restored from the persisted state file when available (see below).

  **Position persistence (`strategy/position_store.py`):** on shutdown the
  driver writes `LIVE_POSITION_STATE_PATH` (default `state/live_position.json`,
  git-ignored) — `position_open`, `avg_entry_price`, and total BTC qty
  (free + locked). On the next startup, **only when `FLATTEN_ON_START = False`**,
  the driver reloads it and — if the saved BTC qty matches the account balance
  within `max(1e-5, 1%)` — passes the saved `avg_entry_price` as the position
  guard's stop-loss anchor instead of the session-start price, so a carried
  position's stop-loss reflects its true cost basis. Purely additive and
  fail-safe: a missing / corrupt / mismatched / wrong-symbol file (or a hard
  kill that skipped the save) falls back to the session-start-price anchor, and
  no strategy/engine code is involved — only the value handed to the existing
  `initial_avg_entry_price` parameter. Atomic write (temp file + `os.replace`)
  prevents a half-written file; deleting the file reverts to Phase-1 behaviour.

  Note the failure mode of `True` with no flatten: if the MARKET sell fails, the
  pre-arm fires on the inherited BTC, so the bot can ONLY exit and can NEVER buy
  — in a flat market it places **zero orders all session**. The pre-arm logic
  (below) is shared by both the `False` path and that fallback.

  **End-of-session report — balance valuation:** the final balances are valued
  free + locked, minus the foreign locked captured at session start (the same
  `locked_*_at_start` correction the equity chart applies), so funds resting in
  the strategy's own open LIMIT orders still count.  Free-only valuation
  counts them as gone, badly understating the true result.

  **End-of-session report — Buy & Hold benchmark:** alongside the `A + B = Total`
  P&L decomposition, the balance report prints a **Buy & Hold** line —
  `(btc_end / btc_start − 1) × 100` — i.e. what the whole starting equity would
  have returned if simply held as BTC, plus **Strategy vs B&H** (positive ⇒ the
  strategy beat holding BTC). This is the SAME baseline drawn in the session P&L
  chart (`session_chart.py`'s `bnh_index`), surfaced as a number so the text
  report and the chart share one frame of reference. It is a benchmark only, NOT
  part of the strategy's P&L.

  **Session P&L chart — filled vs. unfilled markers:** an order marker sits at
  the order's *dispatch* time regardless of whether it traded, so a LIMIT order
  that was placed then cancelled (or never matched) would look identical to one
  that actually moved the position. To remove that ambiguity the chart draws
  **filled** orders as solid markers and **unfilled** ones (final
  `executedQty == 0`) as hollow markers colour-coded by side (green ▲ BUY /
  red ▼ SELL). The final outcome is read from
  `exec_qty`, which `OrderExecutor.order_status_report()` enriches onto each
  `placed_orders` record (from Binance `GET /api/v3/order`) before the chart is
  built; an order with no `exec_qty` is assumed filled so a genuine fill is
  never hidden. This explains why the strategy equity line stays flat at some
  BUY/SELL markers (no fill → no position change) and moves at others (a fill
  opened a position that was then marked to market while held).

## 6. EXECUTION  (execution/order_executor.py — OrderExecutor)
  Places orders via the Binance WebSocket API and maintains real-time balance
  via `outboundAccountPosition` push events (session.logon → userDataStream.subscribe),
  eliminating the need for a listenKey.  Falls back to REST if the WebSocket
  handshake fails.  Because the client's `on_open` callback fires from the
  socket thread DURING the `SpotWebsocketAPIClient` constructor — before
  `ws_api_client` is assigned — the first `session.logon` attempt can be
  silently swallowed (its `if ws_api_client is None: return` guard trips);
  `__init__` therefore re-sends the logon after construction when `_logon_id`
  is still `None`.  Without this retry the user-data stream never activates
  and balances degrade to 60 s REST polls for the whole session.

  **Order types per side:**
  - **BUY**: LIMIT GTC — dip-buy sits on the book until filled or cancelled.
    On dispatch, `_pending_buy_placed_at` is set (both paths) and
    `_pending_buy_id` is set synchronously (REST) or asynchronously via
    `handle_order_response` (WS).
  - **SELL**: order type depends on **urgency** (the `urgent` arg of
    `execute()`):
    - **Urgent (stop-loss exit) → MARKET.** Closes the position immediately by
      filling against Binance's **real server-side book**, so the close always
      executes and never expires. Immune to `local_book` staleness (a marketable
      LIMIT/IOC priced from `local_book` can expire `EXPIRED`/`execQty=0` when
      `core/message_handler.py` — which has no depth-diff **gap recovery** — has
      left a phantom top-of-book bid). Before the MARKET close the stop-loss
      first calls `cancel_stale_sell(timeout_sec=0.0)` to cancel any resting
      LIMIT exit and free the BTC it locked. Trade-off: real-book slippage
      instead of the backtest's single half-spread — accepted for guaranteed exit.
    - **Non-urgent (planned mean-reversion exit) → LIMIT GTC at `micro_price`**,
      symmetric to the BUY: it rests on the book as a maker order. Because a
      resting LIMIT may not fill (the rally can fade before a buyer crosses), it
      is tracked by `_pending_sell_placed_at` / `_pending_sell_id` and resolved
      by **`cancel_stale_sell()`** after a 10 s timeout — exactly like
      `cancel_stale_buy`: cancel succeeds → `"still_long"` (position kept, retry
      on the next signal); cancel fails (-2011) → `"closed"` (order filled,
      strategy now flat). The position guard stays armed and the stop-loss anchor
      intact until the exit actually fills, so a BUY can never stack behind an
      unfilled SELL. At dispatch a **book-health** line is logged (`best_bid`,
      `best_ask`, `lastUpdateId`); a crossed book (`best_bid ≥ best_ask`) is
      direct proof `local_book` is stale.

  **Exchange-filter normalisation:**
  At construction the executor calls `exchange_info()` once to cache the
  symbol's `LOT_SIZE` (`stepSize`, `minQty`) and `(MIN_)NOTIONAL`
  (`minNotional`) filters.  Every order's quantity is then floored DOWN
  to the `stepSize` grid (never up — keeps the order within budget) and
  rejected before dispatch if it is below `minQty` or if its notional
  is below `minNotional`.  This prevents the silent Binance
  `-1013 "Filter failure: LOT_SIZE"` rejection that otherwise drops
  every order at the gateway when the strategy-computed quantity is not
  an exact multiple of the symbol's precision step (e.g. `0.004785`
  for a stepSize of `0.00001`).  BTCUSDT-correct defaults are used if
  the `exchange_info()` call fails so the executor stays operational.

  **Key methods:**
  - `execute()`: validates strategy, caps quantity, dispatches the order.
    Dynamic cap: BUY = `min(aq, usdt × MAX_POSITION_PCT / (price × (1 + fee)))`
    — at most `MAX_POSITION_PCT` (default 20 %) of available USDT per signal,
    with the taker fee reserved.  The budget is then clamped by the
    **cash-reserve floor** so the leg never spends the account below
    `MIN_CASH_RESERVE_PCT` (0.20) of mark-to-market equity, mirroring
    `backtest/pnl.py`; the dispatched `last_buy_qty` / `last_buy_price` are
    exposed for the strategy's pyramiding cost-basis accrual.  SELL =
    `min(bq, btc)`.  `MAX_POSITION_PCT` and `MIN_CASH_RESERVE_PCT` are the same
    constants used by `backtest/pnl.py`, keeping live and backtest BUY sizing
    aligned.  Whether a new leg is dispatched at all (serialization,
    `MAX_PYRAMID_LEGS`, reserve floor) is
    decided upstream by the exposure gate in `strategy/analysis.py`.
  - `cancel_stale_buy(timeout_sec=10.0)`: cancels the outstanding GTC BUY via
    REST if it has been open ≥ 10 s.  Returns True (reset `_position_open`) on
    success; False if cancel fails (order likely filled — keep guard armed).
  - `_refresh_balance_rest()`: fetches free balances from Binance REST and
    updates `state.balance_status` under `thread_balance_lock`.  Called after
    every successful order and after a stale-BUY cancel to keep the free/locked
    split accurate in REST-fallback mode (no WS push events available).

## 7. BACKTESTING  (backtest/)
  Full design, pseudocode, data-flow diagrams, approximation caveats, and
  step-by-step implementation notes → **BACKTESTING.md**.

## 8. TESTS  (tests/)
  Tier B (Core) pytest suite — deterministic unit tests for the project's pure
  logic.  No network calls (Binance clients mocked via pytest-mock); no writes
  outside pytest's tmp_path.  Tests pin CURRENT behaviour.

  tests/
  ├── conftest.py                — project-root import shim + RNG seeding
  ├── fixtures/
  │   ├── fake_order_book.py     — {price: qty} → live {str: str} book builder
  │   └── fake_klines.py         — OHLCV + signals DataFrame builders
  ├── test_book_utils.py         — build_levels / collect_candidates / select_best (12)
  ├── test_indicators.py         — VWAP, trend-pause, stop-loss, REST indicators (9)
  ├── test_param_loader.py       — rows_to_lookback + best_params loader (8)
  ├── test_pnl.py                — buy-and-hold, simulate_pnl, round-trips (7)
  ├── test_message_handler.py    — depth-diff apply + gap-recovery resync (7)
  ├── test_order_book_state.py   — OrderBookState container (5)
  └── test_signals.py            — _add_hmm_features (3)

  Run:  .venv314/bin/python -m pytest tests/ -q      (51 tests)

  Deferred to higher tiers: regime_director.py (HMM fit/predict), analysis.py
  gate threads, order_executor.py (authenticated WS), end-to-end backtest replays.
