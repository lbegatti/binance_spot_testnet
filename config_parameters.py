from binance.client import Client

# =============================================================================
# config_parameters.py — central configuration file
# All tunable constants live here. Edit this file to adjust behaviour across
# the entire project without touching any logic files.
# =============================================================================

# ---------------------------------------------------------------------------
# Symbol configuration
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"  # trading pair
CCY = "USDT"  # quote currency
CRYPTOCCY = "BTC"  # base / cryptocurrency

# ---------------------------------------------------------------------------
# Order book state
# ---------------------------------------------------------------------------
HISTORY_MAXLEN = 3000  # max snapshots in history_order_book
# at 100 ms update intervals this covers ~5 minutes
N_LEVELS = 50  # number of order book levels used in low_latency_analysis()

# ---------------------------------------------------------------------------
# Analysis engine cadence
# ---------------------------------------------------------------------------
HFT_INTERVAL = 1  # seconds between low-latency evaluations
HIST_INTERVAL = 60  # seconds between historical analyses (1 min)
MIN_SNAPSHOTS = 100  # minimum snapshots required before historical analysis runs

# ---------------------------------------------------------------------------
# WebSocket session
# ---------------------------------------------------------------------------
DEFAULT_SESSION_MINUTES = 60  # default session length
# at 20 min: ~1200 low-latency iterations (every 1 s), ~20 historical runs (every 60 s)
HTF_JOIN_TIMEOUT = 10  # s — max wait for low_latency_analysis thread on shutdown
HIST_JOIN_TIMEOUT = 15  # s — max wait for historical_analysis thread on shutdown
# Cadence of the REST balance-refresh daemon (driver-side, defense in depth).
# Only active when the WS user-data push is NOT live (REST-only fallback): it
# polls a fresh account() snapshot so balances — and the end-of-session equity
# chart — stay current during long idle stretches with no orders.  No-op when
# the WS push is healthy.  ~1 account() call per interval (trivial on weight).
BALANCE_REFRESH_INTERVAL = 60  # seconds

# Startup inventory policy (live session only).
#   True  → MARKET-sell any inherited BTC at startup so the session begins flat,
#           matching the backtest's BACKTEST_INITIAL_BTC = 0.  Each session is an
#           isolated skill test; all P&L is trading alpha (report component B ≡ 0).
#   False → keep inherited BTC and let AnalysisEngine's position guard pre-arm on
#           it (carry inventory across restarts; more realistic).  The end-of-
#           session report's component B then attributes the carried bag's market
#           drift separately from trading alpha.
# NOTE: while False, the carried position's stop-loss anchors at the true cost
# basis when a matching persisted position is restored at startup (see
# LIVE_POSITION_STATE_PATH), otherwise it falls back to the session-start price.
FLATTEN_ON_START: bool = False

# Path to the live position-state file (cost basis carried across restarts).
# Written on shutdown; read at startup ONLY when FLATTEN_ON_START is False, so a
# carried position's stop-loss anchors at its true entry price instead of the
# session-start price.  Runtime artifact — git-ignored; deleting it simply
# reverts to the session-start-price anchor.
LIVE_POSITION_STATE_PATH: str = "state/live_position.json"

# ---------------------------------------------------------------------------
# Binance REST / WebSocket connection
# ---------------------------------------------------------------------------
RECV_WINDOW = 5000  # ms — Binance REST request validity window
SNAPSHOT_DEPTH = 100  # number of order book levels in the seed snapshot
WS_SPEED = 100  # ms — WebSocket diff-depth update interval
# Minimum seconds between local-book REST resyncs after a diff-depth gap /
# reconnect.  When the diff stream drops an event, the first
# update ID (U) of the next event exceeds lastUpdateId+1; MessageHandler then
# re-pulls a fresh depth snapshot to rebuild local_book.  This cooldown prevents
# a resync storm when several gapped events arrive in a burst — the book still
# recovers on the next event after the interval elapses.
DEPTH_RESYNC_MIN_INTERVAL_SEC: float = 2.0

# ---------------------------------------------------------------------------
# Quote calculation throttle
# ---------------------------------------------------------------------------
# Number of ticks between calculate_best_quote() calls.
# At WS_SPEED=100 ms, 10 ticks ≈ 1 second — enough to keep the console
# readable without missing meaningful spread changes.
QUOTE_EVERY_N_TICKS = 10

# ---------------------------------------------------------------------------
# End-of-session order report
# ---------------------------------------------------------------------------
# Maximum number of orders shown at the head *and* tail of the report.
# If the session produces more than 2 * ORDER_REPORT_LIMIT orders, the middle
# block is collapsed to a single summary line to avoid flooding the console.
ORDER_REPORT_LIMIT = 100

# ---------------------------------------------------------------------------
# HMM Parameters and Features
# ---------------------------------------------------------------------------
HMM_FEATURE_COLS = ["return", "volatility", "obi_proxy", "trade_density"]
HMM_N_ITERATIONS = 1000
HMM_MAX_REGIMES = len(HMM_FEATURE_COLS) - 1  # BIC search: 2 … 3 states
HMM_RANDOM_STATE = 46
HMM_INTERVAL = Client.KLINE_INTERVAL_5MINUTE
HMM_LOOKBACK = "10 hours ago UTC"  # 120 candles at 5 m — provides stable EM convergence
# while remaining responsive to intraday BTC shifts (10 h × 12 bars/h = 120 rows).
# Regularisation floor added to the diagonal of every state's covariance
# matrix.  Prevents "covars must be symmetric, positive-definite" errors
# when a hidden state has few observations relative to the feature count.
# 1e-1 is the recommended safe default for z-scored financial features.
HMM_MIN_COVAR = 1e-1
# Number of independent random-seed restarts per candidate n_components value
# inside select_hmm_model().  Higher values reduce the chance of degenerate
# EM solutions (e.g. transmat row summing to 0) on flat/low-variance windows.
# 5 seeds balances robustness against degenerate initialisation with speed:
# the retry loop breaks on the first valid fit, so well-conditioned windows
# still cost 1 seed; only pathological IS windows retry up to 5.
HMM_N_INIT = 5
# Legacy train/predict split constant — RETAINED FOR REFERENCE AND DIAGNOSTICS ONLY.
# regime_director.py no longer uses this value to cap the split.
# The split is now computed adaptively per window:
#   train_end = max(2, int(n_rows × 2/3))   (~⅔ train, ~⅓ test)
# At the default 120-row window this gives train_end = 80 (= HMM_TRAIN_ROWS),
# but shorter windows now scale proportionally instead of collapsing to 1 test row.
# HMM_TRAIN_ROWS is derived from HMM_LOOKBACK_ROWS — see below.
# Minimum posterior probability for regime gating.
# predict_proba()[-1][current_regime] < HMM_MIN_CONFIDENCE → iteration skipped.
# 0.60 = 1.8× random for 3 states (33 % baseline). Still requires clear model
# conviction while opening the 50–69 % confidence band that 0.70 fully rejected.
HMM_MIN_CONFIDENCE = 0.60
# Minimum ABSOLUTE mean log-return (per 5 m candle) for a state to earn a
# directional label.  The directional labels in assign_regime_labels() are
# assigned by RELATIVE rank (lowest-ranked state → "trending_down", highest →
# "trending_up"), so with the usual 2 BIC states there is ALWAYS a
# "trending_down" bucket — even in a flat or rising market.  That spuriously
# blocked the BUY side for a whole flat/+0.44 % session (2026-07-14): 38/50
# HMM pulses read "trending_down" while price actually drifted UP.
# A state whose mean return sits inside ±REGIME_DIRECTIONAL_RETURN_THRESHOLD is
# treated as non-directional and falls through to high_volatility / neutral
# (both BUY-eligible), so only a genuinely negative/positive state trips the
# directional gate.  Units: log-return per 5 m bar (0.0005 ≈ 0.05 %/bar).
# NOTE: default is a first cut — calibrate against a few live/backtest sessions.
REGIME_DIRECTIONAL_RETURN_THRESHOLD = 0.0005
# Cadence at which the full HMM re-fit (select_hmm_model()) is triggered
# inside historical_analysis().  Between re-fits only a cheap Viterbi pass
# (predict_current_regime()) runs.  Must be a multiple of HIST_INTERVAL.
HMM_REFIT_INTERVAL = 300  # seconds (= 5 min)

# ---------------------------------------------------------------------------
# Backtesting — all parameters used by backtest/ in one place
# ---------------------------------------------------------------------------

# -- Candle resolution — DECOUPLED (macro brain vs micro body) -------------
# The backtest uses two separate resolutions to isolate the HMM (statistically
# stable, low-frequency) from the PnL simulation (granular, execution-realistic):
#
#   MACRO (5m) — fed to GaussianHMM for regime classification.
#     Fewer rows → fast EM convergence; regime shifts are measured in hours,
#     not seconds, so 5-minute granularity is sufficient.
#
#   MICRO (1m) — fed to VWAP, signal generation, and PnL simulation.
#     1-minute bars expose intra-hour price structure that a 5-minute bar
#     would suppress, giving the entry/exit logic its full resolution.
#
# Row counts per window:
#   IS  5m : 270 days × 288  rows/day ≈  77,760 rows  (HMM input)
#   IS  1m : 270 days × 1440 rows/day ≈ 388,800 rows  (PnL input)
#   OOS 5m :  90 days × 288  rows/day ≈  25,920 rows
#   OOS 1m :  90 days × 1440 rows/day ≈ 129,600 rows
BACKTEST_MACRO_INTERVAL: str = Client.KLINE_INTERVAL_5MINUTE  # "5m" — HMM training
BACKTEST_MICRO_INTERVAL: str = Client.KLINE_INTERVAL_1MINUTE  # "1m" — PnL execution

# -- In-sample / out-of-sample split (Option B) ----------------------------
# ALL 360 days (1 year) are covered.  The pipeline is split cleanly:
#
#   |── days −360 → −90 (270 days, ~77,760 5m / ~388,800 1m rows) ──|── days −90 → today (90 days) ──|
#                  IN-SAMPLE (sensitivity.py)                             OUT-OF-SAMPLE (runner.py)
#
# sensitivity.py fetches [BACKTEST_LOOKBACK … BACKTEST_OOS_START) — never sees OOS data.
# runner.py      fetches [BACKTEST_OOS_START … today]             — genuinely fresh data.
# 360-day IS window captures ~3 full market cycles (bull / bear / sideways), reducing
# the risk that Optuna fits parameters to a single regime.
BACKTEST_LOOKBACK = "360 days ago UTC"  # IS start  — used by sensitivity.py (~1 year)
BACKTEST_OOS_START = "90 days ago UTC"  # OOS start — used by runner.py; IS end cutoff

# Set to an integer to cap the number of replay candles for quick debug runs;
# None = use all candles in the window (full production backtest).
BACKTEST_MAX_ROWS: int | None = None

# -- Synthetic order book (backtest/synthetic_book.py) ---------------------
# Volume at each price level decays exponentially away from the mid.
# Level i carries base_volume × VOLUME_DECAY_FACTOR ** i.
# 0.80 → level 1 retains 80 % of level 0, level 2 retains 64 %, etc.
VOLUME_DECAY_FACTOR = 0.80

# -- HMM cadence inside the signal replay (backtest/signals.py) -----------
# These constants apply to the MACRO (5m) frame — the HMM never sees 1m bars.
# Mirrors the live system's HMM_LOOKBACK / HMM_REFIT_INTERVAL constants so
# the backtest uses the same rolling-window logic as websocket_main.py.
HMM_LOOKBACK_ROWS = 120  # warm-up window (rows) — 10 h at 5 m (120 × 5 min = 600 min)
VWAP_WINDOW = 5  # rolling VWAP window (rows) — 5 min at 1 m (micro frame)
REFIT_EVERY = 480  # full BIC re-fit every N macro candles (= 40 h at 5 m)
# Shared by sensitivity.py (IS sweep, ~162 refits over 270 days) and runner.py
# (OOS backtest, ~54 refits over 90 days) so both pipelines use the same HMM
# refit cadence — IS↔OOS Sharpe figures are directly comparable.
# How many macro candles to skip between cheap Viterbi passes in sensitivity.py.
# Between two predict calls the last known regime label is reused (forward-filled
# onto the 1m micro frame via merge_asof).
# 5-min macro candles → regime changes in ≤ 25 min are missed, but the
# RELATIVE ranking of parameter combinations is preserved (all runs use the
# same cadence).  runner.py always uses predict_every=1 (every macro candle).
#
# ⚠ IS / OOS Sharpe comparability caveat
# ----------------------------------------
# IS sweep (sensitivity.py)  → predict_every=5  → regime re-inferred every 25 min
# OOS run  (runner.py)       → predict_every=1  → regime re-inferred every  5 min
#
# A non-zero IS-vs-OOS Sharpe gap is therefore NOT a pure overfitting signal —
# part of the gap is caused by the cadence difference (the IS sweep responds to
# regime transitions 20 minutes later than the OOS run, which produces a
# different signal mix on the same data).  Treat the IS Sharpe as a parameter
# ranking score, not as an absolute predictor of OOS performance.  Do not lower
# this value to "match" OOS — the IS sweep would slow down ~5× without any
# guarantee that the cadence-gap component of the IS/OOS gap is the dominant one.
SENSITIVITY_PREDICT_EVERY = (
    5  # ~25 min at 5 m — IS sensitivity-sweep speed override only
)
# Between two full-BIC refit calls, predict_current_regime() is called only every
# 5 macro candles; the last known regime label is reused (~5× Viterbi speedup).
# runner.py always uses predict_every=1 (every candle).

# Fee rate applied to ALL sensitivity runs (OAT, full-grid, Bayes).
# Must match BACKTEST_FEE_RATE — Binance charges what it charges regardless.
# Kept as a separate constant so changing BACKTEST_FEE_RATE does not silently
# affect sensitivity results (and vice versa), making any deliberate change obvious.
# IS window: BACKTEST_LOOKBACK → BACKTEST_OOS_START (270 days, ~77,760 5m / ~388,800 1m rows).
SENSITIVITY_FEE_RATE: float = 0.001  # 0.10 % — standard Binance Spot taker fee

# Metric used to rank parameter combinations and select best_params.json.
# "sharpe_ratio" is the standard risk-adjusted return metric.
# Other valid choices: "sortino_ratio", "total_return_pct".
SENSITIVITY_RANK_METRIC: str = "sharpe_ratio"

# |ΔSharpe| threshold that flags a parameter as highly sensitive in the OAT report.
# If any non-default value moves the rank metric by more than this amount,
# the OAT report prints a warning and recommends running --bayes.
SENSITIVITY_OAT_THRESHOLD: float = 0.5

# -- P&L simulation (backtest/pnl.py) -------------------------------------
# Starting balances mirror the live paper-trading account (~250k USDT + 1 BTC).
# BACKTEST_INITIAL_BTC is set to 0.0 to avoid "orphan SELL" signals consuming
# a pre-existing BTC balance before the strategy opens its first BUY.  The 1 BTC
# is instead folded into BACKTEST_INITIAL_CAPITAL at ~65k/BTC so total starting
# equity (~315k) matches the live account without distorting PnL metrics.
BACKTEST_INITIAL_CAPITAL = 315000.0  # starting USDT balance (~250k USDT + 1 BTC @ ~65k)
BACKTEST_INITIAL_BTC = (
    0.0  # starting BTC balance (BTC value folded into BACKTEST_INITIAL_CAPITAL)
)

# Taker fee per side (Binance Spot standard tier).
BACKTEST_FEE_RATE = 0.001  # 0.10 %

# Annualised risk-free rate for Sharpe / Sortino.
# pnl.py converts this to a per-period rate automatically.
# Set to 0.04–0.05 for a US T-bill proxy;
BACKTEST_RISK_FREE_RATE = 0.04  # annualised (0.0 = no risk-free rate adjustment)

# Fill-cost model: simulated bid-ask half-spread in basis points.
#   half_spread = close × BACKTEST_FILL_SPREAD_BPS / 20 000
#   BUY  fill   = close + half_spread   (you pay the synthetic ask)
#   SELL fill   = close - half_spread   (you receive the synthetic bid)
#
# Why NOT (high - low) / 2:
#   A 1-min BTC candle range of $50–$300 gives half_spread $25–$150 —
#   10–100× larger than the real Binance BTCUSDT spread of ~1–5 bps.
#   A LIMIT order fills at or inside the spread, not at the candle extreme.
#
#   2  bps — tight / optimistic
#   5  bps — realistic base case (default)
#   10 bps — conservative / stressed
BACKTEST_FILL_SPREAD_BPS: float = 5.0  # full bid-ask spread in basis points

# Maximum fraction of available USDT deployed per BUY trade.
# Applies to BOTH the live system (execution/order_executor.py) and the
# backtest (backtest/pnl.py) so the simulated and live order sizes stay
# aligned.  Prevents all-in behaviour where fee + spread costs compound on
# 100 % of the balance every round trip.  0.20 → at most 20 % of available
# USDT risked per BUY leg (legs may still pyramid up to the cash-reserve floor).
# Set to 1.0 to revert to full all-in behaviour.
#
# Risk-management parameter — DO NOT add to the Optuna search space.
MAX_POSITION_PCT: float = 0.20  # 20 % of available USDT per BUY leg (live + backtest)

# Cash-reserve floor (fraction of mark-to-market equity that must always remain
# in USDT).  BUY legs may PYRAMID — each leg is still ≤ MAX_POSITION_PCT of the
# available USDT, but successive legs stack until invested exposure reaches
# (1 − MIN_CASH_RESERVE_PCT) of equity.  0.10 → at most 90 % invested, always
# ≥ 10 % cash held back.  Replaces the old single-position rule that left the
# book ~90 % idle in cash, without ever going fully all-in.  Applies to BOTH the
# live system and the backtest so simulated and live sizing stay aligned.
#
# Set to 0.10 to give the strategy the most room to trade (up to 90 % invested).
# Trade-off: exposure — and therefore drawdown — scales with the invested
# fraction (~20 % IS tail drawdown at 90 % invested), so this is the aggressive
# end of the dial.  Raise toward 0.50 to bound the tail if a live drawdown feels
# too large.  Re-run the OOS backtest after changing.  NOTE: the live path also
# needs MAX_PYRAMID_LEGS high enough to reach this floor (see below).
#
# Risk-management parameter — DO NOT add to the Optuna search space.
MIN_CASH_RESERVE_PCT: float = 0.10  # keep ≥ 10 % of equity as USDT (live + backtest)

# -- Pyramiding control (live path — execution/order_executor.py +
#    strategy/analysis.py) ------------------------------------------------
# Hard ceiling on how FAR the LIVE strategy may stack BUY legs, independent of
# the reserve-floor maths above.  A safety net for the case where the reserve
# floor is fed a stale balance: no matter what, no more than MAX_PYRAMID_LEGS
# legs open before a full SELL resets the count.  Serialization (one order in
# flight at a time, in strategy/analysis.py) already bounds how FAST legs stack.
# The backtest fills instantly and does not use this cap.
#
# Risk-management parameter — DO NOT add to the Optuna search space.
# Each leg spends MAX_POSITION_PCT (20 %) of REMAINING free cash, so invested
# exposure after n legs ≈ 1 − 0.8ⁿ of cash: n=7 → ~79 %, n=11 → ~91 %.  To let
# the 10 % MIN_CASH_RESERVE_PCT floor (→ 90 % invested) actually bind, the cap
# must allow ~11 legs; 12 leaves one leg of margin (the reserve clamp trims the
# final leg to land exactly at the floor).  Also keeps the live cap consistent
# with the backtest, which has no leg cap and reaches 90 % from the reserve
# alone.  Lower this to cap live exposure by leg count regardless of the floor.
MAX_PYRAMID_LEGS: int = 12  # hard cap on concurrently-stacked live BUY legs

# -- Trend-pause filter (macro frame, backtest/signals.py) ----------------
# Pauses all new BUY/SELL entries when the macro frame shows N consecutive
# same-direction closes (sustained trend).  Trading resumes after the streak
# breaks AND TREND_COOLDOWN_BARS additional ranging bars have elapsed.
TREND_CONSECUTIVE_BARS: int = (
    3  # N consecutive same-direction 5m closes → pause (= 15 min)
)
# Fixed from Optuna study 2026-05-24 — removed from search space.
TREND_COOLDOWN_BARS: int = (
    4  # extra macro bars to stay paused after trend ends (= 20 min)
)
# Fixed from Optuna study 2026-05-24 — removed from search space.

# -- Adaptive stop-loss (backtest/signals.py + pnl.py) --------------------
# Forces a SELL when the open position's unrealised loss exceeds a dynamic
# threshold derived from rolling daily volatility — no fixed percentage needed.
#   threshold(t) = STOP_LOSS_STD_MULT × rolling_std(daily_abs_return, STOP_LOSS_ROLLING_DAYS)
# In normal BTC vol (~1–1.5 % daily std) this gives ~3–4.5 % stop distance.
# DO NOT add these to the Optuna search space — the formula is self-calibrating.
STOP_LOSS_ROLLING_DAYS: int = 90  # lookback window for rolling std of daily abs returns
STOP_LOSS_STD_MULT: float = 3.0  # multiplier: threshold = rolling_std × mult

# ---------------------------------------------------------------------------
# VWAP gate — applies to BOTH live system AND backtest
# ---------------------------------------------------------------------------
# The bot executes a BUY only when micro_price < bid_vwap × (1 − threshold),
# and a SELL only when micro_price ≥ bid_vwap × (1 + threshold).
# This creates a symmetric dead zone around the VWAP so microscopic noise
# (1-penny vibrations) never triggers an order.
#
# Rule of thumb: threshold must cover at least the round-trip fee.
#   Standard Binance Spot taker fee: 0.10 % per side → 0.20 % round trip.
#   0.002 (0.20 %) = exact round-trip break-even (2 × one-way fee). [default]
#   0.003 (0.30 %) = break-even + 0.10 % profit margin per side.
#   Higher values filter out more marginal signals, reducing trade count.
#
# Increase to 0.005–0.010 in choppy / low-volatility markets.
# Set to 0.0 to disable (reverts to bare VWAP gate — buys any dip).
#
# Imported by: strategy/analysis.py (live WebSocket path),
#              backtest/signals.py  (backtest path),
#              backtest/sensitivity.py (fixed baseline — not tuned in grid).
VWAP_THRESHOLD_MULTIPLIER: float = (
    0.002  # 0.20 % dead zone — exact round-trip break-even
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Backtesting — Output Paths and best_params.json Orchestration
# ---------------------------------------------------------------------------
# Filename of the best-parameter JSON file produced by sensitivity.py and
# consumed by websocket_main.py (live system) and runner.py (backtest).
# Centralized here to prevent divergence between producer and consumers.
# NOTE: This is the FILENAME ONLY (best_params.json); directory paths defined below.
BEST_PARAMS_FILE = "best_params.json"

# Output directories for sensitivity sweep artifacts.
BACKTEST_RESULTS_DIR = "backtest/results"  # Machine: best_params.json, optuna.db
BACKTEST_REPORTING_DIR = "backtest/reporting"  # Human: CSVs, Optuna HTML charts

# Sensitivity sweep — --force-save override flag.
# By default, sensitivity.py only overwrites best_params.json if the new
# Sharpe ratio is strictly better than the value already stored (guard logic
# in backtest/sensitivity.py line ~920). Pass --force-save to override when
# the market regime has shifted and cached params are stale/harmful.
#
# Usage: python -m backtest.sensitivity --force-save
#
# Fields derived from best_params.json:
#
# LIVE SYSTEM (websocket_main.py):
#   - hmm_lookback_rows  → patched to HMM_LOOKBACK (string, via rows_to_lookback)
#   - hmm_max_regimes    → patched to HMM_MAX_REGIMES (int)
#   - vwap_threshold     → patched to VWAP_THRESHOLD_MULTIPLIER (float)
#   - fee_rate          → NOT USED (stored for reference only; Binance charges
#                         its own fees regardless of our simulation parameter)
#
# BACKTESTING (runner.py):
#   - hmm_lookback_rows  → passed as kwarg to run_signals()
#   - hmm_max_regimes    → passed as kwarg to run_signals()
#   - vwap_window        → passed as kwarg to run_signals()
#   - vwap_threshold     → passed as kwarg to run_signals()
#   - fee_rate          → passed as kwarg to simulate_pnl(); ensures OOS Sharpe
#                         is computed with same fee assumption as IS tuning.
#
# Note: HMM_LOOKBACK conversion between storage formats:
#   - Storage (best_params.json): hmm_lookback_rows as int (candle count)
#   - Live system (HMM_LOOKBACK): dateutil string ("10 hours ago UTC")
#   - Conversion (param_loader.py): rows_to_lookback() multiplies rows × 5 min
#   - Example: 120 rows × 5 min/row = 600 min = 10 hours → "10 hours ago UTC"
