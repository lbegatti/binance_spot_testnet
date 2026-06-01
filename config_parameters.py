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
DEFAULT_SESSION_MINUTES = 10  # default session length
# at 10 min: ~600 low-latency iterations (every 1 s), ~10 historical runs (every 60 s)
HTF_JOIN_TIMEOUT = 10  # s — max wait for low_latency_analysis thread on shutdown
HIST_JOIN_TIMEOUT = 15  # s — max wait for historical_analysis thread on shutdown

# ---------------------------------------------------------------------------
# Binance REST / WebSocket connection
# ---------------------------------------------------------------------------
RECV_WINDOW = 5000  # ms — Binance REST request validity window
SNAPSHOT_DEPTH = 100  # number of order book levels in the seed snapshot
WS_SPEED = 100  # ms — WebSocket diff-depth update interval

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
HMM_N_INIT = 10
# Legacy train/predict split constant — RETAINED FOR REFERENCE AND DIAGNOSTICS ONLY.
# regime_director.py no longer uses this value to cap the split.
# The split is now computed adaptively per window:
#   train_end = max(2, int(n_rows × 2/3))   (~⅔ train, ~⅓ test)
# At the default 120-row window this gives train_end = 80 (= HMM_TRAIN_ROWS),
# but shorter windows now scale proportionally instead of collapsing to 1 test row.
# HMM_TRAIN_ROWS is derived from HMM_LOOKBACK_ROWS — see below.
# Minimum posterior probability for regime gating.
# predict_proba()[-1][current_regime] < HMM_MIN_CONFIDENCE → iteration skipped.
# 0.70 = 70 % probability mass required on the winning state.
HMM_MIN_CONFIDENCE = 0.70
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
VWAP_WINDOW = 5  # rolling VWAP window (rows) — 25 min at 5 m (micro frame)
REFIT_EVERY = 360  # full BIC re-fit every N macro candles (= 20 h at 5 m)
# Aligned with SENSITIVITY_REFIT_EVERY so the OOS validation
# in runner.py uses the same HMM cadence as the IS optimisation
# in sensitivity.py — makes IS↔OOS Sharpe comparisons

# Sensitivity-sweep override for backtest/sensitivity.py — kept equal to REFIT_EVERY
# so IS and OOS regime classification use the same cadence.  Defined as a separate
# constant so sensitivity.py can be overridden independently if needed in future.
SENSITIVITY_REFIT_EVERY = 360
# How many macro candles to skip between cheap Viterbi passes in sensitivity.py.
# Between two predict calls the last known regime label is reused (forward-filled
# onto the 1m micro frame via merge_asof).
# 5-min macro candles → regime changes in ≤ 25 min are missed, but the
# RELATIVE ranking of parameter combinations is preserved (all runs use the
# same cadence).  runner.py always uses predict_every=1 (every macro candle).
SENSITIVITY_PREDICT_EVERY = 5  # ~25 min at 5 m — sensitivity-sweep override only
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
# Starting balances.  Total = USDT + BTC × first_close ≈ $10 000 at $77k BTC.
BACKTEST_INITIAL_CAPITAL = 5000.0  # starting USDT balance
BACKTEST_INITIAL_BTC = 0.065  # starting BTC balance

# Taker fee per side (Binance Spot standard tier).
BACKTEST_FEE_RATE = 0.001  # 0.10 %

# Annualised risk-free rate for Sharpe / Sortino.
# pnl.py converts this to a per-period rate automatically.
# Set to 0.04–0.05 for a US T-bill proxy; 0.0 is the standard in crypto research.
BACKTEST_RISK_FREE_RATE = 0.00  # annualised (0.0 = no risk-free rate adjustment)

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
# Prevents all-in behaviour where fee + spread costs compound on 100 % of
# the balance every round trip.  0.10 → at most 10 % risked per signal.
# Set to 1.0 to revert to full all-in behaviour.
BACKTEST_MAX_POSITION_PCT: float = 0.10  # 10 % of USDT per BUY signal

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
#   0.002 (0.20 %) = exact round-trip break-even (2 × one-way fee).
#   0.003 (0.30 %) = break-even + 0.10 % profit margin per side (default).
#   Higher values filter out more marginal signals, reducing trade count.
#
# Increase to 0.005–0.010 in choppy / low-volatility markets.
# Set to 0.0 to disable (reverts to bare VWAP gate — buys any dip).
#
# Imported by: strategy/analysis.py (live WebSocket path),
#              backtest/signals.py  (backtest path),
#              backtest/sensitivity.py (fixed baseline — not tuned in grid).
VWAP_THRESHOLD_MULTIPLIER: float = 0.003  # 0.30 % dead zone — break-even + margin
