from binance.client import Client

# =============================================================================
# config_parameters.py — central configuration file
# All tunable constants live here. Edit this file to adjust behaviour across
# the entire project without touching any logic files.
# =============================================================================

# ---------------------------------------------------------------------------
# Symbol configuration
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"   # trading pair
CCY = "USDT"         # quote currency
CRYPTOCCY = "BTC"    # base / cryptocurrency

# ---------------------------------------------------------------------------
# Order book state
# ---------------------------------------------------------------------------
HISTORY_MAXLEN = 3000  # max snapshots in history_order_book
# at 100 ms update intervals this covers ~5 minutes
N_LEVELS = 50  # number of order book levels used in low_latency_analysis()

# ---------------------------------------------------------------------------
# Analysis engine cadence
# ---------------------------------------------------------------------------
HFT_INTERVAL = 1    # seconds between low-latency evaluations
HIST_INTERVAL = 60  # seconds between historical analyses (1 min)
MIN_SNAPSHOTS = 100 # minimum snapshots required before historical analysis runs

# ---------------------------------------------------------------------------
# WebSocket session
# ---------------------------------------------------------------------------
DEFAULT_SESSION_MINUTES = 10  # default session length
# at 10 min: ~600 low-latency iterations (every 1 s), ~10 historical runs (every 60 s)
HTF_JOIN_TIMEOUT = 10   # s — max wait for low_latency_analysis thread on shutdown
HIST_JOIN_TIMEOUT = 15  # s — max wait for historical_analysis thread on shutdown

# ---------------------------------------------------------------------------
# Binance REST / WebSocket connection
# ---------------------------------------------------------------------------
RECV_WINDOW = 5000    # ms — Binance REST request validity window
SNAPSHOT_DEPTH = 100  # number of order book levels in the seed snapshot
WS_SPEED = 100        # ms — WebSocket diff-depth update interval

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
HMM_INTERVAL = Client.KLINE_INTERVAL_1MINUTE
HMM_LOOKBACK = "2 hours ago UTC"  # 120 candles — responsive to intraday BTC shifts
                                   # while keeping enough data for stable EM convergence
# Regularisation floor added to the diagonal of every state's covariance
# matrix.  Prevents "covars must be symmetric, positive-definite" errors
# when a hidden state has few observations relative to the feature count.
# 1e-1 is the recommended safe default for z-scored financial features.
HMM_MIN_COVAR = 1e-1
# Number of independent random-seed restarts per candidate n_components value
# inside select_hmm_model().  Higher values reduce the chance of degenerate
# EM solutions (e.g. transmat row summing to 0) on flat/low-variance windows.
HMM_N_INIT = 10
# Train / predict split — walk-forward style (no look-ahead bias).
# The HMM is fitted ONLY on the first HMM_TRAIN_ROWS rows of klines_df.
# Viterbi prediction then runs on the remaining rows klines_df[HMM_TRAIN_ROWS:].
# Rule of thumb: ~2/3 for training.  At 120 rows → 80 train, ~40 out-of-sample.
HMM_TRAIN_ROWS = 80
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

# -- Data window -----------------------------------------------------------
# 180 calendar days at 1 m resolution → 259,200 rows.
# Crypto trades 24/7 so there are no weekend gaps.
BACKTEST_LOOKBACK = "180 days ago UTC"

# Set to an integer to cap the number of replay candles for quick debug runs;
# None = use all candles in the window (full production backtest).
BACKTEST_MAX_ROWS: int | None = None

# -- Synthetic order book (backtest/synthetic_book.py) ---------------------
# Volume at each price level decays exponentially away from the mid.
# Level i carries base_volume × VOLUME_DECAY_FACTOR ** i.
# 0.80 → level 1 retains 80 % of level 0, level 2 retains 64 %, etc.
VOLUME_DECAY_FACTOR = 0.80

# -- HMM cadence inside the signal replay (backtest/signals.py) -----------
# Mirrors the live system's HMM_LOOKBACK_ROWS / REFIT_EVERY constants so
# the backtest uses the same rolling-window logic as websocket_main.py.
HMM_LOOKBACK_ROWS = 120  # warm-up window (rows) — 2 h at 1 m
VWAP_WINDOW = 5          # rolling VWAP window (rows) — 5 min at 1 m
REFIT_EVERY = 120        # full BIC re-fit every N replay candles (= 2 h at 1 m)

# Speed-up override used ONLY inside backtest/sensitivity.py.
# At 1 m candles: 480 iterations = 8 h between refits → ~90 refits per 30-day
# run instead of ~360, giving a ~4× speedup.  The relative ranking of parameter
# combinations is preserved; only absolute P&L numbers change slightly.
# config_parameters.py defaults and the live system are NEVER affected.
SENSITIVITY_REFIT_EVERY = 480
# Data fetch window used by sensitivity.py — shorter than BACKTEST_LOOKBACK
# (180 days) to keep each of the 6 OAT runs fast (~36–108 min total).
# 30 days ≈ 43,200 rows at 1 m resolution.
# IMPORTANT: this is passed as start_str to fetch_klines() inside run_signals()
# only when called from sensitivity.py.  run_backtest.py always uses
# BACKTEST_LOOKBACK and is completely unaffected.
SENSITIVITY_LOOKBACK = "30 days ago UTC"
# How many candles to skip between cheap Viterbi passes (predict_current_regime)
# in sensitivity.py.  Between two predict calls the last known regime label is
# reused.  1-min candles → regime changes in ≤5 min are missed, but the
# RELATIVE ranking of parameter combinations is preserved (all runs use the
# same cadence).  run_backtest.py always uses predict_every=1 (every candle).
SENSITIVITY_PREDICT_EVERY = 5  # 8 h at 1 m — sensitivity-sweep override only

# -- P&L simulation (backtest/pnl.py) -------------------------------------
# Starting balances.  Total = USDT + BTC × first_close ≈ $10 000 at $68 k BTC.
BACKTEST_INITIAL_CAPITAL = 5000.0   # starting USDT balance
BACKTEST_INITIAL_BTC = 0.0735       # starting BTC balance

# Taker fee per side (Binance Spot standard tier).
BACKTEST_FEE_RATE = 0.001  # 0.10 %

# Annualised risk-free rate for Sharpe / Sortino.
# pnl.py converts this to a per-period rate automatically.
# Set to 0.04–0.05 for a US T-bill proxy; 0.0 is the standard in crypto research.
BACKTEST_RISK_FREE_RATE = 0.0  # annualised (0.0 = no risk-free rate adjustment)

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
# Set to 1.0 to revert to all-in behaviour.
BACKTEST_MAX_POSITION_PCT: float = 0.10  # 10 % of USDT per BUY signal
