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
# Backtesting Parameters and Features
# ---------------------------------------------------------------------------
# 10 *calendar* days (including weekends) — crypto trades 24/7 so there are
# no weekend gaps in Binance kline data.  At 1 m resolution this yields
# 10 × 24 × 60 = 14 400 rows.  Reduced from 30 days to keep the backtest
# runtime manageable (~45–90 s vs ~2–4 min).
# NOTE: backtest/regime_validation.py (Step 6b) uses its own 10-day fetch and
# bypasses BACKTEST_MAX_ROWS intentionally — this constant only affects
# backtest/runner.py (signal replay) and is independent of regime_validation.
BACKTEST_LOOKBACK = "10 days ago UTC"
BACKTEST_MAX_ROWS: int | None = 500
VOLUME_DECAY_FACTOR = (
    0.80  # each lever down the order book retains 80% of the previous level's volume
)
HMM_LOOKBACK_ROWS = 120  # 2 h at 1 m — matches HMM_LOOKBACK in the live system
VWAP_WINDOW = 5  # 5 candles = 5 min at 1 m — matches live VWAP window
REFIT_EVERY = 5  # HMM_REFIT_INTERVAL // HIST_INTERVAL = 300 // 60

# ---------------------------------------------------------------------------
# Backtesting P&L Parameters  (Step 4 — backtest/pnl.py)
# ---------------------------------------------------------------------------
# TOTAL AMOUNT = USD + BTC = 10,000$ (assuming BTC is 68,000)
BACKTEST_INITIAL_CAPITAL = 5000.0  # starting USDT balance for the simulation
BACKTEST_INITIAL_BTC = 0.0735  # starting BTC balance for the simulation
BACKTEST_FEE_RATE = 0.001  # 0.10 % taker fee per side (Binance Spot standard)
# Annualised risk-free rate used in the Sharpe / Sortino denominator.
# Sharpe = (mean(Rp) - Rf_per_period) / std(Rp) × √(periods_per_year)
# For crypto there is no direct risk-free equivalent; 0.0 is the most common
# choice in academic crypto research.  To use a T-bill proxy (e.g. 4 % p.a.)
# set this to 0.04 — pnl.py converts it to the correct per-period rate
# automatically regardless of the adaptive resampling bucket chosen.
# TODO maybe the Rf rate should be equal to US Treasury bills (4 - 5 %)
BACKTEST_RISK_FREE_RATE = 0.0  # annualized (0.0 = no risk-free rate adjustment)
# Slippage is NOT a fixed constant — it equals half the candle range:
#   half_spread = (high - low) / 2
# This quantity is computed per-candle in synthetic_book.py and stored in
# the signals DataFrame.  Fill prices are then:
#   BUY  fill = close + half_spread   (you pay the synthetic ask)
#   SELL fill = close - half_spread   (you receive the synthetic bid)

# ---------------------------------------------------------------------------
# HMM Parameters and Features
# ---------------------------------------------------------------------------
HMM_FEATURE_COLS = ["return", "volatility", "obi_proxy", "trade_density"]
HMM_N_ITERATIONS = 1000
HMM_MAX_REGIMES = len(HMM_FEATURE_COLS) - 1
HMM_RANDOM_STATE = 46
HMM_INTERVAL = Client.KLINE_INTERVAL_1MINUTE
HMM_LOOKBACK = "2 hours ago UTC"  # 120 candles — responsive to intraday BTC shifts
# while keeping enough data for stable EM convergence
# Regularisation floor added to the diagonal of every state's covariance
# matrix.  Prevents "covars must be symmetric, positive-definite" errors
# when a hidden state has few observations relative to the feature count.
# 1e-3 is a safe default for normalised financial features (return,
# volatility, obi_proxy, trade_density all sit in the [-1, +1] range).
HMM_MIN_COVAR = 1e-3
# Train / predict split for select_hmm_model() — walk-forward style.
# The HMM is fitted ONLY on the FIRST HMM_TRAIN_ROWS rows of klines_df (older,
# "in-sample" data).  Regime prediction (Viterbi) then runs ONLY on the
# remaining rows klines_df[HMM_TRAIN_ROWS:] (most recent, out-of-sample).
# self.current_regime / regime_confidence therefore always reflect a candle
# the model has never seen during fit() — no look-ahead bias.
# Rule of thumb: ~2/3 for training.  At HMM_LOOKBACK="2 hours ago UTC"
# (≈120 rows) this gives 80 training rows and ~40 out-of-sample rows.
HMM_TRAIN_ROWS = 80
# Minimum posterior probability the model must assign to the predicted regime
# before an order is allowed.  When predict_proba()[-1][current_regime] < this
# threshold the regime is treated as "uncertain" and the iteration is skipped.
# 0.70 means at least 70 % probability mass on the winning state — a coin-flip
# (0.50 for 2 states) would always be rejected, a clear signal (0.80+) passes
# comfortably.  Raise to 0.75–0.80 for more conservative gating.
HMM_MIN_CONFIDENCE = 0.70
# Cadence at which the full HMM re-fit (select_hmm_model()) is triggered
# inside historical_analysis().  Between re-fits, only a cheap Viterbi
# prediction (predict_current_regime()) is run on the latest kline features.
# Must be a multiple of HIST_INTERVAL (60 s).  Default: 300 s = 5 minutes.
HMM_REFIT_INTERVAL = 300
# ---------------------------------------------------------------------------
# Order book state
# ---------------------------------------------------------------------------
HISTORY_MAXLEN = 3000  # max snapshots in history_order_book
# at 100 ms update intervals this covers ~5 minutes
N_LEVELS = 50  # number of order book levels used in low_latency_analysis()
# ---------------------------------------------------------------------------
# Analysis engine cadence
# ---------------------------------------------------------------------------
HFT_INTERVAL = 1  # seconds between HFT evaluations
HIST_INTERVAL = 60  # seconds between historical analyses (1 min)
MIN_SNAPSHOTS = 100  # minimum snapshots required before historical analysis runs

# ---------------------------------------------------------------------------
# WebSocket session
# ---------------------------------------------------------------------------
DEFAULT_SESSION_MINUTES = 20  # default session length
# at 20 min: ~1200 low-latency iterations (every 1 s), ~20 historical runs (every 60 s)
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
