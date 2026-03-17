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
HISTORY_MAXLEN = 6000  # max snapshots in history_order_book
# at 100 ms update intervals this covers ~10 minutes
N_LEVELS = 50  # number of order book levels used in low_latency_analysis()
# ---------------------------------------------------------------------------
# Analysis engine cadence
# ---------------------------------------------------------------------------
HFT_INTERVAL = 5  # seconds between HFT evaluations
HIST_INTERVAL = 300  # seconds between historical analyses (5 min)
MIN_SNAPSHOTS = 100  # minimum snapshots required before historical analysis runs

# ---------------------------------------------------------------------------
# WebSocket session
# ---------------------------------------------------------------------------
DEFAULT_SESSION_MINUTES = 15  # default session length
# at 15 min: ~180 HFT iterations, ~3 historical runs
HTF_JOIN_TIMEOUT = 10  # s — max wait for low_latency_analysis thread on shutdown
HIST_JOIN_TIMEOUT = 15  # s — max wait for historical_analysis thread on shutdown

# ---------------------------------------------------------------------------
# Binance REST / WebSocket connection
# ---------------------------------------------------------------------------
RECV_WINDOW = 5000  # ms — Binance REST request validity window
SNAPSHOT_DEPTH = 100  # number of order book levels in the seed snapshot
WS_SPEED = 100  # ms — WebSocket diff-depth update interval
