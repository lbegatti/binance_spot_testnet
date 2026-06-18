# Binance Spot Testnet — Order Book Analysis

---

> ## ⚠️ DISCLAIMER — PLEASE READ BEFORE USING THIS PROJECT
>
> **This project is a personal side project and a technical learning exercise. It is provided strictly for educational and research purposes.**
>
> - 🚫 **Not financial advice.** Nothing in this repository constitutes financial advice, investment advice, trading advice, or any other form of advice. The strategies, signals, metrics, and outputs produced by this code should **not** be interpreted as recommendations to buy, sell, or hold any financial instrument.
>
> - 🚫 **Not a solicitation.** This project is not a solicitation or offer to trade any asset, cryptocurrency, or financial product — on Binance or any other platform. **Binance is not a sponsor, partner, or affiliate of this project in any way.** The use of the Binance Spot Testnet is purely technical and for educational purposes only.
>
> - 🚫 **Not legal advice.** Nothing in this repository constitutes legal advice of any kind. Trading cryptocurrency may be regulated, restricted, or prohibited in your jurisdiction. It is your sole responsibility to ensure compliance with all applicable laws and regulations before engaging in any trading activity.
>
> - 🧪 **Testnet / paper trading only.** All code in this repository is designed and tested exclusively against the **Binance Spot Testnet** (`testnet.binance.vision`), which uses **simulated funds with no real monetary value**. No real Binance account, real funds, or real orders are involved in any way.
>
> - 🏢 **No employer affiliation.** This is an independent personal project. It does not represent, involve, or reflect the views, work, products, or interests of my current or any previous employer in any way.
>
> - 🤖 **AI-assisted development.** Parts of this codebase, including implementation, documentation, and code review, were produced with the assistance of AI language models (e.g. GitHub Copilot / GPT-4 class models) to speed up development and act as a four-eyes check. AI models can be wrong, produce misleading output, introduce subtle bugs, or generate plausible-sounding but incorrect logic. **All AI-generated content should be treated as unverified until independently reviewed.** The author takes no responsibility for errors or omissions that originate from, or were not caught by, AI-assisted tooling.
>
> - ⚡ **Use at your own risk.** If you choose to adapt or extend this code to interact with a real Binance account or any live trading environment, you do so entirely **at your own risk**. The author accepts no liability for any financial loss, legal consequence, or damage arising from such use.
>
> **By cloning, forking, or using this repository you acknowledge that you have read and understood this disclaimer.**

---

# Binance Spot Testnet — Order Book Analysis

A Python toolkit for order book analysis on the **Binance Spot Testnet**.  
Two execution modes are available:

| Mode | Entry point | Data source | Symbol |
|------|-------------|-------------|--------|
| **REST** (polling) | `restapi_main.py` | `client.get_order_book()` snapshots | BTCUSDT |
| **WebSocket** (real-time) | `websocket_main.py` | `diff_book_depth` stream (100 ms) | BTCUSDT |

The REST path is fully wired — metrics → indicators → scores → best quote.  
The WebSocket path runs the full strategy pipeline (metrics → indicators → scores) in near real-time via the `AnalysisEngine`.

---

## 🚀 Four Entry Points — Start Here

The project is driven by **four standalone scripts**. Everything else (strategy, core, execution, backtest modules) is support code called by one of these.

| # | Script | Purpose | Typical runtime | Command |
|---|--------|---------|----------------|---------|
| 1 | `websocket_main.py` | **Live trading session** — connects to the Binance Testnet WebSocket, maintains a real-time order book, detects market regimes via HMM on **5-minute klines** (`HMM_INTERVAL="5m"`, `HMM_LOOKBACK="10 hours ago UTC"`), and places LIMIT orders when the VWAP gate and regime filter both pass. | 10 min (default session length; configurable via `DEFAULT_SESSION_MINUTES`) | `python websocket_main.py` |
| 2 | `backtest/runner.py` | **Offline backtest (OOS validation)** — replays 90 days of data through the **two-resolution pipeline**: 5-minute klines (`BACKTEST_MACRO_INTERVAL`) for the HMM (~25,920 rows) and 1-minute klines (`BACKTEST_MICRO_INTERVAL`) for signals and PnL (~129,600 rows).  Prints P&L, Sharpe ratio, max drawdown, and filter hit-rates.  Uses parameters from `best_params.json` produced by script 3. | ~15–25 min on a laptop (90-day OOS window at 1 m resolution) | `python -m backtest.runner` |
| 3 | `backtest/sensitivity.py` | **Parameter optimisation (IS tuning)** — runs an Optuna Bayesian search (40 trials by default) over `hmm_lookback_rows`, `hmm_max_regimes`, `vwap_window`, and `vwap_threshold` on the **in-sample** window (`BACKTEST_LOOKBACK = "360 days ago UTC"` → `BACKTEST_OOS_START = "90 days ago UTC"`, 270 days, ~77,760 macro rows / ~388,800 micro rows). Saves `best_params.json`, which is automatically loaded by scripts 1 and 2 on their next run. | ~3–6 h on a laptop (~5–8 min per trial; use `--n-trials 20` for a ~1.5–3 h run) | `python -m backtest.sensitivity` |
| 4 | `backtest/diagnostics/regime_validation.py` | **Regime sanity check** — fits the HMM on 730 days of data and runs six statistical tests (direction test, Kruskal-Wallis H-test, cross-correlation, entropy, stationarity, persistence) to confirm the regime labels are statistically meaningful before trusting them in live trading. | ~20–30 min on a laptop (730-day window fetch + fit) | `python -m backtest.diagnostics.regime_validation` |

**Detailed notes on `regime_validation.py`:**
- Uses its own independent **2-year lookback** (`VALIDATION_LOOKBACK = "730 days ago UTC"`) — NOT affected by `BACKTEST_LOOKBACK` or `BACKTEST_OOS_START` used by `sensitivity.py` and `runner.py`.
- Splits the 2-year window into 70% train (~511 days) and 30% test (~219 days).
- Fits a fresh HMM model on the train set, predicts regime labels on the test set, and runs 6 statistical checks.
- Runtime: ~20–30 minutes (dominated by Binance API paginated fetches for ~1,050k rows).

> **Recommended order for a new setup:**
> `regime_validation` → `sensitivity` → `runner` → `websocket_main`

---

## Project Structure

```
binance_spot_testnet/
├── config_parameters.py               # Central configuration — all tunable constants in one place
├── restapi_main.py                    # REST orchestration — loops over depth limits
├── websocket_main.py                  # WebSocket — real-time local order book + session driver
├── README.md
├── BACKTESTING.md                     # Backtesting design, pseudo-code, and implementation roadmap
├── SYSTEM_ARCHITECTURE.md             # Full system architecture, data-flow diagrams, and orchestration notes
├── REGIME_DIRECTOR.md                 # RegimeDirector deep dive: config → startup → threading → data-flow
│
├── core/                              # Shared state and data ingestion
│   ├── __init__.py
│   ├── order_book_state.py            # Shared state container (local book + history + balances + locks)
│   └── message_handler.py             # WebSocket callbacks — maintains local book and balances in real time
│
├── strategy/                          # Analysis and scoring pipeline
│   ├── __init__.py
│   ├── analysis.py                    # AnalysisEngine — low-latency (1 s) and historical (1 min) loops
│   ├── book_utils.py                  # Shared order-book utilities (build_levels, collect_candidates, select_best_opportunity)
│   ├── regime_director.py             # RegimeDirector — HMM regime detection on **5-minute klines** (HMM_INTERVAL="5m", HMM_LOOKBACK="10h"); pre-session fit + two-speed refresh every 60 s / 300 s
│   ├── param_loader.py                # load_best_params() + load_best_params_for_backtest() — reads best_params.json; shared by websocket_main.py and run_backtest.py
│   ├── best_quote_calculator.py       # Live spread printer — prints best_bid | best_ask on every tick
│   ├── metrics.py                     # Order book metric calculations
│   ├── indicators.py                  # Strategy-specific indicator columns
│   ├── scores.py                      # Weighted opportunity scoring
│   └── quotes.py                      # Best quote selection logic
│
├── execution/                         # Order placement
│   ├── __init__.py
│   └── order_executor.py             # OrderExecutor — LIMIT GTC (BUY) / IOC (SELL) orders via WebSocket API; 10-second stale-BUY cancel; balance refresh
│
├── backtest/                          # Offline backtesting framework (see BACKTESTING.md)
│   ├── __init__.py
│   ├── data.py                        # Historical kline downloader — fetch_macro_klines() (5 m, HMM) + fetch_micro_klines() (1 m, PnL); Parquet cache (cache/klines/, 24h TTL); --flush-cache flag
│   ├── synthetic_book.py              # Synthetic 50-level order book builder (per kline row)
│   ├── signals.py                     # Two-frame signal pipeline: Phase 1 HMM walk-forward on 5m, Phase 2 merge_asof stitch, Phase 3 1m execution loop + gates
│   ├── pnl.py                         # P&L simulation — balance guard, bps-based fill, intra-candle whipsaw guard, position guard, equity curve, metrics
│   ├── runner.py                      # Top-level runner — fetches 5m macro + 1m micro OOS frames; plot ON by default (--no-plot to suppress); --csv; --flush-cache
│   ├── visualization.py               # Step 7 — interactive Plotly chart (6-panel); shown by default from CLI, or run_backtest(plot=True) programmatically
│   ├── sensitivity.py                 # Step 8 — Bayesian (Optuna TPE, default) / OAT / full-grid sweep; writes best_params.json
│   ├── regime_validation.py           # Offline long-horizon HMM validation — python -m backtest.regime_validation
│   ├── visualization.py               # Backtest chart (7-panel Plotly): equity + B&H overlay, drawdown, price+signals, regime, VWAP, signal funnel, signals-by-regime.  Writes backtest_chart_<ts>.html (runner.py) or sensitivity_chart_<ts>.html (sensitivity.py)
│   └── reporting/                     # Console report formatting and CSV export (AI-authored)
│       ├── __init__.py
│       └── formatters.py              # fmt(), print_report(), save_csv(), print_sensitivity_table(), print_oat_sensitivity_report(), print_bnh_comparison() — public helpers
│
└── visualization/                     # Plotting utilities
    ├── __init__.py
    ├── plot_helpers.py                # Plotly visualisations (depth, OHLC)
    └── session_chart.py               # End-of-session P&L chart (live): Strategy vs B&H index + BUY/SELL markers + USDT/BTC panel — writes backtest/reporting/session_pnl_<ts>.html
```

---

## Configuration (`config_parameters.py`)

All tunable constants are centralised in `config_parameters.py`. Edit this file to change behaviour across the entire project without touching any logic files.

| Group | Constant | Default | Description |
|---|---|---|---|
| **Symbol** | `SYMBOL` | `"BTCUSDT"` | Trading pair used across all REST and WebSocket calls |
| **Symbol** | `CCY` | `"USDT"` | Quote currency |
| **Symbol** | `CRYPTOCCY` | `"BTC"` | Base / crypto currency |
| **Order book state** | `HISTORY_MAXLEN` | `3000` | Max snapshots in `history_order_book` — at 100 ms intervals this covers ~5 min.  Each entry: `{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}` (all numeric) |
| **Order book state** | `N_LEVELS` | `50` | Number of order book levels used in `low_latency_analysis` |
| **Analysis cadence** | `HFT_INTERVAL` | `1` s | Time between low-latency evaluations |
| **Analysis cadence** | `HIST_INTERVAL` | `60` s | Time between historical analyses (1 min) |
| **Analysis cadence** | `MIN_SNAPSHOTS` | `100` | Minimum snapshots required before historical analysis runs |
| **WebSocket session** | `DEFAULT_SESSION_MINUTES` | `10` min | Default session length (fixed — no startup prompt) |
| **WebSocket session** | `HTF_JOIN_TIMEOUT` | `10` s | Max wait for `low_latency_analysis` thread on shutdown |
| **WebSocket session** | `HIST_JOIN_TIMEOUT` | `15` s | Max wait for `historical_analysis` thread on shutdown |
| **Binance connection** | `RECV_WINDOW` | `5000` ms | Binance REST request validity window |
| **Binance connection** | `SNAPSHOT_DEPTH` | `100` | Order book levels in the seed snapshot |
| **Binance connection** | `WS_SPEED` | `100` ms | WebSocket diff-depth update interval |
| **Quote throttle** | `QUOTE_EVERY_N_TICKS` | `10` | Ticks between `calculate_best_quote()` calls.  At `WS_SPEED=100 ms`, 10 ticks ≈ 1 s |
| **HMM** | `HMM_FEATURE_COLS` | `["return", "volatility", "obi_proxy", "trade_density"]` | Feature columns fed to the `GaussianHMM` |
| **HMM** | `HMM_N_ITERATIONS` | `1000` | Max EM iterations per model fit |
| **HMM** | `HMM_MAX_REGIMES` | `3` | Upper bound on hidden states evaluated during BIC search (2 … 3, = `len(HMM_FEATURE_COLS) − 1`).  Capped at 3 to avoid under-populated states when training on 4 features |
| **HMM** | `HMM_RANDOM_STATE` | `46` | Random seed for reproducible HMM initialisation |
| **HMM** | `HMM_INTERVAL` | `Client.KLINE_INTERVAL_5MINUTE` | Kline granularity for regime detection (5 m — reduces noise vs 1 m without losing intraday granularity) |
| **HMM** | `HMM_LOOKBACK` | `"10 hours ago UTC"` | Kline history window (~120 rows at 5 m — provides stable EM convergence without being stale) |
| **HMM** | `HMM_MIN_COVAR` | `1e-1` | Regularisation floor for covariance matrices — prevents positive-definite errors.  1e-1 is the recommended safe default for z-scored financial features |
| **HMM** | `HMM_N_INIT` | `5` | Number of independent random-seed restarts per candidate `n_components` value inside `select_hmm_model()`.  The retry loop breaks on the first valid (non-degenerate) fit, so well-conditioned windows cost 1 seed; only pathological windows retry up to 5 |
| **HMM** | `HMM_TRAIN_ROWS` | `80` | Legacy constant kept in `config_parameters.py` for reference.  **No longer used** to cap the live split — `regime_director.py` now computes `train_end = max(2, int(n_rows × 2/3))` adaptively per window.  At the default 120-row window this equals 80 (identical result), but shorter windows (e.g. 60 rows → 40 train) are now handled correctly instead of collapsing to a 1-row test set |
| **HMM** | `HMM_MIN_CONFIDENCE` | `0.60` | Minimum posterior probability (`predict_proba()[-1][current_regime]`) required to allow an order.  Below this threshold the regime signal is treated as ambiguous and both BUY and SELL are skipped |
| **HMM** | `HMM_REFIT_INTERVAL` | `300` s | Cadence of **full** HMM re-fit inside `historical_analysis()`.  Between re-fits only a cheap Viterbi prediction runs.  Must be a multiple of `HIST_INTERVAL` |
| **Order report** | `ORDER_REPORT_LIMIT` | `100` | Max orders shown at head *and* tail of the end-of-session report.  Middle block collapsed when total > 2 × limit |
| **Backtesting** | `BACKTEST_MACRO_INTERVAL` | `"5m"` | Kline resolution for the HMM regime frame.  5-minute bars reduce noise vs 1-minute without losing intraday regime granularity.  At 5 m: 270 IS days → ~77,760 rows; 90 OOS days → ~25,920 rows. |
| **Backtesting** | `BACKTEST_MICRO_INTERVAL` | `"1m"` | Kline resolution for the execution/PnL frame (VWAP, signal gates, PnL simulation).  At 1 m: 270 IS days → ~388,800 rows; 90 OOS days → ~129,600 rows. |
| **Backtesting** | `BACKTEST_LOOKBACK` | `"360 days ago UTC"` | **In-sample (IS) start** — how far back `sensitivity.py` fetches klines for parameter tuning.  A 360-day window captures ~3 full market cycles, reducing over-fit risk. |
| **Backtesting** | `BACKTEST_OOS_START` | `"90 days ago UTC"` | **Out-of-sample (OOS) start / IS end cutoff**.  `sensitivity.py` stops fetching here (`end_str`); `runner.py` starts here.  Enforces a clean IS/OOS boundary. |
| **Backtesting** | `VOLUME_DECAY_FACTOR` | `0.80` | Exponential decay factor for synthetic order-book depth — each level retains 80 % of the previous level's volume |
| **Backtesting** | `HMM_LOOKBACK_ROWS` | `120` | Number of macro (5 m) kline rows used as the HMM warm-up window in the backtest (**10 h at 5 m** — 120 × 5 min; matches `HMM_LOOKBACK` in the live system) |
| **Backtesting** | `VWAP_WINDOW` | `5` | Rolling window size (in 1-minute micro bars) for the backtest VWAP computation (**5 min** — 5 × 1 min) |
| **Backtesting** | `REFIT_EVERY` | `480` | Macro-bar iterations between full HMM BIC re-fits. Shared by `sensitivity.py` (IS sweep, ~162 refits over 270 days) and `runner.py` (OOS backtest, ~54 refits over 90 days) so IS↔OOS Sharpe figures use the same HMM cadence and are directly comparable. At 5 m: one re-fit every **40 hours**. |
| **Backtesting P&L** | `BACKTEST_INITIAL_CAPITAL` | `315_000.0` | Starting USDT balance for the simulation (~250k USDT + 1 BTC @ ~65k, mirroring the live paper-trading account) |
| **Backtesting P&L** | `BACKTEST_INITIAL_BTC` | `0.0` | Starting BTC balance. Always `0.0` to avoid orphan SELL signals; the BTC equivalent is folded into `BACKTEST_INITIAL_CAPITAL`. |
| **Backtesting P&L** | `BACKTEST_FEE_RATE` | `0.001` | Taker fee fraction per side (0.10 %).  Also used by `OrderExecutor.execute()` to compute the fee-adjusted BUY quantity cap: `usdt / (micro_price × (1 + BACKTEST_FEE_RATE))` — prevents Binance from rejecting orders with `insufficient balance` when the taker fee pushes the total debit over the available balance |
| **Backtesting P&L** | `BACKTEST_RISK_FREE_RATE` | `0.0` | Annualised risk-free rate for Sharpe / Sortino denominator (0.0 = no adjustment; set to e.g. 0.04 for a 4 % T-bill proxy) |
| **Backtesting P&L** | `BACKTEST_MAX_ROWS` | `None` | Max replay candles in debug mode (`None` = full production run; set to e.g. `500` for a quick debug run) |
| **Trend-pause filter** | `TREND_CONSECUTIVE_BARS` | `3` | Number of consecutive same-direction 5-minute closes required to trigger a trend-pause flag.  When the flag is set, new BUY/SELL entries are suppressed (mean-reversion should not trade into a trending market).  Fixed from Optuna study 2026-05-24 — **not in the Optuna search space** |
| **Trend-pause filter** | `TREND_COOLDOWN_BARS` | `4` | Extra macro bars to remain paused after the last trending bar.  Prevents whipsaw re-entry the instant a streak breaks.  Fixed from Optuna study 2026-05-24 — **not in the Optuna search space** |
| **Adaptive stop-loss** | `STOP_LOSS_ROLLING_DAYS` | `90` | Lookback window (calendar days) for the rolling standard deviation of daily absolute returns used to compute the dynamic stop-loss threshold.  `threshold(t) = rolling_std(abs_daily_return, 90d) × STOP_LOSS_STD_MULT` — calibrates automatically to BTC's current volatility regime.  Enforced in BOTH the backtest (`backtest/pnl.py`) and the live system (`strategy/analysis.py`); `websocket_main.py` fetches 95 days of daily klines at startup and refreshes the threshold once per UTC day inside `historical_analysis()` |
| **Adaptive stop-loss** | `STOP_LOSS_STD_MULT` | `3.0` | Multiplier applied to the rolling daily-return std to set the stop-loss distance.  At typical BTC volatility (~1–1.5 % daily std) this gives a ~3–4.5 % stop distance.  Increase to loosen (fewer fires, more tail risk); decrease to tighten (more fires, more missed rebounds).  Monitor `n_stop_loss_fires` (backtest console report) and the "Stop-loss summary" line at session end (live) to calibrate |
| **Sensitivity** | `SENSITIVITY_PREDICT_EVERY` | `5` | Viterbi predict cadence used **only** by `sensitivity.py`. Between refit calls, `predict_current_regime()` is called only every 5 candles; the last known regime label is reused otherwise (~5× fewer Viterbi calls). `runner.py` always predicts every candle. (`SENSITIVITY_REFIT_EVERY` removed — refit cadence is now the shared `REFIT_EVERY = 480`.) |
| **Sensitivity** | `SENSITIVITY_FEE_RATE` | `0.001` | Fee rate applied to **all** sensitivity runs (OAT, full-grid, Bayes). Fixed at the standard Binance Spot taker fee — not a strategy knob, never included in the search grid. |
| **Sensitivity** | `SENSITIVITY_RANK_METRIC` | `"sharpe_ratio"` | Metric used to rank parameter combinations and select `best_params.json`. Change to `"sortino_ratio"` or `"total_return_pct"` to optimise for a different objective. |
| **Sensitivity** | `SENSITIVITY_OAT_THRESHOLD` | `0.5` | $|\Delta \text{Sharpe}|$ threshold in the OAT sensitivity report. If any parameter change moves the rank metric by more than this, the report recommends running `--bayes` for a wider search. |

For a compact per-category summary and the full per-module import list, see **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)**.

---

## Setup

1. **Install dependencies**

   ```bash
   pip install binance-connector python-dotenv pandas numpy plotly
   ```

   Or install everything from the lockfile:

   ```bash
   pip install -r requirements.txt
   ```

   > **`pip_system_certs`** (added 2026-04-25) — injects the OS/system CA bundle
   > into `pip` and `requests` on macOS/Linux.  Required on machines where the
   > Binance Spot Testnet self-signed certificate chain would otherwise fail
   > standard SSL verification at the OS level.  The three-layer SSL patch in
   > `websocket_main.py` (REST `session.verify=False`, WebSocket
   > `sslopt={"cert_reqs": ssl.CERT_NONE}`, urllib3 warning suppression) handles
   > the *runtime* bypass; `pip_system_certs` ensures the install step itself
   > succeeds in restricted environments.

2. **Create a `.env` file** in the project root:

   ```
   BINANCE_TESTNET_API_KEY=your_api_key
   BINANCE_TESTNET_SECRET_KEY=your_secret_key
   ```

   Keys are generated at <https://testnet.binance.vision/>.

3. **Run**

   ```bash
   # REST (static snapshots)
   python restapi_main.py

   # WebSocket (real-time)
   python websocket_main.py
   ```

---

## Pipeline Overview

### REST path (`restapi_main.py`)

```
Binance REST API
    │  get_order_book()
    ▼
┌──────────────────┐     ┌────────────────┐     ┌──────────────┐     ┌────────────┐
│ restapi_main.py  │────▶│  metrics.py    │────▶│ indicators.py│────▶│  scores.py │
│  (orchestrate)   │     │  (enrich df)   │     │ (strategy)   │     │  (score)   │
└──────────────────┘     └────────────────┘     └──────────────┘     └────────────┘
                                                                           │
                                                                           ▼
                                                                   ┌──────────────┐
                                                                   │  quotes.py   │
                                                                   │ (best quote) │
                                                                   └──────────────┘
```

### WebSocket path (`websocket_main.py`)

```
Binance REST API                   Binance WebSocket (production)
    │  depth() snapshot                │  diff_book_depth (100 ms)
    ▼                                  ▼
┌──────────────────┐       ┌──────────────────────┐
│  OrderBookState  │◀──────│   MessageHandler      │
│  (shared state)  │       │  .handle_depth_msg()  │
│                  │       └──────────────────────┘
│  • local_book    │
│  • history_book  │       ┌──────────────────────┐
│  • balance_status│◀──────│   AnalysisEngine      │  (read-only)
│  • thread_lock   │       │  .low_latency_analysis()  │
│  • thread_       │       │   (balance check +        │
│    balance_lock  │       │    order book strategy     │
│                  │       │    + VWAP filter)          │
│                  │       │  .historical_analysis()    │
│                  │       └──────────┬───────────┘
│                  │                  │  opportunity tuple
│                  │                  ▼
│                  │       ┌──────────────────────────────┐   Binance WS API (testnet)
│                  │◀──────│   OrderExecutor               │──▶ wss://testnet.binance.vision
│                  │       │  .execute()                   │    /ws-api/v3
│  (balance_status │       │   (balance guard + LIMIT GTC) │
│   updated by     │       │                               │◀── handle_order_response()
│   push events)   │       │  session.logon                │    (routes: orders + balance
│                  │       │  userDataStream.subscribe     │     push events)
│                  │       │  _handle_balance_update()     │
│                  │       └──────────────────────────────┘
│                  │                  ▲
│                  │       ┌──────────────────────┐
│                  │       │  websocket_main      │
│                  │       │  (session driver)     │
└──────────────────┘       └──────────────────────┘
```

**Component responsibilities:**

| Class / Module | Role |
|---|---|
| `OrderBookState` | Shared order book (`local_book`, `history_order_book`) and balance data (`balance_status`); serialised via `thread_lock` and `thread_balance_lock` |
| `MessageHandler` | Merges diff-depth WebSocket ticks into `local_book`; appends snapshots to history; triggers throttled quote calculation every 10 ticks |
| `RegimeDirector` | Detects market regime via HMM on 5-minute bars (2/3-1/3 train/test split, adaptive per window). Exposes `regime_label` (`"trending_up"`, `"trending_down"`, `"high_volatility"`, `"neutral"`) and `regime_confidence` (posterior probability) protected by `_regime_lock` |
| `AnalysisEngine` | Runs `low_latency_analysis` (1 s cadence, scores order-book candidates) and `historical_analysis` (60 s cadence, updates HMM + VWAPs). Enforces single-open-position guard (`_position_open` flag); pre-armed at startup when BTC balance ≥ 0.0001 so inherited BTC is treated as an open position |
| `OrderExecutor` | Places LIMIT GTC (BUY) / IOC (SELL) orders via Binance WebSocket API; maintains real-time balance via `outboundAccountPosition` push events; cancels stale GTC BUY orders after 10 s; refreshes balance from REST after every order. Falls back to REST for order placement if WS is unavailable |
| `websocket_main` | Session orchestrator — creates state, injects into handlers, runs pre-session regime fit, opens WebSocket streams, starts threads |

**How they interact (essentials):**

1. `websocket_main.py` creates a single `OrderBookState` instance and injects it into both `MessageHandler` and `AnalysisEngine`.  After construction it immediately seeds `state.balance_status` with the REST-fetched balances before any thread starts.
2. Order-book data (`local_book`, `history_order_book`) is serialised through `state.thread_lock`.  `MessageHandler.handle_depth_message` acquires it to write; `AnalysisEngine.low_latency_analysis` acquires it to take a read-only copy and releases it before any heavy computation.
3. Balance data (`balance_status`) is serialised through the dedicated `state.thread_balance_lock`, completely independent of `thread_lock`.  This prevents the high-frequency WebSocket order-book path (every 100 ms) from blocking on the lower-frequency balance path.
4. `MessageHandler` is the **only writer** to order-book data in `OrderBookState`.  `OrderExecutor._handle_balance_update` is the **only writer** to `balance_status`.  `AnalysisEngine` is **read-only** — it copies data under the appropriate lock and immediately releases it.
5. `AnalysisEngine` delegates order placement to `OrderExecutor`.  When `_select_best_opportunity()` returns a non-`None` 8-element tuple `(level_idx, score, delta, total_depth, obi, micro_price, bq, aq)`, the engine calls `executor.execute("BUY", best_buy)` or `executor.execute("SELL", best_sell)`.  `OrderExecutor` validates the strategy, checks balances under `thread_balance_lock`, computes quantity (`aq` for BUY, `bq` for SELL), and sends a **LIMIT GTC** (BUY) or **LIMIT IOC** (SELL) order via its own `SpotWebsocketAPIClient`.  IOC for SELL ensures the order fills at the best available bid immediately or auto-cancels — it never leaves BTC locked on the book (which would cause Binance error -2010 on the next SELL attempt).  BUY orders are capped at `usdt / (micro_price × (1 + BACKTEST_FEE_RATE))`: dividing by price × (1 + fee) reserves the taker fee so the total debit never exceeds the available USDT balance.  After every successful order, `_refresh_balance_rest()` re-fetches free balances from Binance REST to keep `state.balance_status` accurate in REST-fallback mode (where no `outboundAccountPosition` push events arrive).  GTC BUY orders are automatically cancelled after 10 seconds via `cancel_stale_buy()`, freeing the locked USDT so the strategy can re-enter on the next signal.  The WS order response arrives asynchronously in `handle_order_response`.
6. **VWAP dip/strength confirmation filter** — `AnalysisEngine` owns a private `_vwap_lock` plus two attributes `_bid_vwap` and `_ask_vwap` (initially `None`).  `historical_analysis` computes both VWAPs from `history_order_book` every 1 min and publishes them under `_vwap_lock`.  `low_latency_analysis` reads both under the same lock and gates order execution using a **mean-reversion** strategy with a **dead-zone threshold** (`VWAP_THRESHOLD_MULTIPLIER` δ, default 0.002 = 0.20 %):
   - **BUY**: execute only if `_bid_vwap is None` (first ~1 min) **or** `micro_price < bid_vwap × (1 − δ)`.  BUY is anchored to `bid_vwap` (volume-weighted bid pressure).
   - **SELL**: execute only if `_ask_vwap is None` (first ~1 min) **or** `micro_price ≥ ask_vwap × (1 + δ)`.  SELL is anchored to `ask_vwap` (volume-weighted ask pressure).
   - Each side uses its own VWAP anchor; using a shared `bid_vwap` for both would introduce a cross-side anchoring bias.
   - This logic may be reverted to a momentum strategy (BUY if `micro_price > ask_vwap`, SELL if `micro_price < bid_vwap`) if backtesting favours trend-following over mean-reversion.
7. **Real-time balance tracking (no listenKey)** — `OrderExecutor` owns the single `SpotWebsocketAPIClient` connection (`wss://testnet.binance.vision/ws-api/v3`).  On socket open (`on_open` callback) it sends a signed `session.logon` frame.  On success it immediately sends `userDataStream.subscribe`.  Once confirmed, Binance pushes `outboundAccountPosition` events on the **same** connection whenever a balance changes (e.g. after an order fill).  `handle_order_response` routes these push events (frames with no `"id"` field) to `_handle_balance_update`, which writes to `state.balance_status` under `thread_balance_lock`.  If the testnet doesn't support `session.logon` or `userDataStream.subscribe`, the executor falls back silently to the REST snapshot taken at session startup.
8. **HMM regime filter** — `websocket_main.py` instantiates `RegimeDirector` and calls `get_klines_data()` → `select_hmm_model()` → `assign_regime_labels()` **before** the analysis threads start, so `regime_label` and `regime_confidence` are never `None` on the first `low_latency_analysis` iteration.  Every `historical_analysis` iteration applies a **two-speed** update: cheap Viterbi prediction (`predict_current_regime()`) on most iterations, full model re-fit (`select_hmm_model()`) every `HMM_REFIT_INTERVAL` (300 s, i.e. every 5th iteration).  Features are **z-score scaled** via `StandardScaler` before every `fit()` / `predict()` / `predict_proba()` call — the scaler is fitted on the first `train_end` rows only (in-sample), where `train_end = max(2, int(n_rows × 2/3))`, and applied to the full window (out-of-sample) to avoid data leakage.  At the default 120-row window `train_end = 80` (~⅔ in-sample, ~⅓ out-of-sample); shorter windows scale proportionally.  `assign_regime_labels()` runs inside `_regime_lock` on every iteration.  `low_latency_analysis` reads both `regime_label` and `regime_confidence` under `_regime_lock` and applies **two sequential gates** before any order: (a) **confidence gate** — if `regime_confidence < HMM_MIN_CONFIDENCE` (0.60), both BUY and SELL are skipped regardless of label (the model is uncertain); (b) **direction gate** — BUY orders are suppressed in `"trending_down"` or `"high_volatility"` regimes; SELL orders are suppressed in `"trending_up"` or `"high_volatility"` regimes.

---

## Session Duration

### Rationale

Both analysis loops in `AnalysisEngine` are designed to run indefinitely:

| Loop | Cadence | Purpose |
|------|---------|---------|
| `low_latency_analysis` | every **1 s** | Near-real-time best bid/ask evaluation |
| `historical_analysis` | every **1 min** | VWAP computation over the rolling snapshot window |

Rather than running forever, `websocket_main.py` uses a fixed session duration set by `DEFAULT_SESSION_MINUTES` (no startup prompt).

The **default of 10 minutes** is chosen deliberately:

| Metric | Value at 10 min |
|--------|----------------|
| Low-latency iterations (`low_latency_analysis`) | $10 \times 60 / 1 = \mathbf{600}$ |
| Historical iterations (`historical_analysis`) | $10 \times 60 / 60 = \mathbf{10}$ |
| Order book snapshots in history | up to $10 \times 60 \times 10 = \mathbf{6{,}000}$ ticks (capped at `maxlen=3000` ≈ last 5 min) |

When the session duration elapses, `websocket_main.py` sets `stop_event`, calls `ws_client.stop()` to close the stream cleanly, and joins both analysis threads (with timeouts of 10 s and 15 s respectively). A `KeyboardInterrupt` (Ctrl-C) triggers the same shutdown path early.

> **Thread Timeline** — For a step-by-step t=0s … t=10 min walkthrough of both threads, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

### Deque Fill-Up (`history_order_book`)

The deque size is driven by the **WebSocket tick rate** (~10 entries/sec at 100 ms), not by the historical analysis interval.  `historical_analysis` only *reads* the deque — it never clears it.

| Time elapsed | WebSocket ticks | Deque size | Historical iterations |
|---|---|---|---|
| 1 min | ~600 | 600 | 1st runs (reads 600 entries) |
| 3 min | ~1 800 | 1 800 | 3rd runs (reads 1 800 entries) |
| 5 min | ~3 000 | **3 000 (full)** | 5th runs (reads 3 000 entries) |
| 10 min | ~6 000 sent | **3 000 (capped — oldest evicted)** | 10th runs (reads 3 000 entries) |
| 10 min | ~6 000 sent | **3 000 (capped)** | 10th runs (reads 3 000 entries) |

After ~5 minutes the deque hits `maxlen=3000` and becomes a true **rolling window** of the last ~5 minutes. Each `historical_analysis` iteration operates on whatever is currently in the window — not a fixed block.

---

## WebSocket Execution Flow (`websocket_main.py`)

> **Testnet SSL patch** — applied once at module level, before any client is
> created.  The Binance Spot Testnet serves a self-signed certificate chain
> that fails standard SSL verification.  Two layers are patched:
> 1. `urllib3.disable_warnings(InsecureRequestWarning)` — suppresses per-request noise.
> 2. `BinanceSocketManager.create_ws_connection` is monkey-patched to pass
>    `sslopt={"cert_reqs": ssl.CERT_NONE}` to every `websocket-client`
>    `create_connection()` call — covers both the market-data stream
>    (`SpotWebsocketStreamClient`) and the order/balance WebSocket
>    (`SpotWebsocketAPIClient` inside `OrderExecutor`).
> 3. `rest_client.session.verify = False` — disables SSL verification on the
>    `requests.Session` used by the REST client.

1. Load API keys from `.env`.
2. Connect to Binance Testnet via `binance-connector` REST client.
3. Consolidate non-BTC/USDT balances into USDT via market sell orders.
4. Seed `state.balance_status` with the REST-fetched `usdt_balance` and `btc_balance` before any thread starts.
5. Snapshot `btc_start_price` via `ticker_price(symbol)` and compute `start_total_usdt = usdt_balance + btc_balance × btc_start_price` — stored for the end-of-session P&L decomposition.
6. Session duration fixed at `DEFAULT_SESSION_MINUTES` (10 min) — no user prompt.
6. Fetch a depth snapshot for **BTCUSDT** (100 levels) to seed `OrderBookState.local_book`.
7. **Pre-session regime detection** — instantiate `RegimeDirector` and call `get_klines_data()` → `select_hmm_model()` → `assign_regime_labels()`.  This downloads ~120 rows of **5-minute klines** (last 10 hours) via `HMM_INTERVAL` / `HMM_LOOKBACK`, fits `GaussianHMM` models for 2–3 states (from `HMM_MAX_REGIMES`), selects the best by BIC, and assigns `regime_label` before any thread starts.
8. Instantiate `OrderBookState`, `MessageHandler`, `OrderExecutor`, and `AnalysisEngine` (with `regime_director` injected), wiring the shared state.  `OrderExecutor` creates its own `SpotWebsocketAPIClient` internally (connected to `wss://testnet.binance.vision/ws-api/v3`) with `on_open=self._on_ws_open` and `on_message=self.handle_order_response`.  On socket open it automatically sends `session.logon` → `userDataStream.subscribe` to enable real-time balance push events on the same connection.  Set `stop_event = threading.Event()`.
9. Open one `SpotWebsocketStreamClient` (`ws_client`) on the production stream endpoint; subscribe to `diff_book_depth` at 100 ms intervals; callback: `handle_depth_message`.
10. Wait 1 second for the first diff-depth messages to arrive and populate `local_book["bids"]`.
11. Start daemon threads:
    - `low_latency_thread` → `AnalysisEngine.low_latency_analysis`
    - `hist_thread` → `AnalysisEngine.historical_analysis`
12. On each incoming depth message (`MessageHandler.handle_depth_message`):
    - Skip the initial subscription confirmation (`{"id": 1, "result": null}`).
    - Drop stale updates where `data["u"] <= state.local_book["lastUpdateId"]`.
    - Acquire `state.thread_lock`, apply bid/ask deltas, update `lastUpdateId`, append snapshot (`{timestamp, lastUpdateId, best_bid, best_ask, volume_best_bid, volume_best_ask}`) to `state.history_order_book`, release lock.
    - Call `calculate_best_quote()` every `QUOTE_EVERY_N_TICKS` ticks (~1 s) with the updated book (prints live spread to stdout).  The local book is still updated on every tick.
13. On each incoming `outboundAccountPosition` push event (`OrderExecutor.handle_order_response` → `_handle_balance_update`):
    - Under `state.thread_balance_lock`, update `state.balance_status` for each tracked asset (`CRYPTOCCY`, `CCY`) using the `"f"` (free) field.
14. After `session_seconds`, set `stop_event`, stop `ws_client`, stop `executor` (closes the WS API + user-data connection), and join both analysis threads (10 s low-latency, 15 s historical).  A `KeyboardInterrupt` (Ctrl-C) triggers the same shutdown path early.  After threads exit, two reports are printed:
    - **Order status report** — queries every placed order via REST (`GET /api/v3/order`) and logs fill status.
    - **Balance report** — fetches final balances and `btc_end_price`, then prints a P&L decomposition:
      - **Trading alpha (A)** = `Δusdt + Δbtc × end_price` — the strategy's contribution, independent of BTC price movement.
      - **Price move (B)** = `btc_start × (end_price − start_price)` — gain/loss from the starting BTC position due to market movement.
      - **Total P&L** = A + B (with percentage return on starting portfolio value).

## Notation

| Symbol | Column name | Description |
|--------|-------------|-------------|
| $P_b$, $Q_b$ | `bid_price`, `bid_quantity` | Best bid price and quantity at a level |
| $P_a$, $Q_a$ | `ask_price`, `ask_quantity` | Best ask price and quantity at a level |
| $bq$ | `bq` | Individual bid quantity at a level (carried through the opportunity pipeline for order sizing — SELL uses this) |
| $aq$ | `aq` | Individual ask quantity at a level (carried through the opportunity pipeline for order sizing — BUY uses this) |
| $D$ | `total_depth` | Sum of bid and ask quantities |
| $P_{\text{mid}}$ | `mid_price` | Arithmetic mid-price |
| $P_{\mu}$ | `micro_price` | Volume-weighted micro-price |
| $\text{OBI}$ | `obi` | Order book imbalance |
| $S$ | `bid_ask_spread` | Relative bid-ask spread |
| $\mathbb{1}_{\mu > \text{mid}}$ | `micro_vs_mid` | True when micro-price exceeds mid |
| $\Delta_{\mu}$ | `micro_mid_delta` | Directional micro–mid delta |
| $\mathbb{1}_{\text{thin}}$ | `is_thin_micro_effect` | True when depth is below median |
| $\mathbb{1}_{D \geq 50\%}$ | `is_total_depth_50pct_l0` | True when depth ≥ 50 % of level-0 |

---

## Metrics (`metrics.py`)

`get_order_book_metrics()` enriches a raw order book DataFrame with the columns below.

Let $P_b$, $Q_b$ denote the bid price and quantity and $P_a$, $Q_a$ the ask price and quantity at a given level.

### Total Depth

$$D = Q_b + Q_a$$

### Mid-Price

$$P_{\text{mid}} = \frac{P_b + P_a}{2}$$

### Order Book Imbalance (OBI)

$$\text{OBI} = \frac{Q_b - Q_a}{Q_b + Q_a}$$

Values range in $[-1, 1]$.  
$\text{OBI} > 0$ → excess bid-side liquidity; $\text{OBI} < 0$ → excess ask-side liquidity.

### Micro-Price

$$P_{\mu} = \frac{P_b \cdot Q_a + P_a \cdot Q_b}{Q_b + Q_a}$$

A volume-weighted fair price that shifts towards the side with **less** resting quantity (i.e. the side more likely to be consumed).

### Micro vs Mid (`micro_vs_mid`)

$$\mathbb{1}_{\mu > \text{mid}} = \begin{cases} \text{True}  & \text{if } P_{\mu} > P_{\text{mid}} \\ \text{False} & \text{otherwise} \end{cases}$$

When $P_{\mu} > P_{\text{mid}}$, buy-side pressure is implied (ask quantity is thinner than bid quantity).

### Bid-Ask Spread (relative)

$$S = \frac{P_a - P_b}{P_{\text{mid}}}$$

### Spread Flags

| Flag | Condition | Decimal threshold |
|------|-----------|-------------------|
| `is_large_spread` | $S > 0.10\%$ | $S > 0.001$ |
| `is_small_spread` | $S \leq 0.02\%$ | $S \leq 0.0002$ |

---

## Strategy Indicators (`indicators.py`)

`add_strategy_indicators(df, strategy)` adds directional columns depending on the chosen strategy.

### Micro–Mid Delta

$$\Delta_{\mu} = \begin{cases} P_{\mu} - P_{\text{mid}} & \text{if strategy = buy} \\ P_{\text{mid}} - P_{\mu} & \text{if strategy = sell} \end{cases}$$

A positive $\Delta_{\mu}$ signals that the micro-price diverges from mid in the direction favourable to the strategy.

### Thin Micro Effect (`is_thin_micro_effect`)

$$\mathbb{1}_{\text{thin}} = \begin{cases} \text{True}  & \text{if } D < \widetilde{D} \\ \text{False} & \text{otherwise} \end{cases}$$

where $\widetilde{D}$ is the **median** total depth across all levels in the snapshot.  
When `True`, the level's depth is below median, meaning the micro-price signal may be unreliable (thin book artefact).

### Depth Adequacy (`is_total_depth_50pct_l0`)

$$\mathbb{1}_{D \geq 50\%} = \begin{cases} \text{True}  & \text{if } D \geq 0.5 \cdot D_0 \\ \text{False} & \text{otherwise} \end{cases}$$

where $D_0$ is the total depth at level 0 (best bid/ask).  
Ensures the level carries at least 50 % of the top-of-book liquidity.

---

## Opportunity Scoring (`scores.py`)

`get_weighted_volume_micro_spread_score()` produces a composite score per level:

$$\text{Score} = 0.7 \cdot \frac{D}{D_{\max}} + 0.3 \cdot \frac{\Delta_{\mu}}{\Delta_{\mu,\max}}$$

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Normalised depth $\frac{D}{D_{\max}}$ | 70 % | **Safety** — prefer levels with substantial liquidity |
| Normalised delta $\frac{\Delta_{\mu}}{\Delta_{\mu,\max}}$ | 30 % | **Aggression** — prefer levels where the micro-price divergence is largest |

---

## Historical VWAP & Dip/Strength Filter (`indicators.py` + `analysis.py`)

### Volume-Weighted Average Price (VWAP)

`historical_analysis()` computes two VWAPs from the rolling `history_order_book` deque (up to 3 000 snapshots ≈ last 5 min of ticks):

$$\text{VWAP}_{\text{bid}} = \frac{\displaystyle\sum_{i=1}^{N} P_{\text{bid},i} \cdot V_{\text{bid},i}}{\displaystyle\sum_{i=1}^{N} V_{\text{bid},i}}$$

$$\text{VWAP}_{\text{ask}} = \frac{\displaystyle\sum_{i=1}^{N} P_{\text{ask},i} \cdot V_{\text{ask},i}}{\displaystyle\sum_{i=1}^{N} V_{\text{ask},i}}$$

where $P_{\text{bid},i}$, $V_{\text{bid},i}$ (respectively $P_{\text{ask},i}$, $V_{\text{ask},i}$) are the best bid (ask) price and quantity recorded at tick $i$, and $N$ is the number of snapshots currently in the deque.

The helper `volume_weighted_average_price(price, volume)` in `indicators.py` implements this as:

```python
float(np.sum(price * volume) / np.sum(volume))
```

Both VWAPs are published under `_vwap_lock` so the low-latency thread can read them safely.

### Dip/Strength Confirmation Filter (Mean-Reversion)

After the first `historical_analysis` iteration (≈ 1 min into the session), `_bid_vwap` and `_ask_vwap` are populated.  `low_latency_analysis` reads both on every iteration and applies a **dip/strength confirmation gate** with a symmetric dead zone (`VWAP_THRESHOLD_MULTIPLIER` δ, default 0.002) before sending an order:

| Side | Anchor | Condition to execute | Interpretation |
|------|--------|---------------------|----------------|
| **BUY** | `bid_vwap` | `_bid_vwap is None` **or** `micro_price < bid_vwap × (1 − δ)` | Dip deep enough below volume-weighted bid pressure to cover fees and leave profit |
| **SELL** | `ask_vwap` | `_ask_vwap is None` **or** `micro_price ≥ ask_vwap × (1 + δ)` | Rally strong enough above volume-weighted ask pressure to cover fees and leave profit |

$$\text{VWAP gate (BUY):} \quad P_{\mu} < \text{VWAP}_{\text{bid}} \times (1 - \delta)$$
$$\text{VWAP gate (SELL):} \quad P_{\mu} \geq \text{VWAP}_{\text{ask}} \times (1 + \delta)$$

where $\delta$ = `VWAP_THRESHOLD_MULTIPLIER` (default 0.002 = 0.20 %).  Signals within ±δ of the respective VWAP are rejected as micro-noise that cannot cover the round-trip fee.

Each side is anchored to its own reference price: `bid_vwap` tracks volume-weighted bid pressure (the BUY benchmark); `ask_vwap` tracks volume-weighted ask pressure (the SELL benchmark).  Using a shared `bid_vwap` for both sides would introduce a cross-side anchoring bias.

While `_bid_vwap` / `_ask_vwap` are still `None` (first ~1 min) the filter is transparent and orders execute based on the opportunity score and regime alone.

### ⚠️ Strategy choice: mean-reversion vs momentum

The current dip/strength filter implements a **buy-the-dip / sell-the-strength** (mean-reversion) strategy: buy when the price falls below the historical average, sell when it rises above it.

The **alternative** is a pure-momentum strategy: buy when `micro_price > ask_vwap` (upward momentum above cost-to-buy) and sell when `micro_price < bid_vwap` (falling below historical bid).  Switching between the two strategies requires changing only the VWAP comparison in `low_latency_analysis` (and the matching gate in `backtest/signals.py`).  The best approach depends on the prevailing market regime and is subject to further backtesting.

> For the full threading walkthrough (how `historical_analysis` publishes VWAPs and `low_latency_analysis` reads them under `_vwap_lock`) see **[REGIME_DIRECTOR.md § A.3](REGIME_DIRECTOR.md#a3-inside-analysisengine--strategyanalysispy)**.

---

## HMM Regime Detection (`strategy/regime_director.py`)

> For a full file-by-file trace — startup sequence, threading model, data-flow diagram, and regime label reference — see **[REGIME_DIRECTOR.md](REGIME_DIRECTOR.md)**.

`RegimeDirector` trains a `GaussianHMM` on recent Binance klines to classify the current market into one of several **hidden regimes** and exposes the result as a plain string label that gates order execution in `low_latency_analysis`.

### Theoretical Background

> 📖 **Reference:** The HMM theory is explained with examples in this paper by
> Jurafsky, D. & Martin, J.H. — *Speech and Language Processing*, 3rd ed.,
> Appendix A — *Hidden Markov Models*.
> [Read online (Stanford)](https://web.stanford.edu/~jurafsky/slp3/A.pdf) **[1]**

A **Hidden Markov Model (HMM)** **[1]** is a statistical model for systems that move through a finite set of states over time, where those states are **not directly observable** — they are _hidden_.  What you _can_ observe at each time step is a signal that depends probabilistically on whichever hidden state the system is currently in.

In this project:

- The **hidden states** are the market regimes (`trending_up`, `trending_down`, `high_volatility`, `neutral`).  You cannot observe the regime directly; you can only infer it from market data.
- The **observations** are the four feature vectors computed from 1-minute klines: `[return, volatility, obi_proxy, trade_density]`.

An HMM is fully described by three components:

#### 1. Transition Probability Matrix $A$

$$A_{ij} = P(q_{t+1} = s_j \mid q_t = s_i)$$

The probability of moving from regime $s_i$ at time $t$ to regime $s_j$ at time $t+1$.  Stored in `model.transmat_` (shape $n \times n$, rows sum to 1).

In this context: if the market is currently in `trending_up`, $A$ encodes how likely it is to stay there vs. transition to `neutral` or `high_volatility` on the next candle.  A high diagonal value means the regime is **persistent**; a low diagonal value means it is **short-lived**.

#### 2. Emission Probability $B_i(o)$

Because we use a **Gaussian HMM**, the emission probability for state $i$ is a multivariate Gaussian over the observation vector:

$$B_i(o) = \mathcal{N}(o\,;\,\mu_i,\,\Sigma_i)$$

where $\mu_i$ is the mean feature vector and $\Sigma_i$ is the covariance matrix for state $i$.  Stored in `model.means_` and `model.covars_`.

In this context: given that the market is in regime $i$, $B_i$ describes the distribution of `[return, volatility, obi_proxy, trade_density]` values we expect to observe.  For example, a `trending_up` state would have a $\mu_i$ with a positive `return` and a positive `obi_proxy` — this is exactly what `assign_regime_labels()` reads to assign human-readable names.

#### 3. Initial Probability $\pi$

$$\pi_i = P(q_1 = s_i)$$

The probability of starting in each regime at $t = 1$.  In practice the model learns this from data and it has minimal effect after a sufficiently long observation sequence.

---

#### Learning and Inference

| Step | Algorithm | hmmlearn call | What it does |
|---|---|---|---|
| **Training** | Baum-Welch (Expectation-Maximisation) | `model.fit(features)` | Learns $A$, $B$ ($\mu_i$, $\Sigma_i$), $\pi$ from the kline feature matrix |
| **Inference** | Viterbi | `model.predict(features)` | Finds the most likely hidden state sequence $q_1, q_2, \ldots, q_T$ given the observed features |
| **Model selection** | BIC | `model.bic(features)` | Penalises model complexity; used to choose the best $n$ (number of regimes) |

The last element of the predicted sequence, `regimes[-1]`, corresponds to the **most recent candle** — this becomes `current_regime` and is what `assign_regime_labels()` uses to set `regime_label`.

### Features

| Feature | Formula | Interpretation |
|---|---|---|
| `return` | `close.pct_change()` | Per-candle price momentum |
| `volatility` | `(high - low) / close` | Normalised intra-bar price swing |
| `obi_proxy` | `(taker_buy_base_vol / volume) × 2 − 1` | Taker-flow imbalance proxy ∈ `[-1, +1]`; approximates live OBI from kline data |
| `trade_density` | `num_trades / volume` | Trade fragmentation: high → many small trades (retail/HFT); low → large blocks (institutional) |

### Feature Scaling

Before any `fit()`, `predict()`, or `predict_proba()` call, the four HMM features are **z-score normalised** via `sklearn.preprocessing.StandardScaler`:

- The scaler is **`fit_transform`'d on the first `train_end` rows only** — where `train_end = max(2, int(n_rows × 2/3))`, i.e. the older in-sample ~⅔ of the 10-hour window.  At the default 120-row window `train_end = 80`; the value scales proportionally for shorter windows.
- The same scaler is then **`transform`'d onto the full window** for prediction, so the most recent ~⅓ of rows are genuinely out-of-sample.
- Without scaling, `trade_density` (`num_trades / volume`) can be orders of magnitude larger than `return` or `obi_proxy` (both near `[-1, +1]`), which would distort the covariance structure and bias BIC selection.

### Model Selection (BIC)

`GaussianHMM` models are fitted for `n = 2 … HMM_MAX_REGIMES` (default 3) hidden states with full covariance.  The best model is selected by **Bayesian Information Criterion**:

$$\text{BIC} = -2 \ln \hat{L} + k \ln N$$

where $\hat{L}$ is the model likelihood, $k$ the number of free parameters, and $N$ the number of **training** observations (`train_end = max(2, int(n_rows × 2/3))`).  The model with the **lowest BIC** is retained.

### Two-Speed Update

To avoid re-training the full model on nearly identical data every 60 s, `historical_analysis()` uses a two-speed scheme:

| Cadence | Method called | Cost | When |
|---|---|---|---|
| Every `HIST_INTERVAL` (60 s) | `predict_current_regime()` | O(n × k) — single Viterbi pass | Iterations 1, 2, 3, 4, 6, 7, … |
| Every `HMM_REFIT_INTERVAL` (300 s) | `select_hmm_model()` | O(n × k × `HMM_N_ITERATIONS`) per candidate — expensive | Iterations 5, 10, 15, … |

`assign_regime_labels()` runs inside `_regime_lock` on **every** iteration regardless of path.

### Regime Confidence

After each `predict()` call, `predict_proba(features_scaled)` is called to obtain the **posterior state probabilities** (Forward-Backward algorithm) for the full window.  The probability of the current (latest) candle's state is stored as `regime_confidence`:

$$\text{regime\_confidence} = P(\text{state} = \text{current\_regime} \mid \text{observations})$$

| `regime_confidence` | Interpretation |
|---|---|
| ≥ `HMM_MIN_CONFIDENCE` (0.60) | Model is confident — gates apply normally |
| < `HMM_MIN_CONFIDENCE` | Model is uncertain (e.g. 55 % vs 45 %) — **both BUY and SELL are skipped** |
| `None` (warm-up) | No model fitted yet — gate is transparent |

### State Labelling

State integers (0, 1, 2, …) carry no inherent meaning.  `assign_regime_labels()` assigns labels using a **rank-based** directional scheme and threshold-based secondary rules — no hard-coded price constants:

**Directional labels** — guaranteed to be unique (exactly one each):

A combined **direction score** is computed per state:

$$\text{direction\_score}_i = \text{rank}(\text{return}_i) + \text{rank}(\text{obi\_proxy}_i)$$

| Assignment | Rule |
|---|---|
| `"trending_up"` | State with the **highest** direction score (most bullish: highest return + strongest buy-side flow) |
| `"trending_down"` | State with the **lowest** direction score (most bearish: lowest return + weakest flow) |

Because `idxmax()` and `idxmin()` are exclusive by definition, duplicate directional labels are structurally impossible regardless of `n_components`.

**Secondary labels** — applied to remaining states via volatility and trade-density thresholds:

| Flag | Condition |
|---|---|
| `high_vol` | `volatility > cross-state mean + 1 × std` |
| `high_td` | `trade_density > cross-state mean + 0.5 × std` |

| Condition | Label | Rationale |
|---|---|---|
| `high_vol` **or** `high_td` | `"high_volatility"` | Either large price swings (unpredictable fills) **or** heavy trade fragmentation (noisy, no directional intent) makes the market **unreliable to trade in** |
| *(default)* | `"neutral"` | No dominant signal |

> **Why OR?** Both `high_vol` and `high_td` independently indicate an unreliable market — one from the price side (large swings), the other from the flow side (fragmented activity). Using OR ensures `trade_density` has a meaningful role in the label assignment rather than being used only for model training.

### Regime Gates in `low_latency_analysis`

Two gates are applied **sequentially**:

**Gate 1 — Confidence** (evaluated first, before any label check):

| `regime_confidence` | Result |
|---|---|
| `None` (before first historical run) | ✅ transparent — all orders allowed |
| ≥ `HMM_MIN_CONFIDENCE` (0.60) | ✅ proceed to direction gate |
| < `HMM_MIN_CONFIDENCE` | ❌ **both** BUY and SELL skipped |

**Gate 2 — Direction** (evaluated only if confidence gate passed):

| Regime | BUY | SELL |
|---|---|---|
| `"trending_up"` | ✅ allowed | ❌ suppressed |
| `"trending_down"` | ❌ suppressed | ✅ allowed |
| `"high_volatility"` | ❌ suppressed | ❌ suppressed |
| `"neutral"` | ✅ allowed | ✅ allowed |
| `None` (before first historical run) | ✅ transparent | ✅ transparent |

### Session Timeline with HMM

```
websocket_main.py startup
│
├── Pre-session: RegimeDirector.get_klines_data()     ← fetches last 2 h of 1-min klines (~120 rows)
│               StandardScaler.fit_transform(rows[:80]) ← scale training rows in-sample
│               RegimeDirector.select_hmm_model()     ← fits HMM n=2..4, selects best BIC
│               RegimeDirector.assign_regime_labels() ← sets regime_label + regime_confidence
│               Both are set BEFORE threads start
│
├── low_latency_analysis — every 1 s
│   └── reads regime_label + regime_confidence under _regime_lock  ← never None
│       ├── Gate 1: skip if regime_confidence < HMM_MIN_CONFIDENCE
│       └── Gate 2: skip buy/sell based on regime_label direction
│
└── historical_analysis — every 60 s
    ├── compute bid_vwap / ask_vwap                           ← existing VWAP logic
    ├── check UTC clock boundary:
    │     now = int(time.time())
    │     current_5m_boundary = now - (now % 300)
    │     if current_5m_boundary > _last_hmm_boundary:
    │       get_klines_data()                                 ← refresh features from Binance
    │       if hmm_iteration % hmm_refit_every == 0:
    │         select_hmm_model()                              ← full BIC re-fit + new scaler
    │       else:
    │         predict_current_regime()                        ← cheap Viterbi + predict_proba
    └── assign_regime_labels() under _regime_lock             ← label + confidence write (every boundary)
```

---

## Backtesting

A backtesting framework has been built to replay the live strategy against
historical 5-minute BTCUSDT klines using a clean **in-sample / out-of-sample
(IS/OOS) split**:

| Window | Config constant | Rows at 5 m | Used by |
|---|---|---|---|
| IS: 360 days ago → 90 days ago (270 days) | `BACKTEST_LOOKBACK` → `BACKTEST_OOS_START` | ~77,760 | `sensitivity.py` (parameter tuning) |
| OOS: 90 days ago → today (90 days) | `BACKTEST_OOS_START` | ~25,920 | `runner.py` (validation) |

Because Binance does
not expose historical Level-2 order book data, a **synthetic 50-level depth
ladder** is reconstructed from each kline's OHLCV data and taker volume split,
then fed through the **same production scoring pipeline** used by
`low_latency_analysis()`.

Three independent data flows — opportunity scoring (synthetic book), HMM regime
filter (kline features), and VWAP dip/strength filter (level-0 rolling window) —
are combined into a single signal per candle, exactly mirroring the live
architecture.

All design decisions, pseudo-code, data-flow diagrams, approximation caveats,
and a step-by-step implementation roadmap with progress tracking are documented
in **[`BACKTESTING.md`](BACKTESTING.md)**.

| Module | Status | Purpose |
|---|---|---|
| `backtest/data.py` | ✅ done | Download historical klines — `fetch_macro_klines()` (5m, HMM) + `fetch_micro_klines()` (1m, PnL); Parquet cache (`cache/klines/`, 24h TTL, `--flush-cache`); IS/OOS boundary enforced via `end_str` / `lookback` |
| `backtest/synthetic_book.py` | ✅ done | Build synthetic 50-level order book per candle |
| `backtest/signals.py` | ✅ done | Two-frame signal pipeline: Phase 1 HMM walk-forward on 5m macro frame, Phase 2 `merge_asof` stitch (zero look-ahead), Phase 3 1m execution loop + confidence/regime/VWAP gates → signal +1/−1/0 |
| `backtest/pnl.py` | ✅ done | Simulated P&L — **intra-candle whipsaw guard** (`SELL_WHIPSAW` at `low−half_spread` when same 1m bar hits BUY+SELL zones; `n_whipsaw_exits` in stats); **single-position mean-reversion guard** (`open_strategy_qty` / `_POSITION_DUST_BTC = 1e-6`): BUY fires only when flat, SELL closes the full open position in one shot; balance guard; bps-based `half_spread` fill (`BACKTEST_FILL_SPREAD_BPS`); per-trade position cap (`MAX_POSITION_PCT`, shared with live `execution/order_executor.py`); equity curve; FIFO round-trip pairing; explicit taker-fee deduction in `_pair_round_trips`; Step 5 metrics |
| `backtest/runner.py` | ✅ done | Top-level orchestration — fetches **OOS window** (`BACKTEST_OOS_START → today`, ~25,920 rows at 5 m / ~129,600 rows at 1 m) via `fetch_macro_klines` + `fetch_micro_klines` (cached); chains all modules, delegates report/CSV to `reporting/`; **plot is ON by default** (use `--no-plot` to suppress in headless/CI mode), `--save-png`, and `--flush-cache` flags.  Loads `fee_rate` from `best_params.json` and passes to `simulate_pnl()` |
| `backtest/reporting/formatters.py` | ✅ done | Console report formatting (`print_report`, `print_regime_validation_report`), sensitivity report helpers (`print_sensitivity_table`, `print_oat_sensitivity_report`, `print_bnh_comparison`), and CSV export (`save_csv`) — AI-authored |
| `backtest/diagnostics/regime_validation.py` | ✅ done | Offline long-horizon regime validation — **70/30 train-test split** on 2 years (~210,000 rows, `VALIDATION_LOOKBACK = "730 days ago UTC"`), self-contained (no `RegimeDirector`), fits HMM on full train set, **vectorised** single-pass Viterbi on ~63,000 test candles, six statistical checks, `python -m backtest.diagnostics.regime_validation` |
| `backtest/visualization.py` | ✅ done | Interactive six-panel Plotly chart — equity curve, drawdown, BUY/SELL markers + VWAP lines, regime step-line + scaled confidence + colour bands, VWAP vs micro-price + near-miss dots, signal funnel, signals-by-regime |
| `backtest/sensitivity.py` | ✅ done | **Bayesian optimisation via Optuna TPE (default, 40 trials)**, OAT sweep (`--oat`, 8 runs), and deprecated full-grid (`--full-grid`) over `HMM_LOOKBACK_ROWS`, `HMM_MAX_REGIMES`, `VWAP_WINDOW`, and `VWAP_THRESHOLD_MULTIPLIER` on the **IS window** (270 days, ~77,760 5m / ~388,800 1m rows).  `fee_rate` fixed at `SENSITIVITY_FEE_RATE` (0.001).  `--lookback` flag overrides IS start.  `--flush-cache` clears the Parquet cache.  Writes `best_params.json` — loaded by `websocket_main.py` (live) and `runner.py` (backtest). Use Case B deferred. |

> **Parameter Flow**
> ```
> sensitivity.py  ──►  best_params.json  ──►  strategy/param_loader.py
>                                                   ├── load_best_params()              → websocket_main.py  (live)
>                                                   └── load_best_params_for_backtest() → run_backtest.py    (backtest)
> ```
> Both consumers fall back silently to `config_parameters.py` defaults if
> `best_params.json` is absent.  Do **not** commit `best_params.json` to git
> (add `backtest/results/best_params.json` to `.gitignore`).

---

### Bayesian Parameter Optimisation (Optuna)

`backtest/sensitivity.py` uses the **Optuna TPE (Tree-structured Parzen
Estimator)** sampler to find the parameter combination that maximises the
Sharpe ratio over the **IS window** (`BACKTEST_LOOKBACK = "360 days ago UTC"` → `BACKTEST_OOS_START = "90 days ago UTC"`, 270 days, ~77,760 5m macro rows / ~388,800 1m micro rows).

**Why Bayesian, not full-grid?**

| Approach | Combinations / Trials | Typical wall time | Search space |
|---|---|---|---|
| Full-grid (deprecated) | ~36 fixed combos | ~70–180 min | Discrete, narrow |
| OAT sweep | 8 runs | ~1–2 h | One param at a time |
| **Bayesian / Optuna (default)** | **40 trials (configurable)** | **~3–6 h** | **Continuous, wider** |

The TPE sampler builds a probabilistic surrogate of the objective (Sharpe ratio)
and focuses new trials in regions where it predicts high reward — equivalent to
guided grid search but far more efficient in high-dimensional spaces.

**Search space (`_OPTUNA_SPACE`):**

| Parameter | Low | High | Step |
|---|---|---|---|
| `hmm_lookback_rows` | 30 | 240 | 10 |
| `hmm_max_regimes` | 2 | 4 | 1 |
| `vwap_window` | 5 | 60 | 5 |
| `vwap_threshold` | 0.001 | 0.005 | 0.0005 |

`fee_rate` is always fixed at `BACKTEST_FEE_RATE = 0.001` — it is a fixed
exchange cost, not a strategy knob.

**Key design details:**

- Data pre-fetched **once** before the study; shared across all trials via a
  `_make_objective(prefetched_df)` factory closure — no redundant Binance API
  calls during the study.
- Study persisted to `backtest/results/optuna.db` (`load_if_exists=True`) —
  an interrupted run **resumes automatically** from the last completed trial.
- `_check_existing_best_params()` guard inspects any existing `best_params.json`
  (age, Sharpe, params) before every run and prompts `[y/N]` for confirmation.
- `--lookback "180 days ago UTC"` overrides `BACKTEST_LOOKBACK` (IS start) for a
  single deep-calibration run without touching `config_parameters.py`.  The IS end
  (`BACKTEST_OOS_START`) is always fixed to preserve the OOS boundary.

**Running the sensitivity analysis:**

```bash
# Bayesian (default) — 40 Optuna trials on IS window (~3–6 h)
python -m backtest.sensitivity

# Bayesian with custom trial count
# Optuna HTML charts (history, importance, contour) are always saved
# automatically to backtest/reporting/ — no flag needed.
python -m backtest.sensitivity --bayes --n-trials 50

# Override IS start (IS end always = BACKTEST_OOS_START)
python -m backtest.sensitivity --lookback "720 days ago UTC"

# OAT sweep (Phase 1, quick sanity check)
python -m backtest.sensitivity --oat

# Full-grid (deprecated — prefer --bayes)
python -m backtest.sensitivity --full-grid
```

> For a full description of all execution modes, the `_OPTUNA_SPACE` definition,
> the `_check_existing_best_params()` guard logic, optional Plotly charts, and
> step-by-step implementation notes, see **[BACKTESTING.md — Step 8](BACKTESTING.md#step-8--sensitivity-analysis)**.

**Running the backtest:**

```bash
# Console report only (default)
python -m backtest.runner

# Console report + open interactive Plotly chart in browser
python -c "from backtest.runner import run_backtest; run_backtest(plot=True)"

# Console report + chart + save PNG (requires kaleido) or HTML fallback
python -c "from backtest.runner import run_backtest; run_backtest(plot=True, save_png=True)"

# Console report + export trade/equity CSVs
python -c "from backtest.runner import run_backtest; run_backtest(export_csv=True)"
```

> ⚠️ **Important — re-run after any change to `strategy/regime_director.py`**  
> `regime_validation.py` is the **sanity check** for the HMM regime filter.  
> Whenever `RegimeDirector` logic is modified (feature columns, BIC search range,
> label-assignment rules, confidence threshold, etc.) the validation tool **must**
> be re-run to confirm that the frozen model still produces statistically meaningful
> labels on out-of-sample data.  A failing check (especially Check 1 — direction
> test, or Check 2 — Kruskal-Wallis H-test) is a strong signal that the change broke the
> regime filter's discriminative power and should be reviewed before deploying live.
>
> ```bash
> python -m backtest.diagnostics.regime_validation
> ```

---

## Appendix A — `RegimeDirector` Deep Dive

For a full file-by-file trace of `RegimeDirector` — configuration constants,
pre-session startup, threading model, data-flow diagram, regime label reference,
and threading safety summary — see:

**[REGIME_DIRECTOR.md](REGIME_DIRECTOR.md)**

---
---

## References

**[1]** Jurafsky, D. & Martin, J.H. (2024). *Speech and Language Processing*, 3rd edition, Appendix A — *Hidden Markov Models*. Stanford University. Available at: <https://web.stanford.edu/~jurafsky/slp3/A.pdf>

> This appendix provides the formal definitions of the Transition Probability
> Matrix $A$, the Emission Probability $B$, the Initial Probability $\pi$, the
> Baum-Welch (EM) training algorithm, and the Viterbi decoding algorithm that
 > underpin the `RegimeDirector` implementation in `strategy/regime_director.py`.
> Recommended reading for anyone who wants to follow the HMM theory described
> in the *HMM Regime Detection* section and in **[REGIME_DIRECTOR.md](REGIME_DIRECTOR.md)**.

---

*Document last updated: 2026-05-17*

