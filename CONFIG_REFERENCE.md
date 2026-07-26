# Configuration Reference (`config_parameters.py`)

Complete reference for all tunable constants. Default values are as of 2026-07-22.

## Live System Parameters

| Constant | Default | Purpose |
|---|---|---|
| `SYMBOL` | `"BTCUSDT"` | Trading pair |
| `HFT_INTERVAL` | 1 s | Low-latency analysis cadence |
| `HIST_INTERVAL` | 60 s | Historical analysis + VWAP update cadence |
| `N_LEVELS` | 50 | Order book depth (50 bids + 50 asks) |
| `DEFAULT_SESSION_MINUTES` | 60 | Session duration (live trading) |
| `BALANCE_REFRESH_INTERVAL` | 60 s | Cadence of the driver-side REST balance-refresh daemon. Active only in REST-only mode (WS user-data push down): polls `account()` so balances and the equity chart stay current during idle stretches. No-op when the WS push is healthy. |
| `FLATTEN_ON_START` | `False` | Startup inventory policy. `True` = MARKET-sell inherited BTC so the session starts flat (matches `BACKTEST_INITIAL_BTC = 0`; per-session skill test, report component B ≡ 0). `False` = carry inherited BTC across restarts; the position guard pre-arms on it and report component B attributes its market drift. While `False`, a carried position's stop-loss is anchored at the session-start price until position persistence (Phase 2) is added. |
| `DEPTH_RESYNC_MIN_INTERVAL_SEC` | 2.0 s | Minimum interval between local-book REST resyncs after a diff-depth gap/reconnect. When the diff stream drops an event, the next event's first update ID `U` exceeds `lastUpdateId + 1`; `MessageHandler` re-pulls a fresh `depth()` snapshot to rebuild `local_book`. The cooldown prevents a resync storm on a burst of gapped events (the book still recovers on the next event after the interval). |

## HMM Regime Detection

| Constant | Default | Purpose |
|---|---|---|
| `HMM_INTERVAL` | 5-minute klines | Regime detection granularity |
| `HMM_LOOKBACK` | "10 hours ago UTC" | Rolling window (~120 bars at 5 m); stable EM convergence |
| `HMM_MAX_REGIMES` | 3 | Upper bound on hidden states (BIC search 2–3) |
| `HMM_MIN_CONFIDENCE` | 0.60 | Gate: block orders if regime confidence < 60% posterior |
| `REGIME_DIRECTIONAL_RETURN_THRESHOLD` | 0.0005 | Min abs. mean log-return (per 5 m bar) for a state to earn a directional label; inside ±this a state is `neutral`/`high_volatility` (BUY-eligible), so a flat market is not mis-tagged `trending_down` |
| `HMM_N_ITERATIONS` | 1000 | Max EM iterations per model fit |
| `HMM_RANDOM_STATE` | 46 | Random seed for reproducible state numbering |
| `HMM_MIN_COVAR` | 1e-1 | Regularisation floor for covariance matrices |
| `HMM_N_INIT` | 5 | Random restarts per candidate `n_components` |
| `HMM_REFIT_INTERVAL` | 300 s | Full re-fit cadence (every 5 min at 5 m bars) |
| `HMM_FEATURE_COLS` | `["return", "volatility", "obi_proxy", "trade_density"]` | Features fed to GaussianHMM |

## Strategy Filters

| Constant | Default | Purpose |
|---|---|---|
| `VWAP_THRESHOLD_MULTIPLIER` | 0.002 (0.20 %) | Dead-zone threshold for VWAP mean-reversion |
| `VWAP_WINDOW` | 5 | 1-minute rolling window for VWAP (5 min total) |
| `CANDIDATE_MEDIAN_FRAC` | 0.5 | `collect_candidates()` thin-book filter: keep a book level only if `depth ≥ FRAC × median_depth`. Lower = more permissive. **Currently redundant** — every thin-book rejection is already a relative-depth rejection (thin ⊆ depth), so this gate is inert at the current `CANDIDATE_DEPTH_FRAC`. Shared live + backtest (`strategy/book_utils.py`). |
| `CANDIDATE_DEPTH_FRAC` | 0.10 | `collect_candidates()` relative-depth filter: keep a book level only if `depth ≥ FRAC × level_0_depth`. **Sole binding liquidity gate.** Loosened 0.5 → 0.25 → 0.10; live instrumentation (2026-07-18) showed 0.25 rejected 98.2 % of levels, starving 83 % of buy ticks of candidates. Lower = more candidates live but thinner/worse-quality levels and more backtest trades. Shared live + backtest (`strategy/book_utils.py`). |
| `CANDIDATE_FILTER_DEBUG` | `False` | Diagnostic only: when `True`, `collect_candidates()` accumulates per-gate reject counts and logs a cumulative `[cand-filter]` summary every `CANDIDATE_FILTER_DEBUG_EVERY` (300) calls, revealing which gate starves candidates. Read-only (does not change filtering). Set `False` before backtests — it fires ~430 log lines per `runner.py` run. |
| `TREND_CONSECUTIVE_BARS` | 3 | Consecutive same-direction bars to trigger trend-pause |
| `TREND_COOLDOWN_BARS` | 4 | Bars to maintain paused state after trend ends |
| `STOP_LOSS_ROLLING_DAYS` | 90 | Rolling window for stop-loss volatility calculation |
| `STOP_LOSS_STD_MULT` | 2.0 | Volatility multiplier for stop-loss distance |
| `MACRO_TREND_ENABLED` | `True` | Master switch for the symmetric macro-trend overlay (`down`→cash, `up`→hold & ride, `neutral`→trade). `False` = fully inert (ablation baseline) |
| `MACRO_TREND_SMA_DAYS` | 20 | Daily-close SMA window (days) anchoring the macro-trend state |
| `MACRO_TREND_SLOPE_DAYS` | 5 | SMA slope lookback (days): `sign(SMA − SMA.shift(N))` |
| `MACRO_TREND_BAND_PCT` | 0.02 | ±band dead-zone around the SMA before a `down`/`up` trend fires |

## Backtesting Parameters

| Constant | Default | Purpose |
|---|---|---|
| `BACKTEST_MACRO_INTERVAL` | "5m" | HMM regime frame (5-minute) |
| `BACKTEST_MICRO_INTERVAL` | "1m" | Execution frame (1-minute) |
| `BACKTEST_LOOKBACK` | "360 days ago UTC" | In-sample window start for sensitivity tuning |
| `BACKTEST_OOS_START` | "90 days ago UTC" | Out-of-sample boundary (IS/OOS cutoff) |
| `BACKTEST_INITIAL_BTC` | 0.0 | Starting BTC balance. Set to `0.0` to avoid orphan SELL signals consuming a pre-existing balance before the first strategy BUY. The equivalent BTC value is folded into `BACKTEST_INITIAL_CAPITAL`. |
| `BACKTEST_INITIAL_CAPITAL` | 315,000 USDT | Starting USDT balance (~250k USDT + 1 BTC @ ~65k), mirroring the live paper-trading account. |
| `BACKTEST_FEE_RATE` | 0.001 | Taker fee per side (0.10 %) |
| `MAX_POSITION_PCT` | 0.20 | Per-BUY-leg cap: at most 20 % of *available* USDT spent per BUY signal. Legs may pyramid (stack) — see `MIN_CASH_RESERVE_PCT`. Shared live + backtest. |
| `MIN_CASH_RESERVE_PCT` | 0.30 | Cash-reserve floor: stacked BUY legs invest until exposure reaches (1 − 0.30) = 70 % of mark-to-market equity, always keeping ≥ 30 % as USDT. **Primary drawdown dial** — sets the invested-exposure ceiling identically on BOTH paths (`backtest/pnl.py` + live `execution/order_executor.py` sizing clamp + `strategy/analysis.py` exposure gate). Raised 2026-07-17 from 0.10 (90 % invested) to 0.20 (80 %), then 2026-07-18 to 0.35 (65 %) after loosening `CANDIDATE_DEPTH_FRAC` (0.25→0.10) fed more BUY signals into the leg-cap-free backtest and re-inflated the IS tail DD. **Lowered 2026-07-26 from 0.35 to 0.30 (65 %→70 % invested)** now that the macro-trend overlay handles downtrend tail risk directly (goes fully to cash in a persistent downtrend) — the always-on reserve no longer needs to be the primary DD brake, so exposure is loosened to recover return in neutral/up regimes. Lower to 0.10 for the most room to trade. Risk-management — NOT in the Optuna search space. |
| `MAX_PYRAMID_LEGS` | 12 | **Live-only** hard cap on concurrently-stacked BUY legs. With legs of 20 % of *remaining* cash, invested ≈ 1 − 0.8ⁿ, so reaching the current 70 % ceiling (30 % reserve) needs ~6 legs; the 12-leg cap leaves comfortable margin (the reserve clamp trims the final leg exactly to the floor). Keeps live consistent with the backtest (no leg cap, reaches 70 % from the reserve alone). Guards the 2026-07-08 runaway-pyramiding path. Not in the Optuna search space. |
| `BACKTEST_FILL_SPREAD_BPS` | 5 | Synthetic fill cost (basis points) |
| `BACKTEST_MAX_ROWS` | None | Max kline rows (unlimited; use for testing) |

## Output Paths

| Constant | Default | Purpose |
|---|---|---|
| `BEST_PARAMS_FILE` | `"best_params.json"` | Sensitivity sweep output **filename only** — joined with `BACKTEST_RESULTS_DIR` to form the full path (centralized to keep producer and consumers in sync) |
| `BACKTEST_RESULTS_DIR` | `"backtest/results"` | Machine artefacts (JSON, Optuna SQLite) |
| `BACKTEST_REPORTING_DIR` | `"backtest/reporting"` | Human reports (CSVs, Plotly charts) |
| `LIVE_POSITION_STATE_PATH` | `"state/live_position.json"` | Live carried-position state (cost basis). Written on shutdown; read at startup only when `FLATTEN_ON_START = False`. Git-ignored runtime artifact. |

## Sensitivity Analysis Parameters

| Constant | Default | Purpose |
|---|---|---|
| `SENSITIVITY_LOOKBACK` | *(removed)* | Use `BACKTEST_LOOKBACK` + `BACKTEST_OOS_START` instead |
| `REFIT_EVERY` | 480 | Full HMM re-fit cadence shared by IS sweep and OOS backtest (40 h at 5 m, ~162 IS refits / ~54 OOS refits) |
| `SENSITIVITY_PREDICT_EVERY` | 5 | Viterbi prediction cadence during IS sweep (every 5 candles) |
| `SENSITIVITY_FEE_RATE` | 0.001 | Fee rate fixed during Optuna search (standard Binance Spot taker) |
| `SENSITIVITY_RANK_METRIC` | "sharpe_ratio" | Metric used to rank trials (Sharpe ratio) |

## Example best_params.json Output

```json
{
  "hmm_lookback_rows": 120,
  "hmm_max_regimes": 3,
  "vwap_window": 5,
  "vwap_threshold": 0.003,
  "fee_rate": 0.001,
  "generated_at": "2026-06-04T18:00:00+00:00",
  "source_metric": "sharpe_ratio",
  "source_value": 1.42
}
```

## Imported By

Which modules import each constant from `config_parameters.py`:

- `core/order_book_state.py` — `HISTORY_MAXLEN`, `CRYPTOCCY`, `CCY`
- `core/message_handler.py` — `CRYPTOCCY`, `CCY`, `QUOTE_EVERY_N_TICKS`
- `strategy/analysis.py` — `HFT_INTERVAL`, `HIST_INTERVAL`, `MIN_SNAPSHOTS`, `N_LEVELS`, `CCY`, `CRYPTOCCY`, `HMM_REFIT_INTERVAL`, `HMM_MIN_CONFIDENCE`, `VWAP_THRESHOLD_MULTIPLIER`, `TREND_CONSECUTIVE_BARS`, `TREND_COOLDOWN_BARS` (stop-loss constants read indirectly via the refresher closure injected by `websocket_main.py`)
- `strategy/book_utils.py` — `N_LEVELS`
- `strategy/regime_director.py` — `HMM_FEATURE_COLS`, `HMM_N_ITERATIONS`, `HMM_RANDOM_STATE`, `HMM_MAX_REGIMES`, `HMM_INTERVAL`, `HMM_LOOKBACK`, `HMM_MIN_COVAR`, `HMM_N_INIT`, `REGIME_DIRECTIONAL_RETURN_THRESHOLD`
- `execution/order_executor.py` — `SYMBOL`, `CRYPTOCCY`, `CCY`, `RECV_WINDOW`, `ORDER_REPORT_LIMIT`, `BACKTEST_FEE_RATE`, `MAX_POSITION_PCT`, `MIN_CASH_RESERVE_PCT`
- `strategy/analysis.py` — `MAX_POSITION_PCT`, `MIN_CASH_RESERVE_PCT`, `MAX_PYRAMID_LEGS` (+ VWAP / regime / stop-loss / trend-pause constants; the stop-loss threshold and the macro-trend state are both read indirectly via refresher closures injected by `websocket_main.py`, not imported here)
- `backtest/signals.py` — `HMM_LOOKBACK_ROWS`, `VWAP_WINDOW`, `REFIT_EVERY`, `BACKTEST_MAX_ROWS`, `BACKTEST_LOOKBACK`, `HMM_MIN_CONFIDENCE`, `BACKTEST_FILL_SPREAD_BPS`, `HMM_MAX_REGIMES`, `VWAP_THRESHOLD_MULTIPLIER`, `TREND_CONSECUTIVE_BARS`, `TREND_COOLDOWN_BARS`, `STOP_LOSS_ROLLING_DAYS`, `STOP_LOSS_STD_MULT`, `MACRO_TREND_ENABLED`, `MACRO_TREND_SMA_DAYS`, `MACRO_TREND_SLOPE_DAYS`, `MACRO_TREND_BAND_PCT`
- `backtest/data.py` — `SYMBOL`, `BACKTEST_LOOKBACK`, `BACKTEST_MACRO_INTERVAL`, `BACKTEST_MICRO_INTERVAL`
- `backtest/synthetic_book.py` — `N_LEVELS`, `VOLUME_DECAY_FACTOR`
- `backtest/pnl.py` — `BACKTEST_FEE_RATE`, `BACKTEST_INITIAL_BTC`, `BACKTEST_INITIAL_CAPITAL`, `BACKTEST_RISK_FREE_RATE`, `MAX_POSITION_PCT`, `MIN_CASH_RESERVE_PCT`, `HMM_MIN_CONFIDENCE`
- `backtest/runner.py` — `BACKTEST_FEE_RATE`, `BACKTEST_INITIAL_BTC`, `BACKTEST_INITIAL_CAPITAL`, `SYMBOL`, `BACKTEST_OOS_START`
- `backtest/reporting/formatters.py` — `BACKTEST_FEE_RATE`, `BACKTEST_INITIAL_BTC`, `BACKTEST_INITIAL_CAPITAL`, `SYMBOL`
- `backtest/sensitivity.py` — `REFIT_EVERY`, `BACKTEST_LOOKBACK`, `BACKTEST_OOS_START`, `SENSITIVITY_PREDICT_EVERY`, `SENSITIVITY_FEE_RATE`, `SENSITIVITY_RANK_METRIC`, `SENSITIVITY_OAT_THRESHOLD`, `VWAP_THRESHOLD_MULTIPLIER`
- `websocket_main.py` — `SYMBOL`, `CCY`, `CRYPTOCCY`, `STOP_LOSS_ROLLING_DAYS`, `STOP_LOSS_STD_MULT`, `MACRO_TREND_ENABLED`, `MACRO_TREND_SMA_DAYS`, `MACRO_TREND_SLOPE_DAYS`, `MACRO_TREND_BAND_PCT`, and all session / connection constants
