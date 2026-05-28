from collections import deque
import logging

import numpy as np
import pandas as pd

from backtest.data import fetch_macro_klines, fetch_micro_klines
from backtest.synthetic_book import build_synthetic_book
from strategy.book_utils import (
    build_levels,
    collect_candidates,
    select_best_opportunity,
)
from strategy.indicators import volume_weighted_average_price
from strategy.regime_director import RegimeDirector

from config_parameters import (
    HMM_LOOKBACK_ROWS,
    VWAP_WINDOW,
    REFIT_EVERY,
    BACKTEST_MAX_ROWS,
    BACKTEST_LOOKBACK,
    HMM_MIN_CONFIDENCE,
    BACKTEST_FILL_SPREAD_BPS,
    HMM_MAX_REGIMES,
    VWAP_THRESHOLD_MULTIPLIER,
    TREND_CONSECUTIVE_BARS,
    TREND_COOLDOWN_BARS,
    STOP_LOSS_ROLLING_DAYS,
    STOP_LOSS_STD_MULT,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _add_hmm_features(k_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the four HMM features on a klines DataFrame.

    Mirrors the feature engineering in ``RegimeDirector.get_klines_data()``
    but operates on an already-fetched DataFrame instead of downloading
    fresh klines.  The first row is dropped because ``pct_change()``
    produces ``NaN``.

    Args:
        k_df (pd.DataFrame): Raw klines DataFrame returned by
            ``backtest.data.fetch_klines()``.  Must contain columns:
            ``close``, ``high``, ``low``, ``volume``,
            ``taker_buy_base_vol``, ``num_trades``.

    Returns:
        pd.DataFrame: A copy of *df* with four extra columns —
            ``return``, ``volatility``, ``obi_proxy``, ``trade_density``
            — and no ``NaN`` rows.
    """
    klines_df = k_df.copy()
    klines_df["return"] = klines_df["close"].pct_change()
    klines_df["volatility"] = (klines_df["high"] - klines_df["low"]) / klines_df[
        "close"
    ]
    klines_df["obi_proxy"] = (
        klines_df["taker_buy_base_vol"] / klines_df["volume"]
    ) * 2 - 1
    klines_df["trade_density"] = klines_df["num_trades"] / klines_df["volume"]
    return klines_df.dropna()


def _add_trend_pause_flag(
    df_macro: pd.DataFrame,
    n: int,
    cooldown: int,
) -> pd.Series:
    """
    Compute a boolean ``trend_pause`` Series on the macro (5 m) frame.

    True  → ``n`` or more consecutive same-direction closes detected;
             mean-reversion entries should be suppressed.
    False → market is ranging; normal signal logic applies.

    The streak counter resets automatically when the close direction flips,
    so no explicit "trend end" detection is required.  The cooldown then keeps
    ``paused=True`` for ``cooldown`` bars after the last trending bar, preventing
    whipsaw re-entry the instant the streak breaks.

    NaN rows at the start (HMM warm-up period) have direction filled to 0 so
    they never artificially trigger a trend pause.  Flat streaks (direction == 0,
    i.e. consecutive equal closes) are excluded from trend detection.
    """
    close = df_macro["close"]
    # +1 up-close, -1 down-close, 0 flat; NaN at row 0 filled to 0 (no direction)
    direction = np.sign(close.diff()).fillna(0)

    # Cumulative group ID increments on every direction change.
    # cumcount()+1 gives the streak length: 1 on the first bar of each run.
    streak = (
        direction.groupby(
            (direction != direction.shift()).cumsum()
        ).cumcount() + 1
    )

    # A bar "is in trend" when streak ≥ n AND direction is not flat.
    in_trend = (streak >= n) & (direction != 0)

    # Cooldown: OR with shifted versions to stay paused for `cooldown` extra bars
    # after the last in_trend=True bar (counted from the END of the streak).
    paused = in_trend.copy()
    for lag in range(1, cooldown + 1):
        paused = paused | in_trend.shift(lag).fillna(False)

    return paused.astype(bool)


def run_signals(
    prefetched_macro: pd.DataFrame | None = None,
    prefetched_micro: pd.DataFrame | None = None,
    lookback: str | None = None,
    end_str: str | None = None,
    hmm_lookback_rows: int | None = None,
    hmm_max_regimes: int | None = None,
    vwap_window: int | None = None,
    refit_every: int | None = None,
    predict_every: int | None = None,
    vwap_threshold: float | None = None,
    trend_consecutive_bars: int | None = None,
    trend_cooldown_bars: int | None = None,
) -> pd.DataFrame:
    """
    Replay the full trading pipeline on historical klines using a
    **two-resolution architecture**:

    * **Macro frame (5 m)** — ``GaussianHMM`` walk-forward.  Regime labels
      are produced once per 5-minute bar and stored in ``regime_df``.
    * **Micro frame (1 m)** — Synthetic order-book scoring, rolling VWAP,
      and PnL signal generation.  Each 1-minute bar is annotated with
      the most recent 5-minute regime label via ``pd.merge_asof`` (zero
      look-ahead bias, backward direction).

    Decoupling the HMM cadence from the execution cadence yields two
    improvements over the previous single-timeframe design:

    1. **Faster EM convergence** — the GaussianHMM sees 5× fewer rows
       (5 m bars) with richer information content, so the EM algorithm
       converges earlier and is less sensitive to microstructure noise.
    2. **Cleaner signal gating** — the VWAP / synthetic-book signals
       still operate at full 1-minute resolution, while the macro regime
       filter advances only on meaningful structural changes.

    Parameters
    ----------
    prefetched_macro : pd.DataFrame | None
        Pre-fetched **raw** 5-minute OHLCV DataFrame (columns: ``open``,
        ``high``, ``low``, ``close``, ``volume``, ``taker_buy_base_vol``,
        ``num_trades``).  When provided, skips ``fetch_macro_klines()``.
        Pass raw klines — HMM features are computed internally.
    prefetched_micro : pd.DataFrame | None
        Pre-fetched **raw** 1-minute OHLCV DataFrame.  When provided,
        skips ``fetch_micro_klines()``.
    lookback : str | None
        Fetch window start passed to both ``fetch_macro_klines()`` and
        ``fetch_micro_klines()`` when the pre-fetched frames are ``None``.
        Defaults to ``BACKTEST_LOOKBACK`` (``"360 days ago UTC"``).
    end_str : str | None
        Optional fetch window end (e.g. ``BACKTEST_OOS_START``).
        ``None`` → fetch up to the most recent complete candle.
    hmm_lookback_rows : int | None
        Rolling window for HMM warm-up and the macro walk-forward slice
        ``features_macro.iloc[i - _lookback : i]``.
        Counts **5-minute bars** (default ``HMM_LOOKBACK_ROWS`` = 120
        → 10 h at 5 m).
    hmm_max_regimes : int | None
        Upper bound on the BIC state search inside ``select_hmm_model()``.
        Defaults to ``HMM_MAX_REGIMES`` (3).
    vwap_window : int | None
        ``deque(maxlen=…)`` size for the rolling VWAP on 1-minute bars.
        Defaults to ``VWAP_WINDOW`` (5).
    refit_every : int | None
        Full BIC re-fit cadence counted in **5-minute macro bars**.
        Defaults to ``REFIT_EVERY`` (120 → re-fit every 10 h).
    predict_every : int | None
        Cheap Viterbi pass cadence in macro bars.  Sensitivity sweeps
        may pass a higher value to cut Viterbi overhead.  Default 1
        (predict every macro bar).
    vwap_threshold : float | None
        Minimum fractional dip / rally around VWAP before a signal fires.
        Defaults to ``VWAP_THRESHOLD_MULTIPLIER`` (0.003 → 0.30 % dead
        zone).

    Internal pipeline
    -----------------
    **Phase 1 — HMM walk-forward on df_macro (5 m)**:
        ``_add_hmm_features`` → initial fit on warm-up window →
        rolling ``[i-_lookback : i]`` slices → ``select_hmm_model``
        every ``_refit_every`` bars / ``predict_current_regime`` every
        ``_predict_every`` bars → ``assign_regime_labels`` →
        ``regime_df (timestamp → regime, regime_confidence)``.

    **Phase 2 — Temporal stitch**:
        ``pd.merge_asof(df_micro, regime_df, direction='backward')`` →
        ``df_exec``.  Each 1-minute bar receives the latest 5-minute
        regime label that preceded it (no look-ahead bias).
        1-minute bars that predate the first regime label are discarded.

    **Phase 3 — Execution loop on df_exec (1 m)**:
        ``build_synthetic_book`` → ``build_levels`` →
        ``collect_candidates`` → ``select_best_opportunity`` (Flow A),
        rolled VWAP on top-of-book prices (Flow C), regime / confidence
        read from stitched columns (no HMM calls in this loop).

    Returns
    -------
    pd.DataFrame
        One row per 1-minute bar (from the first valid regime label
        onward), indexed by ``timestamp``, with columns:
        ``close``, ``high``, ``low``,
        ``half_spread``, ``signal`` (+1 / −1 / 0),
        ``regime``, ``regime_confidence``,
        ``bid_vwap``, ``ask_vwap``,
        ``best_buy_micro``, ``best_sell_micro``,
        ``buy_qty``, ``sell_qty``.

        ``high`` and ``low`` are retained for the intra-candle whipsaw
        guard in ``backtest/pnl.py`` (Step 4).
    """
    # ── Resolve parameter overrides ──────────────────────────────────────────
    _lookback = (
        hmm_lookback_rows if hmm_lookback_rows is not None else HMM_LOOKBACK_ROWS
    )
    _max_regimes = hmm_max_regimes if hmm_max_regimes is not None else HMM_MAX_REGIMES
    _vwap_window = vwap_window if vwap_window is not None else VWAP_WINDOW
    _refit_every = refit_every if refit_every is not None else REFIT_EVERY
    _predict_every = predict_every if predict_every is not None else 1
    _fetch_lookback = lookback if lookback is not None else BACKTEST_LOOKBACK
    _vwap_threshold = (
        vwap_threshold if vwap_threshold is not None else VWAP_THRESHOLD_MULTIPLIER
    )
    _trend_consecutive = (
        trend_consecutive_bars if trend_consecutive_bars is not None
        else TREND_CONSECUTIVE_BARS
    )
    _trend_cooldown = (
        trend_cooldown_bars if trend_cooldown_bars is not None
        else TREND_COOLDOWN_BARS
    )

    # ── 1. Fetch / accept raw OHLCV frames ───────────────────────────────────
    if prefetched_macro is not None:
        df_macro_raw = prefetched_macro
        logging.info("Using pre-fetched macro (5 m) frame: %d rows.", len(df_macro_raw))
    else:
        df_macro_raw = fetch_macro_klines(lookback=_fetch_lookback, end_str=end_str)
        logging.info("Fetched macro (5 m) frame: %d rows.", len(df_macro_raw))

    if prefetched_micro is not None:
        df_micro_raw = prefetched_micro
        logging.info("Using pre-fetched micro (1 m) frame: %d rows.", len(df_micro_raw))
    else:
        df_micro_raw = fetch_micro_klines(lookback=_fetch_lookback, end_str=end_str)
        logging.info("Fetched micro (1 m) frame: %d rows.", len(df_micro_raw))

    # Cap micro frame for debug / sensitivity runs.
    # Macro frame is intentionally left uncapped so the HMM always sees the
    # full warm-up history even when BACKTEST_MAX_ROWS is very small.
    df_micro = df_micro_raw.copy()
    if BACKTEST_MAX_ROWS is not None:
        df_micro = df_micro.iloc[-BACKTEST_MAX_ROWS:]
        logging.info("Debug mode: micro frame capped at %d rows.", len(df_micro))

    # ── Phase 1 — HMM walk-forward on 5-minute macro bars ────────────────────
    features_macro = _add_hmm_features(df_macro_raw)
    logging.info(
        "Macro features ready: %d rows (dropped %d NaN rows from pct_change).",
        len(features_macro),
        len(df_macro_raw) - len(features_macro),
    )

    rd = RegimeDirector(max_regimes=_max_regimes)
    rd.klines_df = features_macro.iloc[:_lookback]
    rd.select_hmm_model()
    rd.assign_regime_labels()
    logging.info(
        "Initial HMM fit on 5 m bars — regime: '%s' (confidence: %.2f)",
        rd.regime_label,
        rd.regime_confidence or 0.0,
    )

    # Compute trend_pause vectorially on the full macro frame BEFORE the loop.
    # Indexed by position so trend_pause_series.iloc[i] aligns with features_macro[i].
    trend_pause_series = _add_trend_pause_flag(
        features_macro, n=_trend_consecutive, cooldown=_trend_cooldown
    )
    logging.info(
        "Trend-pause flag computed (n=%d consecutive bars, cooldown=%d bars): "
        "%d / %d macro bars flagged as trending.",
        _trend_consecutive,
        _trend_cooldown,
        int(trend_pause_series.sum()),
        len(trend_pause_series),
    )

    macro_timestamps = features_macro.index
    macro_records: list[dict] = []
    macro_iteration = 0

    for i in range(_lookback, len(features_macro)):
        macro_iteration += 1
        rd.klines_df = features_macro.iloc[i - _lookback : i]
        rd.max_states = _max_regimes

        if macro_iteration % _refit_every == 0:
            try:
                rd.select_hmm_model()  # full BIC re-fit
            except RuntimeError as exc:
                logging.warning(
                    "signals: HMM refit failed at macro iteration %d — "
                    "keeping previous model. (%s)",
                    macro_iteration,
                    exc,
                )
        elif macro_iteration % _predict_every == 0:
            rd.predict_current_regime()  # cheap Viterbi pass
        # else: reuse regime / confidence from the previous macro bar

        rd.assign_regime_labels()
        macro_records.append(
            {
                "timestamp": macro_timestamps[i],
                "regime": rd.regime_label,
                "regime_confidence": rd.regime_confidence,
                "trend_pause": bool(trend_pause_series.iloc[i]),
            }
        )

    regime_df = pd.DataFrame(macro_records).set_index("timestamp")
    logging.info(
        "Phase 1 complete: %d 5-minute regime labels produced.", len(regime_df)
    )

    # ── Phase 2 — Temporal stitch (5 m → 1 m, zero look-ahead) ──────────────
    # merge_asof with direction='backward' assigns each 1-minute bar the most
    # recent 5-minute regime label that precedes (or matches) its timestamp.
    # 1-minute bars that predate the very first regime label are discarded so
    # the execution loop never reads a NaN regime.
    df_exec = pd.merge_asof(
        df_micro.sort_index(),
        regime_df.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    first_regime_ts = regime_df.index[0]
    df_exec = df_exec[df_exec.index >= first_regime_ts].copy()
    logging.info(
        "Phase 2 complete: %d 1-minute execution bars after stitch "
        "(discarded %d pre-regime bars).",
        len(df_exec),
        len(df_micro) - len(df_exec),
    )

    # Fill any NaN trend_pause values left by the stitch with False (no pause).
    df_exec["trend_pause"] = df_exec["trend_pause"].fillna(False).astype(bool)

    # ── Adaptive stop-loss threshold (daily rolling std of |daily return|) ────
    # Resamples df_macro_raw to daily bars, computes the rolling std of
    # daily absolute pct-changes, multiplies by STOP_LOSS_STD_MULT, and
    # forward-fills onto df_exec via merge_asof.

    # Result: each 1-minute bar
    # carries the most recent daily-updated stop-loss threshold — no look-ahead.
    _sl_daily = df_macro_raw[["close"]].resample("1D").last().dropna()
    _sl_daily["abs_return"] = _sl_daily["close"].pct_change().abs()
    _sl_daily["stop_loss_pct"] = (
        _sl_daily["abs_return"]
        .rolling(STOP_LOSS_ROLLING_DAYS, min_periods=1)
        .std()
        * STOP_LOSS_STD_MULT
    )
    df_exec = pd.merge_asof(
        df_exec.sort_index(),
        _sl_daily[["stop_loss_pct"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    df_exec["stop_loss_pct"] = df_exec["stop_loss_pct"].fillna(0.0)
    logging.info(
        "Stop-loss pct merged onto execution frame "
        "(daily rolling %d-day std × %.1f); "
        "median threshold: %.4f (%.2f%%).",
        STOP_LOSS_ROLLING_DAYS,
        STOP_LOSS_STD_MULT,
        df_exec["stop_loss_pct"].median(),
        df_exec["stop_loss_pct"].median() * 100,
    )

    # ── Phase 3 — Execution loop on 1-minute bars ────────────────────────────
    # Pre-extract columns as numpy arrays to avoid ~390k pandas iloc() calls.
    close_arr = df_exec["close"].to_numpy()
    high_arr = df_exec["high"].to_numpy()
    low_arr = df_exec["low"].to_numpy()
    vol_arr = df_exec["volume"].to_numpy()
    tbv_arr = df_exec["taker_buy_base_vol"].to_numpy()
    regime_arr = df_exec["regime"].to_numpy()
    confidence_arr = df_exec["regime_confidence"].to_numpy(dtype=object)
    trend_pause_arr = df_exec["trend_pause"].to_numpy(dtype=bool)
    stop_loss_pct_arr = df_exec["stop_loss_pct"].to_numpy(dtype=float)
    timestamps = df_exec.index

    vwap_deque: deque = deque(maxlen=_vwap_window)
    records: list[dict] = []

    for i in range(len(df_exec)):
        regime = regime_arr[i]
        # regime_confidence may be a float/NaN for the very first stitched bar;
        # treat that the same as None (transparent gate).
        raw_conf = confidence_arr[i]
        regime_confidence = (
            None
            if (
                raw_conf is None or (isinstance(raw_conf, float) and np.isnan(raw_conf))
            )
            else float(raw_conf)
        )

        # Flow A: synthetic order book → levels → candidates → best
        row_dict = {
            "high": high_arr[i],
            "low": low_arr[i],
            "close": close_arr[i],
            "volume": vol_arr[i],
            "taker_buy_base_vol": tbv_arr[i],
        }
        order_book = build_synthetic_book(row_dict)
        levels, median_depth, level_0_depth = build_levels(
            order_book["bids"], order_book["asks"]
        )
        buy_candidates, sell_candidates = collect_candidates(
            levels, median_depth, level_0_depth
        )
        best_buy = select_best_opportunity(buy_candidates, "buy", i)
        best_sell = select_best_opportunity(sell_candidates, "sell", i)

        # Flow C: rolling VWAP on top-of-book prices
        best_bid_price = order_book["_best_bid"]
        best_ask_price = order_book["_best_ask"]
        vol_best_bid = order_book["_vol_best_bid"]
        vol_best_ask = order_book["_vol_best_ask"]

        vwap_deque.append(
            {
                "best_bid": best_bid_price,
                "vol_bid": vol_best_bid,
                "best_ask": best_ask_price,
                "vol_ask": vol_best_ask,
            }
        )

        if len(vwap_deque) >= _vwap_window:
            bid_prices = np.array([s["best_bid"] for s in vwap_deque])
            bid_vols = np.array([s["vol_bid"] for s in vwap_deque])
            ask_prices = np.array([s["best_ask"] for s in vwap_deque])
            ask_vols = np.array([s["vol_ask"] for s in vwap_deque])
            bid_vwap = volume_weighted_average_price(bid_prices, bid_vols)
            ask_vwap = volume_weighted_average_price(ask_prices, ask_vols)
        else:
            bid_vwap = None
            ask_vwap = None

        close_price = close_arr[i]
        half_spread = close_price * BACKTEST_FILL_SPREAD_BPS / 20_000.0

        signal = 0
        best_buy_micro = None
        best_sell_micro = None
        buy_qty = None
        sell_qty = None

        if best_buy:
            best_buy_micro = float(best_buy[5])  # micro_price
            buy_qty = float(best_buy[7])  # aq
        if best_sell:
            best_sell_micro = float(best_sell[5])  # micro_price
            sell_qty = float(best_sell[6])  # bq

        # Confidence gate — mirrors low_latency_analysis in analysis.py.
        # Transparent (True) when confidence is None (before first regime label).
        confidence_ok = (
            regime_confidence is None or regime_confidence >= HMM_MIN_CONFIDENCE
        )

        # Combined gate — regime + VWAP dip/rally threshold.
        if confidence_ok:
            if best_buy and best_buy_micro is not None:
                regime_ok = regime not in ("trending_down", "high_volatility")
                vwap_floor = (
                    bid_vwap * (1.0 - _vwap_threshold) if bid_vwap is not None else None
                )
                vwap_ok = vwap_floor is None or best_buy_micro < vwap_floor
                if regime_ok and vwap_ok:
                    signal = 1

            if best_sell and best_sell_micro is not None and signal == 0:
                regime_ok = regime not in ("trending_up", "high_volatility")
                vwap_ceil = (
                    ask_vwap * (1.0 + _vwap_threshold) if ask_vwap is not None else None
                )
                vwap_ok = vwap_ceil is None or best_sell_micro >= vwap_ceil
                if regime_ok and vwap_ok:
                    signal = -1

        records.append(
            {
                "timestamp": timestamps[i],
                "close": close_price,
                "high": high_arr[i],  # retained for whipsaw guard in pnl.py
                "low": low_arr[i],  # retained for whipsaw guard in pnl.py
                "half_spread": half_spread,
                "signal": signal,
                "regime": regime,
                "regime_confidence": regime_confidence,
                "bid_vwap": bid_vwap,
                "ask_vwap": ask_vwap,
                "best_buy_micro": best_buy_micro,
                "best_sell_micro": best_sell_micro,
                "buy_qty": buy_qty,
                "sell_qty": sell_qty,
                "trend_pause": trend_pause_arr[i],
                "stop_loss_pct": stop_loss_pct_arr[i],
            }
        )

    result = pd.DataFrame(records).set_index("timestamp")
    logging.info(
        "Backtest complete: %d candles processed | BUY signals: %d | "
        "SELL signals: %d | HOLD: %d",
        len(result),
        (result["signal"] == 1).sum(),
        (result["signal"] == -1).sum(),
        (result["signal"] == 0).sum(),
    )
    return result
