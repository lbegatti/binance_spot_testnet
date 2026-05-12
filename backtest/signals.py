from collections import deque
import logging

import numpy as np
import pandas as pd

from backtest.data import fetch_klines
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


def run_signals(
    hmm_lookback_rows: int | None = None,
    hmm_max_regimes: int | None = None,
    vwap_window: int | None = None,
    refit_every: int | None = None,
    predict_every: int | None = None,
    lookback: str | None = None,
    prefetched_df: pd.DataFrame | None = None,
    vwap_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Replay the full trading pipeline on historical 1-minute klines.

    All four optional parameters default to ``None``, in which case each falls
    back to the module-level constant from ``config_parameters.py``.  This
    allows ``backtest/sensitivity.py`` to override individual values per run
    without modifying ``config_parameters.py`` or the live system.

    Parameters
    ----------
    hmm_lookback_rows : int | None
        Rolling window size (candles) passed to the HMM warm-up and the main
        loop slice ``features_df.iloc[i - _lookback : i]``.
        Defaults to ``HMM_LOOKBACK_ROWS`` (120 — 2 h at 1 m).
    hmm_max_regimes : int | None
        Upper bound on the BIC state search inside ``select_hmm_model()``.
        Sets ``rd.max_states`` before each fit.
        Defaults to ``HMM_MAX_REGIMES`` (3).
    vwap_window : int | None
        ``deque(maxlen=…)`` size for the rolling VWAP.
        Defaults to ``VWAP_WINDOW`` (5 — 5 min at 1 m).
    refit_every : int | None
        Full BIC re-fit cadence in iterations.  Sensitivity runs pass
        ``SENSITIVITY_REFIT_EVERY`` (480) here for a ~4× speedup; the live
        system cadence (``REFIT_EVERY`` = 120) is used otherwise.
        Defaults to ``REFIT_EVERY`` (120).
    lookback : str | None
        Data fetch window passed to ``fetch_klines()`` as ``start_str``.
        Sensitivity runs pass ``SENSITIVITY_LOOKBACK`` ("30 days ago UTC",
        ~43,200 rows) so each of the 6 OAT runs only processes 30 days
        instead of the full 180-day backtest window.
        Defaults to ``BACKTEST_LOOKBACK`` ("180 days ago UTC").
    predict_every : int | None
        How many candles to skip between cheap Viterbi passes
        (``predict_current_regime()``).  1 = predict every candle (default,
        used by ``runner.py``).  Sensitivity runs may pass a higher
        value (e.g. ``SENSITIVITY_PREDICT_EVERY = 5``) to cut Viterbi
        overhead by ~5× while preserving relative parameter rankings.
    prefetched_df : pd.DataFrame | None
        If provided, skip ``fetch_klines()`` and ``_add_hmm_features()``
        entirely and use this DataFrame directly.  The caller is responsible
        for passing a features-enriched DataFrame
        (columns: ``close``, ``high``, ``low``, ``volume``,
        ``taker_buy_base_vol``, ``return``, ``volatility``, ``obi_proxy``,
        ``trade_density``).

        **This is the key optimisation for sensitivity sweeps**: the caller
        pre-fetches and feature-engineers the data *once* before the grid
        loop, then passes the same ``prefetched_df`` to every run.  Without
        this, ``fetch_klines()`` would hit the Binance API once per
        combination (24× for the full grid, 6× for OAT) for data that is
        identical across all runs.

        When ``None`` (default), the normal fetch path runs and the
        ``lookback`` parameter controls the window.  Existing callers
        (``runner.py``) are completely unaffected.
    vwap_threshold : float | None
        Minimum fractional dip / rally required around the VWAP before a
        signal fires.  Creates a symmetric dead zone:

        - BUY  fires only when ``micro_price < bid_vwap × (1 − threshold)``.
        - SELL fires only when ``micro_price ≥ bid_vwap × (1 + threshold)``.

        Rule of thumb: set to at least 2 × one-way fee to guarantee a
        profitable round trip (e.g. 0.002 for 0.10 % / side fees).
        Default ``VWAP_THRESHOLD_MULTIPLIER`` from ``config_parameters.py``
        (0.003 → 0.30 % dead zone).

    For every candle from ``_lookback`` onward the function
    reproduces the three concurrent flows of the live system:

    **Flow A — Order-book scoring** (mirrors ``low_latency_analysis``):
        ``build_synthetic_book`` → ``build_levels`` → ``collect_candidates``
        → ``select_best_opportunity`` for each side.

    **Flow B — HMM regime** (mirrors ``historical_analysis`` regime update):
        The ``RegimeDirector`` is fitted on the initial warm-up window
        (``HMM_LOOKBACK_ROWS`` candles).  On subsequent iterations it is
        updated using the cheap Viterbi path, with a full BIC re-fit every
        ``REFIT_EVERY`` iterations — identical cadence to the live system.

    **Flow C — VWAP dip/strength filter** (mirrors ``historical_analysis`` VWAP):
        A rolling ``deque(maxlen=VWAP_WINDOW)`` accumulates best-bid/ask
        prices and volumes from the synthetic book.  ``bid_vwap`` and
        ``ask_vwap`` are computed once the window is full; ``None`` before
        that (filter is transparent, same as live).  Both gate conditions use
        ``bid_vwap`` exclusively (mean-reversion strategy); ``ask_vwap`` is
        retained in the output record for diagnostics only.

    **Combined gate** (mirrors the live ``low_latency_analysis`` decision):
        * ``signal = +1`` (BUY)  when ``best_buy`` exists **and**
          ``regime_confidence ≥ HMM_MIN_CONFIDENCE`` (model is certain enough)
          **and** regime ∉ {``trending_down``, ``high_volatility``}
          **and** (``bid_vwap`` is ``None`` or
          ``micro_price < bid_vwap × (1 − vwap_threshold)``)
          (dip deep enough to cover fees and leave profit).
        * ``signal = -1`` (SELL) when ``best_sell`` exists **and**
          ``regime_confidence ≥ HMM_MIN_CONFIDENCE``
          **and** regime ∉ {``trending_up``, ``high_volatility``}
          **and** (``bid_vwap`` is ``None`` or
          ``micro_price ≥ bid_vwap × (1 + vwap_threshold)``)
          (rally strong enough to cover fees and leave profit).
        * ``signal = 0``  otherwise (no trade).

    Returns:
        pd.DataFrame: One row per candle (from ``HMM_LOOKBACK_ROWS`` onward),
            indexed by ``timestamp``, with columns:
            ``close`` (candle close — VWAP anchor),
            ``half_spread`` (bps-based fill cost: ``close × BACKTEST_FILL_SPREAD_BPS /
            20_000``; default 5 bps ≈ $20 at $80k BTC. BUY fills at
            ``close + half_spread``, SELL fills at ``close - half_spread``),
            ``signal`` (+1 BUY / -1 SELL / 0 HOLD),
            ``regime``, ``regime_confidence`` (posterior probability of the
            predicted regime from ``predict_proba()``, ``None`` before the
            first fit), ``bid_vwap``, ``ask_vwap``,
            ``best_buy_micro``, ``best_sell_micro``,
            ``buy_qty`` (ask-side quantity at the best-buy level),
            ``sell_qty`` (bid-side quantity at the best-sell level).
            ``buy_qty`` / ``sell_qty`` are ``None`` when no signal fired.
    """
    # Resolve parameter overrides — fall back to config constants when None.
    # This is the only place where overrides are applied; callers that pass
    # no arguments (e.g. runner.py) are completely unaffected.
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

    # 1. Fetch and prepare data
    if prefetched_df is not None:
        # Caller pre-fetched the data (e.g. sensitivity.py fetches once for all
        # runs and passes the same DataFrame to every _run_one() call).
        # Skip fetch_klines() and _add_hmm_features() entirely.
        features_df = prefetched_df
        logging.info(
            "Using pre-fetched features_df: %d rows (no API call).", len(features_df)
        )
    else:
        klines = fetch_klines(start_str=_fetch_lookback)
        features_df = _add_hmm_features(klines)
        if BACKTEST_MAX_ROWS is not None:
            features_df = features_df.iloc[-(_lookback + BACKTEST_MAX_ROWS) :]
            logging.info(
                "Debug mode: capped at %d rows (%d replay candles).",
                len(features_df),
                BACKTEST_MAX_ROWS,
            )
        logging.info(
            "Fetched %d klines; %d rows after HMM features.",
            len(klines),
            len(features_df),
        )

    # 2. Initial HMM fit on the warm-up window (first _lookback rows)
    # NOTE: the train/predict split and StandardScaler are applied INSIDE
    # select_hmm_model() — no explicit split is needed here.
    # Concretely, select_hmm_model() does:
    #   train_features = klines_df[:HMM_TRAIN_ROWS]   (first 80 rows — in-sample)
    #   scaler.fit_transform(train_features)            (scale fitted on training rows only;
    #                                                    recent candles cannot leak into mean/std)
    #   test_features  = klines_df[HMM_TRAIN_ROWS:]    (last ~40 rows — out-of-sample)
    #   scaler.transform(test_features)                 (same scale applied to test rows)
    #   model.fit(train_scaled)                         (EM on 80 rows only)
    #   model.predict(test_scaled)                      (Viterbi on ~40 test rows only)
    #   model.predict_proba(test_scaled)                (confidence on ~40 test rows only)
    # current_regime / regime_confidence reflect the LAST of those ~40 test rows,
    # which was genuinely out-of-sample during training (walk-forward split).
    rd = RegimeDirector(max_regimes=_max_regimes)
    rd.klines_df = features_df.iloc[:_lookback]
    rd.select_hmm_model()
    rd.assign_regime_labels()
    logging.info(
        "Initial HMM fit complete — regime: '%s' (confidence: %.2f)",
        rd.regime_label,
        rd.regime_confidence or 0.0,
    )

    # 3. Rolling VWAP window
    vwap_deque: deque = deque(maxlen=_vwap_window)

    # 4. Pre-extract feature columns as numpy arrays so the hot loop avoids
    #    43 k pandas iloc() calls.  Dict-style row access still works for
    #    build_synthetic_book() — passing a plain dict is ~3× faster than
    #    passing a pd.Series because it skips the pandas Index lookup.
    close_arr = features_df["close"].to_numpy()
    high_arr = features_df["high"].to_numpy()
    low_arr = features_df["low"].to_numpy()
    vol_arr = features_df["volume"].to_numpy()
    tbv_arr = features_df["taker_buy_base_vol"].to_numpy()
    timestamps = features_df.index

    # 5. Main loop
    records: list[dict] = []
    hist_iteration = 0  # counts iterations since warm-up (for refit cadence)

    for i in range(_lookback, len(features_df)):
        timestamp = timestamps[i]
        hist_iteration += 1

        # Lightweight dict — avoids pd.Series overhead inside build_synthetic_book.
        row_dict = {
            "high": high_arr[i],
            "low": low_arr[i],
            "close": close_arr[i],
            "volume": vol_arr[i],
            "taker_buy_base_vol": tbv_arr[i],
        }

        # Flow A: synthetic book → levels → candidates → best
        order_book = build_synthetic_book(row_dict)
        levels, median_depth, level_0_depth = build_levels(
            order_book["bids"], order_book["asks"]
        )
        buy_candidates, sell_candidates = collect_candidates(
            levels, median_depth, level_0_depth
        )
        best_buy = select_best_opportunity(buy_candidates, "buy", hist_iteration)
        best_sell = select_best_opportunity(sell_candidates, "sell", hist_iteration)

        # Flow C: VWAP — use pre-computed top-of-book values (no max/min key scan)
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

        # ── Rolling VWAP ─────────────────────────────────────────────────────
        # deque(maxlen=_vwap_window) acts as a self-evicting rolling window.
        # The else branch (None) fires during the first _vwap_window-1 iterations
        # while the window is still warming up.  None makes the VWAP gate
        # transparent, mirroring the live system's warm-up behaviour.
        #
        # VWAP formula: bid_vwap = Σ(best_bid[t] × vol_bid[t]) / Σ(vol_bid[t])
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

        # Flow B: HMM regime update — two-speed refit.
        # _predict_every lets sensitivity sweeps skip N-1 Viterbi passes
        # between refits (reusing the previous regime label) for a speed-up
        # proportional to _predict_every while preserving relative rankings.
        rd.klines_df = features_df.iloc[i - _lookback : i]
        rd.max_states = _max_regimes
        if hist_iteration % _refit_every == 0:
            try:
                rd.select_hmm_model()  # full BIC re-fit
            except RuntimeError as exc:
                logging.warning(
                    "signals: HMM refit failed at iteration %d — keeping "
                    "previous model. (%s)",
                    hist_iteration,
                    exc,
                )
        elif hist_iteration % _predict_every == 0:
            rd.predict_current_regime()  # cheap Viterbi pass
        # else: reuse regime / confidence from the previous iteration
        rd.assign_regime_labels()
        regime = rd.regime_label
        regime_confidence = rd.regime_confidence

        close_price = close_arr[i]
        half_spread = close_price * BACKTEST_FILL_SPREAD_BPS / 20_000.0

        # Always capture raw candidate details for reporting (needed by
        # _compute_stats in pnl.py to count raw candidates and blocked rates
        # independently of which gates fired).
        signal = 0
        best_buy_micro = None
        best_sell_micro = None
        buy_qty = None  # aq at the best-buy level  (ask-side qty we consume)
        sell_qty = None  # bq at the best-sell level (bid-side qty we consume)

        if best_buy:
            best_buy_micro = float(best_buy[5])  # micro_price
            buy_qty = float(best_buy[7])  # aq
        if best_sell:
            best_sell_micro = float(best_sell[5])  # micro_price
            sell_qty = float(best_sell[6])  # bq

        # --- confidence gate (mirrors low_latency_analysis in analysis.py) ---
        # predict_proba()[-1][current_regime] < HMM_MIN_CONFIDENCE means the
        # model cannot distinguish the current state clearly enough to justify
        # an order.  Skip both sides — ambiguous HMM signal treated as flat.
        # While regime_confidence is None (warm-up / before first fit) the
        # gate is transparent, matching the live system's behaviour.
        confidence_ok = (
            regime_confidence is None or regime_confidence >= HMM_MIN_CONFIDENCE
        )

        # Combined gate (mirrors live low_latency_analysis)
        # Threshold-gated mean-reversion: the dip / rally must be large enough
        # to cover round-trip fees and leave a profit margin.
        #   BUY  → price < bid_vwap × (1 − threshold)   [deep enough dip]
        #          anchored to bid VWAP — the relevant pressure for buy-side.
        #   SELL → price ≥ ask_vwap × (1 + threshold)   [strong enough rally]
        #          anchored to ask VWAP — the relevant pressure for sell-side.
        # Using separate anchors avoids cross-side VWAP bias.
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
                "timestamp": timestamp,
                "close": close_price,  # candle close — VWAP anchor
                "half_spread": half_spread,  # bps-based taker fill cost: close × BACKTEST_FILL_SPREAD_BPS / 20_000
                "signal": signal,
                "regime": regime,
                "regime_confidence": regime_confidence,  # posterior prob of winning state
                "bid_vwap": bid_vwap,
                "ask_vwap": ask_vwap,
                "best_buy_micro": best_buy_micro,
                "best_sell_micro": best_sell_micro,
                "buy_qty": buy_qty,  # aq from best_buy candidate
                "sell_qty": sell_qty,  # bq from best_sell candidate
            }
        )

    result = pd.DataFrame(records).set_index("timestamp")
    logging.info(
        "Backtest complete: %d candles processed | BUY signals: %d | SELL signals: %d | HOLD: %d",
        len(result),
        (result["signal"] == 1).sum(),
        (result["signal"] == -1).sum(),
        (result["signal"] == 0).sum(),
    )
    return result
