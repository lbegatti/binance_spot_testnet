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
    HMM_MIN_CONFIDENCE,
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


def run_signals() -> pd.DataFrame:
    """
    Replay the full trading pipeline on 30 days of historical 1-minute klines.

    For every candle from ``HMM_LOOKBACK_ROWS`` onward the function
    reproduces the three concurrent flows of the live system:

    **Flow A — Order-book scoring** (mirrors ``low_latency_analysis``):
        ``build_synthetic_book`` → ``build_levels`` → ``collect_candidates``
        → ``select_best_opportunity`` for each side.

    **Flow B — HMM regime** (mirrors ``historical_analysis`` regime update):
        The ``RegimeDirector`` is fitted on the initial warm-up window
        (``HMM_LOOKBACK_ROWS`` candles).  On subsequent iterations it is
        updated using the cheap Viterbi path, with a full BIC re-fit every
        ``REFIT_EVERY`` iterations — identical cadence to the live system.

    **Flow C — VWAP momentum filter** (mirrors ``historical_analysis`` VWAP):
        A rolling ``deque(maxlen=VWAP_WINDOW)`` accumulates best-bid/ask
        prices and volumes from the synthetic book.  ``bid_vwap`` and
        ``ask_vwap`` are computed once the window is full; ``None`` before
        that (filter is transparent, same as live).

    **Combined gate** (mirrors the live ``low_latency_analysis`` decision):
        * ``signal = +1`` (BUY)  when ``best_buy`` exists **and**
          ``regime_confidence ≥ HMM_MIN_CONFIDENCE`` (model is certain enough)
          **and** regime ∉ {``trending_down``, ``high_volatility``}
          **and** (``ask_vwap`` is ``None`` or ``micro_price > ask_vwap``).
        * ``signal = -1`` (SELL) when ``best_sell`` exists **and**
          ``regime_confidence ≥ HMM_MIN_CONFIDENCE``
          **and** regime ∉ {``trending_up``, ``high_volatility``}
          **and** (``bid_vwap`` is ``None`` or ``micro_price < bid_vwap``).
        * ``signal = 0``  otherwise (no trade).

    Returns:
        pd.DataFrame: One row per candle (from ``HMM_LOOKBACK_ROWS`` onward),
            indexed by ``timestamp``, with columns:
            ``close`` (candle close — VWAP anchor),
            ``half_spread`` (``(high-low)/2`` — taker fill cost; BUY fills at
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
    # 1. Fetch and prepare data
    klines = fetch_klines()
    features_df = _add_hmm_features(klines)
    if BACKTEST_MAX_ROWS is not None:
        features_df = features_df.iloc[-(HMM_LOOKBACK_ROWS + BACKTEST_MAX_ROWS) :]
        logging.info(
            "Debug mode: capped at %d rows (%d replay candles).",
            len(features_df),
            BACKTEST_MAX_ROWS,
        )
    logging.info(
        "Fetched %d klines; %d rows after HMM features.", len(klines), len(features_df)
    )

    # 2. Initial HMM fit on the warm-up window (first HMM_LOOKBACK_ROWS rows)
    # NOTE: the train/predict split and StandardScaler are applied INSIDE
    # select_hmm_model() — no explicit split is needed here.
    # Concretely, select_hmm_model() does:
    #   train_features = klines_df[:HMM_TRAIN_ROWS]   (first 80 rows — in-sample)
    #   scaler.fit_transform(train_features)            (scale on training rows only)
    #   scaler.transform(klines_df)                     (apply same scale to all 120 rows)
    #   model.fit(train_scaled)                         (EM on 80 rows)
    #   model.predict(full_scaled)                      (Viterbi on all 120 rows)
    #   model.predict_proba(full_scaled)                (confidence on all 120 rows)
    # current_regime / regime_confidence reflect the LAST (most recent) row,
    # which was genuinely out-of-sample during training.
    rd = RegimeDirector()
    rd.klines_df = features_df.iloc[:HMM_LOOKBACK_ROWS]
    rd.select_hmm_model()
    rd.assign_regime_labels()
    logging.info(
        "Initial HMM fit complete — regime: '%s' (confidence: %.2f)",
        rd.regime_label,
        rd.regime_confidence or 0.0,
    )

    # 3. Rolling VWAP window
    vwap_deque: deque = deque(maxlen=VWAP_WINDOW)

    # 4. Main loop
    records: list[dict] = []
    hist_iteration = 0  # counts iterations since warm-up (for refit cadence)

    for i in range(HMM_LOOKBACK_ROWS, len(features_df)):
        row = features_df.iloc[i]
        timestamp = features_df.index[i]
        hist_iteration += 1

        # Flow A: synthetic book → levels → candidates → best
        order_book = build_synthetic_book(row)
        levels, median_depth, level_0_depth = build_levels(
            order_book["bids"], order_book["asks"]
        )
        buy_candidates, sell_candidates = collect_candidates(
            levels, median_depth, level_0_depth
        )
        best_buy = select_best_opportunity(buy_candidates, "buy", hist_iteration)
        best_sell = select_best_opportunity(sell_candidates, "sell", hist_iteration)

        # Flow C: VWAP from synthetic best bid/ask
        best_bid_price = float(max(order_book["bids"].keys(), key=float))
        best_ask_price = float(min(order_book["asks"].keys(), key=float))
        vol_best_bid = float(order_book["bids"][f"{best_bid_price:.2f}"])
        vol_best_ask = float(order_book["asks"][f"{best_ask_price:.2f}"])

        vwap_deque.append(
            {
                "best_bid": best_bid_price,
                "vol_bid": vol_best_bid,
                "best_ask": best_ask_price,
                "vol_ask": vol_best_ask,
            }
        )

        # ── Rolling VWAP ─────────────────────────────────────────────────────
        # deque(maxlen=VWAP_WINDOW) acts as a self-evicting rolling window:
        # once it holds VWAP_WINDOW entries the oldest is dropped automatically
        # on every append, so len() == VWAP_WINDOW for the rest of the loop.
        #
        # The else branch (None) only fires during the first VWAP_WINDOW-1
        # iterations while the window is still warming up.  Setting None makes
        # the downstream VWAP gate transparent (ask_vwap is None → any
        # micro_price passes), mirroring the live system's warm-up behaviour.
        #
        # VWAP formula (bid side shown; ask side is identical):
        #   bid_vwap = Σ(best_bid[t] × vol_bid[t]) / Σ(vol_bid[t])
        #
        # Heavier candles (larger vol_bid) pull the average more, giving a
        # momentum-aware reference price: if recent large volume traded at
        # high bid prices, bid_vwap is elevated, tightening the SELL gate.
        if len(vwap_deque) >= VWAP_WINDOW:
            bid_prices = np.array([s["best_bid"] for s in vwap_deque])
            bid_vols = np.array([s["vol_bid"] for s in vwap_deque])
            ask_prices = np.array([s["best_ask"] for s in vwap_deque])
            ask_vols = np.array([s["vol_ask"] for s in vwap_deque])
            bid_vwap = volume_weighted_average_price(bid_prices, bid_vols)
            ask_vwap = volume_weighted_average_price(ask_prices, ask_vols)
        else:
            # Window still filling — filter is transparent for both sides.
            bid_vwap = None
            ask_vwap = None

        # Flow B: HMM regime update
        # A rolling window of HMM_LOOKBACK_ROWS (120) candles is assigned to
        # rd.klines_df on every iteration.  The train/predict split and
        # StandardScaler are handled INSIDE select_hmm_model() /
        # predict_current_regime() — see regime_director.py for details.
        # In brief: oldest HMM_TRAIN_ROWS (80) rows → fit; all 120 → predict.

        # we want to catch 1m regime changes on a 120 rolling window
        rd.klines_df = features_df.iloc[i - HMM_LOOKBACK_ROWS : i]
        if hist_iteration % REFIT_EVERY == 0:
            rd.select_hmm_model()  # full BIC re-fit
        else:
            rd.predict_current_regime()  # cheap Viterbi pass
        rd.assign_regime_labels()
        regime = rd.regime_label
        # Posterior probability for the predicted regime — same value that
        # low_latency_analysis reads from rd.regime_confidence under _regime_lock.
        regime_confidence = rd.regime_confidence

        # Fill price anchor and spread for Step-4 P&L.
        # half_spread = (high - low) / 2 is the same quantity computed in
        # synthetic_book.py Step 1.  It represents the natural taker cost:
        #   BUY  fill = close + half_spread  (≡ synthetic_best_ask)
        #   SELL fill = close - half_spread  (≡ synthetic_best_bid)
        close_price = float(row["close"])
        half_spread = (float(row["high"]) - float(row["low"])) / 2.0

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
        if confidence_ok:
            if best_buy:
                regime_ok = regime not in ("trending_down", "high_volatility")
                vwap_ok = ask_vwap is None or best_buy_micro > ask_vwap
                if regime_ok and vwap_ok:
                    signal = 1

            if best_sell and signal == 0:
                regime_ok = regime not in ("trending_up", "high_volatility")
                vwap_ok = bid_vwap is None or best_sell_micro < bid_vwap
                if regime_ok and vwap_ok:
                    signal = -1

        records.append(
            {
                "timestamp": timestamp,
                "close": close_price,  # candle close — VWAP anchor
                "half_spread": half_spread,  # (high-low)/2 — taker fill cost
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
