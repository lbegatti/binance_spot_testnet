import numpy as np
from binance.client import Client
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from config_parameters import (
    HMM_FEATURE_COLS,
    HMM_N_ITERATIONS,
    HMM_RANDOM_STATE,
    HMM_MAX_REGIMES,
    HMM_INTERVAL,
    HMM_LOOKBACK,
    SYMBOL,
)
import logging


class RegimeDirector:
    """
    Detects the current market regime by training a Gaussian Hidden Markov
    Model (``GaussianHMM``) on recent Binance kline data.

    The model is fitted on four features derived from OHLCV candles:

    * ``return``        — per-candle percentage price change
      (``close.pct_change()``).
    * ``volatility``    — normalised intra-bar range
      (``(high - low) / close``).
    * ``obi_proxy``     — taker-flow imbalance proxy, rescaled to ``[-1, +1]``
      (``(taker_buy_base_vol / volume) × 2 − 1``).  Approximates the live OBI
      when full order-book history is unavailable.
    * ``trade_density`` — trade fragmentation
      (``num_trades / volume``): high → many small trades (retail / HFT);
      low → few large trades (institutional blocks).

    The best number of hidden states (2 … ``HMM_MAX_REGIMES``) is selected
    automatically via the **Bayesian Information Criterion (BIC)** — the model
    with the lowest BIC is retained.

    State labels are assigned in ``assign_regime_labels()`` by comparing each
    state's feature means against cross-state statistics (mean ± k × std),
    so no domain-specific price constants are hard-coded.  Possible labels:
    ``"trending_up"``, ``"trending_down"``, ``"high_volatility"``,
    ``"active_neutral"``, ``"neutral"``.

    **Intended usage**:

    1. Instantiate once in ``websocket_main.py`` before threads start.
    2. Call ``get_klines_data()`` → ``select_hmm_model()`` →
       ``assign_regime_labels()`` for the initial fit.
    3. ``AnalysisEngine.historical_analysis()`` repeats steps 2–4 every
       ``HIST_INTERVAL`` seconds so the label stays current throughout the
       trading session.
    4. ``AnalysisEngine.low_latency_analysis()`` reads ``regime_label`` under
       ``_regime_lock`` and uses it to gate order execution.

    Attributes:
        symbol (str): Trading pair (default ``"BTCUSDT"``).
        interval (str): Kline interval for data download (``HMM_INTERVAL``).
        lookback (str): How far back to fetch klines (``HMM_LOOKBACK``).
        max_states (int): Maximum number of HMM states to evaluate (``HMM_MAX_REGIMES``).
        random_state (int): Random seed for reproducible HMM initialisation.
        n_iterations (int): Maximum EM iterations per model fit.
        client (Client): Binance public REST client (no auth needed for klines).
        model (GaussianHMM | None): Best-BIC fitted model; ``None`` until
            ``select_hmm_model()`` is called.
        klines_df (pd.DataFrame | None): Feature DataFrame; ``None`` until
            ``get_klines_data()`` is called.
        regimes (np.ndarray | None): Predicted state sequence for the full
            kline window; ``None`` until ``select_hmm_model()`` is called.
        current_regime (int | None): State index of the most recent candle.
        regime_label (str | None): Human-readable label for ``current_regime``;
            ``None`` until ``assign_regime_labels()`` is called.
    """

    def __init__(
        self,
        symbol: str = SYMBOL,
        interval: str = HMM_INTERVAL,
        lookback: str = HMM_LOOKBACK,
        random_state: int = HMM_RANDOM_STATE,
        n_iterations: int = HMM_N_ITERATIONS,
        max_regimes: int = HMM_MAX_REGIMES,
    ):
        """
        Args:
            symbol (str): Binance trading pair to fetch klines for.
            interval (str): Kline granularity (e.g. ``Client.KLINE_INTERVAL_1MINUTE``).
                Pulled from ``HMM_INTERVAL`` in ``config_parameters.py``.
            lookback (str): How far back to download data
                (e.g. ``"4 hours ago UTC"``).  Pulled from ``HMM_LOOKBACK``.
            random_state (int): Seed for ``GaussianHMM`` initialisation to
                ensure reproducible state numbering across runs.
            n_iterations (int): Maximum number of Expectation–Maximisation
                iterations per model.
            max_regimes (int): Upper bound on the number of hidden states
                evaluated during BIC search (states tested: 2 … max_regimes).
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.max_states = max_regimes
        self.random_state = random_state
        self.n_iterations = n_iterations
        self.client = Client()

        self.model: GaussianHMM | None = None
        self.klines_df: pd.DataFrame | None = None
        self.regimes: np.ndarray | None = None
        self.current_regime: int | None = None
        self.regime_label: str | None = None

    def get_klines_data(self):
        """
        Download recent klines from Binance and compute the four HMM features.

        Uses the public Binance REST endpoint (no API key required).  Fetches
        ``HMM_LOOKBACK`` worth of ``HMM_INTERVAL`` candles for ``self.symbol``
        and builds the following columns:

        * ``return``        — ``close.pct_change()``
        * ``volatility``    — ``(high - low) / close``
        * ``obi_proxy``     — ``(taker_buy_base_vol / volume) × 2 − 1``
        * ``trade_density`` — ``num_trades / volume``

        The first row is dropped (``pct_change`` produces ``NaN``).
        The resulting DataFrame is stored in ``self.klines_df``.

        Raises:
            Any exception raised by ``binance.client.Client.get_historical_klines``
            (e.g. network errors, rate-limit violations) propagates to the caller.
        """
        raw_data = self.client.get_historical_klines(
            symbol=self.symbol,
            interval=self.interval,
            start_str=self.lookback,
        )
        df = pd.DataFrame(
            raw_data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "num_trades",
                "taker_buy_base_vol",
                "taker_buy_quote_vol",
                "ignore",
            ],
        ).astype(float)
        df[["open_time", "close_time"]] = df[["open_time", "close_time"]].apply(
            pd.to_datetime, unit="ms"
        )
        df.set_index("open_time", inplace=True)
        df["return"] = df["close"].pct_change()
        df["volatility"] = (df["high"] - df["low"]) / df["close"]
        df["obi_proxy"] = (df["taker_buy_base_vol"] / df["volume"]) * 2 - 1
        df["trade_density"] = df["num_trades"] / df["volume"]
        df.dropna(inplace=True)

        self.klines_df = df

    def select_hmm_model(self):
        """
        Fit ``GaussianHMM`` models for ``n = 2 … self.max_states`` hidden states
        and select the best one by **Bayesian Information Criterion (BIC)**.

        For each candidate ``n``:
        * A ``GaussianHMM`` with full covariance is fitted on the feature
          matrix from ``self.klines_df[HMM_FEATURE_COLS]``.
        * BIC is computed as ``model.bic(features)`` — lower is better.

        The model with the lowest BIC is stored in ``self.model``.  The full
        state sequence is predicted via ``self.model.predict(features)`` and
        stored in ``self.regimes``.  The state of the **last candle** is stored
        in ``self.current_regime`` as a plain ``int``.

        Requires ``get_klines_data()`` to have been called first.

        Raises:
            AttributeError: if ``self.klines_df`` is ``None`` (``get_klines_data``
                not yet called).
        """
        features = self.klines_df[HMM_FEATURE_COLS].values
        best_hmm_model, best_hmm_bic = None, np.inf
        for n in range(2, self.max_states + 1):
            m = GaussianHMM(
                n_components=n,
                covariance_type="full",
                n_iter=self.n_iterations,
                random_state=self.random_state,
            )
            m.fit(features)
            # TODO numpy.linalg.LinAlgError: 2-th leading minor of the array is not positive definite
            bic = m.bic(features)
            logging.info("RegimeDetector: n=%d  BIC=%.1f", n, bic)
            if bic < best_hmm_bic:
                best_hmm_bic, best_hmm_model = bic, m
        self.model = best_hmm_model
        self.regimes = self.model.predict(features)
        self.current_regime = int(self.regimes[-1])
        logging.info(
            "RegimeDirector: best n=%d current_regime=%d",
            self.model.n_components,
            self.current_regime,
        )

    def assign_regime_labels(self):
        """
        Assign a human-readable label to each HMM state and expose the label
        for the current (latest) candle via ``self.regime_label``.

        Labels are derived **entirely** from the model's own learned parameters
        (``model.means_``) — no domain-specific price thresholds are hard-coded.
        For each feature the cross-state mean and standard deviation are
        computed, then boolean flags are set per state:

        * ``high_return``   — return  > cross-state mean + 0.5 × std
        * ``low_return``    — return  < cross-state mean − 0.5 × std
        * ``buy_pressure``  — obi     > cross-state mean + 0.5 × std
        * ``sell_pressure`` — obi     < cross-state mean − 0.5 × std
        * ``high_vol``      — volatility > cross-state mean + 1 × std
        * ``high_td``       — trade_density > cross-state mean + 0.5 × std
          (many small trades — fragmented / retail activity)
        * ``low_td``        — trade_density < cross-state mean − 0.5 × std
          (few large trades — institutional blocks)

        Label assignment priority (first matching rule wins):

        ===========================  ==========================================
        Condition                    Label
        ===========================  ==========================================
        high_return + buy_pressure   ``"trending_up"``
        low_return + sell_pressure   ``"trending_down"``
        high_vol                     ``"high_volatility"``
        high_td + not high_vol       ``"active_neutral"``
        (default)                    ``"neutral"``
        ===========================  ==========================================

        .. note::
            ``low_td`` is **not** required for trending regimes.  Requiring
            all three features to simultaneously exceed cross-state thresholds
            is too strict in practice (most states fall through to ``"neutral"``
            with a triple-AND rule).  ``high_vol`` alone is sufficient to
            classify a state as ``"high_volatility"`` — adding ``high_td`` as a
            second required condition suppresses that label too aggressively.

        ``self.regime_label`` is set to the label of ``self.current_regime``
        and is safe to read from another thread once this method returns,
        provided the caller holds ``_regime_lock`` (as ``historical_analysis``
        does).

        Requires ``select_hmm_model()`` to have been called first.

        Returns:
            str: The regime label for the current candle.
        """
        means = pd.DataFrame(self.model.means_, columns=HMM_FEATURE_COLS)

        # --- cross-state thresholds, fully data-driven ---
        # so this is the mean across the states != the self.model.means_ so we do not need to hardcode the threshold.
        std_r = means["return"].std()
        mean_r = means["return"].mean()

        mean_obi = means["obi_proxy"].mean()
        std_obi = means["obi_proxy"].std()

        mean_vol = means["volatility"].mean()
        std_vol = means["volatility"].std()

        mean_td = means["trade_density"].mean()
        std_td = means["trade_density"].std()

        labels = {}

        for state in range(self.model.n_components):
            r = means.loc[state, "return"]
            obi = means.loc[state, "obi_proxy"]
            vol = means.loc[state, "volatility"]
            td = means.loc[state, "trade_density"]

            # boolean - thresholds are derived from cross-state statistics and are dynamically adapted to the data.
            high_return = r > mean_r + 0.5 * std_r
            low_return = r < mean_r - 0.5 * std_r
            buy_pressure = obi > mean_obi + 0.5 * std_obi
            sell_pressure = obi < mean_obi - 0.5 * std_obi
            high_vol = vol > mean_vol + std_vol  # 1std above avg volatility
            high_td = td > mean_td + 0.5 * std_td  # above-average fragmentation
            low_td = (
                td < mean_td - 0.5 * std_td
            )  # below-average fragmentation (large blocks)

            # Priority: direction first, then volatility, then activity, then default.
            # Only 2 conditions required for directional regimes — requiring a
            # third (low_td) was too strict and caused almost every state to
            # fall through to "neutral".
            if high_return and buy_pressure:
                labels[state] = "trending_up"
            elif low_return and sell_pressure:
                labels[state] = "trending_down"
            elif high_vol:                          # vol alone is enough
                labels[state] = "high_volatility"
            elif high_td and not high_vol:
                labels[state] = "active_neutral"
            else:
                labels[state] = "neutral"

            logging.info(
                "RegimeDirector: state=%d label='%s' | r=%.5f vol=%.5f obi=%.4f td=%.6f",
                state, labels[state], r, vol, obi, td,
            )

        self.regime_label = labels[self.current_regime]
        logging.info(
            "RegimeDirector: current_regime state=%d → label='%s'",
            self.current_regime,
            self.regime_label,
        )
        logging.info("RegimeDirector: state labels → %s", labels)
        logging.info("RegimeDirector: current regime → '%s'", self.regime_label)

        return self.regime_label
