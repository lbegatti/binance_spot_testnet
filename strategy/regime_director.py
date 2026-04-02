import numpy as np
from binance.client import Client
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from config_parameters import (
    HMM_FEATURE_COLS,
    HMM_N_ITERATIONS,
    HMM_RANDOM_STATE,
    HMM_MAX_REGIMES,
    HMM_INTERVAL,
    HMM_LOOKBACK,
    HMM_MIN_COVAR,
    HMM_TRAIN_ROWS,
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

    A **train / predict split** is applied in ``select_hmm_model()``: the model
    is fitted only on the first ``HMM_TRAIN_ROWS`` rows of the downloaded
    window (older, in-sample data).  Regime inference (Viterbi) then runs on
    the **full** window, so ``current_regime`` always reflects the latest
    candle, which was genuinely out-of-sample during training.

    State labels are assigned in ``assign_regime_labels()`` using a rank-based
    directional assignment and volatility thresholds.  Possible labels:
    ``"trending_up"``, ``"trending_down"``, ``"high_volatility"``, ``"neutral"``.

    **Intended usage**:

    1. Instantiate once in ``websocket_main.py`` before threads start.
    2. Call ``get_klines_data()`` → ``select_hmm_model()`` →
       ``assign_regime_labels()`` for the initial fit.
    3. ``AnalysisEngine.historical_analysis()`` then applies a **two-speed**
       update every ``HIST_INTERVAL`` seconds:

       * **Cheap path (every iteration):** ``get_klines_data()`` →
         ``predict_current_regime()`` → ``assign_regime_labels()``.
         Runs the Viterbi algorithm on the existing model — O(n × k).
       * **Full re-fit (every ``HMM_REFIT_INTERVAL`` seconds, default 5 min):**
         ``get_klines_data()`` → ``select_hmm_model()`` →
         ``assign_regime_labels()``.  Replaces the model entirely.

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
        scaler (StandardScaler | None): ``sklearn`` scaler fitted on the
            training rows only.  Used to z-score all four features before
            ``fit()`` and ``predict()``.  ``None`` until
            ``select_hmm_model()`` is called.
        regime_confidence (float | None): Posterior probability assigned by
            ``predict_proba()`` to ``current_regime`` for the latest candle
            (``proba[-1][current_regime]``).  Range ``[0.0, 1.0]``.
            ``None`` until ``select_hmm_model()`` is called.
            Used by ``AnalysisEngine`` to gate orders: executions are skipped
            when ``regime_confidence < HMM_MIN_CONFIDENCE``.
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
                (e.g. ``"2 hours ago UTC"``).  Pulled from ``HMM_LOOKBACK``.
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
        self.scaler: StandardScaler | None = None
        self.regime_confidence: float | None = None

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

        A **train / predict split** is applied to avoid overfitting the model
        to the most recent candles:

        * **Training set** — the first ``HMM_TRAIN_ROWS`` rows of
          ``klines_df`` (older, in-sample data).  All ``fit()`` and ``bic()``
          calls use only this slice.  At the default ``HMM_LOOKBACK`` of 2 h
          (≈ 120 rows) and ``HMM_TRAIN_ROWS = 80``, roughly the oldest ⅔ of
          the window is used for training.
        * **Prediction set** — the **full** feature matrix.  After the best
          model is selected, ``model.predict()`` runs on all rows so that
          ``self.current_regime`` reflects the very latest candle, which was
          genuinely out-of-sample during training.

        **Feature scaling** via ``StandardScaler`` is applied before any
        model call:

        * The scaler is ``fit_transform``'d on **training rows only** —
          mean and standard deviation from held-out (recent) candles cannot
          leak into the model.
        * The same scaler is ``transform``'d onto the full window for
          prediction so that both sets live in the same z-score space.
        * This is necessary because ``trade_density`` (``num_trades / volume``)
          can be orders of magnitude larger than ``return`` or ``obi_proxy``
          (which sit in ``[-1, +1]``).  Without scaling the largest feature
          would dominate the covariance structure and inflate BIC for all
          candidate ``n``.
        * The fitted ``StandardScaler`` is stored in ``self.scaler`` so that
          ``predict_current_regime()`` can reuse the exact same transform
          without refitting.

        For each candidate ``n``:

        * A ``GaussianHMM`` with full covariance is fitted on
          ``train_features`` (not the whole window).
        * ``min_covar=HMM_MIN_COVAR`` is passed to the model so that a small
          regularisation constant is added to the diagonal of every state's
          covariance matrix, preventing ``"covars must be symmetric,
          positive-definite"`` errors when a state has too few observations.
        * If ``fit()`` or ``bic()`` still raise ``ValueError`` or
          ``numpy.linalg.LinAlgError`` (e.g. extreme feature values), that
          ``n`` is skipped with a ``WARNING`` log and the search continues.
        * BIC is computed as ``model.bic(train_features)`` — lower is better.

        The model with the lowest BIC is stored in ``self.model``.  The full
        state sequence is predicted via ``self.model.predict(features)`` and
        stored in ``self.regimes``.  The state of the **last candle** is stored
        in ``self.current_regime`` as a plain ``int``.

        Requires ``get_klines_data()`` to have been called first.

        Raises:
            AttributeError: if ``self.klines_df`` is ``None`` (``get_klines_data``
                not yet called).
            RuntimeError: if every candidate ``n`` fails to produce a valid
                covariance matrix.  Increase ``HMM_MIN_COVAR`` or decrease
                ``HMM_MAX_REGIMES`` in ``config_parameters.py`` to resolve.
        """
        features = self.klines_df[HMM_FEATURE_COLS].values

        # --- train / predict split ---
        # Train only on the older in-sample rows; the most recent rows are
        # intentionally held out so that self.current_regime reflects a regime
        # that the model has never "seen" during fitting.
        train_features = features[:HMM_TRAIN_ROWS]

        # --- feature scaling ---
        # Fit the scaler ONLY on the training rows so that the mean and std
        # of recent (held-out) candles cannot leak into the model.
        # transform() is then applied to the full window for prediction.
        # This is important because trade_density (num_trades / volume) can
        # be orders of magnitude larger than return or obi_proxy [-1, +1],
        # which would otherwise dominate the covariance structure.
        self.scaler = StandardScaler()
        train_features_scaled = self.scaler.fit_transform(train_features)
        features_scaled = self.scaler.transform(features)

        logging.info(
            "RegimeDirector: fitting on %d rows (train), predicting on %d rows (full)",
            len(train_features),
            len(features),
        )

        best_hmm_model, best_hmm_bic = None, np.inf
        for n in range(2, self.max_states + 1):
            m = GaussianHMM(
                n_components=n,
                covariance_type="full",
                n_iter=self.n_iterations,
                random_state=self.random_state,
                min_covar=HMM_MIN_COVAR,  # regularisation floor — keeps covariance
                # matrices positive-definite when a state
                # has few observations
            )
            try:
                m.fit(train_features_scaled)
                bic = m.bic(train_features_scaled)
            except (ValueError, np.linalg.LinAlgError) as exc:
                # Covariance matrix became singular for this n — skip and try next
                logging.warning(
                    "RegimeDirector: n=%d skipped — covariance error (%s)", n, exc
                )
                continue
            logging.info("RegimeDirector: n=%d  BIC=%.1f", n, bic)
            if bic < best_hmm_bic:
                best_hmm_bic, best_hmm_model = bic, m

        if best_hmm_model is None:
            raise RuntimeError(
                "RegimeDirector: all GaussianHMM fits failed. "
                "Try increasing HMM_MIN_COVAR or reducing HMM_MAX_REGIMES."
            )

        self.model = best_hmm_model
        # Predict on the FULL scaled window — out-of-sample rows included
        self.regimes = self.model.predict(features_scaled)
        self.current_regime = int(self.regimes[-1])

        # Posterior probability of the winning state for the latest candle.
        # predict_proba() runs Forward-Backward (vs Viterbi for predict()):
        # it returns the marginal probability of each state at each time step.
        # We only need the last row [-1] and the current state's column.
        proba = self.model.predict_proba(features_scaled)
        self.regime_confidence = float(proba[-1, self.current_regime])

        logging.info(
            "RegimeDirector: best n=%d current_regime=%d confidence=%.2f",
            self.model.n_components,
            self.current_regime,
            self.regime_confidence,
        )

    def predict_current_regime(self) -> None:
        """
        Update ``current_regime`` using the **already-fitted** model — no
        re-training.

        This is the cheap alternative to ``select_hmm_model()``.  It runs the
        Viterbi algorithm (``model.predict()``) on the feature matrix that was
        last downloaded by ``get_klines_data()`` and updates
        ``self.current_regime`` to the state of the **last candle**.
        ``self.regimes`` (the full state sequence) is also refreshed so that
        ``assign_regime_labels()`` always operates on a current sequence.

        Complexity vs. ``select_hmm_model()``:

        * ``select_hmm_model()`` — O(n × k × iterations) per candidate model,
          repeated for every ``n = 2 … max_states``.  Expensive.
        * ``predict_current_regime()`` — O(n × k) single Viterbi pass on the
          existing model.  ~1000× cheaper for ``n_iter = 1000``.

        Call pattern expected by ``AnalysisEngine.historical_analysis()``:

        .. code-block:: text

            every HIST_INTERVAL (60 s):
                get_klines_data()          # refresh self.klines_df
                predict_current_regime()   # cheap Viterbi inference
                assign_regime_labels()     # map state → label (under _regime_lock)

            every HMM_REFIT_INTERVAL (300 s):
                get_klines_data()          # refresh self.klines_df
                select_hmm_model()         # full re-fit (replaces predict step)
                assign_regime_labels()     # under _regime_lock

        Requires:
            * ``get_klines_data()`` must have been called first so that
              ``self.klines_df`` is populated.
            * ``select_hmm_model()`` must have been called at least once so
              that ``self.model`` and ``self.scaler`` are not ``None``.

        Raises:
            RuntimeError: if ``self.model`` is ``None`` (no prior fit exists).
            AttributeError: if ``self.klines_df`` is ``None`` (``get_klines_data``
                not yet called).
        """
        if self.model is None:
            raise RuntimeError(
                "predict_current_regime() called before select_hmm_model(). "
                "Run the initial fit first."
            )
        features = self.klines_df[HMM_FEATURE_COLS].values
        # Reuse the scaler fitted during select_hmm_model() so the feature
        # distribution seen by the model is identical to training time.
        features_scaled = self.scaler.transform(features)
        self.regimes = self.model.predict(features_scaled)
        self.current_regime = int(self.regimes[-1])

        # Refresh the posterior confidence for the latest candle.
        proba = self.model.predict_proba(features_scaled)
        self.regime_confidence = float(proba[-1, self.current_regime])

        logging.info(
            "RegimeDirector: predict (no refit) → current_regime=%d confidence=%.2f",
            self.current_regime,
            self.regime_confidence,
        )

    def assign_regime_labels(self):
        """
        Assign a human-readable label to each HMM state and expose the label
        for the current (latest) candle via ``self.regime_label``.

        **Directional labels** are assigned by ranking all states on a combined
        score of ``return.rank() + obi_proxy.rank()``:

        * ``trending_up``   — state with the **highest** combined rank
          (highest mean return + strongest buy-side order flow).
        * ``trending_down`` — state with the **lowest** combined rank
          (lowest mean return + weakest / most negative order flow).

        This guarantees **exactly one state per directional label** regardless
        of ``n_components``, eliminating the duplicate-label problem that
        arises when multiple states simultaneously exceed a threshold.

        **Secondary labels** are assigned to the remaining states:

        ==========================================  ===========================
        Condition                                   Label
        ==========================================  ===========================
        high_vol OR high_td                         ``"high_volatility"``
        (default)                                   ``"neutral"``
        ==========================================  ===========================

        ``"high_volatility"`` is triggered by **either**:

        * Large intra-bar price swings (``volatility > mean + 1 × std``) — the
          market is moving too fast for limit orders to fill predictably.
        * High trade fragmentation (``trade_density > mean + 0.5 × std``) —
          many small retail/HFT orders with no clear directional intent, making
          the order-book signal unreliable.

        Both conditions independently indicate an **unreliable market** to
        trade in: one from the price side (large swings), the other from the
        flow side (fragmented activity).  Together they give ``trade_density``
        a meaningful role beyond logging.

        ``self.regime_label`` is set to the label of ``self.current_regime``
        and is safe to read from another thread once this method returns,
        provided the caller holds ``_regime_lock`` (as ``historical_analysis``
        does).

        Requires ``select_hmm_model()`` to have been called first.

        Returns:
            str: The regime label for the current candle.
        """
        means = pd.DataFrame(self.model.means_, columns=HMM_FEATURE_COLS)

        # Thresholds for secondary labels
        mean_vol = means["volatility"].mean()
        std_vol = means["volatility"].std()
        mean_td = means["trade_density"].mean()
        std_td = means["trade_density"].std()

        # --- rank-based directional assignment ---
        # Sum the ordinal rank of each state on return and obi_proxy.
        # idxmax / idxmin are exclusive by definition → no duplicate
        # trending_up / trending_down labels regardless of n_components.
        direction_score = means["return"].rank() + means["obi_proxy"].rank()
        best_state = int(direction_score.idxmax())
        worst_state = int(direction_score.idxmin())

        labels = {}

        for state in range(self.model.n_components):
            vol = means.loc[state, "volatility"]
            td = means.loc[state, "trade_density"]

            # large intra-bar swings → unpredictable fills
            high_vol = vol > mean_vol + std_vol
            # many small fragmented trades → no clear directional intent
            high_td = td > mean_td + 0.5 * std_td

            if state == best_state:
                labels[state] = "trending_up"
            elif state == worst_state:
                labels[state] = "trending_down"
            elif high_vol or high_td:
                # Either large price swings OR heavy fragmentation makes the
                # market unreliable to trade in — both signal noise over signal.
                labels[state] = "high_volatility"
            else:
                labels[state] = "neutral"

            logging.info(
                "RegimeDirector: state=%d label='%s' | r=%.5f vol=%.5f obi=%.4f td=%.6f"
                " | direction_score=%.2f",
                state,
                labels[state],
                means.loc[state, "return"],
                vol,
                means.loc[state, "obi_proxy"],
                td,
                direction_score[state],
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
