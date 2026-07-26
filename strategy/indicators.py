import logging
import pandas as pd
import numpy as np


def add_strategy_indicators(order_book_df: pd.DataFrame, strategy: str = "buy"):
    """Add indicators to the order book DataFrame to identify potential opportunities based on the specified strategy.

    .. note::
        **REST PATH ONLY** — called exclusively by ``strategy/quotes.py`` which
        is used by the REST execution path (``restapi_main.py``).  The WebSocket
        and backtesting paths use ``strategy/book_utils.py`` directly.
        Do not remove — ``restapi_main.py`` depends on this.

    Indicators added:
    - micro_mid_delta: The difference between the micro price and the mid-price.
    - is_thin_micro_effect: Boolean indicating if the total depth is below the median depth, hinting a thin order book.
    - is_total_depth_50pct_l0: Boolean if the total depth is at least 50% of the level 0 depth, hinting enough liq.
    - w_volume_micro_spread_score: A weighted score with total depth and micro-mid delta to get potential opportunities.
    Args:
        order_book_df (pd.DataFrame): DataFrame containing order book data with calculated metrics.
        strategy (str): The trading strategy to apply. Supports "buy" for buy-side and "sell" for sell-side opportunities.
    Returns:
        pd.DataFrame: The input DataFrame with additional columns for the opportunity indicators.
    """
    if strategy == "buy":
        if not order_book_df["micro_vs_mid"].any():
            logging.info("No micro price above mid-price. No opportunities detected.")
            return order_book_df  # Return early if no opportunities
        order_book_df["micro_mid_delta"] = (
            order_book_df["micro_price"] - order_book_df["mid_price"]
        )
    elif strategy == "sell":
        if order_book_df["micro_vs_mid"].all():
            logging.info(
                "Micro price is above mid-price everywhere. No sell opportunities detected."
            )
            return order_book_df

        order_book_df["micro_mid_delta"] = (
            order_book_df["mid_price"] - order_book_df["micro_price"]
        )
    else:
        raise ValueError("Unsupported strategy. Use 'buy' or 'sell'.")

    order_book_df["is_thin_micro_effect"] = np.where(
        order_book_df["total_depth"] >= np.median(order_book_df["total_depth"]),
        False,
        True,
    ).astype(bool)
    order_book_df["is_total_depth_50pct_l0"] = (
        order_book_df["total_depth"] >= 0.5 * order_book_df["total_depth"].iloc[0]
    ).astype(bool)

    return order_book_df


def volume_weighted_average_price(
    price: pd.Series | int | float | list | np.ndarray,
    volume: pd.Series | int | float | list | np.ndarray,
) -> float:
    """Calculate the volume-weighted average price (VWAP).
    VWAP is calculated as the sum of (price * volume) divided by the sum of volume.
    Args:
        price (pd.Series | int | float | list | np.ndarray): The price(s) to be weighted.
        volume (pd.Series | int | float | list | np.ndarray): The corresponding volume(s) for the price(s).
    Returns:
        float: The calculated volume-weighted average price.
    """

    total_vol = float(np.sum(volume))
    if total_vol == 0.0:
        # Guard against all-zero volume arrays (e.g. a deque window full of
        # candles where one side of the synthetic book had buy_ratio = 0 or 1).
        # Returning NaN explicitly avoids a silent numpy RuntimeWarning and
        # makes the downstream VWAP filter transparently pass-through
        # (NaN comparisons always evaluate to False in Python).
        return float("nan")
    return float(np.sum(price * volume) / total_vol)


def add_trend_pause_flag(
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

    Shared by:
        backtest/signals.py    — vectorised over the full IS/OOS macro frame.
        strategy/analysis.py   — computed at every HMM pulse on the live klines
                                  so the live system gates entries identically
                                  to the backtest.

    Args:
        df_macro (pd.DataFrame): Must contain a ``close`` column.  Expected to
            be a 5-minute OHLCV frame (live: ``regime_director.klines_df``;
            backtest: ``features_macro``).
        n (int): Number of consecutive same-direction closes that trigger a pause.
        cooldown (int): Extra bars to keep ``paused=True`` after the streak ends.

    Returns:
        pd.Series: Boolean Series indexed identically to ``df_macro``.
    """
    close = df_macro["close"]
    # +1 up-close, -1 down-close, 0 flat; NaN at row 0 filled to 0 (no direction)
    direction = np.sign(close.diff()).fillna(0)

    # Cumulative group ID increments on every direction change.
    # cumcount()+1 gives the streak length: 1 on the first bar of each run.
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1

    # A bar "is in trend" when streak ≥ n AND direction is not flat.
    in_trend = (streak >= n) & (direction != 0)

    # Cooldown: OR with shifted versions to stay paused for `cooldown` extra bars
    # after the last in_trend=True bar (counted from the END of the streak).
    paused = in_trend.copy()
    for lag in range(1, cooldown + 1):
        paused = paused | in_trend.shift(lag).fillna(False)

    return paused.astype(bool)


def compute_live_stop_loss_pct(
    rest_client,
    symbol: str,
    rolling_days: int,
    std_mult: float,
    fetch_extra_days: int = 5,
) -> float:
    """
    Compute the adaptive stop-loss threshold (fraction) from recent
    daily Binance klines.  Mirrors the formula used by backtest/signals.py
    so the live system protects open positions with the same
    volatility-scaled threshold the IS / OOS pipelines were tuned with:

        threshold = rolling_std(daily |pct_change|, rolling_days) × std_mult

    Args:
        rest_client: A ``binance.spot.Spot`` client (only ``.klines`` is used).
            The Binance Spot REST endpoint is public for kline data, so no
            authentication is required, but reusing the live client avoids a
            second TCP connection.
        symbol (str): Trading pair (e.g. ``"BTCUSDT"``).
        rolling_days (int): Lookback window for the rolling std
            (= ``STOP_LOSS_ROLLING_DAYS``).
        std_mult (float): Multiplier applied to the rolling std
            (= ``STOP_LOSS_STD_MULT``).
        fetch_extra_days (int): Buffer added to ``rolling_days`` so the
            rolling window is warm by the time the last bar is reached.

    Returns:
        float: The latest threshold (e.g. ``0.038`` = 3.8 %).  Returns ``0.0``
            if the computation fails or yields NaN, which disables the
            stop-loss until the next refresh — fail-safe rather than fail-noisy
            because the live system should never crash on a transient data
            issue.
    """
    try:
        limit = rolling_days + fetch_extra_days
        # binance.spot.Spot.klines returns a list of 12-element lists:
        # [open_time, open, high, low, close, volume, close_time, ...].
        raw = rest_client.klines(symbol=symbol, interval="1d", limit=limit)
        if not raw:
            return 0.0
        closes = pd.Series([float(row[4]) for row in raw])  # column 4 = close
        abs_ret = closes.pct_change().abs()
        rolling_std = abs_ret.rolling(rolling_days, min_periods=1).std()
        last = float(rolling_std.iloc[-1]) * float(std_mult)
        if last != last:  # NaN check (NaN != NaN)
            return 0.0
        return last
    except Exception:
        # Live system MUST NOT crash on a transient REST failure; returning
        # 0.0 disables the stop-loss for this refresh cycle but keeps the
        # session alive.  Caller logs the failure separately.
        return 0.0


def add_macro_trend_state(
    df_macro_raw: pd.DataFrame,
    sma_days: int,
    slope_days: int,
    band_pct: float,
) -> pd.Series:
    """
    Classify each day into a macro-trend state ``{"down", "neutral", "up"}``
    on the DAILY-resampled macro close.  Drives the symmetric macro-trend
    overlay (see config_parameters.MACRO_TREND_* and backtest/signals.py):

        down    → suppress BUYs + force-liquidate to cash
        neutral → normal mean-reversion
        up      → suppress mean-reversion SELLs (hold & ride)

    Detector::

        SMA   = daily_close.rolling(sma_days).mean()
        slope = sign(SMA - SMA.shift(slope_days))
        down    if daily_close < SMA × (1 - band_pct) AND slope < 0
        up      if daily_close > SMA × (1 + band_pct) AND slope > 0
        neutral otherwise

    The band dead-zone + slope requirement provide hysteresis so the state
    does not flip every time price grazes the SMA in chop.

    NO look-ahead is applied inside this function — it returns the state on the
    daily bar it was computed from.  The CALLER (backtest/signals.py) must
    ``shift(1)`` the returned Series before merging onto the intraday frame so
    that intraday bars only ever see COMPLETED prior days.  The live sibling
    ``compute_live_macro_trend`` enforces the same rule by dropping Binance's
    in-progress current-day bar.

    Args:
        df_macro_raw (pd.DataFrame): Datetime-indexed macro (5 m) frame with a
            ``close`` column — the same frame the stop-loss resamples.
        sma_days (int): Daily-close SMA window (= ``MACRO_TREND_SMA_DAYS``).
        slope_days (int): SMA slope lookback (= ``MACRO_TREND_SLOPE_DAYS``).
        band_pct (float): ±band dead-zone around the SMA (= ``MACRO_TREND_BAND_PCT``).

    Returns:
        pd.Series: Daily-indexed object Series named ``macro_trend`` with values
            in ``{"down", "neutral", "up"}``.  Warm-up days (SMA/slope still NaN)
            fall through to ``"neutral"`` — the fail-safe "keep trading normally"
            state.
    """
    daily_close = df_macro_raw[["close"]].resample("1D").last().dropna()["close"]
    sma = daily_close.rolling(sma_days, min_periods=sma_days).mean()
    slope = np.sign(sma - sma.shift(slope_days))

    down = (daily_close < sma * (1 - band_pct)) & (slope < 0)
    up = (daily_close > sma * (1 + band_pct)) & (slope > 0)

    # np.select resolves to "neutral" wherever neither condition holds, which
    # includes the SMA/slope warm-up rows (NaN comparisons → False on both).
    state = np.select([down, up], ["down", "up"], default="neutral")
    return pd.Series(state, index=daily_close.index, name="macro_trend")


def compute_live_macro_trend(
    rest_client,
    symbol: str,
    sma_days: int,
    slope_days: int,
    band_pct: float,
    fetch_extra_days: int = 5,
) -> str:
    """
    Compute the latest COMPLETED-day macro-trend state from recent daily
    Binance klines.  Mirrors ``add_macro_trend_state`` so the live system
    gates entries and liquidates identically to the backtest.

    Look-ahead / parity note: Binance returns the in-progress current day as the
    last kline.  That bar is dropped before evaluating the state, so the live
    system reads exactly the same "last completed day" the backtest sees after
    its ``shift(1)`` — no partial-day leakage.

    Args:
        rest_client: A ``binance.spot.Spot`` client (only ``.klines`` is used).
        symbol (str): Trading pair (e.g. ``"BTCUSDT"``).
        sma_days (int): Daily-close SMA window (= ``MACRO_TREND_SMA_DAYS``).
        slope_days (int): SMA slope lookback (= ``MACRO_TREND_SLOPE_DAYS``).
        band_pct (float): ±band dead-zone (= ``MACRO_TREND_BAND_PCT``).
        fetch_extra_days (int): Buffer added to ``sma_days + slope_days`` so the
            rolling window is warm at the last completed bar.

    Returns:
        str: ``"down"``, ``"neutral"``, or ``"up"``.  Returns ``"neutral"`` on
            any failure or insufficient data — the fail-safe "keep trading
            normally" state, so a transient REST issue never crashes the session
            or spuriously liquidates the book.
    """
    try:
        # +2: one extra for the current (incomplete) day that gets dropped, one
        # so pct/slope diffs are warm at the final completed bar.
        limit = sma_days + slope_days + fetch_extra_days + 2
        raw = rest_client.klines(symbol=symbol, interval="1d", limit=limit)
        if not raw or len(raw) < sma_days + slope_days + 1:
            return "neutral"
        # Drop the last row — Binance's in-progress current day (live shift(1)).
        closes = pd.Series([float(row[4]) for row in raw[:-1]])  # col 4 = close
        sma = closes.rolling(sma_days, min_periods=sma_days).mean()
        slope = np.sign(sma - sma.shift(slope_days))

        last_close = closes.iloc[-1]
        last_sma = sma.iloc[-1]
        last_slope = slope.iloc[-1]
        if last_sma != last_sma:  # NaN check → warm-up not complete
            return "neutral"

        if last_close < last_sma * (1 - band_pct) and last_slope < 0:
            return "down"
        if last_close > last_sma * (1 + band_pct) and last_slope > 0:
            return "up"
        return "neutral"
    except Exception:
        # Live system MUST NOT crash on a transient REST failure; "neutral"
        # keeps mean-reversion running normally until the next refresh.
        return "neutral"
