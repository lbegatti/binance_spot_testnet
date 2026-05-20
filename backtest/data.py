"""
backtest/data.py
----------------
Downloads historical klines from Binance and returns clean pandas DataFrames
for the two-resolution backtest pipeline:

  • fetch_macro_klines() — 5-minute bars for GaussianHMM regime classification
  • fetch_micro_klines() — 1-minute bars for VWAP, signal generation, and PnL

Both wrappers call fetch_klines() internally and cache the result as a Parquet
file under ``cache/klines/``.  On subsequent calls within CACHE_TTL_HOURS (24 h)
the file is served from disk without touching the Binance API, saving ~2–3 min
of network I/O per session restart.

Cache invalidation
------------------
Call ``flush_kline_cache()`` (or pass ``--flush-cache`` on the CLI of
sensitivity.py / runner.py) whenever BACKTEST_LOOKBACK or BACKTEST_OOS_START
change, to force a fresh download.

Uses ``binance.client.Client.get_historical_klines`` — public endpoint, no API
key is required.
"""

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
from binance.client import Client

from config_parameters import (
    SYMBOL,
    BACKTEST_LOOKBACK,
    BACKTEST_MACRO_INTERVAL,
    BACKTEST_MICRO_INTERVAL,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parquet cache configuration
# ---------------------------------------------------------------------------
_CACHE_DIR = Path("cache/klines")
_CACHE_TTL_HOURS: float = 24.0  # re-fetch from Binance after this many hours

# Column names returned by get_historical_klines — fixed by the Binance API.
_KLINE_COLUMNS = [
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
]


# ---------------------------------------------------------------------------
# Internal cache helpers
# ---------------------------------------------------------------------------


def _cache_path(
    symbol: str, interval: str, start_str: str, end_str: str | None
) -> Path:
    """
    Return a deterministic Parquet file path for the given fetch parameters.

    Uses a short MD5 hash of the full parameter tuple to avoid filesystem
    issues with long or special-character strings, while keeping a
    human-readable prefix (``BTCUSDT_5m_<hash>.parquet``) for easy inspection.
    """
    key = f"{symbol}|{interval}|{start_str}|{end_str or 'now'}"
    slug = hashlib.md5(key.encode()).hexdigest()[:12]
    prefix = f"{symbol}_{interval}".replace(" ", "_")
    return _CACHE_DIR / f"{prefix}_{slug}.parquet"


def _is_cache_fresh(path: Path) -> bool:
    """Return True if ``path`` exists and was written within CACHE_TTL_HOURS."""
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours < _CACHE_TTL_HOURS


def flush_kline_cache() -> None:
    """
    Delete all cached Parquet files under ``cache/klines/``.

    Call this whenever BACKTEST_LOOKBACK or BACKTEST_OOS_START change so
    the next fetch pulls a fresh window from the Binance API.
    """
    if not _CACHE_DIR.exists():
        log.info("flush_kline_cache: cache directory does not exist — nothing to do.")
        return
    removed = 0
    for f in _CACHE_DIR.glob("*.parquet"):
        f.unlink()
        removed += 1
    log.info(
        "flush_kline_cache: removed %d cached file(s) from %s", removed, _CACHE_DIR
    )


def _cached_fetch(
    symbol: str,
    interval: str,
    start_str: str,
    end_str: str | None,
) -> pd.DataFrame:
    """
    Fetch klines with parquet caching.

    Cache hit  → load from ``cache/klines/<file>.parquet`` (fast, no API call).
    Cache miss → fetch from Binance API → save to parquet → return.
    """
    path = _cache_path(symbol, interval, start_str, end_str)
    if _is_cache_fresh(path):
        log.info(
            "Cache HIT  [%s %s start=%r end=%r] — loading from %s",
            symbol,
            interval,
            start_str,
            end_str,
            path.name,
        )
        return pd.read_parquet(path)

    log.info(
        "Cache MISS [%s %s start=%r end=%r] — fetching from Binance API…",
        symbol,
        interval,
        start_str,
        end_str,
    )
    df = fetch_klines(
        symbol=symbol, interval=interval, start_str=start_str, end_str=end_str
    )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log.info("Cached → %s  (%d rows)", path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_klines(
    symbol: str = SYMBOL,
    interval: str = BACKTEST_MACRO_INTERVAL,
    start_str: str = BACKTEST_LOOKBACK,
    end_str: str | None = None,
) -> pd.DataFrame:
    """
    Download historical klines from Binance and return a clean DataFrame.

    This is the low-level fetch function.  Prefer the typed wrappers
    ``fetch_macro_klines()`` and ``fetch_micro_klines()`` at call sites in
    ``sensitivity.py`` and ``runner.py`` — they enforce the correct interval
    and route through the parquet cache automatically.

    Uses the public REST endpoint — no API key is needed.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTCUSDT"``.
    interval : str
        Kline interval constant from ``binance.client.Client``,
        e.g. ``Client.KLINE_INTERVAL_5MINUTE``.
        Defaults to ``BACKTEST_MACRO_INTERVAL`` (5 minutes).
    start_str : str
        How far back to fetch, in plain English understood by the
        ``python-binance`` library, e.g. ``"360 days ago UTC"``.
    end_str : str | None
        Optional end date/time string.  When provided, only klines up to
        (but not including) this point are returned — use this to create a
        clean in-sample window that does not overlap with the OOS period.
        Example: ``BACKTEST_OOS_START = "90 days ago UTC"`` → returns rows
        from ``start_str`` to 90 days ago.
        ``None`` (default) → fetch up to the most recent complete candle.

    Returns
    -------
    pd.DataFrame
        Indexed by ``open_time`` (UTC datetime).  Numeric columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``,
        ``quote_volume``, ``num_trades``, ``taker_buy_base_vol``,
        ``taker_buy_quote_vol``.
        ``close_time`` retained as a datetime column.
        ``ignore`` column dropped.

    Examples
    --------
    >>> df_5m = fetch_klines(interval="5m", start_str="360 days ago UTC",
    ...                      end_str="90 days ago UTC")   # IS macro frame
    >>> df_1m = fetch_klines(interval="1m", start_str="360 days ago UTC",
    ...                      end_str="90 days ago UTC")   # IS micro frame
    """
    client = Client()  # public endpoint — no api_key / api_secret needed

    klines_kwargs: dict = {
        "symbol": symbol,
        "interval": interval,
        "start_str": start_str,
    }
    if end_str is not None:
        klines_kwargs["end_str"] = end_str  # omit entirely when None — avoids
        # library versions that treat explicit None differently from absent arg

    raw = client.get_historical_klines(**klines_kwargs)

    df = pd.DataFrame(raw, columns=_KLINE_COLUMNS)

    # Cast all price / volume / count columns to float.
    numeric_cols = [
        c for c in _KLINE_COLUMNS if c not in ("open_time", "close_time", "ignore")
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)

    # Parse timestamps.
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df.set_index("open_time", inplace=True)
    df.drop(columns=["ignore"], inplace=True)

    return df


def fetch_macro_klines(
    symbol: str = SYMBOL,
    lookback: str = BACKTEST_LOOKBACK,
    end_str: str | None = None,
) -> pd.DataFrame:
    """
    Fetch 5-minute klines for HMM regime classification (macro frame).

    Results are cached in ``cache/klines/`` for ``CACHE_TTL_HOURS`` (24 h).
    IS usage  : ``fetch_macro_klines(end_str=BACKTEST_OOS_START)``
    OOS usage : ``fetch_macro_klines(lookback=BACKTEST_OOS_START)``

    Parameters
    ----------
    symbol : str
        Trading pair.  Defaults to ``SYMBOL`` (``"BTCUSDT"``).
    lookback : str
        Start of the fetch window, e.g. ``"360 days ago UTC"``.
    end_str : str | None
        End of the fetch window.  ``None`` → up to today.

    Returns
    -------
    pd.DataFrame
        5-minute OHLCV DataFrame indexed by ``open_time`` (UTC).
        IS window  (~77,760 rows): ``lookback="360 days ago UTC"``,
                                    ``end_str="90 days ago UTC"``.
        OOS window (~25,920 rows): ``lookback="90 days ago UTC"``,
                                    ``end_str=None``.
    """
    return _cached_fetch(symbol, BACKTEST_MACRO_INTERVAL, lookback, end_str)


def fetch_micro_klines(
    symbol: str = SYMBOL,
    lookback: str = BACKTEST_LOOKBACK,
    end_str: str | None = None,
) -> pd.DataFrame:
    """
    Fetch 1-minute klines for VWAP, signal generation, and PnL (micro frame).

    Results are cached in ``cache/klines/`` for ``CACHE_TTL_HOURS`` (24 h).
    IS usage  : ``fetch_micro_klines(end_str=BACKTEST_OOS_START)``
    OOS usage : ``fetch_micro_klines(lookback=BACKTEST_OOS_START)``

    Parameters
    ----------
    symbol : str
        Trading pair.  Defaults to ``SYMBOL`` (``"BTCUSDT"``).
    lookback : str
        Start of the fetch window, e.g. ``"360 days ago UTC"``.
    end_str : str | None
        End of the fetch window.  ``None`` → up to today.

    Returns
    -------
    pd.DataFrame
        1-minute OHLCV DataFrame indexed by ``open_time`` (UTC).
        IS window  (~388,800 rows): ``lookback="360 days ago UTC"``,
                                     ``end_str="90 days ago UTC"``.
        OOS window (~129,600 rows): ``lookback="90 days ago UTC"``,
                                     ``end_str=None``.
    """
    return _cached_fetch(symbol, BACKTEST_MICRO_INTERVAL, lookback, end_str)
