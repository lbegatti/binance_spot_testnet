"""
backtest/data.py
----------------
Downloads historical 1-minute klines from Binance and returns a clean
``pandas.DataFrame`` ready for synthetic order book construction
(``backtest/synthetic_book.py``) and HMM feature computation.

Uses ``binance.client.Client.get_historical_klines`` — the same library and
method already used by ``RegimeDirector`` in ``strategy/regime_director.py``.
No API key is required; kline data is a public endpoint.
"""

import pandas as pd
from binance.client import Client

from config_parameters import SYMBOL, BACKTEST_LOOKBACK

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


def fetch_klines(
    symbol: str = SYMBOL,
    interval: str = Client.KLINE_INTERVAL_1MINUTE,
    start_str: str = BACKTEST_LOOKBACK,
) -> pd.DataFrame:
    """
    Download historical klines from Binance and return a clean DataFrame.

    Uses the public REST endpoint — no API key is needed.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTCUSDT"``.
    interval : str
        Kline interval constant from ``binance.client.Client``,
        e.g. ``Client.KLINE_INTERVAL_1MINUTE``.  Defaults to 1 minute.
    start_str : str
        How far back to fetch, in plain English understood by the
        ``python-binance`` library, e.g. ``"30 days ago UTC"``.
        At 1 m resolution, 30 days ≈ 43 200 rows.

    Returns
    -------
    pd.DataFrame
        Indexed by ``open_time`` (UTC datetime).  Numeric columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``,
        ``quote_volume``, ``num_trades``, ``taker_buy_base_vol``,
        ``taker_buy_quote_vol``.
        ``close_time`` retained as a datetime column.
        ``ignore`` column dropped.
        No NaN rows — first row dropped because ``return`` (pct_change)
        would be NaN.

    Examples
    --------
    >>> df = fetch_klines()          # 30 days of BTCUSDT 1 m klines
    >>> df.shape                     # (~43 200, 10)
    >>> df.columns.tolist()
    ['open', 'high', 'low', 'close', 'volume', 'close_time',
     'quote_volume', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol']
    """
    client = Client()  # public endpoint — no api_key / api_secret needed

    raw = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
    )

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
