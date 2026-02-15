import plotly.graph_objects as go

from plotly.subplots import make_subplots
from binance.client import Client

import pandas as pd


def plot_depth_bid_ask(df: pd.DataFrame):
    """Plot bid/ask prices and volumes using Plotly.
    Args:
        df (pd.DataFrame): DataFrame containing bid/ask prices (depth) and quantities.
    return:
        None: Displays the plot.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Bid/Ask Price", "Volume"),
        row_width=[0.2, 0.7],
    )  # Volume takes 20% height
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["bid_price"],
            name="Bid",
            line=dict(color="rgba(255, 0, 0, 0.5)"),
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ask_price"],
            name="Ask",
            line=dict(color="rgba(0, 255, 0, 0.5)"),
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["bid_quantity"],
            name="bid_volume",
            marker=dict(color="red"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["ask_quantity"],
            name="ask_volume",
            marker=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Bid/Ask and Volume",
        xaxis_rangeslider_visible=False,  # Hide rangeslider
        height=800,
    )
    fig.show()


def plot_ohlc_with_volume(client: Client, symbol: str, interval: str, lookback: str):
    """Fetch historical klines for a symbol and plot OHLC with volume.
    Args:
        client (Client): Binance API client instance.
        symbol (str): Trading pair symbol (e.g., "BNBUSDT").
        interval (str): Kline interval (e.g., Client.KLINE_INTERVAL_1MINUTE).
        lookback (str): Lookback period for historical data (e.g., "5 day ago UTC").
    return:
        None: Displays the plot.
    """
    klines = pd.DataFrame(
        client.get_historical_klines(
            symbol=symbol, interval=interval, start_str=lookback
        ),
        columns=[
            "Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
        ],
    )[["Time", "Open", "High", "Low", "Close", "Volume"]]
    klines["Time"] = pd.to_datetime(klines["Time"], unit="ms")
    klines.set_index("Time", inplace=True)
    ohlc_plot = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]
    )
    ohlc_plot.add_trace(
        go.Ohlc(
            x=klines.index,
            open=klines["Open"].astype(float),
            high=klines["High"].astype(float),
            low=klines["Low"].astype(float),
            close=klines["Close"].astype(float),
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    ohlc_plot.add_trace(
        go.Bar(
            x=klines.index,
            y=klines["Volume"].astype(float),
            name="Volume",
            marker=dict(color="black"),
        ),
        row=2,
        col=1,
    )
    ohlc_plot.update_layout(
        title=f"OHLC & Volume, {symbol} - {interval} - {lookback}",
        xaxis_rangeslider_visible=False,  # Hide rangeslider
        height=800,
    )
    ohlc_plot.show()
