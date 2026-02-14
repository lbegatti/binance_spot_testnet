import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from binance.client import Client

# 1. Load environment variables from .env file
load_dotenv()

api_key = os.getenv("BINANCE_TESTNET_API_KEY")
api_secret = os.getenv("BINANCE_TESTNET_SECRET_KEY")

if not api_key or not api_secret:
    raise ValueError("API keys not found. Check your .env file.")

# 2. Initialize the Client
# The 'testnet=True' parameter tells the library to use the testnet.binance.vision URL
client = Client(api_key, api_secret, testnet=True)

# 3. Example: Get account information
balance = client.get_asset_balance(asset="BNB", recvWindow=5000)

# 4. Get different bids and asks for a symbol.
depth = client.get_order_book(symbol="BNBUSDT")
max_depth = min(len(depth["bids"]), len(depth["asks"]))
depth_bids = pd.DataFrame(
    depth["bids"][:max_depth], columns=["bid_price", "bid_quantity"], dtype=float
)
depth_asks = pd.DataFrame(
    depth["asks"][:max_depth], columns=["ask_price", "ask_quantity"], dtype=float
)
depths_bid_ask = pd.concat([depth_bids, depth_asks], axis=1)

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
        x=depths_bid_ask.index,
        y=depths_bid_ask["bid_price"],
        name="Bid",
        line=dict(color="rgba(255, 0, 0, 0.5)"),
        mode="lines",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=depths_bid_ask.index,
        y=depths_bid_ask["ask_price"],
        name="Ask",
        line=dict(color="rgba(0, 255, 0, 0.5)"),
        mode="lines",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=depths_bid_ask.index,
        y=depths_bid_ask["bid_quantity"],
        name="bid_volume",
        marker=dict(color="red"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=depths_bid_ask.index,
        y=depths_bid_ask["ask_quantity"],
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


## 4.0 Get historical klines for a symbol
# klines_1s = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1SECOND, "1 day ago UTC")


def plot_ohlc_with_volume(symbol: str, interval: str, lookback: str):
    """Fetch historical klines for a symbol and plot OHLC with volume."""
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


plot_ohlc_with_volume('BNBUSDT', Client.KLINE_INTERVAL_1MINUTE, "5 day ago UTC")

# klines_5m = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_5MINUTE, "1 day ago UTC")
# klines_3m = client.get_historical_klines_generator("BNBBTC", Client.KLINE_INTERVAL_3MINUTE, "1 day ago UTC")

# 4.1 Get order book depth for a symbol
# 4.1.1. Analyze the order book depth for a symbol
# 4.2 Get recent trades for a symbol
# 4.3 Get historical trades for a symbol
# 4.4 Get aggregate trades for a symbol
