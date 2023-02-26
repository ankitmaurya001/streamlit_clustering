import streamlit as st
import requests
import os
import sys
import subprocess

# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):
    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    os.system("ls -la /app/equity/")
    # build
    os.system("./configure --prefix=/home/appuser")
    os.system("make")
    # install
    os.system("make install")
    # back to the cwd
    os.chdir(default_cwd)
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

os.system("ls -la /home/appuser/lib/libta*")
lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
# import library
try:
    import talib
except ImportError:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--global-option=build_ext",
            "--global-option=-L/home/appuser/lib/",
            "--global-option=-I/home/appuser/include/",
            "ta-lib==0.4.0",
        ]
    )
finally:
    import talib

# here goes your code


import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# import talib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import plotly.express as px

snp500 = pd.read_csv("Datasets/ind_nifty500list.csv")
symbols = snp500["Symbol"].sort_values().tolist()


def get_percent_wick_in_opposite_direction(pct_change, close, open, high, low):
    if pct_change >= 0:
        # get the wick from open to low
        return (open - low) / open * 100
    else:
        # get the wick from close to high
        return (high - close) / close * 100


def get_whether_body_more_than_wick(pct_change, wick_pct_change):
    if abs(pct_change) > wick_pct_change:
        return 1
    else:
        return 0


st.set_page_config(page_title="Market Profile Chart (India Nifty 500)", layout="wide")


ticker = st.sidebar.selectbox("Choose a Nifty 500 Stock", symbols)

i = st.sidebar.selectbox("Interval", ("1h", "1d", "5d", "1wk", "1mo"))


p = st.sidebar.number_input(
    "How many days (21-6000)", min_value=21, max_value=6000, step=1
)

clustering = st.sidebar.selectbox("Clustering", ("PCA", "T-SNE"))
ticks = st.sidebar.number_input(
    "Number of ticks to visualize", min_value=10, max_value=400, step=1
)

stock = yf.Ticker(ticker + ".NS")
history_data = stock.history(interval=i, period=str(p) + "d")

prices = history_data["Close"]
volumes = history_data["Volume"]

lower = prices.min()
upper = prices.max()

prices_ax = np.linspace(lower, upper, num=20)

vol_ax = np.zeros(20)

for i in range(0, len(volumes)):
    if prices[i] >= prices_ax[0] and prices[i] < prices_ax[1]:
        vol_ax[0] += volumes[i]

    elif prices[i] >= prices_ax[1] and prices[i] < prices_ax[2]:
        vol_ax[1] += volumes[i]

    elif prices[i] >= prices_ax[2] and prices[i] < prices_ax[3]:
        vol_ax[2] += volumes[i]

    elif prices[i] >= prices_ax[3] and prices[i] < prices_ax[4]:
        vol_ax[3] += volumes[i]

    elif prices[i] >= prices_ax[4] and prices[i] < prices_ax[5]:
        vol_ax[4] += volumes[i]

    elif prices[i] >= prices_ax[5] and prices[i] < prices_ax[6]:
        vol_ax[5] += volumes[i]

    elif prices[i] >= prices_ax[6] and prices[i] < prices_ax[7]:
        vol_ax[6] += volumes[i]

    elif prices[i] >= prices_ax[7] and prices[i] < prices_ax[8]:
        vol_ax[7] += volumes[i]

    elif prices[i] >= prices_ax[8] and prices[i] < prices_ax[9]:
        vol_ax[8] += volumes[i]

    elif prices[i] >= prices_ax[9] and prices[i] < prices_ax[10]:
        vol_ax[9] += volumes[i]

    elif prices[i] >= prices_ax[10] and prices[i] < prices_ax[11]:
        vol_ax[10] += volumes[i]

    elif prices[i] >= prices_ax[11] and prices[i] < prices_ax[12]:
        vol_ax[11] += volumes[i]

    elif prices[i] >= prices_ax[12] and prices[i] < prices_ax[13]:
        vol_ax[12] += volumes[i]

    elif prices[i] >= prices_ax[13] and prices[i] < prices_ax[14]:
        vol_ax[13] += volumes[i]

    elif prices[i] >= prices_ax[14] and prices[i] < prices_ax[15]:
        vol_ax[14] += volumes[i]

    elif prices[i] >= prices_ax[15] and prices[i] < prices_ax[16]:
        vol_ax[15] += volumes[i]

    elif prices[i] >= prices_ax[16] and prices[i] < prices_ax[17]:
        vol_ax[16] += volumes[i]

    elif prices[i] >= prices_ax[17] and prices[i] < prices_ax[18]:
        vol_ax[17] += volumes[i]

    elif prices[i] >= prices_ax[18] and prices[i] < prices_ax[19]:
        vol_ax[18] += volumes[i]

    else:
        vol_ax[19] += volumes[i]

fig = make_subplots(
    rows=1, cols=2, column_widths=[0.2, 0.8], specs=[[{}, {}]], horizontal_spacing=0.01
)

fig.add_trace(
    go.Bar(
        x=vol_ax,
        y=prices_ax,
        text=np.around(prices_ax, 2),
        textposition="auto",
        orientation="h",
    ),
    row=1,
    col=1,
)


dateStr = history_data.index.strftime("%d-%m-%Y %H:%M:%S")

fig.add_trace(
    go.Candlestick(
        x=dateStr,
        open=history_data["Open"],
        high=history_data["High"],
        low=history_data["Low"],
        close=history_data["Close"],
        yaxis="y2",
    ),
    row=1,
    col=2,
)


fig.update_layout(
    title_text=f"Market Profile Chart {ticker}",  # title of plot
    bargap=0.01,  # gap between bars of adjacent location coordinates,
    showlegend=False,
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False),
    yaxis2=dict(title="Price (INR)", side="right"),
)

fig.update_yaxes(nticks=20)
fig.update_yaxes(side="right")
fig.update_layout(height=800)

config = {"modeBarButtonsToAdd": ["drawline"]}

st.plotly_chart(fig, use_container_width=True, config=config)

## Feature engineering
df = history_data.copy()
df = df.rename(
    columns={
        "Close": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    }
)

df = df.dropna()

df["7_day_sma_ratio"] = (
    talib.SMA(df["close"].values, timeperiod=7)
    / (talib.SMA(df["close"], timeperiod=7)).rolling(7).mean()
)
df["21_day_sma_ratio"] = (
    talib.SMA(df["close"].values, timeperiod=21)
    / (talib.SMA(df["close"], timeperiod=21)).rolling(21).mean()
)
df["50_day_sma_ratio"] = (
    talib.SMA(df["close"].values, timeperiod=50)
    / (talib.SMA(df["close"], timeperiod=50)).rolling(50).mean()
)

df["9_day_rsi"] = talib.RSI(df["close"], timeperiod=9)
df["14_day_rsi"] = talib.RSI(df["close"], timeperiod=14)

df["bop"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])

df["9_day_aroon_oscillator"] = talib.AROONOSC(df["high"], df["low"], timeperiod=9)
df["14_day_aroon_oscillator"] = talib.AROONOSC(df["high"], df["low"], timeperiod=14)

df["bbands_u"] = (
    talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
    / (talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0])
    .rolling(20)
    .mean()
)

df["bbands_m"] = (
    talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
    / (talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1])
    .rolling(20)
    .mean()
)

df["bbands_l"] = (
    talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
    / (talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2])
    .rolling(20)
    .mean()
)

df["ADX"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
df["pct_change"] = df["close"].pct_change(periods=1)

macd, macd_signal, macd_hist = talib.MACD(
    df["close"], fastperiod=12, slowperiod=26, signalperiod=9
)
df["macd"] = macd


df["wick_pct_change"] = df.apply(
    lambda x: get_percent_wick_in_opposite_direction(
        x["pct_change"], x["close"], x["open"], x["high"], x["low"]
    ),
    axis=1,
)
st.write("Total data points")
st.write(df.shape)

st.write("Starting data")
st.dataframe(df.head(2))

st.write("End data")
st.dataframe(df.tail(2))
df_new = df.drop(columns=["open", "high", "low"])
df_new = df_new.dropna()
df_new = df_new.drop(columns=["close", "volume"])

st.write("Final data points")
st.write(df_new.shape)


X = df_new.values
scaler = StandardScaler()
X = scaler.fit_transform(X)


if clustering == "PCA":
    st.write(clustering)
    pca = PCA(n_components=2).fit(X)
    X_embedded = pca.transform(X)
    st.write(f"Explained variance: {pca.explained_variance_ratio_}")
    st.write(f"Sum variance: {sum(pca.explained_variance_ratio_):.4f}")

elif clustering == "T-SNE":
    st.write(clustering)
    st.write("Perfroming T-SNE clustering. Please wait....")
    X_embedded = TSNE(n_components=2, init="pca", perplexity=250).fit_transform(X)
# X_embedded = TSNE(n_components=2, init="pca", perplexity=250).fit_transform(X)


gm = GaussianMixture(n_components=2, random_state=0, max_iter=5000).fit(X_embedded)
st.write(f"Gm means: {gm.means_}")

predict_clust = []
for val in X_embedded:
    pred = gm.predict_proba([val])
    if np.max(pred) > 0.8:
        predict_clust.append(np.argmax(pred))
    else:
        predict_clust.append(2)


st.write("Low Dimension visualization")
st.write(pd.Series(predict_clust).value_counts())
fig1 = px.scatter(
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    color=predict_clust,
    color_discrete_sequence=px.colors.qualitative.G10,
)
st.plotly_chart(fig1, use_container_width=True, config=config)


df_new["cluster"] = predict_clust
df_new["close"] = df["close"]


st.write("Starting data")
st.dataframe(df_new.head(2))

st.write("End data")
st.dataframe(df_new.tail(2))

st.write("All data")
t = np.arange(0, df_new.shape[0])
fig1 = px.scatter(
    x=t,
    y=df_new["close"],
    color=df_new["cluster"],
    color_discrete_sequence=px.colors.qualitative.G10,
)
# st.plotly_chart(fig1, use_container_width=True, config=config)
st.plotly_chart(fig1, use_container_width=True)


ticks = int(ticks)
st.write(f"Last {ticks} ticks")
fig1 = px.scatter(
    x=t[-ticks:],
    y=df_new["close"][-ticks:],
    color=df_new["cluster"][-ticks:],
    color_discrete_sequence=px.colors.qualitative.G10,
)
st.plotly_chart(fig1, use_container_width=True, config=config)


st.write("Sample prediction of last tick from GMM")
st.write(f"Cluster: {gm.predict([X_embedded[-1]])[0]}")
st.write(f"Cluster probability: {np.max(gm.predict_proba([X_embedded[-1]])):.4f}")
