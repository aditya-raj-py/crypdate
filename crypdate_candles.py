"""
crypdate_candles.py
====================
v5.0 - Candlestick charts for any OHLCV series stored in the CRYPDATE
databases, optionally with the buy/sell markers from a backtest.

Plain matplotlib only (no extra dependency). Works with crypdate_binance.db
today, and with stock data later as long as it uses the same table layout
(use --table / --symbol-col if the names differ).

Backtest markers: run backtest_xgboost.py once. It saves its decision points
to results/<symbol>_decisions.csv, and this script picks that file up
automatically. Green triangle = buy signal, red triangle = sell signal.

Usage:
    python crypdate_candles.py --symbol bitcoin
    python crypdate_candles.py --symbol bitcoin --days 365 --timeframe W
    python crypdate_candles.py --symbol ethereum --days 400 --save eth_candles.png
    python crypdate_candles.py --symbol bitcoin --days 3000 --timeframe W --log
"""

import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

from crypdate_markets import MARKETS, get_market, resolve_symbol

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"


def format_price(value, _position=None):
    """Readable price labels for any scale: BTC-sized, stock-sized or sub-dollar coins."""
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def format_volume(value, _position=None):
    if abs(value) >= 1e9:
        return f"{value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_ohlcv(db_path, symbol, table="ohlcv", symbol_col="coin"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT timestamp, open, high, low, close, volume FROM {table} "
        f"WHERE {symbol_col} = ? ORDER BY timestamp ASC",
        conn, params=(symbol,),
    )
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for '{symbol}' in {db_path} (table '{table}', column '{symbol_col}').")

    ts = pd.to_datetime(df["timestamp"], format="ISO8601")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)

    df = df.drop(columns="timestamp")
    df.index = ts.dt.normalize()
    df.index.name = "Date"
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df[~df.index.duplicated(keep="first")]


def resample_ohlcv(df, timeframe):
    """Aggregate daily candles into weekly (W) or monthly (M) candles."""
    if timeframe == "D":
        return df
    rules = {"W": ["W"], "M": ["ME", "M"]}[timeframe]  # 'ME' on newer pandas, 'M' on older
    last_error = None
    for rule in rules:
        try:
            return df.resample(rule).agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna()
        except ValueError as e:
            last_error = e
    raise last_error


def build_window(df_daily, timeframe, days, ma_periods):
    """Resample, compute moving averages on the FULL history, then cut to the window."""
    res = resample_ohlcv(df_daily, timeframe)
    for p in ma_periods:
        res[f"MA{p}"] = res["Close"].rolling(p).mean()
    window_start = df_daily.index[-1] - pd.Timedelta(days=days)
    return res[res.index >= window_start], window_start


def load_signals(path):
    sig = pd.read_csv(path, parse_dates=["date"])
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    return sig


def signal_positions(window, signals, kind, window_start):
    """Map each signal date to the index of the candle that contains it."""
    rows = signals[(signals["signal"] == kind)
                   & (signals["date"] >= window_start)
                   & (signals["date"] <= window.index[-1])]
    positions = []
    for d in rows["date"]:
        i = int(window.index.searchsorted(d))
        if 0 <= i < len(window):
            positions.append(i)
    return sorted(set(positions))


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot_candles(window, window_start, symbol, timeframe, ma_periods, show_volume,
                 signals, log_scale, save_path):
    n = len(window)
    x = np.arange(n)

    if show_volume:
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        axv = None

    opens, highs, lows, closes = (window[c].values for c in ["Open", "High", "Low", "Close"])
    colors = np.where(closes >= opens, UP_COLOR, DOWN_COLOR)

    # Wicks, then bodies (a tiny minimum height keeps flat candles visible)
    ax.vlines(x, lows, highs, color=colors, linewidth=1)
    body_height = np.maximum(np.abs(closes - opens), closes * 0.0003)
    ax.bar(x, body_height, bottom=np.minimum(opens, closes), width=0.7, color=colors)

    for p in ma_periods:
        ax.plot(x, window[f"MA{p}"].values, linewidth=1.3, label=f"MA{p}")

    if signals is not None:
        buys = signal_positions(window, signals, "buy", window_start)
        sells = signal_positions(window, signals, "sell", window_start)
        if buys:
            ax.scatter(buys, lows[buys] * 0.97, marker="^", s=70, color="#1b8f3a",
                       zorder=5, label=f"Buy signal ({len(buys)})")
        if sells:
            ax.scatter(sells, highs[sells] * 1.03, marker="v", s=70, color="#c62828",
                       zorder=5, label=f"Sell signal ({len(sells)})")

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5), numticks=20))
        ax.yaxis.set_minor_formatter(FuncFormatter(format_price))
        ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(format_price))
    ax.set_ylabel("Price" + (" (log scale)" if log_scale else ""))
    ax.grid(True, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", ncol=4, fontsize=9)

    names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
    ax.set_title(f"{symbol} - {names[timeframe]} candles, "
                 f"{window.index[0]:%Y-%m-%d} to {window.index[-1]:%Y-%m-%d}")

    if axv is not None:
        axv.bar(x, window["Volume"].values, width=0.7, color=colors)
        axv.yaxis.set_major_formatter(FuncFormatter(format_volume))
        axv.set_ylabel("Volume")
        axv.grid(True, alpha=0.25)

    # Date labels on the bottom axis
    bottom = axv if axv is not None else ax
    tick_idx = np.unique(np.linspace(0, n - 1, num=min(10, n)).astype(int))
    fmt = "%Y-%m-%d" if (window.index[-1] - window.index[0]).days <= 730 else "%Y-%m"
    bottom.set_xticks(tick_idx)
    bottom.set_xticklabels([window.index[i].strftime(fmt) for i in tick_idx], rotation=0)
    labels = bottom.get_xticklabels()
    if len(labels) > 1:
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
    ax.set_xlim(-1, n)

    # Hover readout in the window toolbar: date and the candle's open/high/low/close
    def candle_at(xv):
        i = int(round(xv))
        return i if 0 <= i < n else None

    def price_readout(xv, yv):
        i = candle_at(xv)
        if i is None:
            return ""
        return (f"{window.index[i]:%Y-%m-%d}  O {format_price(opens[i])}  H {format_price(highs[i])}  "
                f"L {format_price(lows[i])}  C {format_price(closes[i])}")

    def volume_readout(xv, yv):
        i = candle_at(xv)
        return "" if i is None else f"{window.index[i]:%Y-%m-%d}  volume {format_volume(window['Volume'].values[i])}"

    ax.format_coord = price_readout
    if axv is not None:
        axv.format_coord = volume_readout

    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.07)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRYPDATE v5.0 - candlestick charts with optional backtest signals")
    parser.add_argument("--market", choices=list(MARKETS), default="crypto",
                        help="crypto (default), india or us. Picks the database")
    parser.add_argument("--symbol", required=True, help="Asset name: bitcoin, or a ticker like RELIANCE / AAPL")
    parser.add_argument("--db", default=None, help="OHLCV database path (default: the market's database)")
    parser.add_argument("--table", default="ohlcv", help="Table name")
    parser.add_argument("--symbol-col", default="coin", help="Column holding the asset name")
    parser.add_argument("--days", type=int, default=180, help="How many calendar days back to show")
    parser.add_argument("--timeframe", choices=["D", "W", "M"], default="D", help="Candle size: daily, weekly or monthly")
    parser.add_argument("--ma", type=int, nargs="*", default=[20, 50], help="Moving average periods, in candles (pass none to disable)")
    parser.add_argument("--no-volume", action="store_true", help="Hide the volume panel")
    parser.add_argument("--log", action="store_true", help="Log-scale price axis (good for long ranges)")
    parser.add_argument("--signals", default=None, help="Backtest decisions CSV (default: results/<symbol>_decisions.csv if it exists)")
    parser.add_argument("--no-signals", action="store_true", help="Do not draw backtest signals")
    parser.add_argument("--save", default=None, help="Save the chart as a PNG instead of opening a window")
    args = parser.parse_args()

    market = get_market(args.market)
    symbol = resolve_symbol(market, args.symbol)
    df_daily = load_ohlcv(args.db or market.db_path, symbol, args.table, args.symbol_col)
    window, window_start = build_window(df_daily, args.timeframe, args.days, args.ma)

    signals = None
    if not args.no_signals:
        signals_path = args.signals or f"results/{symbol}_decisions.csv"
        if os.path.exists(signals_path):
            signals = load_signals(signals_path)
            print(f"Using backtest signals from {signals_path}")
        elif args.signals:
            raise FileNotFoundError(f"Signals file not found: {args.signals}")

    plot_candles(window, window_start, symbol, args.timeframe, args.ma,
                 not args.no_volume, signals, args.log, args.save)
