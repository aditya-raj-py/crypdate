"""
stock_fetch.py
===============
v5.1 - Downloads daily OHLCV history for stocks into a market-specific
database (crypdate_india.db or crypdate_us.db), using the SAME table layout as
the crypto data (table `ohlcv`, column `coin` holds the ticker). That means the
forecaster, backtester and candle charts work on stocks unchanged, just add
--market india or --market us.

Data source: Yahoo Finance through the yfinance package. It is free, but it is
an unofficial interface that can change or throttle without notice, so expect
the occasional failure and rerun. Prices are ADJUSTED for splits and dividends
(auto_adjust), which is what you want for returns and backtests.

Survivorship warning: the default ticker lists are large, long-listed
companies chosen TODAY. Companies that failed or were delisted are not in
them, which flatters any backtest. Treat results on these lists as optimistic.

Usage:
    python stock_fetch.py --market india
    python stock_fetch.py --market us --symbols AAPL MSFT
    python stock_fetch.py --market india --symbols RELIANCE TCS --exchange bse
    python stock_fetch.py --market us --start 2005-01-01
"""

import argparse
import sqlite3
import sys
import time

import pandas as pd

from crypdate_markets import MARKETS, get_market, resolve_symbol

try:
    import yfinance as yf
except ImportError:
    print("yfinance is not installed. Run: pip install yfinance")
    sys.exit(1)


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            timestamp TEXT,
            UNIQUE(coin, timestamp)
        )
    """)
    conn.commit()


def drop_unfinished_session(hist, market, now=None):
    """
    Yahoo includes the session in progress as today's row. Its close and volume
    are partial (a morning fetch shows a tiny volume), and the forecaster uses
    the newest row as the "last known price". Drop it until the market has
    closed, with a 30-minute buffer for the final numbers to settle.
    Returns (history, dropped_flag).
    """
    if hist.empty or not market.close_time:
        return hist, False
    try:
        now = now if now is not None else pd.Timestamp.now(tz=market.timezone)
    except Exception:  # timezone database missing: keep the data rather than guess
        print("  Could not check the exchange clock; keeping the latest row")
        return hist, False

    hour, minute = (int(x) for x in market.close_time.split(":"))
    cutoff = now.normalize() + pd.Timedelta(hours=hour, minutes=minute + 30)
    if hist.index[-1].date() == now.date() and now < cutoff:
        return hist.iloc[:-1], True
    return hist, False


def fetch_history(symbol: str, start: str, market, retries: int = 3, pause: float = 2.0) -> list:
    """Returns rows of (open, high, low, close, volume, 'YYYY-MM-DD')."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            hist = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
            break
        except Exception as e:  # yfinance raises a variety of errors when throttled
            last_error = e
            print(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(pause * attempt)
    else:
        raise RuntimeError(f"Could not download {symbol}: {last_error}")

    if hist is None or hist.empty:
        return []

    # Drop junk rows: missing or non-positive prices, and zero-volume days
    # (placeholder rows Yahoo sometimes returns for non-trading days). They
    # break the volume features and distort returns.
    before = len(hist)
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"])
    hist = hist[(hist[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (hist["Volume"] > 0)]
    if len(hist) < before:
        print(f"  Dropped {before - len(hist)} rows with missing prices or zero volume")
    hist, dropped_today = drop_unfinished_session(hist, market)
    if dropped_today:
        print("  Dropped today's unfinished session (the market has not closed yet)")
    return [
        (float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]),
         float(r["Volume"]), ts.strftime("%Y-%m-%d"))
        for ts, r in hist.iterrows()
    ]


def fetch_market(market, symbols: list, db_path: str, start: str, pause: float) -> None:
    conn = sqlite3.connect(db_path)
    ensure_table(conn)

    stored = 0
    for symbol in symbols:
        print(f"Fetching {symbol}...")
        try:
            rows = fetch_history(symbol, start, market)
        except RuntimeError as e:
            print(f"  {e}")
            continue

        if not rows:
            print("  No data returned. Check the ticker (Indian stocks need .NS or .BO).")
            continue

        # Replace just this ticker's rows so a refresh never touches the others
        conn.execute("DELETE FROM ohlcv WHERE coin = ?", (symbol,))
        conn.executemany(
            "INSERT OR IGNORE INTO ohlcv (coin, open, high, low, close, volume, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(symbol,) + row for row in rows],
        )
        conn.commit()
        stored += 1
        print(f"  Stored {len(rows)} rows ({rows[0][-1]} to {rows[-1][-1]})")
        time.sleep(pause)

    total = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    print(f"\nDone. {stored}/{len(symbols)} tickers stored. Total rows in {db_path}: {total}")
    conn.close()


if __name__ == "__main__":
    stock_markets = [m for m in MARKETS if m != "crypto"]
    parser = argparse.ArgumentParser(description="Download daily stock OHLCV data into a market database")
    parser.add_argument("--market", choices=stock_markets, default="india", help="Which market to fetch")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Tickers to fetch (default: the market's built-in list)")
    parser.add_argument("--exchange", choices=["nse", "bse"], default="nse",
                        help="India only: use NSE (.NS) or BSE (.BO) for tickers without a suffix")
    parser.add_argument("--start", default="2010-01-01", help="Earliest date to download (YYYY-MM-DD)")
    parser.add_argument("--db", default=None, help="Database path (default: the market's database)")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to wait between tickers")
    args = parser.parse_args()

    market = get_market(args.market)
    raw_symbols = args.symbols or list(market.default_symbols)
    symbols = [resolve_symbol(market, s) for s in raw_symbols]
    if args.market == "india" and args.exchange == "bse":
        symbols = [s[:-3] + ".BO" if s.endswith(".NS") else s for s in symbols]

    db_path = args.db or market.db_path
    print(f"Market: {market.description}\nDatabase: {db_path}\nTickers: {', '.join(symbols)}\n")
    fetch_market(market, symbols, db_path, args.start, args.pause)
