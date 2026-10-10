"""
binance_fetch.py
=================
Pulls FULL historical daily OHLCV (Open, High, Low, Close, Volume) data from
Binance's public market data API - no API key required, no 365-day cap like
CoinGecko's free tier. Goes back to each symbol's Binance listing date.

Writes to crypdate_binance.db, a NEW separate database (table: ohlcv), kept
apart from crypdate.db (live per-minute data) and crypdate_historical.db
(CoinGecko backfill) so nothing existing gets disturbed.

Why OHLCV instead of just price: this unlocks real candlestick charts later,
AND gives the XGBoost model real features to engineer from (daily range,
volume trends) instead of just a single closing price.

Usage:
    python binance_fetch.py
    python binance_fetch.py --symbols BTCUSDT ETHUSDT DOGEUSDT
"""

import argparse
import sqlite3
import time
from datetime import datetime, timezone

import requests

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
DB_PATH = "crypdate_binance.db"
BASE_URL = "https://api.binance.com/api/v3/klines"
MS_PER_DAY = 86_400_000


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


def fetch_all_klines(symbol: str, interval: str = "1d") -> list[dict]:
    """
    Fetch the FULL available history for a symbol by paginating Binance's
    klines endpoint (max 1000 candles per request) from the earliest
    available data forward to now.
    """
    all_rows = []
    start_time = 0  # 0 = let Binance return from the earliest available candle

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": 1000,
        }
        response = requests.get(BASE_URL, params=params, timeout=30)

        if response.status_code == 429:
            print("  Rate limited, waiting 10s...")
            time.sleep(10)
            continue

        response.raise_for_status()
        data = response.json()

        if not data:
            break

        for candle in data:
            open_time_ms = candle[0]
            all_rows.append({
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "timestamp": datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat(),
            })

        last_open_time = data[-1][0]
        start_time = last_open_time + MS_PER_DAY

        if len(data) < 1000:
            break  # reached the end

        time.sleep(0.3)  # be polite between pages

    return all_rows


def backfill(symbols: list[str], db_path: str = DB_PATH, clear_first: bool = True) -> None:
    conn = sqlite3.connect(db_path)
    ensure_table(conn)
    cursor = conn.cursor()

    if clear_first:
        cursor.execute("DELETE FROM ohlcv")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='ohlcv'")
        conn.commit()
        print("Cleared existing rows before backfilling.\n")

    for symbol in symbols:
        coin = symbol.replace("USDT", "").lower()  # BTCUSDT -> bitcoin-style label "btc"
        print(f"Fetching full history for {symbol}...")

        try:
            rows = fetch_all_klines(symbol)
        except requests.RequestException as e:
            print(f"  Failed to fetch {symbol}: {e}")
            continue

        cursor.executemany(
            "INSERT OR IGNORE INTO ohlcv (coin, open, high, low, close, volume, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(coin, r["open"], r["high"], r["low"], r["close"], r["volume"], r["timestamp"]) for r in rows],
        )
        conn.commit()

        if rows:
            print(f"  Inserted {len(rows)} rows for {coin} ({rows[0]['timestamp'][:10]} to {rows[-1]['timestamp'][:10]})")
        else:
            print(f"  No rows returned for {symbol}")

        time.sleep(1)

    total = cursor.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    print(f"\nDone. Total rows in 'ohlcv' table: {total}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill crypdate_binance.db with full OHLCV history from Binance")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Binance symbols, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing rows before backfilling")

    args = parser.parse_args()
    backfill(args.symbols, args.db, clear_first=not args.no_clear)
