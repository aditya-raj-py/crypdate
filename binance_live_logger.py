"""
binance_live_logger.py
========================
Live price logger - functionally replaces crypdate_v3.py, but pulls from
Binance's public API instead of CoinGecko, so your live data source matches
the historical data source (binance_fetch.py) now that both the "brain"
(moving to XGBoost) and the API are being switched.

Writes into crypdate.db, SAME table/schema as v1-v3 (prices: id, coin,
price, timestamp) - so crypdate_v4.0.py, backtest.py, view_db.py all keep
working unchanged. Only the SOURCE of the data changes.

Note: crypdate_v2.py/v3.py are left untouched (they're part of your git
history/tags) - this is a new script that supersedes them going forward.

Usage:
    python binance_live_logger.py
"""

import sqlite3
import time
from datetime import datetime

import requests

SYMBOLS = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "DOGEUSDT": "dogecoin"}
DB_PATH = "crypdate.db"
TICKER_URL = "https://api.binance.com/api/v3/ticker/price"


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT,
            price REAL,
            timestamp TEXT
        )
    """)
    conn.commit()


def fetch_and_save(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    for symbol, coin in SYMBOLS.items():
        try:
            response = requests.get(TICKER_URL, params={"symbol": symbol}, timeout=10)
            response.raise_for_status()
            price = float(response.json()["price"])
        except requests.RequestException as e:
            print(f"  Failed to fetch {coin}: {e}")
            continue

        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO prices (coin, price, timestamp) VALUES (?, ?, ?)",
            (coin, price, timestamp),
        )
        print(f"Saved {coin}: {price} at {timestamp}")

    conn.commit()


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)

    try:
        while True:
            fetch_and_save(conn)
            print("Crypdating (via Binance)\n")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopped by user.")
        conn.close()
