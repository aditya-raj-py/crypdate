import requests
import sqlite3
import time
from datetime import datetime

# -----------------------------
# Connect to SQLite database
# -----------------------------
conn = sqlite3.connect("crypdate.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT,
    price REAL,
    timestamp TEXT
)
""")

# -----------------------------
# Function: Fetch and save prices
# -----------------------------


def fetch_and_save():
    # API endpoint for live prices
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,dogecoin&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()

    # Insert each coin into DB
    for coin in data:
        price = data[coin]["usd"]
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO prices (coin, price, timestamp) VALUES (?, ?, ?)",
                       (coin, price, timestamp))
        print(f"Saved {coin}: {price} at {timestamp}")

    # Commit changes after each cycle
    conn.commit()

# -----------------------------
# Query Functions
# -----------------------------


def get_latest_price(coin):
    cursor.execute(
        "SELECT price, timestamp FROM prices WHERE coin=? ORDER BY id DESC LIMIT 1", (coin,))
    return cursor.fetchone()


def get_last_n_prices(coin, n=10):
    cursor.execute(
        "SELECT price, timestamp FROM prices WHERE coin=? ORDER BY id DESC LIMIT ?", (coin, n))
    return cursor.fetchall()


def get_high_low(coin):
    cursor.execute(
        "SELECT MAX(price), MIN(price) FROM prices WHERE coin=?", (coin,))
    return cursor.fetchone()


def get_average_price(coin):
    cursor.execute("SELECT AVG(price) FROM prices WHERE coin=?", (coin,))
    return cursor.fetchone()[0]


# -----------------------------
# Auto-update loop
# -----------------------------
if __name__ == "__main__":
    try:
        while True:
            fetch_and_save()
            print("Crypdating\n")
            time.sleep(60)  # wait 1 minute before next fetch
    except KeyboardInterrupt:
        print("Stopped by user.")
        conn.close()
