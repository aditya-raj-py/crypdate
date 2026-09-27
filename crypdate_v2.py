import requests
import sqlite3
from datetime import datetime

# Connect to SQLite
conn = sqlite3.connect("crypdate.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT,
    price REAL,
    timestamp TEXT
)
""")

# Fetch live price from CoinGecko
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

# Commit and close
conn.commit()
conn.close()
