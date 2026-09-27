import sqlite3
conn = sqlite3.connect("crypdate.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT,
    price REAL,
    timestamp TEXT
)
""")
cursor.execute("INSERT INTO prices (coin, price, timestamp) VALUES (?, ?, ?)",
               ("bitcoin", 26500, "2026-09-20T13:46:00"))
conn.commit()
cursor.execute("SELECT * FROM prices")
rows = cursor.fetchall()
print(rows)
conn.close()
