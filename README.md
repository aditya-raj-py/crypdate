# CRYPDATE

A crypto price logger and forecaster, built incrementally.

## Versions

- **v1** — `crypdate_v1.py`: creates the SQLite database (`crypdate.db`) and `prices` table, inserts one sample row.
- **v2** — `crypdate_v2.py`: fetches live prices (bitcoin, ethereum, dogecoin) from the CoinGecko API and logs them once.
- **v3** — `crypdate_v3.py`: wraps fetching in a loop (`fetch_and_save`, logs every 60s) plus query helpers (`get_latest_price`, `get_last_n_prices`, `get_high_low`, `get_average_price`).
- **v4.0** — `crypdate_v4.py`: forecasts prices with [Prophet](https://facebook.github.io/prophet/), plotting historical + predicted prices with confidence intervals. Modular (`load_data`, `train_model`, `forecast_prices`, `plot_forecast`) and designed to extend to:
  - **v4.1** (planned): decision percentages (buy/sell/hold signal from forecast vs. current price)
  - **v4.2** (planned): confidence scoring (derived from interval width)
  - **v5** (planned): stock price support, using the same pipeline with a different asset table

## Schema

```sql
CREATE TABLE prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT,        -- e.g. "bitcoin", "ethereum", "dogecoin"
    price REAL,
    timestamp TEXT     -- ISO 8601, from datetime.now().isoformat()
);
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the logger (v3) to start collecting data:

```bash
python crypdate_v3.py
```

Once you have some history, forecast prices (v4.0):

```bash
python crypdate_v4.py --symbol bitcoin --periods 30
```

Useful flags:

- `--symbol` — coin name as stored in the DB (`bitcoin`, `ethereum`, `dogecoin`)
- `--periods` — how many future steps to forecast
- `--freq` — forecast step size (`D` daily, `H` hourly)
- `--resample-freq` — how raw logs are aggregated before fitting (default `H`; pass `none` to fit on raw rows)
- `--save path.png` — save the plot instead of displaying it

## ⚠️ Note on live trading use

Prophet forecasts extrapolate from historical trend/seasonality in your own logged data — it has no awareness of news, exchange-specific liquidity, or sudden market shocks. Treat forecast output as one input among many, not a standalone trading signal.

## Roadmap

- [ ] v4.1 — decision percentages
- [ ] v4.2 — confidence scoring
- [ ] v5 — multi-asset support (stocks)
