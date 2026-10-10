# CRYPDATE

A crypto price logger, forecaster and signal generator, built incrementally from a simple price database into the early stages of a quantitative trading pipeline.

**Status: working research prototype. Not trading-ready.** See [Status and results](#status-and-results) for an honest picture of what has and hasn't been shown.

## Version history

| Version | File | What it added |
|---|---|---|
| v1 | `crypdate_v1.py` | SQLite database (`prices` table) |
| v2 | `crypdate_v2.py` | Live price fetching from CoinGecko |
| v3 | `crypdate_v3.py` | 60-second logging loop and query helpers |
| v4.0 | `crypdate_v4.0.py` | Prophet forecasting with confidence intervals and plots |
| v4.1 | `crypdate_v4.0.py` | Buy/sell/hold decision signals |
| v4.2 | `crypdate_v4.0.py`, `backtest.py` | Confidence scoring (absolute thresholds) and a walk-forward backtester |
| v4.3 | `crypdate_xgboost.py`, `backtest_xgboost.py`, `binance_fetch.py`, `binance_live_logger.py` | XGBoost model, Binance data (years of daily OHLCV), confidence-weighted position sizing |

Each version is tagged in git (`v4.0`, `v4.1`, `v4.2`, `v4.3`), so any earlier state can be restored with `git checkout <tag>`.

## How the current pipeline works (v4.3)

```
Binance daily OHLCV  ->  feature engineering  ->  XGBoost  ->  signal + confidence  ->  position size  ->  backtest
```

1. **Data:** `binance_fetch.py` downloads the full daily OHLCV history for bitcoin, ethereum and dogecoin from Binance's public API (no key needed) into `crypdate_binance.db`.
2. **Features:** `crypdate_xgboost.py` builds 14 scale-invariant features from OHLCV: returns over several windows, distance from moving averages, volatility, daily range, volume change and RSI. Everything is relative or percentage-based on purpose, because prices spanned several orders of magnitude over the history.
3. **Model:** XGBoost predicts the percent price change over a chosen horizon (default 7 days).
4. **Confidence:** a band around the prediction is built from the model's validation error. Its half-width as a percentage of the predicted price is mapped to a 0-1 confidence score between two thresholds (`--high-threshold-pct`, `--low-threshold-pct`).
5. **Signal and sizing:** a predicted change above the buy threshold (default +3%) gives a buy, below the sell threshold (default -3%) a sell, otherwise a hold. A buy gets a position size equal to its confidence score (0-100% of capital). Sells and holds size to zero: the project is long-only and does not short.
6. **Backtest:** `backtest_xgboost.py` walks forward through history, retraining on only the data available at each step, and compares full-size trading, confidence-weighted sizing and buy-and-hold.

The earlier Prophet pipeline (`crypdate_v4.0.py`, `backtest.py`) is kept and still runs. It still uses the older rule where a low-confidence signal is downgraded to hold, rather than position sizing.

## Setup

```bash
pip install -r requirements.txt
```

Coin names are used as stored in the databases: `bitcoin`, `ethereum`, `dogecoin`.

## Usage

Download historical data (once, then rerun to refresh):

```bash
python binance_fetch.py
```

Log live prices every 60 seconds into `crypdate.db`:

```bash
python binance_live_logger.py
```

Generate a forecast, signal and position size with XGBoost:

```bash
python crypdate_xgboost.py --symbol bitcoin
```

Backtest it:

```bash
python backtest_xgboost.py --symbol bitcoin --high-threshold-pct 14 --low-threshold-pct 25
```

Useful flags for both XGBoost scripts: `--horizon-days`, `--buy-threshold`, `--sell-threshold`, `--high-threshold-pct`, `--low-threshold-pct`. The backtester also takes `--step-days` (spacing between decision points) and `--min-training-days`.

Prophet version, run against the live log in `crypdate.db`:

```bash
python crypdate_v4.0.py --symbol bitcoin --periods 14 --freq D
python backtest.py --symbol bitcoin
```

Databases (`*.db`) are not committed; see `.gitignore`.

## Status and results

On bitcoin, with the confidence thresholds calibrated to the model's observed interval widths (14% / 25%), a walk-forward backtest over about 9 years of daily data (232 decision points) gave:

| Strategy | Compounded return |
|---|---|
| Buy-and-hold | +1090.5% |
| XGBoost, full size on every buy | +61.0% |
| XGBoost, confidence-weighted size | +164.3% |

How to read this:

- Confidence-weighted sizing beat full-size trading, and trades the model rated Medium confidence did best (57% win rate). That suggests the confidence score carries some information.
- **Buy-and-hold still won by a wide margin.** The model has not shown an edge over simply holding during this period, which was a strong bull market.
- The thresholds were calibrated on the same history they were scored on, so the +164% is optimistic. Some confidence buckets contain very few trades (6 in the High bucket), so their win rates are noisy.
- The backtest ignores trading fees and slippage, and does not model price moves between decision points.
- Ethereum and dogecoin have not been calibrated per coin and did not show a consistent edge.

## Not yet in place

- Fees, slippage, max drawdown, Sharpe ratio, and a clean calibration/test split in the backtester
- Bear-market testing
- Risk management (volatility-based sizing, stop-losses, drawdown limits)
- Paper trading and live execution

## Roadmap

- [x] v1-v3: data collection
- [x] v4.0-v4.2: Prophet forecasting, signals, confidence scoring, backtester
- [x] v4.3: XGBoost, Binance data, position sizing
- [ ] Backtester upgrade: fees, drawdown, Sharpe, out-of-sample calibration
- [ ] Per-coin threshold calibration
- [ ] Model improvements (time-series cross-validation, quantile-based confidence)
- [ ] Risk management layer
- [ ] Paper trading
- [ ] Stocks (v5)

## Disclaimer

This is an educational project. Forecasts extrapolate patterns from historical price data and know nothing about news, liquidity or market shocks. Nothing here is financial advice, and it should not be used to trade real money.
