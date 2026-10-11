# CRYPDATE

A crypto price logger, forecaster and signal generator, built incrementally from a simple price database into the early stages of a quantitative trading pipeline.

**Status: research prototype. Not trading-ready. No consistent predictive edge has been found yet.** See [Status and results](#status-and-results) for what has and hasn't been shown.

## Version history

| Version | File | What it added |
|---|---|---|
| v1 | `crypdate_v1.py` | SQLite database (`prices` table) |
| v2 | `crypdate_v2.py` | Live price fetching from CoinGecko |
| v3 | `crypdate_v3.py` | 60-second logging loop and query helpers |
| v4.0 | `crypdate_v4.0.py` | Prophet forecasting with confidence intervals and plots |
| v4.1 | `crypdate_v4.0.py` | Buy/sell/hold decision signals |
| v4.2 | `crypdate_v4.0.py`, `backtest.py` | Confidence scoring (absolute thresholds) and a walk-forward backtester |
| v4.3 | `crypdate_xgboost.py`, `backtest_xgboost.py`, `binance_fetch.py`, `binance_live_logger.py` | XGBoost model, Binance data (years of daily OHLCV), confidence-weighted position sizing, and a backtester with costs, risk metrics and out-of-sample testing |

Each version is tagged in git (`v4.0`, `v4.1`, `v4.2`, `v4.3`), so any earlier state can be restored with `git checkout <tag>`.

## How the current pipeline works (v4.3)

```
Binance daily OHLCV  ->  feature engineering  ->  XGBoost  ->  signal + confidence  ->  position size  ->  backtest
```

1. **Data:** `binance_fetch.py` downloads the full daily OHLCV history for bitcoin, ethereum and dogecoin from Binance's public API (no key needed) into `crypdate_binance.db`.
2. **Features:** `crypdate_xgboost.py` builds 14 scale-invariant features from OHLCV: returns over several windows, distance from moving averages, volatility, daily range, volume change and RSI. Everything is relative or percentage-based on purpose, because prices spanned several orders of magnitude over the history.
3. **Model:** XGBoost predicts the percent price change over a chosen horizon (default 7 days).
4. **Confidence:** a band around the prediction is built from the model's validation error. Its half-width as a percentage of the predicted price is mapped to a 0-1 confidence score between two thresholds.
5. **Signal and sizing:** a predicted change above the buy threshold (default +3%) gives a buy, below the sell threshold (default -3%) a sell, otherwise a hold. A buy gets a position size equal to its confidence score. Sells and holds size to zero: the project is long-only and does not short.
6. **Backtest:** `backtest_xgboost.py` walks forward through history, retraining at each decision point on only the data available then (no look-ahead). It charges trading costs, reports max drawdown and Sharpe ratio, calibrates the confidence thresholds on earlier decision points and scores only later ones, and compares the model against cash, buy-and-hold, always-buying and a random-entry baseline.

The earlier Prophet pipeline (`crypdate_v4.0.py`, `backtest.py`) is kept and still runs. It still uses the older rule where a low-confidence signal is downgraded to hold, and `backtest.py` has not been upgraded with the newer evaluation features.

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
# Calibrate on the earlier half of history, score the later half
python backtest_xgboost.py --symbol bitcoin

# Weekly decision points (more samples, slower)
python backtest_xgboost.py --symbol bitcoin --step-days 7

# Score a specific period, calibrating on everything before it
python backtest_xgboost.py --symbol bitcoin --test-start-date 2022-01-01 --test-end-date 2022-12-31
```

Backtester flags: `--horizon-days`, `--step-days`, `--fee-pct`, `--slippage-pct`, `--calibration-fraction`, `--test-start-date`, `--test-end-date`, `--random-trials`, `--seed`, and manual `--high-threshold-pct` / `--low-threshold-pct` overrides. A full run retrains the model at every decision point and can take several minutes.

On Windows PowerShell, `>` saves output as UTF-16. To save it as UTF-8 instead: `python backtest_xgboost.py --symbol bitcoin | Out-File -Encoding utf8 results.txt`.

Prophet version, run against the live log in `crypdate.db`:

```bash
python crypdate_v4.0.py --symbol bitcoin --periods 14 --freq D
python backtest.py --symbol bitcoin
```

Databases (`*.db`) are not committed; see `.gitignore`.

## Status and results

Setup for the results below: weekly decision points over about 9 years of daily data, a 7-day horizon, confidence thresholds calibrated on the earlier half and scored only on the later half (2022-04 to 2026-09), with a 0.1% fee plus 0.05% slippage per side.

| | Net return | Max drawdown | Sharpe | Buys | Beat random-entry sets |
|---|---|---|---|---|---|
| **Bitcoin** buy-and-hold | +102.2% | -61.5% | 0.57 | | |
| **Bitcoin** XGBoost | +68.3% | -21.9% | 0.73 | 43 | 89% |
| **Ethereum** buy-and-hold | -12.7% | -66.1% | 0.29 | | |
| **Ethereum** XGBoost | -66.1% | -75.8% | -0.54 | 79 | 12% |

How to read this:

- **No consistent edge.** Bitcoin looks mildly positive and ethereum looks clearly negative. Averaged across the two, the model's timing is about what chance would give.
- **Bitcoin is suggestive, not significant.** Beating 89% of random entry sets means roughly 1 in 9 random strategies did as well, short of the usual 95% bar. The Sharpe ratio is uncertain by roughly +/-0.5 over this period, so 0.73 versus 0.57 is not a real difference.
- **Ethereum is the opposite.** The average buy lost money before costs, and the strategy's drawdown was worse than buy-and-hold's.
- **Position sizing had no effect.** In the later test period every buy was rated High confidence, so full-size and confidence-weighted results are identical. The thresholds learned from earlier years did not carry over, which is why rolling calibration is on the roadmap.
- **An earlier result was removed.** A previous version of this README quoted +164% for bitcoin with confidence-weighted sizing. That used hand-picked thresholds on the same data it was scored on, and it did not hold up out of sample.
- **2022 bear-market check (bitcoin, bi-weekly decisions).** Only 4 buys, far too few to conclude anything. The strategy lost less than buy-and-hold mostly by being in cash.
- Results from the later half of history are also not fully clean, since the setup was adjusted while looking at this period. New, unseen data (paper trading) is the only fully out-of-sample test.

Simplifications in the backtester: strategies are in the market only for the horizon (7 days) of each decision step; every buy is charged a full round trip; Sharpe and drawdown use one return per decision point, so swings between decision points are not captured; Sharpe assumes a 0% risk-free rate.

## Not yet in place

- Rolling confidence calibration so position sizing adapts to changing volatility
- Dogecoin results with the upgraded backtester
- A second model version: cross-asset features (bitcoin returns as an input for other coins), trend-regime features, a longer horizon
- Risk management (volatility-based sizing, stop-losses, drawdown limits)
- Paper trading and live execution

## Roadmap

- [x] v1-v3: data collection
- [x] v4.0-v4.2: Prophet forecasting, signals, confidence scoring, backtester
- [x] v4.3: XGBoost, Binance data, position sizing
- [x] Backtester upgrade: costs, drawdown, Sharpe, out-of-sample split, random-entry baseline
- [ ] Rolling confidence calibration
- [ ] Model v2: cross-asset and regime features, longer horizon
- [ ] Risk management layer
- [ ] Paper trading
- [ ] Stocks (v5)

## Disclaimer

This is an educational project. Forecasts extrapolate patterns from historical price data and know nothing about news, liquidity or market shocks. Nothing here is financial advice, and it should not be used to trade real money.
