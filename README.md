# CRYPDATE

A crypto and stock forecasting, signal and backtesting toolkit, built incrementally from a simple price database into the early stages of a quantitative trading pipeline. One switch (`--market`) moves every script between crypto, Indian stocks and US stocks.

**Status: research prototype. Not trading-ready. No consistent predictive edge has been found yet.** See [Status and results](#status-and-results) for what has and hasn't been shown.

## Project history

| Version | What it added |
|---|---|
| v1 | SQLite price database |
| v2 | Live price fetching from CoinGecko |
| v3 | 60-second logging loop and query helpers |
| v4.0 | Prophet forecasting with confidence intervals and plots |
| v4.1 | Buy/sell/hold decision signals |
| v4.2 | Confidence scoring and the first walk-forward backtester |
| v4.3 | XGBoost model, Binance data, confidence-weighted position sizing, backtester with costs, risk metrics and out-of-sample testing |
| v5.0 | Candlestick charts with backtest signal overlay |
| v5.1 | Stock support with a switchable market: crypto, Indian stocks (NSE/BSE) and US stocks |

The main branch only contains the current XGBoost pipeline. The earlier scripts (CoinGecko logging, the Prophet pipeline and its backtester) were removed to keep it focused, but they remain in git history and tags. `git checkout v4.2` restores all of them.

## Files

| File | Purpose |
|---|---|
| `crypdate_markets.py` | Market profiles: database, calendar, costs, signal thresholds and symbol rules for crypto, India and US |
| `binance_fetch.py` | Downloads full daily crypto OHLCV history from Binance into `crypdate_binance.db` |
| `stock_fetch.py` | Downloads daily stock OHLCV history (Yahoo Finance) into `crypdate_india.db` or `crypdate_us.db` |
| `crypdate_xgboost.py` | Feature engineering, XGBoost model, confidence band, signal and position size |
| `backtest_xgboost.py` | Walk-forward backtester with costs, drawdown, Sharpe, out-of-sample split and random-entry baseline |
| `backtest_batch.py` | Runs the backtest over a whole ticker list and judges the results together, with a pooled test against random buy timing |
| `crypdate_candles.py` | Candlestick charts, optionally with the backtest's buy/sell markers |

## How the pipeline works

```
OHLCV data  ->  feature engineering  ->  XGBoost  ->  signal + confidence  ->  position size  ->  backtest
```

1. **Data:** crypto comes from Binance's public API (no key needed). Stocks come from Yahoo Finance through the `yfinance` package, adjusted for splits and dividends. Every market uses the same table layout, so everything downstream is identical.
2. **Features:** 14 scale-invariant features: returns over several windows, distance from moving averages, volatility, daily range, volume change and RSI. Everything is relative or percentage-based on purpose, because prices span several orders of magnitude.
3. **Model:** XGBoost predicts the percent price change over a chosen horizon (default 7 days; for stocks, 7 trading days).
4. **Confidence:** a band around the prediction is built from the model's validation error. Its half-width as a percentage of the predicted price is mapped to a 0-1 confidence score between two thresholds.
5. **Signal and sizing:** a predicted change above the buy threshold gives a buy, below the sell threshold a sell, otherwise a hold. A buy gets a position size equal to its confidence score. Sells and holds size to zero: the project is long-only and does not short.
6. **Backtest:** walks forward through history, retraining at each decision point on only the data available then (no look-ahead). It charges trading costs, reports max drawdown and Sharpe ratio, calibrates the confidence thresholds on earlier decision points and scores only later ones, and compares the model against cash, buy-and-hold, always-buying and a random-entry baseline.

## Markets

Pick a market with `--market crypto` (the default), `--market india` or `--market us`. The market decides which database is read, how symbols are completed, how the calendar works, and the default costs and thresholds.

| | crypto | india | us |
|---|---|---|---|
| Database | `crypdate_binance.db` | `crypdate_india.db` | `crypdate_us.db` |
| Symbols | `bitcoin`, `ethereum`, `dogecoin` | `RELIANCE` becomes `RELIANCE.NS` (NSE); `TCS.BO` for BSE | `AAPL` |
| Calendar | every day | trading days, about 250 a year | trading days, about 252 a year |
| Default buy/sell thresholds | +/-3% | +/-1.5% (placeholder) | +/-1.5% (placeholder) |
| Default fee + slippage per side | 0.1% + 0.05% | 0.15% + 0.05% (placeholder) | 0.05% + 0.03% (placeholder) |

Things to know about stocks:

- **Horizons and steps count trading days**, not calendar days. `--horizon-days 7` means 7 trading days. Exchange holidays are ignored when the forecast's target date is labelled, so that label can be a day or two off.
- **Costs and thresholds are placeholders.** Real costs depend on your broker, platform and country. Check them and pass `--fee-pct`, `--slippage-pct`, `--buy-threshold` and `--sell-threshold` to override.
- **Survivorship bias.** The built-in ticker lists are large, long-listed companies picked today. Failed or delisted companies are missing, which flatters any backtest. Treat results on these lists as optimistic.
- **Yahoo Finance is an unofficial source.** It is free but can change or throttle without notice, so expect the occasional failed download and rerun.
- **Unfinished sessions are dropped.** Fetching while the market is open would store a partial day (tiny volume, mid-session close), so `stock_fetch.py` drops today's row until 30 minutes after the close. Crypto has the same quirk: Binance's newest candle is today's unfinished one, so a live crypto forecast uses a partial day. Backtests are unaffected.
- **Confidence thresholds for the forecast command** are crypto-tuned defaults. For a stock, run the backtester first: it prints calibrated thresholds you can pass to `crypdate_xgboost.py`.

## Setup

```bash
pip install -r requirements.txt
```

All scripts import `crypdate_markets.py`, so keep the files in one folder.

## Usage

Download data (once, then rerun to refresh):

```bash
python binance_fetch.py                          # crypto
python stock_fetch.py --market india             # the built-in Indian list
python stock_fetch.py --market us --symbols AAPL MSFT
python stock_fetch.py --market india --symbols RELIANCE TCS --exchange bse
```

Generate a forecast, signal and position size:

```bash
python crypdate_xgboost.py --symbol bitcoin
python crypdate_xgboost.py --market india --symbol RELIANCE
python crypdate_xgboost.py --market us --symbol AAPL
```

Backtest it:

```bash
# Calibrate on the earlier half of history, score the later half
python backtest_xgboost.py --symbol bitcoin
python backtest_xgboost.py --market india --symbol RELIANCE

# Weekly decision points (more samples, slower)
python backtest_xgboost.py --symbol bitcoin --step-days 7

# Score a specific period, calibrating on everything before it
python backtest_xgboost.py --symbol bitcoin --test-start-date 2022-01-01 --test-end-date 2022-12-31
```

Backtester flags: `--market`, `--horizon-days`, `--step-days`, `--fee-pct`, `--slippage-pct`, `--buy-threshold`, `--sell-threshold`, `--calibration-fraction`, `--test-start-date`, `--test-end-date`, `--random-trials`, `--seed`, and manual `--high-threshold-pct` / `--low-threshold-pct` overrides. A full run retrains the model at every decision point and can take several minutes. Each run saves its decision points to `results/<symbol>_decisions.csv`.

Test a whole market at once and judge the tickers together (one ticker looking good proves little, because with enough tickers some look good by luck):

```bash
python backtest_batch.py --market india
python backtest_batch.py --market us
python backtest_batch.py --market crypto
```

It prints a per-ticker table and a pooled result: how much better the model's buys did than randomly timed buys, with a p-value, how many tickers show a positive edge, and how many reach p < 0.05 against the roughly 5% expected by luck. Each ticker takes a few minutes. Results are cached in `results/batch_<market>/`, so an interrupted run resumes where it stopped, and refetching data or changing a setting recomputes automatically (`--refresh` forces it). Keep `--step-days` at or above `--horizon-days` so windows do not overlap.

Draw candlesticks (and the backtest's signals, if a decisions file exists):

```bash
python crypdate_candles.py --symbol bitcoin
python crypdate_candles.py --symbol ethereum --days 1700 --timeframe W --log
python crypdate_candles.py --market india --symbol RELIANCE
```

On Windows PowerShell, `>` saves output as UTF-16. To save it as UTF-8 instead: `python backtest_xgboost.py --symbol bitcoin | Out-File -Encoding utf8 results.txt`.

Databases (`*.db`) and backtest output (`results/`) are not committed; see `.gitignore`.

## Status and results

All results below are for **crypto**. Stock support was added in v5.1; pooled stock results from `backtest_batch.py` are still pending.

Setup: weekly decision points over about 9 years of daily data, a 7-day horizon, confidence thresholds calibrated on the earlier half and scored only on the later half (2022-04 to 2026-09), with a 0.1% fee plus 0.05% slippage per side.

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

Simplifications in the backtester: strategies are in the market only for the horizon of each decision step; every buy is charged a full round trip; Sharpe and drawdown use one return per decision point, so swings between decision points are not captured; Sharpe assumes a 0% risk-free rate.

## Not yet in place

- Pooled results across the full Indian and US ticker lists (`backtest_batch.py` is built; the runs are pending)
- Rolling confidence calibration so position sizing adapts to changing volatility
- Dogecoin results with the upgraded backtester
- A second model version: cross-asset features, trend-regime features, a longer horizon
- Liquidity and scheduled-event awareness (v5.2), news sentiment as a forward-only experiment (v5.3)
- Risk management (volatility-based sizing, stop-losses, drawdown limits)
- Paper trading and live execution

## Roadmap

- [x] v1-v4.2: data collection, Prophet forecasting, signals, confidence scoring, first backtester
- [x] v4.3: XGBoost, Binance data, position sizing, upgraded backtester
- [x] v5.0: candlestick charts with signal overlay
- [x] v5.1: stock support with switchable markets (India, US) and a batch evaluator; pooled results pending
- [ ] v5.2: liquidity measures and a scheduled-event calendar
- [ ] v5.3: news sentiment (forward-only experiment)
- [ ] Rolling confidence calibration
- [ ] Model v2: cross-asset and regime features, longer horizon
- [ ] Risk management layer
- [ ] Paper trading

## Disclaimer

This is an educational project. Forecasts extrapolate patterns from historical price data and know nothing about news, liquidity or market shocks. Nothing here is financial advice, and it should not be used to trade real money.
