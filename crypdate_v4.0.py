"""
CRYPDATE v4.0
=============
Forecasts asset prices (crypto in v4.0, stocks planned for v5) using Prophet,
reading historical price data from crypdate.db (SQLite) and plotting
historical + predicted prices with confidence intervals.

Design notes for future versions:
- All functions accept an `asset_type` / `symbol` parameter and a generic
  `table_name` / query so this can extend to stocks (v5) without rewriting
  the core pipeline.
- `forecast_prices()` returns the full Prophet forecast dataframe (not just
  plotted values) so v4.1 (decision percentages) and v4.2 (confidence
  scoring) can consume it directly — see the stub functions at the bottom.
- Each stage (load -> train -> forecast -> plot) is a standalone function
  so any stage can be swapped, tested, or reused independently.

Requirements:
    pip install prophet pandas matplotlib

Usage:
    python crypdate_v4.py --db crypdate.db --symbol BTC --periods 30
"""

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

try:
    from prophet import Prophet
except ImportError:
    print("Prophet is not installed. Run: pip install prophet")
    sys.exit(1)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class ForecastConfig:
    """Holds settings for one forecasting run. Extend this for v4.1/v4.2
    (e.g. add `decision_threshold`, `confidence_method`) without touching
    function signatures elsewhere."""
    db_path: str
    symbol: str
    asset_type: str = "crypto"     # "crypto" now, "stock" in v5
    table_name: str = "prices"     # matches v1-v3 schema
    timestamp_col: str = "timestamp"
    price_col: str = "price"
    # v1-v3 use "coin" (values: "bitcoin", "ethereum", "dogecoin")
    symbol_col: str = "coin"
    periods: int = 30              # number of future points to forecast
    freq: str = "D"                # forecast step: "D" daily, "H" hourly, etc.
    interval_width: float = 0.95   # confidence interval width
    # aggregate raw logs before fitting; None = no resampling
    resample_freq: Optional[str] = "h"


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------

def load_data(config: ForecastConfig) -> pd.DataFrame:
    """
    Load historical price data for a given symbol from crypdate.db.

    Returns a DataFrame with columns ['ds', 'y'] as Prophet requires:
        ds -> datetime
        y  -> price

    Asset-agnostic: works for any table with a symbol/timestamp/price
    schema, so v5 can point this at a `stocks` table (or pass a different
    table_name) with no other changes.
    """
    query = f"""
        SELECT {config.timestamp_col} AS ds, {config.price_col} AS y
        FROM {config.table_name}
        WHERE {config.symbol_col} = ?
        ORDER BY {config.timestamp_col} ASC
    """
    with sqlite3.connect(config.db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(config.symbol,))

    if df.empty:
        raise ValueError(
            f"No data found for symbol '{config.symbol}' in "
            f"table '{config.table_name}' of {config.db_path}"
        )

    df["ds"] = pd.to_datetime(df["ds"], format="ISO8601")
    if df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    # Raw logs (e.g. one row per minute from v3's logging loop) are noisy
    # and can have gaps if the logger was stopped/restarted. Resampling to
    # a coarser, regular frequency (mean price per bucket) gives Prophet a
    # cleaner signal for daily-horizon forecasts. Set resample_freq=None
    # to skip this and fit on raw rows instead.
    if config.resample_freq:
        df = (
            df.set_index("ds")
            .resample(config.resample_freq)["y"]
            .mean()
            .dropna()
            .reset_index()
        )

    return df


# --------------------------------------------------------------------------
# 2. Train
# --------------------------------------------------------------------------

def train_model(df: pd.DataFrame, config: ForecastConfig) -> Prophet:
    """
    Fit a Prophet model on the historical data.

    Kept separate from forecast_prices() so v4.1/v4.2 can reuse a trained
    model for multiple purposes (e.g. re-forecasting at different horizons,
    inspecting model.params for confidence scoring) without retraining.
    """
    model = Prophet(interval_width=config.interval_width)
    model.fit(df)
    return model


# --------------------------------------------------------------------------
# 3. Forecast
# --------------------------------------------------------------------------

def forecast_prices(model: Prophet, config: ForecastConfig) -> pd.DataFrame:
    """
    Generate future price predictions with confidence intervals.

    Returns the full Prophet forecast DataFrame, including:
        ds, yhat, yhat_lower, yhat_upper, trend, ...

    v4.1 can compute decision percentages from (yhat vs last known price).
    v4.2 can compute confidence scores from (yhat_upper - yhat_lower).
    """
    future = model.make_future_dataframe(
        periods=config.periods, freq=config.freq)
    forecast = model.predict(future)
    return forecast


# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------

def plot_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    config: ForecastConfig,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot historical prices plus predicted prices with confidence intervals.

    Draws its own chart (rather than relying on Prophet's built-in plot)
    so styling/labels stay consistent across asset types and future
    versions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical actuals
    ax.plot(history["ds"], history["y"], "k.",
            label="Historical", markersize=3)

    # Forecast line
    ax.plot(forecast["ds"], forecast["yhat"],
            "b-", label="Forecast", linewidth=1.5)

    # Confidence interval band
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="blue",
        alpha=0.2,
        label=f"{int(config.interval_width * 100)}% Confidence Interval",
    )

    ax.set_title(f"{config.symbol} ({config.asset_type}) Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_forecast(config: ForecastConfig, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Full pipeline: load -> train -> forecast -> plot.
    Returns the forecast DataFrame for further use (e.g. by v4.1/v4.2, or
    by a caller that wants to inspect/save predictions separately).
    """
    history = load_data(config)
    model = train_model(history, config)
    forecast = forecast_prices(model, config)
    plot_forecast(history, forecast, config, save_path=save_path)
    return forecast


# --------------------------------------------------------------------------
# Stubs for future versions (not implemented in v4.0)
# --------------------------------------------------------------------------

def compute_decision_percentages(forecast: pd.DataFrame, last_known_price: float) -> pd.DataFrame:
    """
    v4.1 placeholder.
    Intended to compute e.g. % change of yhat vs last_known_price for
    each forecasted point, to support buy/sell/hold signals.
    """
    raise NotImplementedError("Planned for v4.1")


def compute_confidence_scores(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    v4.2 placeholder.
    Intended to derive a confidence score per forecasted point, e.g. from
    the width of (yhat_upper - yhat_lower) relative to yhat.
    """
    raise NotImplementedError("Planned for v4.2")


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def parse_args() -> ForecastConfig:
    parser = argparse.ArgumentParser(
        description="CRYPDATE v4.0 - Prophet price forecasting")
    parser.add_argument("--db", default="crypdate.db",
                        help="Path to SQLite database")
    parser.add_argument("--symbol", required=True,
                        help="Coin name as stored in DB, e.g. bitcoin")
    parser.add_argument("--asset-type", default="crypto",
                        help="Asset type, e.g. crypto or stock")
    parser.add_argument("--table", default="prices",
                        help="Table name to query")
    parser.add_argument("--timestamp-col", default="timestamp",
                        help="Timestamp column name")
    parser.add_argument("--price-col", default="price",
                        help="Price column name")
    parser.add_argument("--symbol-col", default="coin",
                        help="Symbol/coin column name")
    parser.add_argument("--periods", type=int, default=30,
                        help="Number of future periods to forecast")
    parser.add_argument("--freq", default="D",
                        help="Forecast frequency (D=daily, H=hourly, ...)")
    parser.add_argument("--interval-width", type=float,
                        default=0.95, help="Confidence interval width")
    parser.add_argument("--resample-freq", default="h",
                        help="Resample raw logs before fitting (e.g. H, D); 'none' to disable")
    parser.add_argument("--save", default=None,
                        help="Path to save the plot instead of showing it")

    args = parser.parse_args()

    config = ForecastConfig(
        db_path=args.db,
        symbol=args.symbol,
        asset_type=args.asset_type,
        table_name=args.table,
        timestamp_col=args.timestamp_col,
        price_col=args.price_col,
        symbol_col=args.symbol_col,
        periods=args.periods,
        freq=args.freq,
        interval_width=args.interval_width,
        resample_freq=None if args.resample_freq.lower() == "none" else args.resample_freq,
    )
    return config, args.save


if __name__ == "__main__":
    cfg, save_path = parse_args()
    run_forecast(cfg, save_path=save_path)
