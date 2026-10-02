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
    freq: str = "D"                # forecast step: "D" daily, "h" hourly, etc.
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

def compute_decision_percentages(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    horizon_days: int = 7,
    buy_threshold: float = 3.0,
    sell_threshold: float = -3.0,
) -> dict:
    """
    v4.1 - Decision percentages.

    Compares the forecasted price `horizon_days` ahead against the last
    known actual price, expresses the difference as a percentage change,
    and classifies it into a buy/sell/hold signal using simple thresholds.

    This is intentionally simple (a first pass): it only looks at where
    Prophet's point forecast (yhat) lands relative to today, with no
    awareness of confidence width, volatility, or risk. Treat this as a
    starting signal to backtest, not a trading instruction.

    Returns a dict with:
        last_price       - most recent known actual price
        target_date      - the forecasted date being evaluated
        predicted_price  - yhat at that date
        percent_change   - % difference vs last_price
        signal           - "buy" / "sell" / "hold"
    """
    last_price = history["y"].iloc[-1]
    last_date = history["ds"].iloc[-1]

    target_date = last_date + pd.Timedelta(days=horizon_days)

    # Find the forecast row closest to the target date
    forecast = forecast.copy()
    forecast["date_diff"] = (forecast["ds"] - target_date).abs()
    closest_row = forecast.loc[forecast["date_diff"].idxmin()]

    predicted_price = closest_row["yhat"]
    percent_change = ((predicted_price - last_price) / last_price) * 100

    if percent_change > buy_threshold:
        signal = "buy"
    elif percent_change < sell_threshold:
        signal = "sell"
    else:
        signal = "hold"

    return {
        "last_price": round(float(last_price), 2),
        "target_date": closest_row["ds"].strftime("%Y-%m-%d"),
        "predicted_price": round(float(predicted_price), 2),
        "percent_change": round(float(percent_change), 2),
        "signal": signal,
    }


def compute_confidence_scores(
    forecast: pd.DataFrame,
    high_threshold_pct: float = 8.0,
    low_threshold_pct: float = 20.0,
) -> pd.DataFrame:
    """
    v4.2 - Confidence scoring (absolute-threshold version).

    Adds 'interval_half_width_pct' (how wide the confidence interval is,
    as a percentage of the predicted price), 'confidence_score' (0-1,
    higher = more confident), and 'confidence_label' ("Low"/"Medium"/
    "High") columns to the forecast dataframe.

    IMPORTANT: this uses FIXED, absolute thresholds you set (as a % of
    predicted price), not a ranking relative to other points in the same
    forecast. An earlier version normalized against this forecast's own
    min/max interval widths - but since Prophet's uncertainty always
    grows toward the forecast horizon, the future point we actually care
    about was nearly always the widest in its own comparison group,
    structurally biasing it toward "Low" regardless of whether the
    forecast was actually any good. Absolute thresholds avoid that.

    Defaults (8% / 20%) are a starting point for daily crypto forecasts -
    recalibrate based on what interval_half_width_pct actually looks like
    for your data (it's included in the output so you can inspect it).
    """
    forecast = forecast.copy()

    half_width = (forecast["yhat_upper"] - forecast["yhat_lower"]) / 2
    half_width_pct = (
        half_width / forecast["yhat"].abs().replace(0, pd.NA)) * 100
    half_width_pct = half_width_pct.fillna(half_width_pct.mean())

    forecast["interval_half_width_pct"] = half_width_pct.round(2)

    def score_row(pct):
        if pct <= high_threshold_pct:
            return 1.0
        elif pct >= low_threshold_pct:
            return 0.0
        else:
            # linear interpolation between the two thresholds
            return round(1 - (pct - high_threshold_pct) / (low_threshold_pct - high_threshold_pct), 3)

    forecast["confidence_score"] = half_width_pct.apply(score_row)

    def label_row(score):
        if score >= 0.66:
            return "High"
        elif score >= 0.33:
            return "Medium"
        else:
            return "Low"

    forecast["confidence_label"] = forecast["confidence_score"].apply(
        label_row)

    return forecast


def apply_confidence_to_decision(decision: dict, forecast_with_confidence: pd.DataFrame) -> dict:
    """
    Looks up the confidence score/label at the decision's target_date and
    adds them to the decision dict. If confidence is "Low", downgrades a
    buy/sell signal to "hold" rather than acting on a shaky forecast -
    the decision still reports what the raw signal WOULD have been, via
    'raw_signal', so nothing is hidden, just not blindly acted on.
    """
    decision = dict(decision)  # don't mutate caller's dict

    target_date = pd.to_datetime(decision["target_date"])
    forecast_with_confidence = forecast_with_confidence.copy()
    forecast_with_confidence["date_diff"] = (
        forecast_with_confidence["ds"] - target_date).abs()
    closest_row = forecast_with_confidence.loc[forecast_with_confidence["date_diff"].idxmin(
    )]

    confidence_score = float(closest_row["confidence_score"])
    confidence_label = str(closest_row["confidence_label"])

    decision["confidence_score"] = round(confidence_score, 3)
    decision["confidence_label"] = confidence_label
    decision["interval_half_width_pct"] = float(
        closest_row["interval_half_width_pct"])
    decision["raw_signal"] = decision["signal"]

    if confidence_label == "Low" and decision["signal"] != "hold":
        decision["signal"] = "hold"
        decision["downgraded"] = True
    else:
        decision["downgraded"] = False

    return decision


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
                        help="Resample raw logs before fitting (e.g. h, D); 'none' to disable")
    parser.add_argument("--save", default=None,
                        help="Path to save the plot instead of showing it")
    parser.add_argument("--horizon-days", type=int, default=7,
                        help="Days ahead to evaluate for the buy/sell/hold decision (v4.1)")
    parser.add_argument("--buy-threshold", type=float, default=3.0,
                        help="Percent gain above which signal is 'buy'")
    parser.add_argument("--sell-threshold", type=float, default=-
                        3.0, help="Percent loss below which signal is 'sell'")

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
    return config, args.save, args.horizon_days, args.buy_threshold, args.sell_threshold


if __name__ == "__main__":
    cfg, save_path, horizon_days, buy_threshold, sell_threshold = parse_args()
    history_df = load_data(cfg)
    model = train_model(history_df, cfg)
    forecast_df = forecast_prices(model, cfg)
    plot_forecast(history_df, forecast_df, cfg, save_path=save_path)

    decision = compute_decision_percentages(
        forecast_df,
        history_df,
        horizon_days=horizon_days,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    forecast_scored = compute_confidence_scores(forecast_df)
    decision = apply_confidence_to_decision(decision, forecast_scored)

    print(f"\n--- CRYPDATED ---")
    print(f"Last known price:     ${decision['last_price']:,}")
    print(
        f"Target date:          {decision['target_date']} ({horizon_days} days ahead)")
    print(f"Predicted price:      ${decision['predicted_price']:,}")
    print(f"Predicted change:     {decision['percent_change']:+.2f}%")
    print(
        f"Confidence:           {decision['confidence_label']} ({decision['confidence_score']:.2f})")
    print(
        f"Interval half-width:  {decision.get('interval_half_width_pct', 'n/a')}% of predicted price")
    if decision["downgraded"]:
        print(
            f"Raw signal:            {decision['raw_signal'].upper()} (downgraded due to low confidence)")
    print(f"Signal:               {decision['signal'].upper()}")
