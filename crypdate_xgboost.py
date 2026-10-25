"""
crypdate_xgboost.py
====================
v4.3 - Replaces Prophet with XGBoost as the forecasting "brain".

Unlike Prophet (which only sees price + time), this builds engineered
features from OHLCV data - lagged prices, rolling averages, volatility,
momentum, volume trends - and trains a gradient-boosted tree model to
predict the price `horizon_days` ahead.

Confidence is derived differently than Prophet's built-in interval: this
holds out a validation slice, measures how far off the model's predictions
were on data it didn't train on, and uses that error spread to build a
yhat +/- band - serving the same role Prophet's yhat_lower/yhat_upper did,
so the same downstream decision-signal logic can be reused.

Data source: crypdate_binance.db (from binance_fetch.py), NOT
crypdate_historical.db - this needs OHLCV, which only the Binance pipeline
provides.

Usage:
    python crypdate_xgboost.py --symbol bitcoin
    python crypdate_xgboost.py --symbol ethereum --horizon-days 7
"""

import argparse
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from crypdate_markets import MARKETS, get_market, resolve_symbol

warnings.filterwarnings("ignore")


def load_ohlcv(db_path: str, coin: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE coin = ? ORDER BY timestamp ASC",
        conn, params=(coin,),
    )
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for '{coin}' in {db_path}. Run binance_fetch.py (crypto) or stock_fetch.py (stocks) first.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Builds predictive features from raw OHLCV, and a target column, ALL in
    relative/percentage terms rather than raw dollar price.

    Why this matters: bitcoin ranged from ~$1 to ~$90,000+ over the
    available history - a ~90,000x range. Features or targets expressed in
    raw dollars (e.g. "30-day rolling average price") make the model mostly
    learn "what price regime are we in" rather than genuine patterns that
    transfer across time. Expressing everything as a percentage (distance
    from a moving average, % volatility, % return) makes a pattern from
    when bitcoin was $1,000 comparable to one from when it was $90,000.
    """
    df = df.copy()

    # Returns (already scale-invariant - % change, not raw price)
    for lag in [1, 2, 3, 7, 14]:
        df[f"return_{lag}d"] = df["close"].pct_change(lag)

    # Rolling stats, expressed RELATIVE to current price, not raw dollars
    for window in [7, 14, 30]:
        rolling_mean = df["close"].rolling(window).mean()
        rolling_std = df["close"].rolling(window).std()
        df[f"dist_from_ma_{window}_pct"] = (df["close"] / rolling_mean - 1) * 100
        df[f"volatility_{window}_pct"] = (rolling_std / df["close"]) * 100

    # Daily range and volume trend (already percentage-based)
    df["daily_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    # Zero-volume days (common in real stock data) would divide by zero and
    # produce inf, which XGBoost rejects. Treat a zero as "no information",
    # and a missing change as 0 (no change).
    df["volume_change_7d"] = (
        df["volume"].replace(0, np.nan).pct_change(7, fill_method=None)
        .replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # Simple RSI (14-day) - already bounded 0-100, scale-invariant
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Safety net: no feature may contain inf (bad prices, zero divisions)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # Target: PERCENT change `horizon_days` ahead, not raw future price
    df["target"] = (df["close"].shift(-horizon_days) - df["close"]) / df["close"] * 100

    return df


FEATURE_COLUMNS = [
    "return_1d", "return_2d", "return_3d", "return_7d", "return_14d",
    "dist_from_ma_7_pct", "dist_from_ma_14_pct", "dist_from_ma_30_pct",
    "volatility_7_pct", "volatility_14_pct", "volatility_30_pct",
    "daily_range_pct", "volume_change_7d", "rsi_14",
]


def train_and_predict(df_features: pd.DataFrame, horizon_days: int, val_fraction: float = 0.15):
    """
    Trains XGBoost on all rows with a known target (now a PERCENT change,
    not raw price - see engineer_features), holds out the most recent
    `val_fraction` to measure prediction error in percentage terms (used
    for the confidence band), then predicts forward from the very last row.

    Returns: predicted_price, lower_bound, upper_bound, model, last_date
    (prices/bounds are converted back from the model's % prediction using
    the current price, so downstream code is unchanged).
    """
    labeled = df_features.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    if len(labeled) < 50:
        raise ValueError(f"Only {len(labeled)} usable rows after feature engineering - need more history.")

    split_idx = int(len(labeled) * (1 - val_fraction))
    train_df = labeled.iloc[:split_idx]
    val_df = labeled.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["target"])

    # Validation error (in PERCENT terms now) -> confidence band width
    val_preds_pct = model.predict(val_df[FEATURE_COLUMNS])
    residuals_pct = val_df["target"].values - val_preds_pct
    error_std_pct = residuals_pct.std()

    # Refit on ALL labeled data before predicting the real future point
    model.fit(labeled[FEATURE_COLUMNS], labeled["target"])

    # Predict from the most recent row (features exist, target doesn't)
    latest_row = df_features.iloc[[-1]]
    if latest_row[FEATURE_COLUMNS].isna().any(axis=1).iloc[0]:
        raise ValueError(
            "The most recent row has missing features. Check the latest data for "
            "zero or missing prices or volumes, then refetch."
        )
    last_price = float(latest_row["close"].iloc[0])
    predicted_pct = float(model.predict(latest_row[FEATURE_COLUMNS])[0])

    predicted_price = last_price * (1 + predicted_pct / 100)
    lower_pct = predicted_pct - 1.96 * error_std_pct  # ~95% band, assuming ~normal residuals
    upper_pct = predicted_pct + 1.96 * error_std_pct
    lower_bound = last_price * (1 + lower_pct / 100)
    upper_bound = last_price * (1 + upper_pct / 100)

    last_date = df_features["timestamp"].iloc[-1]

    return predicted_price, lower_bound, upper_bound, model, last_date


def generate_decision(
    df: pd.DataFrame,
    horizon_days: int = 7,
    buy_threshold: float = 3.0,
    sell_threshold: float = -3.0,
    high_threshold_pct: float = 8.0,
    low_threshold_pct: float = 20.0,
    business_days: bool = False,
) -> dict:
    df_features = engineer_features(df, horizon_days)
    predicted_price, lower, upper, model, last_date = train_and_predict(df_features, horizon_days)

    last_price = df["close"].iloc[-1]
    percent_change = ((predicted_price - last_price) / last_price) * 100

    if percent_change > buy_threshold:
        raw_signal = "buy"
    elif percent_change < sell_threshold:
        raw_signal = "sell"
    else:
        raw_signal = "hold"

    half_width_pct = ((upper - lower) / 2) / abs(predicted_price) * 100 if predicted_price != 0 else 0.0

    if half_width_pct <= high_threshold_pct:
        confidence_score, confidence_label = 1.0, "High"
    elif half_width_pct >= low_threshold_pct:
        confidence_score, confidence_label = 0.0, "Low"
    else:
        confidence_score = round(1 - (half_width_pct - high_threshold_pct) / (low_threshold_pct - high_threshold_pct), 3)
        confidence_label = "Medium"

    # Position sizing instead of a hard confidence cutoff: a low-confidence
    # buy still gets SOME exposure rather than being thrown out entirely.
    # This matters because the biggest, most profitable moves (e.g. a
    # meme-coin pump) are often the least predictable in advance - a hard
    # "skip if uncertain" filter was found to systematically discard
    # exactly those trades. No shorting, so sell/hold always size to 0.
    position_size_pct = round(confidence_score * 100, 1) if raw_signal == "buy" else 0.0

    # Stocks do not trade on weekends: count the horizon in business days there
    # (ignores exchange holidays, so the label can be a day or two off).
    if business_days:
        target = last_date + pd.offsets.BDay(horizon_days)
    else:
        target = last_date + pd.Timedelta(days=horizon_days)

    return {
        "last_price": round(float(last_price), 2),
        "target_date": target.strftime("%Y-%m-%d"),
        "predicted_price": round(predicted_price, 2),
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
        "percent_change": round(percent_change, 2),
        "interval_half_width_pct": round(half_width_pct, 2),
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "signal": raw_signal,
        "position_size_pct": position_size_pct,
        "feature_importance": dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(3).tolist())),
    }


def plot_forecast(df: pd.DataFrame, decision: dict, symbol: str, save_path: str = None, lookback_days: int = 180) -> None:
    """
    Plots recent historical close prices plus the single predicted future
    point with an error bar showing the confidence range. Unlike Prophet's
    day-by-day forecast curve, XGBoost here predicts one target point, so
    the plot reflects that directly rather than faking a smooth line.
    """
    recent = df.tail(lookback_days)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(recent["timestamp"], recent["close"], "k-", label="Historical", linewidth=1)

    target_date = pd.to_datetime(decision["target_date"])
    predicted = decision["predicted_price"]
    lower = decision["lower_bound"]
    upper = decision["upper_bound"]

    ax.errorbar(
        [target_date], [predicted],
        yerr=[[predicted - lower], [upper - predicted]],
        fmt="o", color="blue", ecolor="blue", elinewidth=2, capsize=6,
        markersize=8, label=f"Predicted ({decision['confidence_label']} confidence)",
    )

    ax.axhline(decision["last_price"], color="gray", linestyle="--", linewidth=0.8, label="Last known price")

    ax.set_title(f"{symbol} Price Forecast (XGBoost, v4.3) - Signal: {decision['signal'].upper()}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost-based forecasting + decision signal (v4.3)")
    parser.add_argument("--market", choices=list(MARKETS), default="crypto",
                        help="crypto (default), india or us. Picks the database, costs and calendar")
    parser.add_argument("--db", default=None, help="OHLCV database path (default: the market's database)")
    parser.add_argument("--symbol", required=True,
                        help="Asset name: bitcoin, ethereum, dogecoin, or a ticker like RELIANCE / AAPL")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--buy-threshold", type=float, default=None, help="Predicted %% change that counts as a buy (default: the market profile)")
    parser.add_argument("--sell-threshold", type=float, default=None, help="Predicted %% change that counts as a sell (default: the market profile)")
    parser.add_argument("--high-threshold-pct", type=float, default=8.0)
    parser.add_argument("--low-threshold-pct", type=float, default=20.0)

    args = parser.parse_args()

    market = get_market(args.market)
    symbol = resolve_symbol(market, args.symbol)
    cur = market.currency_prefix
    day_word = "trading days" if market.uses_business_days else "days"

    df = load_ohlcv(args.db or market.db_path, symbol)
    print(f"Loaded {len(df)} days of OHLCV history for {symbol}" + ("" if market.name == "crypto" else f" ({market.name})") + "\n")

    decision = generate_decision(
        df,
        horizon_days=args.horizon_days,
        buy_threshold=market.buy_threshold if args.buy_threshold is None else args.buy_threshold,
        sell_threshold=market.sell_threshold if args.sell_threshold is None else args.sell_threshold,
        high_threshold_pct=args.high_threshold_pct,
        low_threshold_pct=args.low_threshold_pct,
        business_days=market.uses_business_days,
    )

    print("--- CRYPDATED (XGBoost) ---")
    print(f"Last known price:     {cur}{decision['last_price']:,}")
    print(f"Target date:          {decision['target_date']} ({args.horizon_days} {day_word} ahead)")
    print(f"Predicted price:      {cur}{decision['predicted_price']:,}  (range: {cur}{decision['lower_bound']:,} - {cur}{decision['upper_bound']:,})")
    print(f"Predicted change:     {decision['percent_change']:+.2f}%")
    print(f"Confidence:           {decision['confidence_label']} ({decision['confidence_score']:.2f})")
    print(f"Interval half-width:  {decision['interval_half_width_pct']}% of predicted price")
    print(f"Signal:               {decision['signal'].upper()}")
    print(f"Position size:        {decision['position_size_pct']}% of capital")

    print("\nTop 5 features by importance:")
    top_features = sorted(decision["feature_importance"].items(), key=lambda x: -x[1])[:5]
    for name, importance in top_features:
        print(f"  {name:<20} {importance}")

    if market.name != "crypto":
        print("\nNote: the buy/sell and confidence thresholds here are placeholders for stocks. "
              "Run backtest_xgboost.py for this ticker to see calibrated confidence thresholds.")
