"""
backtest.py
===========
Walk-forward backtest for CRYPDATE's v4.1 (decision signals) + v4.2
(confidence scoring) logic, combined.

Concept: step through crypdate_historical.db's daily price history. At each
step, train Prophet on ONLY the data available up to that point (no
look-ahead), generate a decision signal AND a confidence score exactly like
crypdate_v4.0.py does, then check what the price ACTUALLY did `horizon_days`
later. This measures whether the signal logic would have been profitable
historically, and - critically - whether confidence filtering (only acting
on High/Medium confidence signals, per v4.2) actually improves results
compared to acting on every signal regardless of confidence.

Reports TWO win rates and TWO return totals: "all signals" (v4.1 alone) vs
"confidence-filtered" (v4.1 + v4.2), so you can see whether v4.2 is actually
pulling its weight.

This is a first-pass backtester: it ignores trading fees, slippage, and
partial position sizing (every "buy" signal is treated as if you went
all-in, every "sell" as if you went fully to cash). Treat results as a
rough signal of whether the underlying logic has any merit at all, not a
realistic profit projection.

Usage:
    python backtest.py --symbol bitcoin
    python backtest.py --symbol bitcoin --step-days 14 --horizon-days 7
"""

import argparse
import sqlite3
import warnings

import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")  # Prophet is noisy about minor warnings


def load_full_history(db_path: str, coin: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp AS ds, price AS y FROM prices WHERE coin = ? ORDER BY timestamp ASC",
        conn,
        params=(coin,),
    )
    conn.close()

    df["ds"] = pd.to_datetime(df["ds"], format="ISO8601")
    if df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"]).drop_duplicates(subset=["ds"]).reset_index(drop=True)
    return df


def get_signal(
    history_slice: pd.DataFrame,
    horizon_days: int,
    buy_threshold: float,
    sell_threshold: float,
    high_threshold_pct: float = 8.0,
    low_threshold_pct: float = 20.0,
):
    """
    Train on history_slice only, forecast horizon_days ahead.
    Returns (raw_signal, predicted_pct_change, confidence_score, confidence_label, interval_half_width_pct).

    Confidence uses ABSOLUTE thresholds (interval half-width as a % of
    predicted price), matching crypdate_v4.0.py's compute_confidence_scores.
    An earlier version normalized against this forecast's own min/max
    interval widths, which structurally biased the target (always-widest)
    point toward "Low" regardless of forecast quality - fixed here.
    """
    model = Prophet(interval_width=0.95)
    model.fit(history_slice)

    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future)

    last_price = history_slice["y"].iloc[-1]
    target_row = forecast.iloc[-1]
    predicted_price = target_row["yhat"]
    pct_change = ((predicted_price - last_price) / last_price) * 100

    if pct_change > buy_threshold:
        signal = "buy"
    elif pct_change < sell_threshold:
        signal = "sell"
    else:
        signal = "hold"

    half_width = (target_row["yhat_upper"] - target_row["yhat_lower"]) / 2
    half_width_pct = (half_width / abs(predicted_price)) * 100 if predicted_price != 0 else 0.0

    if half_width_pct <= high_threshold_pct:
        confidence_score = 1.0
    elif half_width_pct >= low_threshold_pct:
        confidence_score = 0.0
    else:
        confidence_score = round(1 - (half_width_pct - high_threshold_pct) / (low_threshold_pct - high_threshold_pct), 3)

    if confidence_score >= 0.66:
        confidence_label = "High"
    elif confidence_score >= 0.33:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return signal, pct_change, confidence_score, confidence_label, round(half_width_pct, 2)


def run_backtest(
    db_path: str,
    coin: str,
    horizon_days: int = 7,
    step_days: int = 14,
    buy_threshold: float = 3.0,
    sell_threshold: float = -3.0,
    min_training_days: int = 90,
    high_threshold_pct: float = 8.0,
    low_threshold_pct: float = 20.0,
):
    df = load_full_history(db_path, coin)
    print(f"Loaded {len(df)} days of history for {coin}\n")

    results = []
    i = min_training_days

    while i + horizon_days < len(df):
        history_slice = df.iloc[:i].copy()
        future_row = df.iloc[i + horizon_days]

        raw_signal, predicted_pct, confidence_score, confidence_label, half_width_pct = get_signal(
            history_slice, horizon_days, buy_threshold, sell_threshold,
            high_threshold_pct=high_threshold_pct, low_threshold_pct=low_threshold_pct,
        )

        # v4.2 rule: downgrade buy/sell to hold if confidence is Low
        if confidence_label == "Low" and raw_signal != "hold":
            filtered_signal = "hold"
        else:
            filtered_signal = raw_signal

        actual_price_now = history_slice["y"].iloc[-1]
        actual_price_future = future_row["y"]
        actual_pct = ((actual_price_future - actual_price_now) / actual_price_now) * 100

        def simulated_return_for(signal):
            if signal == "buy":
                return actual_pct
            elif signal == "sell":
                return -actual_pct
            return 0.0

        def correct_for(signal):
            return (
                (signal == "buy" and actual_pct > 0)
                or (signal == "sell" and actual_pct < 0)
                or (signal == "hold" and abs(actual_pct) <= abs(buy_threshold))
            )

        results.append({
            "date": history_slice["ds"].iloc[-1].strftime("%Y-%m-%d"),
            "raw_signal": raw_signal,
            "filtered_signal": filtered_signal,
            "confidence_label": confidence_label,
            "confidence_score": confidence_score,
            "interval_half_width_pct": half_width_pct,
            "predicted_pct": round(predicted_pct, 2),
            "actual_pct": round(actual_pct, 2),
            "raw_return": round(simulated_return_for(raw_signal), 2),
            "filtered_return": round(simulated_return_for(filtered_signal), 2),
            "raw_correct": correct_for(raw_signal),
            "filtered_correct": correct_for(filtered_signal),
        })

        downgrade_note = " (downgraded)" if filtered_signal != raw_signal else ""
        print(f"{results[-1]['date']}  raw={raw_signal:<5} conf={confidence_label:<6} (±{half_width_pct:.1f}%)  "
              f"predicted={predicted_pct:+.2f}%  actual={actual_pct:+.2f}%  "
              f"{'✓' if results[-1]['raw_correct'] else '✗'}{downgrade_note}")

        i += step_days

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNot enough data to run any backtest steps. Need more history or a smaller step-days/horizon-days.")
        return

    # Buy-and-hold baseline over the same overall period the backtest covered
    start_price = df["y"].iloc[min_training_days]
    end_price = df["y"].iloc[-1]
    buy_hold_return = ((end_price - start_price) / start_price) * 100

    def summarize(signal_col, return_col, correct_col, label):
        trades = results_df[results_df[signal_col] != "hold"]
        win_rate = trades[correct_col].mean() * 100 if len(trades) > 0 else 0.0
        total_return = results_df[return_col].sum()
        print(f"\n--- {label} ---")
        print(f"Buy/Sell signals:          {len(trades)}  (Hold: {len(results_df) - len(trades)})")
        print(f"Win rate (buy/sell only):  {win_rate:.1f}%")
        print(f"Sum of simulated returns:  {total_return:+.2f}%  (naive sum, not compounded)")
        return total_return

    print(f"\n=== Backtest Summary ({coin}) ===")
    print(f"Total decision points: {len(results_df)}")
    print(f"Buy-and-hold over period: {buy_hold_return:+.2f}%")
    print(f"Interval half-width %% range seen: {results_df['interval_half_width_pct'].min():.1f}% - "
          f"{results_df['interval_half_width_pct'].max():.1f}% (median {results_df['interval_half_width_pct'].median():.1f}%)")

    raw_total = summarize("raw_signal", "raw_return", "raw_correct", "v4.1 only (all signals, no confidence filtering)")
    filtered_total = summarize("filtered_signal", "filtered_return", "filtered_correct", "v4.1 + v4.2 (confidence-filtered)")

    print(f"\n--- Comparison ---")
    print(f"Buy-and-hold:              {buy_hold_return:+.2f}%")
    print(f"v4.1 only:                 {raw_total:+.2f}%")
    print(f"v4.1 + v4.2 filtered:      {filtered_total:+.2f}%")

    if filtered_total > raw_total:
        print("\n→ Confidence filtering (v4.2) improved results over acting on every signal.")
    elif filtered_total < raw_total:
        print("\n→ Confidence filtering (v4.2) made results worse than acting on every signal - worth revisiting the thresholds.")
    else:
        print("\n→ Confidence filtering made no difference in this run.")

    best = max(buy_hold_return, raw_total, filtered_total)
    if best == buy_hold_return:
        print("→ Simple buy-and-hold still beat both signal strategies over this period.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest CRYPDATE's decision logic against historical data")
    parser.add_argument("--db", default="crypdate_historical.db", help="Path to historical SQLite database")
    parser.add_argument("--symbol", required=True, help="Coin name as stored in DB, e.g. bitcoin")
    parser.add_argument("--horizon-days", type=int, default=7, help="Days ahead each signal predicts")
    parser.add_argument("--step-days", type=int, default=14, help="Days between each backtest decision point")
    parser.add_argument("--buy-threshold", type=float, default=3.0)
    parser.add_argument("--sell-threshold", type=float, default=-3.0)
    parser.add_argument("--min-training-days", type=int, default=90, help="Minimum history before the first signal is generated")
    parser.add_argument("--high-threshold-pct", type=float, default=8.0, help="Interval half-width %% at or below this = High confidence")
    parser.add_argument("--low-threshold-pct", type=float, default=20.0, help="Interval half-width %% at or above this = Low confidence")

    args = parser.parse_args()

    run_backtest(
        db_path=args.db,
        coin=args.symbol,
        horizon_days=args.horizon_days,
        step_days=args.step_days,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        min_training_days=args.min_training_days,
        high_threshold_pct=args.high_threshold_pct,
        low_threshold_pct=args.low_threshold_pct,
    )
