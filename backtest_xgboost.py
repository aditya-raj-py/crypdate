"""
backtest_xgboost.py
====================
Walk-forward backtest for crypdate_xgboost.py's (v4.3) decision logic,
mirroring what backtest.py does for the Prophet pipeline.

Concept: step through crypdate_binance.db's OHLCV history. At each step,
slice the data up to that point only (no look-ahead), call
generate_decision() from crypdate_xgboost.py - which engineers features,
trains XGBoost, and produces a buy/sell/hold signal plus confidence - then
check what the price ACTUALLY did `horizon_days` later.

Reports results both WITH and WITHOUT confidence filtering, same as
backtest.py, so you can compare: does v4.2-style confidence filtering help
XGBoost too, and does XGBoost's signal beat Prophet's on the same coins?

Usage:
    python backtest_xgboost.py --symbol bitcoin
    python backtest_xgboost.py --symbol bitcoin --step-days 7
"""

import argparse
import warnings

import pandas as pd

from crypdate_xgboost import load_ohlcv, generate_decision

warnings.filterwarnings("ignore")


def run_backtest(
    db_path: str,
    coin: str,
    horizon_days: int = 7,
    step_days: int = 14,
    buy_threshold: float = 3.0,
    sell_threshold: float = -3.0,
    high_threshold_pct: float = 8.0,
    low_threshold_pct: float = 20.0,
    min_training_days: int = 90,
):
    df = load_ohlcv(db_path, coin)
    print(f"Loaded {len(df)} days of OHLCV history for {coin}\n")

    results = []
    i = min_training_days

    while i + horizon_days < len(df):
        history_slice = df.iloc[: i + 1].reset_index(drop=True)
        future_row = df.iloc[i + horizon_days]

        try:
            decision = generate_decision(
                history_slice,
                horizon_days=horizon_days,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                high_threshold_pct=high_threshold_pct,
                low_threshold_pct=low_threshold_pct,
            )
        except ValueError as e:
            # Not enough labeled rows yet at this point in history - skip ahead
            i += step_days
            continue

        signal = decision["signal"]
        position_size_pct = decision["position_size_pct"]

        actual_price_now = history_slice["close"].iloc[-1]
        actual_price_future = future_row["close"]
        actual_pct = ((actual_price_future - actual_price_now) / actual_price_now) * 100

        # No shorting: "buy" participates in the actual move, scaled by
        # position size; "sell"/"hold" sit in cash (0%). "Full-size" ignores
        # confidence entirely (always 100% on any buy) - the baseline to
        # compare against confidence-WEIGHTED sizing, which scales exposure
        # by how confident the model actually was (replaces the old hard
        # cutoff, which was found to discard the rare, high-payoff trades
        # that happen to come with wide/uncertain intervals).
        full_size_return = actual_pct if signal == "buy" else 0.0
        weighted_return = (position_size_pct / 100) * actual_pct if signal == "buy" else 0.0

        correct = (
            (signal == "buy" and actual_pct > 0)
            or (signal == "sell" and actual_pct < 0)
            or (signal == "hold" and abs(actual_pct) <= abs(buy_threshold))
        )

        results.append({
            "date": history_slice["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
            "signal": signal,
            "position_size_pct": position_size_pct,
            "confidence_label": decision["confidence_label"],
            "interval_half_width_pct": decision["interval_half_width_pct"],
            "predicted_pct": decision["percent_change"],
            "actual_pct": round(actual_pct, 2),
            "full_size_return": round(full_size_return, 2),
            "weighted_return": round(weighted_return, 2),
            "correct": correct,
        })

        r = results[-1]
        print(f"{r['date']}  signal={signal:<5} size={position_size_pct:>5.1f}% conf={r['confidence_label']:<6} (±{r['interval_half_width_pct']:.1f}%)  "
              f"predicted={r['predicted_pct']:+.2f}%  actual={r['actual_pct']:+.2f}%  {'OK' if r['correct'] else 'X'}")

        i += step_days

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNot enough data to run any backtest steps. Need more history or a smaller min-training-days.")
        return

    start_price = df["close"].iloc[min_training_days]
    end_price = df["close"].iloc[-1]
    buy_hold_return = ((end_price - start_price) / start_price) * 100

    def summarize(return_col, label, size_note=""):
        # "trades" here is buy+sell combined (both are directional calls we
        # score for win-rate purposes); only "buy" actually takes a
        # position/return, which position_size_pct and the return columns
        # already reflect.
        trades = results_df[results_df["signal"] != "hold"]
        buy_count = (results_df["signal"] == "buy").sum()
        sell_count = (results_df["signal"] == "sell").sum()
        win_rate = trades["correct"].mean() * 100 if len(trades) > 0 else 0.0

        # Compound sequentially: start with 1.0 "unit" of capital, apply each
        # decision point's return in order. This replaces naive summing,
        # which could produce nonsense like "-107%" - impossible for a
        # long-only, no-leverage strategy. NOTE: still ignores price
        # movement during the gap between a signal's horizon_days window and
        # the next decision point - a simplification, not a full simulation.
        portfolio = 1.0
        for r in results_df[return_col]:
            portfolio *= (1 + r / 100)
        total_return = (portfolio - 1) * 100

        print(f"\n--- {label} ---")
        print(f"Buy: {buy_count}  Sell: {sell_count}  Hold: {len(results_df) - len(trades)}{size_note}")
        print(f"Win rate (buy/sell calls): {win_rate:.1f}%")
        print(f"Compounded return:         {total_return:+.2f}%")
        return total_return

    print(f"\n=== Backtest Summary ({coin}, XGBoost) ===")
    print(f"Total decision points: {len(results_df)}")
    print(f"Buy-and-hold over period: {buy_hold_return:+.2f}%")
    print(f"Interval half-width %% range seen: {results_df['interval_half_width_pct'].min():.1f}% - "
          f"{results_df['interval_half_width_pct'].max():.1f}% (median {results_df['interval_half_width_pct'].median():.1f}%)")

    buy_rows = results_df[results_df["signal"] == "buy"]
    avg_size_note = f"  (avg size on buys: {buy_rows['position_size_pct'].mean():.1f}%)" if len(buy_rows) else ""

    # Does confidence actually correlate with being right? This is the key
    # question position sizing depends on - if High-confidence buys aren't
    # meaningfully more likely to be correct than Low-confidence ones, the
    # whole sizing approach is built on a shaky assumption.
    print(f"\n--- Win rate by confidence level (buy signals only) ---")
    for label in ["High", "Medium", "Low"]:
        bucket = buy_rows[buy_rows["confidence_label"] == label]
        if len(bucket) == 0:
            print(f"{label:<8} 0 trades")
            continue
        bucket_win_rate = bucket["correct"].mean() * 100
        avg_actual = bucket["actual_pct"].mean()
        print(f"{label:<8} {len(bucket):>4} trades   win rate: {bucket_win_rate:5.1f}%   avg actual move: {avg_actual:+6.2f}%")

    full_size_total = summarize("full_size_return", "Full-size (ignore confidence, 100% on every buy)")
    weighted_total = summarize("weighted_return", "Confidence-WEIGHTED position sizing", size_note=avg_size_note)

    print(f"\n--- Comparison ---")
    print(f"Buy-and-hold:              {buy_hold_return:+.2f}%")
    print(f"XGBoost full-size:         {full_size_total:+.2f}%")
    print(f"XGBoost weighted-size:     {weighted_total:+.2f}%")

    if weighted_total > full_size_total:
        print("\n→ Confidence-weighted sizing improved results over ignoring confidence entirely.")
    elif weighted_total < full_size_total:
        print("\n→ Confidence-weighted sizing made results worse - worth revisiting thresholds.")
    else:
        print("\n→ Confidence-weighted sizing made no difference in this run.")

    best = max(buy_hold_return, full_size_total, weighted_total)
    if best == buy_hold_return:
        print("→ Simple buy-and-hold still beat both XGBoost strategies over this period.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the XGBoost (v4.3) decision logic against historical OHLCV data")
    parser.add_argument("--db", default="crypdate_binance.db", help="Path to Binance OHLCV database")
    parser.add_argument("--symbol", required=True, help="Coin name, e.g. bitcoin, ethereum, dogecoin")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument("--buy-threshold", type=float, default=3.0)
    parser.add_argument("--sell-threshold", type=float, default=-3.0)
    parser.add_argument("--high-threshold-pct", type=float, default=8.0)
    parser.add_argument("--low-threshold-pct", type=float, default=20.0)
    parser.add_argument("--min-training-days", type=int, default=90, help="Minimum history before the first signal is generated")

    args = parser.parse_args()

    run_backtest(
        db_path=args.db,
        coin=args.symbol,
        horizon_days=args.horizon_days,
        step_days=args.step_days,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        high_threshold_pct=args.high_threshold_pct,
        low_threshold_pct=args.low_threshold_pct,
        min_training_days=args.min_training_days,
    )
