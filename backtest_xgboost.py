"""
backtest_xgboost.py
====================
Walk-forward backtest for crypdate_xgboost.py (v4.3), upgraded with:

  * Trading costs: a fee plus slippage on every side of every trade.
  * Risk metrics: max drawdown and annualized Sharpe ratio, shown for
    buy-and-hold, full-size trading and confidence-weighted sizing.
  * An out-of-sample split: confidence thresholds are calibrated on EARLIER
    decision points and every result is scored only on LATER ones.
  * Date-range testing: --test-start-date / --test-end-date let you score a
    specific regime (e.g. the 2022 bear market) while still calibrating on
    everything before it.

How it works: step through the OHLCV history. At each decision point, train
XGBoost on ONLY the data available then (no look-ahead), get a buy/sell/hold
signal and an interval width, then check what price actually did
`horizon_days` later. Position size for a buy is its confidence score
(narrow interval = bigger size). The project is long-only: sell and hold
both mean cash.

Simplifications to keep in mind:
  * Strategies are only in the market for `horizon_days` out of every
    `step_days`; buy-and-hold is in for the whole step. Compare drawdown and
    Sharpe with that in mind.
  * Every buy is charged a full round trip (buy and sell). Real consecutive
    buys could hold through, so costs are slightly overstated.
  * Max drawdown and Sharpe use one return per decision point, so
    swings between decision points are not captured.
  * Sharpe assumes a 0% risk-free rate.

Usage:
    python backtest_xgboost.py --symbol bitcoin
    python backtest_xgboost.py --symbol bitcoin --test-start-date 2022-01-01 --test-end-date 2022-12-31
    python backtest_xgboost.py --symbol bitcoin --step-days 7
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from crypdate_xgboost import load_ohlcv, generate_decision

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # safe when output is redirected to a file on Windows
except Exception:
    pass


# --------------------------------------------------------------------------
# Confidence helpers
# --------------------------------------------------------------------------

def confidence_score(half_width_pct, high_pct, low_pct):
    """1.0 at/below the high threshold, 0.0 at/above the low threshold, linear in between."""
    if half_width_pct <= high_pct:
        return 1.0
    if half_width_pct >= low_pct:
        return 0.0
    return 1 - (half_width_pct - high_pct) / (low_pct - high_pct)


def confidence_label(half_width_pct, high_pct, low_pct):
    if half_width_pct <= high_pct:
        return "High"
    if half_width_pct >= low_pct:
        return "Low"
    return "Medium"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def total_return_pct(returns):
    return ((1 + returns).prod() - 1) * 100


def max_drawdown_pct(returns):
    equity = np.concatenate([[1.0], (1 + returns).cumprod().values])
    peak = np.maximum.accumulate(equity)
    return ((equity / peak) - 1).min() * 100


def sharpe_ratio(returns, step_days):
    if len(returns) < 3:
        return float("nan")
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return returns.mean() / sd * np.sqrt(365 / step_days)


# --------------------------------------------------------------------------
# Walk-forward data collection (the slow part: one model fit per point)
# --------------------------------------------------------------------------

def collect_decision_points(df, horizon_days, step_days, buy_threshold, sell_threshold,
                            min_training_days, end_date=None):
    points = []
    i = min_training_days

    while i + horizon_days < len(df):
        date = df["timestamp"].iloc[i]
        if end_date is not None and date > end_date:
            break

        history_slice = df.iloc[: i + 1].reset_index(drop=True)
        try:
            decision = generate_decision(
                history_slice,
                horizon_days=horizon_days,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
        except ValueError:
            i += step_days
            continue

        price_now = df["close"].iloc[i]
        price_future = df["close"].iloc[i + horizon_days]
        price_step = df["close"].iloc[min(i + step_days, len(df) - 1)]

        point = {
            "date": date,
            "signal": decision["signal"],
            "predicted_pct": decision["percent_change"],
            "half_width_pct": decision["interval_half_width_pct"],
            "actual_pct": (price_future / price_now - 1) * 100,
            "bh_step_pct": (price_step / price_now - 1) * 100,
        }
        points.append(point)

        print(f"{date:%Y-%m-%d}  signal={point['signal']:<5} predicted={point['predicted_pct']:+7.2f}%  "
              f"actual={point['actual_pct']:+7.2f}%  width=+/-{point['half_width_pct']:.1f}%")

        i += step_days

    return pd.DataFrame(points)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_backtest(args):
    df = load_ohlcv(args.db, args.symbol)
    print(f"Loaded {len(df)} days of OHLCV history for {args.symbol}\n")

    end_date = pd.Timestamp(args.test_end_date) if args.test_end_date else None
    points = collect_decision_points(
        df, args.horizon_days, args.step_days, args.buy_threshold, args.sell_threshold,
        args.min_training_days, end_date,
    )

    if points.empty:
        print("\nNot enough data to run any decision points.")
        return

    # ---- Split into calibration (earlier) and test (later) ----
    if args.test_start_date:
        test_start = pd.Timestamp(args.test_start_date)
    else:
        cut = min(int(len(points) * args.calibration_fraction), len(points) - 1)
        test_start = points["date"].iloc[cut]

    calib = points[points["date"] < test_start]
    test = points[points["date"] >= test_start].reset_index(drop=True)

    if test.empty:
        print("\nNo decision points fall in the test period. Check --test-start-date / --test-end-date.")
        return

    # ---- Confidence thresholds ----
    manual = args.high_threshold_pct is not None and args.low_threshold_pct is not None
    if manual:
        high_thr, low_thr = args.high_threshold_pct, args.low_threshold_pct
        source = "manual (only out-of-sample if you chose them before seeing these results)"
    elif len(calib) >= 10:
        high_thr = float(np.quantile(calib["half_width_pct"], args.calib_high_quantile))
        low_thr = float(np.quantile(calib["half_width_pct"], args.calib_low_quantile))
        source = f"calibrated on {len(calib)} earlier decision points (out-of-sample)"
    else:
        high_thr = float(np.quantile(test["half_width_pct"], args.calib_high_quantile))
        low_thr = float(np.quantile(test["half_width_pct"], args.calib_low_quantile))
        source = "IN-SAMPLE (too few earlier points to calibrate separately) - treat results as optimistic"

    if low_thr <= high_thr:
        low_thr = high_thr + 1e-6

    # ---- Position sizes and returns ----
    is_buy = test["signal"] == "buy"
    scores = test["half_width_pct"].apply(lambda hw: confidence_score(hw, high_thr, low_thr))
    weighted_size = pd.Series(np.where(is_buy, scores, 0.0), index=test.index)
    full_size = pd.Series(np.where(is_buy, 1.0, 0.0), index=test.index)

    cost_per_side = (args.fee_pct + args.slippage_pct) / 100
    actual = test["actual_pct"] / 100

    def returns_for(size):
        gross = size * actual
        net = gross - size * 2 * cost_per_side
        return gross, net

    full_gross, full_net = returns_for(full_size)
    weighted_gross, weighted_net = returns_for(weighted_size)
    bh_returns = test["bh_step_pct"] / 100

    # ---- Report ----
    sells = (test["signal"] == "sell").sum()
    buys = int(is_buy.sum())
    holds = len(test) - buys - sells
    directional = test[test["signal"] != "hold"]
    correct = ((directional["signal"] == "buy") & (directional["actual_pct"] > 0)) | \
              ((directional["signal"] == "sell") & (directional["actual_pct"] < 0))
    win_rate = correct.mean() * 100 if len(directional) else 0.0

    print(f"\n=== Backtest Summary ({args.symbol}, XGBoost) ===")
    print(f"Decision points total: {len(points)}  (calibration: {len(calib)}, test: {len(test)})")
    print(f"Test period: {test['date'].iloc[0]:%Y-%m-%d} to {test['date'].iloc[-1]:%Y-%m-%d}")
    print(f"Confidence thresholds: high <= {high_thr:.1f}%, low >= {low_thr:.1f}%  [{source}]")
    print(f"Costs: {args.fee_pct}% fee + {args.slippage_pct}% slippage per side (charged on every buy, round trip)")
    print(f"Test signals: buy {buys}, sell {sells}, hold {holds}   "
          f"invested at {buys / len(test) * 100:.0f}% of decision points")
    if buys:
        print(f"Average size on buys: {weighted_size[is_buy].mean() * 100:.1f}%")
    print(f"Win rate (buy/sell calls): {win_rate:.1f}%")

    always_returns = actual - 2 * cost_per_side  # full-size buy at EVERY decision point
    cash_returns = pd.Series(0.0, index=test.index)

    rows = [
        ("Cash (no trading)", cash_returns),
        ("Buy-and-hold", bh_returns),
        ("Always buy each window", always_returns),
        ("XGBoost full-size", full_net),
        ("XGBoost weighted", weighted_net),
    ]
    print(f"\n{'Strategy':<22}{'Net return':>12}{'Max drawdown':>15}{'Sharpe':>9}")
    for name, rets in rows:
        print(f"{name:<22}{total_return_pct(rets):>+11.1f}%{max_drawdown_pct(rets):>14.1f}%"
              f"{sharpe_ratio(rets, args.step_days):>9.2f}")

    print(f"\nCost drag (return before costs -> after):")
    print(f"  Full-size: {total_return_pct(full_gross):+.1f}% -> {total_return_pct(full_net):+.1f}%")
    print(f"  Weighted:  {total_return_pct(weighted_gross):+.1f}% -> {total_return_pct(weighted_net):+.1f}%")

    # ---- Does confidence predict outcomes? (test buys only) ----
    print(f"\nWin rate by confidence level (test-period buys only):")
    buy_rows = test[is_buy]
    for label in ["High", "Medium", "Low"]:
        bucket = buy_rows[buy_rows["half_width_pct"].apply(
            lambda hw: confidence_label(hw, high_thr, low_thr) == label)]
        if len(bucket) == 0:
            print(f"  {label:<7}  0 trades")
            continue
        print(f"  {label:<7} {len(bucket):>3} trades   win rate: {(bucket['actual_pct'] > 0).mean() * 100:5.1f}%   "
              f"avg actual move: {bucket['actual_pct'].mean():+6.2f}%")

    # ---- Random-entry baseline: is the model's choice of WHEN to buy better than chance? ----
    n_buys = int(is_buy.sum())
    beat_full = None
    print(f"\nRandom-entry baseline ({args.random_trials} trials, same number of buys at random decision points):")
    if n_buys == 0 or n_buys >= len(test):
        print("  Skipped: needs at least one buy and at least one non-buy decision point.")
    else:
        rng = np.random.default_rng(args.seed)
        actual_values = actual.values
        model_buy_sizes = weighted_size[is_buy].values
        full_random = np.empty(args.random_trials)
        weighted_random = np.empty(args.random_trials)
        for t in range(args.random_trials):
            idx = rng.choice(len(test), size=n_buys, replace=False)
            net_each = actual_values[idx] - 2 * cost_per_side
            full_random[t] = total_return_pct(pd.Series(net_each))
            weighted_random[t] = total_return_pct(pd.Series(rng.permutation(model_buy_sizes) * net_each))

        def beat_share(model_value, random_values):
            return ((random_values < model_value).mean() + 0.5 * (random_values == model_value).mean()) * 100

        for name, model_series, random_values in [
            ("Full-size", full_net, full_random),
            ("Weighted ", weighted_net, weighted_random),
        ]:
            model_value = total_return_pct(model_series)
            share = beat_share(model_value, random_values)
            lo, med, hi = np.percentile(random_values, [5, 50, 95])
            print(f"  {name}: model {model_value:+.1f}% | random median {med:+.1f}% "
                  f"(5th-95th pct: {lo:+.1f}% to {hi:+.1f}%) | model beat {share:.0f}% of random sets")
            if name.strip() == "Full-size":
                beat_full = share

    # ---- Plain-language reading ----
    w_total = total_return_pct(weighted_net)
    always_total = total_return_pct(always_returns)
    exposure = n_buys / len(test) * 100
    print("\nReading this:")
    if w_total < 0:
        print(f"  - Net return was {w_total:+.1f}%: holding cash (0%) would have beaten this strategy.")
    else:
        print(f"  - Net return was {w_total:+.1f}%, ahead of holding cash (0%).")
    print(f"  - Buying at every decision point returned {always_total:+.1f}%. That row pays far more in costs and is "
          f"invested more often, so use the random-entry baseline above for the question of whether the model's timing helps.")
    if max_drawdown_pct(weighted_net) > max_drawdown_pct(bh_returns):
        if exposure < 50:
            print(f"  - Max drawdown was shallower than buy-and-hold, but the strategy was invested at only "
                  f"{exposure:.0f}% of decision points, so that mostly reflects being in cash, not skill.")
        else:
            print("  - Max drawdown was shallower than buy-and-hold.")
    else:
        print("  - Max drawdown was not better than buy-and-hold.")
    if beat_full is not None:
        if beat_full >= 95:
            print(f"  - Timing: the model beat {beat_full:.0f}% of random entry sets, which looks better than chance.")
        elif beat_full >= 75:
            print(f"  - Timing: the model beat {beat_full:.0f}% of random entry sets, somewhat better than chance but not conclusive.")
        elif beat_full >= 25:
            print(f"  - Timing: the model beat {beat_full:.0f}% of random entry sets, so its entries look no better than random.")
        else:
            print(f"  - Timing: the model beat only {beat_full:.0f}% of random entry sets, so its entries look worse than random.")
    if len(test) < 30 or n_buys < 20:
        print(f"  - Small sample ({len(test)} decision points, {n_buys} buys): treat all of this as noisy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the XGBoost decision logic with costs, risk metrics and an out-of-sample split")
    parser.add_argument("--db", default="crypdate_binance.db", help="Path to Binance OHLCV database")
    parser.add_argument("--symbol", required=True, help="Coin name, e.g. bitcoin, ethereum, dogecoin")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=14, help="Days between decision points (smaller = more samples, slower)")
    parser.add_argument("--buy-threshold", type=float, default=3.0)
    parser.add_argument("--sell-threshold", type=float, default=-3.0)
    parser.add_argument("--min-training-days", type=int, default=90)
    parser.add_argument("--fee-pct", type=float, default=0.1, help="Exchange fee per side, in percent")
    parser.add_argument("--slippage-pct", type=float, default=0.05, help="Assumed slippage per side, in percent")
    parser.add_argument("--random-trials", type=int, default=1000, help="Trials for the random-entry baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the random-entry baseline")
    parser.add_argument("--calibration-fraction", type=float, default=0.5,
                        help="Share of decision points (earliest first) used to calibrate thresholds; the rest are the test set")
    parser.add_argument("--test-start-date", default=None, help="YYYY-MM-DD; overrides --calibration-fraction")
    parser.add_argument("--test-end-date", default=None, help="YYYY-MM-DD; stop the backtest after this date")
    parser.add_argument("--calib-high-quantile", type=float, default=0.25)
    parser.add_argument("--calib-low-quantile", type=float, default=0.75)
    parser.add_argument("--high-threshold-pct", type=float, default=None, help="Manual override (use with --low-threshold-pct)")
    parser.add_argument("--low-threshold-pct", type=float, default=None, help="Manual override (use with --high-threshold-pct)")

    run_backtest(parser.parse_args())
