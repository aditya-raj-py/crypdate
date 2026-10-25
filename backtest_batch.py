"""
backtest_batch.py
==================
v5.1 - Walk-forward backtests over a whole list of tickers, judged TOGETHER.

Why: one ticker that looks good proves little. Test enough tickers and some
will look good by luck. This runs the same model over every ticker in a market
(or the list you give), saves each ticker's decision points, and then asks the
question that matters: pooled across ALL tickers, do the model's buy signals
beat randomly chosen windows?

The test is deliberately simple and does not depend on any tuned threshold:

  * Every decision point is a window: "what did the price do over the next
    `horizon_days`?" The model's walk-forward predictions are out-of-sample
    throughout (it only ever sees data from before each decision).
  * For each ticker, compare the average move after the model's BUYS with the
    average move over ALL of that ticker's windows. The gap is the "edge".
  * p-value: shuffle which windows count as buys (same number of buys per
    ticker, random timing) many times and see how often chance alone does as
    well. Per ticker, and pooled across all tickers.
  * Also reported: how many tickers show a positive edge, and how many reach
    p < 0.05 compared with the 5% you would expect from luck alone.

Results are cached in results/batch_<market>/, so an interrupted run resumes
where it stopped and a rerun is instant. The cache key includes the settings and
the last date in the data, so refetching data (or changing a setting) recomputes
automatically. Use --refresh to force it.

Caveats the numbers cannot fix: the ticker lists are today's survivors
(survivorship bias flatters results), tickers move together (the pooled p-value
is somewhat too optimistic), and costs are placeholder assumptions for stocks.

Usage:
    python backtest_batch.py --market india
    python backtest_batch.py --market us --symbols AAPL MSFT JPM
    python backtest_batch.py --market crypto
    python backtest_batch.py --market india --step-days 7     (more samples, slower)
"""

import argparse
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from crypdate_markets import MARKETS, get_market, resolve_symbol
from crypdate_xgboost import load_ohlcv
from backtest_xgboost import collect_decision_points

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # safe when output is redirected to a file on Windows
except Exception:
    pass


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def permutation_pvalue(moves, n_buys, observed_mean, trials, rng):
    """P(chance alone, picking n_buys random windows, averages at least observed_mean)."""
    n = len(moves)
    hits = 0
    for _ in range(trials):
        if moves[rng.choice(n, n_buys, replace=False)].mean() >= observed_mean:
            hits += 1
    return (hits + 1) / (trials + 1)


def pooled_pvalue(groups, trials, rng):
    """
    groups: list of (moves, buy_mask) per ticker. The statistic is the average
    EXCESS move of the buys, where excess = move minus that ticker's own average
    window, so tickers with different drifts are comparable. Under the null, each
    ticker keeps its number of buys but their timing is random.
    """
    observed = np.concatenate([m[mask] - m.mean() for m, mask in groups]).mean()
    hits = 0
    for _ in range(trials):
        parts = []
        for m, mask in groups:
            idx = rng.choice(len(m), int(mask.sum()), replace=False)
            parts.append(m[idx] - m.mean())
        if np.concatenate(parts).mean() >= observed:
            hits += 1
    return observed, (hits + 1) / (trials + 1)


def sign_test_pvalue(n_positive, n_total):
    """One-sided exact binomial: P(at least n_positive of n_total positive if each is a coin flip)."""
    return sum(math.comb(n_total, j) for j in range(n_positive, n_total + 1)) / 2 ** n_total


def analyze_ticker(symbol, d, cost_per_side, trials, rng):
    moves = d["actual_pct"].values
    mask = (d["signal"] == "buy").values
    k = int(mask.sum())

    row = {"symbol": symbol, "windows": len(d), "buys": k, "win_pct": np.nan,
           "avg_buy": np.nan, "avg_window": float(moves.mean()), "edge": np.nan, "p": np.nan,
           "strat_later_half": np.nan, "bh_later_half": np.nan}

    # Later half of the ticker's history: full-size strategy vs buy-and-hold, with costs
    later = d.iloc[len(d) // 2:]
    a = later["actual_pct"] / 100
    r = np.where(later["signal"] == "buy", a - 2 * cost_per_side, 0.0)
    row["strat_later_half"] = (np.prod(1 + r) - 1) * 100
    row["bh_later_half"] = (np.prod(1 + later["bh_step_pct"] / 100) - 1) * 100

    if k == 0:
        return row, (moves, mask)

    buy_moves = moves[mask]
    row["win_pct"] = float((buy_moves > 0).mean() * 100)
    row["avg_buy"] = float(buy_moves.mean())
    row["edge"] = row["avg_buy"] - row["avg_window"]
    row["p"] = permutation_pvalue(moves, k, buy_moves.mean(), trials, rng)
    return row, (moves, mask)


# --------------------------------------------------------------------------
# Running one ticker (with caching)
# --------------------------------------------------------------------------

def cache_path(market, symbol, last_date, args):
    name = (f"{symbol}_h{args.horizon_days}_s{args.step_days}_t{args.buy_threshold:g}_"
            f"{args.sell_threshold:g}_m{args.min_training_days}_d{last_date:%Y%m%d}.csv")
    return os.path.join("results", f"batch_{market.name}", name)


def get_decision_points(symbol, args, market):
    df = load_ohlcv(args.db, symbol)
    path = cache_path(market, symbol, df["timestamp"].iloc[-1], args)

    if os.path.exists(path) and not args.refresh:
        return pd.read_csv(path, parse_dates=["date"]), "cached"

    d = collect_decision_points(
        df, args.horizon_days, args.step_days, args.buy_threshold, args.sell_threshold,
        args.min_training_days, verbose=False,
    )
    if d.empty:
        raise ValueError("not enough history for any decision point")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d.to_csv(path, index=False)
    return d, "computed"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_batch(args):
    market = get_market(args.market)
    symbols = [resolve_symbol(market, s) for s in (args.symbols or market.default_symbols)]
    cost = (args.fee_pct + args.slippage_pct) / 100
    rng = np.random.default_rng(args.seed)

    print(f"Market: {market.description}")
    print(f"Tickers: {len(symbols)}   horizon {args.horizon_days}, step {args.step_days}, "
          f"buy above {args.buy_threshold:+g}%, sell below {args.sell_threshold:+g}%")
    print(f"Costs: {args.fee_pct}% fee + {args.slippage_pct}% slippage per side\n")
    if args.step_days < args.horizon_days:
        print("WARNING: step-days is smaller than horizon-days, so windows overlap and the p-values "
              "below are too optimistic. Use step-days >= horizon-days.\n")

    rows, groups, skipped = [], [], []
    started = time.time()
    compute_times = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol} ", end="", flush=True)
        t0 = time.time()
        try:
            d, source = get_decision_points(symbol, args, market)
        except Exception as e:
            print(f" SKIPPED: {e}")
            skipped.append(symbol)
            continue
        took = time.time() - t0
        if source == "computed":
            compute_times.append(took)
        left = len(symbols) - i
        eta = f", about {np.mean(compute_times) * left / 60:.0f} min left" if compute_times and left else ""
        print(f" {source} ({took:.0f}s{eta})")

        row, group = analyze_ticker(symbol, d, cost, args.trials, rng)
        rows.append(row)
        if row["buys"] > 0:
            groups.append(group)

    if not rows:
        print("\nNothing to analyze. Fetch data first (stock_fetch.py or binance_fetch.py).")
        return

    table = pd.DataFrame(rows)

    # ---- Per-ticker table ----
    print(f"\n{'Ticker':<14}{'Pts':>6}{'Buys':>6}{'Win%':>7}{'AvgBuy%':>9}{'AvgWin%':>9}{'Edge%':>8}{'p':>8}"
          f"{'Strat%*':>9}{'B&H%*':>9}")
    for r in rows:
        if r["buys"] == 0:
            print(f"{r['symbol']:<14}{r['windows']:>6}{0:>6}   (no buy signals)"
                  f"{'':>25}{r['strat_later_half']:>+9.1f}{r['bh_later_half']:>+9.1f}")
            continue
        print(f"{r['symbol']:<14}{r['windows']:>6}{r['buys']:>6}{r['win_pct']:>7.1f}{r['avg_buy']:>+9.2f}"
              f"{r['avg_window']:>+9.2f}{r['edge']:>+8.2f}{r['p']:>8.3f}"
              f"{r['strat_later_half']:>+9.1f}{r['bh_later_half']:>+9.1f}")
    print("  Pts = decision points; AvgBuy/AvgWin = average move after buys / over all points; Edge = their difference (percentage points)")
    print("  * later half of each ticker's history, full-size, after costs (not the p-value test)")

    # ---- Pooled result ----
    with_buys = table[table["buys"] > 0]
    if with_buys.empty:
        print("\nNo ticker produced a buy signal, so there is nothing to pool.")
        return

    n_t = len(with_buys)
    n_pos = int((with_buys["edge"] > 0).sum())
    n_sig = int((with_buys["p"] < 0.05).sum())
    total_buys = int(with_buys["buys"].sum())
    pooled_edge, pooled_p = pooled_pvalue(groups, args.trials, rng)
    avg_buy_all = float(np.concatenate([m[mask] for m, mask in groups]).mean())

    print(f"\n=== Pooled across {n_t} tickers ({total_buys} buys) ===")
    print(f"Average excess move per buy (vs each ticker's own average): {pooled_edge:+.2f} percentage points")
    print(f"Pooled p-value (random buy timing, same buys per ticker):   {pooled_p:.4f}")
    print(f"Tickers with a positive edge: {n_pos} of {n_t}   (coin-flip sign test p = {sign_test_pvalue(n_pos, n_t):.3f})")
    print(f"Tickers with p < 0.05: {n_sig} of {n_t}   (about {0.05 * n_t:.1f} expected from luck alone)")
    print(f"Average buy: {avg_buy_all:+.2f}% before costs, {avg_buy_all - 2 * cost * 100:+.2f}% after "
          f"{2 * cost * 100:.2f}% round-trip costs")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")

    print("\nReading this:")
    if pooled_p < 0.01 and n_pos / n_t >= 0.6:
        print("  - Pooled evidence that the buy signals pick better-than-random windows, and it is spread across tickers.")
    elif pooled_p < 0.05:
        print("  - Some pooled evidence of a timing edge, but not overwhelming. Treat it as a lead, not a result.")
    else:
        print("  - No convincing pooled evidence that the buy signals beat random timing.")
    if n_sig <= 0.05 * n_t + 1 and pooled_p < 0.05:
        print("  - Few individual tickers are significant on their own; the evidence comes from pooling many small edges.")
    if avg_buy_all - 2 * cost * 100 <= 0:
        print("  - The average buy does not cover costs, so even a real timing edge would not be tradable as set up.")
    print("  - Survivorship: these are today's surviving companies. Tickers moving together also makes the pooled p-value optimistic.")
    print("  - A backtest, even a clean one, is not forward evidence. Paper trading on unseen data is the real test.")

    out = os.path.join("results", f"batch_{market.name}_summary.csv")
    os.makedirs("results", exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nSummary saved to {out}. Total time {time.time() - started:.0f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest every ticker in a market and judge the results together")
    parser.add_argument("--market", choices=list(MARKETS), default="india", help="Which market to test")
    parser.add_argument("--symbols", nargs="+", default=None, help="Tickers to test (default: the market's built-in list)")
    parser.add_argument("--db", default=None, help="OHLCV database path (default: the market's database)")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=14, help="Days between decision points (keep >= horizon)")
    parser.add_argument("--min-training-days", type=int, default=90)
    parser.add_argument("--buy-threshold", type=float, default=None, help="Default: the market profile")
    parser.add_argument("--sell-threshold", type=float, default=None, help="Default: the market profile")
    parser.add_argument("--fee-pct", type=float, default=None, help="Fee per side in percent (default: the market profile)")
    parser.add_argument("--slippage-pct", type=float, default=None, help="Slippage per side in percent (default: the market profile)")
    parser.add_argument("--trials", type=int, default=5000, help="Random trials for the p-values")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true", help="Ignore cached results and recompute")
    args = parser.parse_args()

    mk = get_market(args.market)
    args.db = args.db or mk.db_path
    args.buy_threshold = mk.buy_threshold if args.buy_threshold is None else args.buy_threshold
    args.sell_threshold = mk.sell_threshold if args.sell_threshold is None else args.sell_threshold
    args.fee_pct = mk.fee_pct if args.fee_pct is None else args.fee_pct
    args.slippage_pct = mk.slippage_pct if args.slippage_pct is None else args.slippage_pct

    run_batch(args)
