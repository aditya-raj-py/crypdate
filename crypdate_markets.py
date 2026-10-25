"""
crypdate_markets.py
====================
v5.1 - Market profiles. One place that says what differs between crypto,
Indian stocks and US stocks, so every CRYPDATE script can switch markets
with a single flag:

    python crypdate_xgboost.py --market india --symbol RELIANCE
    python backtest_xgboost.py --market us --symbol AAPL
    python crypdate_candles.py --market crypto --symbol bitcoin

What a market profile controls:
  * which database holds its data (each market has its own file, same table layout)
  * how many decision days there are per year (used to annualize the Sharpe ratio)
  * default trading costs per side
  * default buy/sell signal thresholds (stocks move less than crypto, so they are tighter)
  * how bare symbols are completed (RELIANCE -> RELIANCE.NS on the NSE)
  * whether horizons count trading days (stocks) or calendar days (crypto)

The cost and threshold numbers for stocks are PLACEHOLDER assumptions. Real costs depend on
your broker, platform and country, so check them and override with
--fee-pct / --slippage-pct / --buy-threshold / --sell-threshold.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    name: str
    description: str
    db_path: str
    periods_per_year: int
    fee_pct: float
    slippage_pct: float
    buy_threshold: float
    sell_threshold: float
    symbol_suffix: str
    uses_business_days: bool
    currency_prefix: str
    timezone: str
    close_time: str  # local exchange close, 'HH:MM'; empty for crypto (trades 24/7)
    default_symbols: tuple


MARKETS = {
    "crypto": Market(
        name="crypto",
        description="Crypto (Binance daily candles, trades every day)",
        db_path="crypdate_binance.db",
        periods_per_year=365,
        fee_pct=0.1,
        slippage_pct=0.05,
        buy_threshold=3.0,
        sell_threshold=-3.0,
        symbol_suffix="",
        uses_business_days=False,
        currency_prefix="$",
        timezone="UTC",
        close_time="",
        default_symbols=("bitcoin", "ethereum", "dogecoin"),
    ),
    "india": Market(
        name="india",
        description="Indian stocks (NSE by default, BSE with the .BO suffix)",
        db_path="crypdate_india.db",
        periods_per_year=250,  # approximate number of NSE trading days per year
        fee_pct=0.15,          # placeholder, check your broker's charges
        slippage_pct=0.05,     # placeholder
        buy_threshold=1.5,     # placeholder: stocks move less than crypto over a week
        sell_threshold=-1.5,
        symbol_suffix=".NS",
        uses_business_days=True,
        currency_prefix="INR ",
        timezone="Asia/Kolkata",
        close_time="15:30",
        default_symbols=(
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "ITC.NS", "SBIN.NS", "HINDUNILVR.NS", "LT.NS", "BHARTIARTL.NS",
        ),
    ),
    "us": Market(
        name="us",
        description="US stocks (NYSE/Nasdaq)",
        db_path="crypdate_us.db",
        periods_per_year=252,  # approximate number of US trading days per year
        fee_pct=0.05,          # placeholder, check your broker/platform charges
        slippage_pct=0.03,     # placeholder
        buy_threshold=1.5,     # placeholder: stocks move less than crypto over a week
        sell_threshold=-1.5,
        symbol_suffix="",
        uses_business_days=True,
        currency_prefix="$",
        timezone="America/New_York",
        close_time="16:00",
        default_symbols=(
            "AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "KO", "INTC", "T", "PFE",
        ),
    ),
}


def get_market(name: str) -> Market:
    try:
        return MARKETS[name]
    except KeyError:
        raise ValueError(f"Unknown market '{name}'. Choose from: {', '.join(MARKETS)}")


def resolve_symbol(market: Market, symbol: str) -> str:
    """
    Turn what the user typed into the symbol as stored in the database.
    Crypto names are lower-case (bitcoin). Stock tickers are upper-case, and a
    bare Indian ticker gets the market's default suffix (RELIANCE -> RELIANCE.NS).
    A ticker that already has a suffix (TCS.BO) is left alone.
    """
    if market.name == "crypto":
        return symbol.lower()
    symbol = symbol.upper()
    if market.symbol_suffix and "." not in symbol:
        symbol += market.symbol_suffix
    return symbol
