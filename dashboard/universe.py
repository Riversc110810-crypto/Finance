"""
Hedge-fund style stock and ETF universe organised by category.
Each category can be selected independently for screening,
backtesting, and Monte Carlo simulation.
"""

UNIVERSE: dict[str, list[str]] = {
    "Mega Cap Tech": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
        "ORCL", "AVGO", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN",
    ],
    "Financials": [
        "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP",
        "V", "MA", "PYPL", "COF", "USB", "PNC", "TFC",
    ],
    "Healthcare": [
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO",
        "ABT", "BMY", "GILD", "CVS", "ISRG", "SYK", "MDT",
    ],
    "Consumer Discretionary": [
        "HD", "NKE", "MCD", "SBUX", "COST", "TGT", "LOW",
        "DIS", "NFLX", "BKNG", "CMG", "YUM", "ABNB", "UBER",
    ],
    "Consumer Staples": [
        "WMT", "PG", "KO", "PEP", "PM", "MO", "MDLZ",
        "CL", "GIS", "KMB", "SYY", "KR",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC",
        "PSX", "VLO", "OXY", "HES", "DVN", "FANG",
    ],
    "Industrials": [
        "CAT", "DE", "BA", "GE", "HON", "MMM",
        "UPS", "FDX", "RTX", "LMT", "NOC", "GD", "ETN",
    ],
    "Materials & Utilities": [
        "LIN", "APD", "ECL", "NEM", "FCX", "NUE",
        "NEE", "DUK", "SO", "D", "AEP", "EXC",
    ],
    "Growth & Momentum": [
        "PLTR", "COIN", "SHOP", "RBLX", "DKNG",
        "IONQ", "SMCI", "ARM", "MSTR", "HOOD",
    ],
    "Broad Market ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "SCHB", "RSP",
    ],
    "Sector ETFs": [
        "XLK", "XLF", "XLE", "XLV", "XLU",
        "XLI", "XLB", "XLRE", "XLC", "XLY", "XLP",
    ],
    "Factor ETFs": [
        "MTUM", "QUAL", "USMV", "VLUE", "VTV",
        "VUG", "IWF", "IWD", "DGRO", "NOBL",
    ],
    "Bond ETFs": [
        "TLT", "IEF", "SHY", "AGG", "BND",
        "HYG", "LQD", "TIPS", "MUB", "EMB",
    ],
    "Commodity ETFs": [
        "GLD", "SLV", "USO", "GDX", "IAU", "PDBC", "CPER",
    ],
    "International ETFs": [
        "EEM", "EFA", "VEA", "VWO", "EWJ",
        "FXI", "VGK", "EWZ", "INDA", "MCHI",
    ],
}

ALL_TICKERS: list[str] = sorted({t for tickers in UNIVERSE.values() for t in tickers})

DEFAULT_CATEGORIES: list[str] = [
    "Mega Cap Tech", "Broad Market ETFs", "Sector ETFs", "Factor ETFs",
]


def tickers_for(categories: list[str]) -> list[str]:
    result = []
    for cat in categories:
        result.extend(UNIVERSE.get(cat, []))
    return sorted(set(result))
