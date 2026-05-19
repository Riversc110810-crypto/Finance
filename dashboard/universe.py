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

# Equity-only sectors used for Long/Short pair generation.
# Each sector must have >= 4 stocks to generate meaningful pairs.
LS_SECTORS: dict[str, list[str]] = {
    "Technology":            ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ADBE", "CRM", "AMD", "INTC", "QCOM", "ORCL", "TXN"],
    "Consumer Internet":     ["AMZN", "NFLX", "UBER", "ABNB", "BKNG", "SHOP", "COIN", "HOOD", "RBLX", "PLTR"],
    "Financials":            ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP", "V", "MA", "PYPL", "COF"],
    "Healthcare":            ["UNH", "LLY", "ABBV", "JNJ", "MRK", "TMO", "PFE", "ISRG", "SYK", "ABT", "BMY", "GILD"],
    "Consumer Discretionary":["HD", "NKE", "MCD", "COST", "TGT", "LOW", "SBUX", "CMG", "DIS", "YUM", "BKNG"],
    "Consumer Staples":      ["WMT", "PG", "KO", "PEP", "PM", "MDLZ", "CL", "GIS", "KR", "MO"],
    "Energy":                ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "PSX", "VLO", "DVN"],
    "Industrials":           ["CAT", "DE", "BA", "GE", "HON", "RTX", "LMT", "UPS", "FDX", "ETN", "NOC"],
    "Semiconductors":        ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC"],
    "Growth & Speculative":  ["TSLA", "MSTR", "SMCI", "IONQ", "ARM", "DKNG", "HOOD", "COIN", "PLTR"],
}

# Inverse ETFs retail investors can use as a short proxy (no margin account needed)
INVERSE_ETFS: dict[str, str] = {
    "Technology":            "PSQ",    # Inverse QQQ
    "Financials":            "SEF",    # Short Financials
    "Healthcare":            "RXD",    # Short Health Care
    "Energy":                "DDG",    # Short Oil & Gas
    "Broad Market":          "SH",     # Short S&P 500
    "Small Cap":             "RWM",    # Short Russell 2000
    "20yr Treasury":         "TBF",    # Short 20+ Year Treasury
}


def tickers_for(categories: list[str]) -> list[str]:
    result = []
    for cat in categories:
        result.extend(UNIVERSE.get(cat, []))
    return sorted(set(result))


def ls_tickers_for(sectors: list[str]) -> list[str]:
    result = []
    for s in sectors:
        result.extend(LS_SECTORS.get(s, []))
    return sorted(set(result))
