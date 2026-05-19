"""Company research and market data helpers — all cached."""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


@st.cache_data(ttl=300)
def get_company_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        return yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_financials(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    result = {}
    for attr, key in [("income_stmt", "income"), ("balance_sheet", "balance"),
                      ("cashflow", "cashflow")]:
        try:
            result[key] = getattr(t, attr)
        except Exception:
            result[key] = pd.DataFrame()
    return result


@st.cache_data(ttl=180)
def get_news(ticker: str) -> list:
    try:
        news = yf.Ticker(ticker).news or []
        return news[:10]
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_analyst_ratings(ticker: str) -> pd.DataFrame:
    try:
        recs = yf.Ticker(ticker).recommendations
        if recs is not None and not recs.empty:
            return recs.tail(20)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=600)
def get_sector_performance() -> pd.DataFrame:
    """Fetch performance for all sector ETFs."""
    sector_etfs = {
        "Technology": "XLK", "Financials": "XLF", "Healthcare": "XLV",
        "Energy": "XLE", "Industrials": "XLI", "Consumer Disc.": "XLY",
        "Consumer Staples": "XLP", "Utilities": "XLU", "Materials": "XLB",
        "Real Estate": "XLRE", "Communications": "XLC",
    }
    rows = []
    tickers = list(sector_etfs.values())
    try:
        data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        closes = closes.ffill().dropna(axis=1, how="all")
    except Exception:
        return pd.DataFrame()

    for sector, etf in sector_etfs.items():
        if etf not in closes.columns:
            continue
        col = closes[etf].dropna()
        if len(col) < 2:
            continue
        p = lambda n: (col.iloc[-1] / col.iloc[-min(n, len(col))] - 1) * 100
        rows.append({
            "Sector": sector, "ETF": etf,
            "1D %":  round(p(2),  2),
            "1W %":  round(p(5),  2),
            "1M %":  round(p(21), 2),
            "3M %":  round(p(63), 2),
            "6M %":  round(p(126),2),
            "1Y %":  round(p(252),2),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def get_macro_snapshot() -> dict:
    """Proxy macro indicators via ETFs/indices."""
    symbols = {
        "S&P 500":    "^GSPC",
        "NASDAQ":     "^IXIC",
        "Russell 2k": "^RUT",
        "VIX":        "^VIX",
        "10Y Yield":  "^TNX",
        "Gold":       "GLD",
        "Oil (WTI)":  "USO",
        "USD Index":  "UUP",
    }
    result = {}
    for name, sym in symbols.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if len(hist) >= 2:
                price = float(hist["Close"].iloc[-1])
                prev  = float(hist["Close"].iloc[-2])
                chg   = (price - prev) / prev * 100
                result[name] = {"price": price, "chg": chg, "symbol": sym}
        except Exception:
            pass
    return result


@st.cache_data(ttl=600)
def get_correlation_matrix(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        return closes.ffill().dropna(axis=1, how="all").pct_change().dropna().corr().round(2)
    except Exception:
        return pd.DataFrame()


def fmt_large(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    n = float(n)
    if abs(n) >= 1e12: return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:  return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:  return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def fmt_pct(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    return f"{float(n)*100:.1f}%"


def fmt_num(n, decimals=2) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    return f"{float(n):.{decimals}f}"
