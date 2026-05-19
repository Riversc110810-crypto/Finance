import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data
    # ensure all columns are present even if some tickers had no data
    return close.ffill().dropna(axis=1, how="all")


def momentum_score(prices: pd.DataFrame) -> pd.Series:
    """12-1 month momentum (skip last month to avoid reversal)."""
    ret_12m = prices.pct_change(252)
    ret_1m = prices.pct_change(21)
    return (ret_12m - ret_1m).iloc[-1].sort_values(ascending=False)


def quality_filter(tickers: list[str], prices: pd.DataFrame) -> dict:
    """
    Proxy quality score using price-derived volatility and trend consistency.
    Lower 90-day volatility = more stable = higher quality proxy.
    """
    scores = {}
    for t in tickers:
        if t not in prices.columns:
            scores[t] = 0
            continue
        try:
            col = prices[t].dropna()
            if len(col) < 21:
                scores[t] = 0
                continue
            vol_90 = col.pct_change().rolling(90).std().iloc[-1]
            # count how many of the last 6 months were positive
            monthly = col.resample("ME").last().pct_change().dropna()
            positive_months = (monthly.tail(6) > 0).sum()
            score = 0
            if vol_90 < 0.025:   # low daily vol = stable
                score += 1
            if positive_months >= 4:  # trending up consistently
                score += 1
            scores[t] = score
        except Exception:
            scores[t] = 0
    return scores


def rank_stocks(tickers: list[str]) -> pd.DataFrame:
    prices = fetch_prices(tickers, period="2y")
    prices.index = pd.DatetimeIndex(prices.index)

    mom = momentum_score(prices)
    quality = quality_filter(tickers, prices)

    df = pd.DataFrame({
        "Momentum Score": mom,
        "Quality Score": pd.Series(quality, dtype=float),
    })
    df.index.name = "Ticker"
    df = df.dropna(subset=["Momentum Score"])
    df["Quality Score"] = df["Quality Score"].fillna(0)

    if len(df) > 1:
        mom_z = stats.zscore(df["Momentum Score"])
        qual_z = stats.zscore(df["Quality Score"]) if df["Quality Score"].std() > 0 else np.zeros(len(df))
    else:
        mom_z = np.zeros(len(df))
        qual_z = np.zeros(len(df))

    df["Combined Score"] = 0.7 * mom_z + 0.3 * qual_z
    df["Signal"] = df["Combined Score"].apply(
        lambda x: "BUY" if x > 0.5 else ("HOLD" if x > -0.5 else "AVOID")
    )
    df["1M Return %"]  = (prices.pct_change(21).iloc[-1] * 100).round(2)
    df["3M Return %"]  = (prices.pct_change(63).iloc[-1] * 100).round(2)
    df["12M Return %"] = (prices.pct_change(252).iloc[-1] * 100).round(2)
    return df.sort_values("Combined Score", ascending=False).round(3)


def backtest_momentum(tickers: list[str], top_n: int = 3, rebalance_days: int = 21) -> pd.Series:
    """Simple backtest: hold top N momentum stocks, rebalance monthly."""
    prices = fetch_prices(tickers, period="5y")
    prices = prices.dropna(axis=1, how="all").ffill()

    portfolio_returns = []
    dates = prices.index[252:]  # need 1yr of history first

    for i in range(0, len(dates) - rebalance_days, rebalance_days):
        window = prices.iloc[: prices.index.get_loc(dates[i])]
        ret_12m = window.pct_change(252).iloc[-1]
        ret_1m = window.pct_change(21).iloc[-1]
        mom = (ret_12m - ret_1m).dropna().sort_values(ascending=False)
        top = mom.head(top_n).index.tolist()

        if not top:
            continue

        end_idx = min(i + rebalance_days, len(dates) - 1)
        period_prices = prices[top].loc[dates[i]: dates[end_idx]]
        period_ret = period_prices.pct_change().mean(axis=1)
        portfolio_returns.append(period_ret)

    if not portfolio_returns:
        return pd.Series(dtype=float)

    combined = pd.concat(portfolio_returns).dropna()
    equity = (1 + combined).cumprod() * 100
    equity = equity.dropna()
    return equity


def monte_carlo(equity_curve: pd.Series, n_simulations: int = 1000,
                days_forward: int = 252, capital: float = 250.0) -> dict:
    """Monte Carlo projection from backtest equity curve."""
    daily_returns = equity_curve.pct_change().dropna()
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    simulations = np.zeros((days_forward, n_simulations))
    for i in range(n_simulations):
        shocks = np.random.normal(mu, sigma, days_forward)
        simulations[:, i] = capital * np.cumprod(1 + shocks)

    final_values = simulations[-1, :]
    return {
        "simulations": simulations,
        "p10": np.percentile(final_values, 10),
        "p50": np.percentile(final_values, 50),
        "p90": np.percentile(final_values, 90),
        "prob_profit": (final_values > capital).mean() * 100,
        "mu": mu,
        "sigma": sigma,
        "capital": capital,
    }


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> dict:
    risk_dollars = capital * (risk_pct / 100)
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return {"shares": 0, "risk_dollars": 0, "position_value": 0}
    shares = risk_dollars / stop_distance
    fractional_shares = round(shares, 4)
    return {
        "shares": fractional_shares,
        "risk_dollars": round(risk_dollars, 2),
        "position_value": round(fractional_shares * entry, 2),
    }
