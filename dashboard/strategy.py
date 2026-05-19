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


def ls_score(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Score every ticker for Long/Short suitability.
    Positive score = long candidate.  Negative score = short candidate.

    Components (all z-scored within the passed universe):
      60% — 12-1M momentum (cross-sectional)
      20% — 3-month trend consistency (% positive weeks)
      20% — relative strength vs. equal-weight universe
    """
    prices = prices.copy()
    prices.index = pd.DatetimeIndex(prices.index)

    ret_12m = prices.pct_change(252).iloc[-1]
    ret_1m  = prices.pct_change(21).iloc[-1]
    mom     = ret_12m - ret_1m

    # 3M trend consistency: fraction of weekly returns > 0 over last 13 weeks
    weekly = prices.resample("W").last().pct_change()
    trend_consistency = (weekly.tail(13) > 0).mean()

    # Relative strength: ticker 3M return minus universe median 3M return
    ret_3m  = prices.pct_change(63).iloc[-1]
    median_3m = ret_3m.median()
    rel_strength = ret_3m - median_3m

    df = pd.DataFrame({
        "Momentum":         mom,
        "TrendConsistency": trend_consistency,
        "RelStrength":      rel_strength,
        "1M%":              prices.pct_change(21).iloc[-1]  * 100,
        "3M%":              ret_3m * 100,
        "12M%":             ret_12m * 100,
    }).dropna(subset=["Momentum"])

    def zscore_safe(s: pd.Series) -> pd.Series:
        return pd.Series(stats.zscore(s), index=s.index) if s.std() > 0 else pd.Series(0.0, index=s.index)

    df["Score"] = (
        0.60 * zscore_safe(df["Momentum"]) +
        0.20 * zscore_safe(df["TrendConsistency"]) +
        0.20 * zscore_safe(df["RelStrength"])
    )

    df["Signal"] = df["Score"].apply(
        lambda x: "LONG" if x > 0.5 else ("SHORT" if x < -0.5 else "NEUTRAL")
    )
    df.index.name = "Ticker"
    return df.sort_values("Score", ascending=False).round(4)


def generate_pairs(sector_map: dict[str, list[str]], prices: pd.DataFrame) -> list[dict]:
    """
    For each sector: score all tickers, pair the top scorer (LONG)
    against the bottom scorer (SHORT).  Returns a list of pair dicts.
    """
    pairs = []
    scores_all = ls_score(prices)

    for sector, tickers in sector_map.items():
        available = [t for t in tickers if t in scores_all.index]
        if len(available) < 2:
            continue

        sector_scores = scores_all.loc[available].sort_values("Score", ascending=False)
        long_t  = sector_scores.index[0]
        short_t = sector_scores.index[-1]

        if long_t == short_t:
            continue

        long_row  = sector_scores.loc[[long_t]].iloc[0]
        short_row = sector_scores.loc[[short_t]].iloc[0]

        # Historical spread: daily long return minus daily short return
        if long_t in prices.columns and short_t in prices.columns:
            spread_ret = prices[long_t].pct_change() - prices[short_t].pct_change()
            spread_cum = (1 + spread_ret.dropna()).cumprod() * 100
            spread_3m  = (prices[long_t].iloc[-1] / prices[long_t].iloc[-63] - 1) * 100 - \
                         (prices[short_t].iloc[-1] / prices[short_t].iloc[-63] - 1) * 100
            spread_1m  = float(long_row["1M%"]) - float(short_row["1M%"])
        else:
            spread_cum = pd.Series(dtype=float)
            spread_3m  = 0.0
            spread_1m  = 0.0

        pairs.append({
            "sector":      sector,
            "long":        long_t,
            "short":       short_t,
            "long_score":  float(long_row["Score"]),
            "short_score": float(short_row["Score"]),
            "long_1m":     float(long_row["1M%"]),
            "short_1m":    float(short_row["1M%"]),
            "long_3m":     float(long_row["3M%"]),
            "short_3m":    float(short_row["3M%"]),
            "long_12m":    float(long_row["12M%"]),
            "short_12m":   float(short_row["12M%"]),
            "spread_1m":   round(spread_1m, 2),
            "spread_3m":   round(spread_3m, 2),
            "spread_curve":spread_cum,
        })

    return sorted(pairs, key=lambda x: x["spread_3m"], reverse=True)


def backtest_long_short(
    sector_map: dict[str, list[str]],
    top_n: int = 2,
    rebalance_days: int = 21,
) -> dict:
    """
    Long/Short backtest.
    Each rebalance: go long top_n and short top_n within each sector.
    L/S daily return = avg(long_daily) - avg(short_daily)  [market neutral].
    Also returns long-only equity for comparison.
    """
    all_tickers = sorted({t for tickers in sector_map.values() for t in tickers})
    prices = fetch_prices(all_tickers, period="5y")
    prices = prices.dropna(axis=1, how="all").ffill()
    prices.index = pd.DatetimeIndex(prices.index)

    ls_returns   = []
    long_returns = []
    dates = prices.index[252:]

    for i in range(0, len(dates) - rebalance_days, rebalance_days):
        window = prices.iloc[: prices.index.get_loc(dates[i])]
        if len(window) < 252:
            continue

        try:
            scores = ls_score(window)
        except Exception:
            continue

        long_book  = []
        short_book = []

        for sector, tickers in sector_map.items():
            avail = [t for t in tickers if t in scores.index and t in prices.columns]
            if len(avail) < 2:
                continue
            ranked = scores.loc[avail].sort_values("Score", ascending=False)
            long_book.extend(ranked.index[:top_n].tolist())
            short_book.extend(ranked.index[-top_n:].tolist())

        # de-duplicate
        long_book  = list(set(long_book))
        short_book = list(set(short_book) - set(long_book))

        if not long_book or not short_book:
            continue

        end_idx = min(i + rebalance_days, len(dates) - 1)
        period  = prices.loc[dates[i]: dates[end_idx]]

        long_ret  = period[long_book].pct_change().mean(axis=1)
        short_ret = period[short_book].pct_change().mean(axis=1)
        ls_ret    = long_ret - short_ret   # market-neutral spread

        ls_returns.append(ls_ret)
        long_returns.append(long_ret)

    if not ls_returns:
        return {"ls": pd.Series(dtype=float), "long_only": pd.Series(dtype=float)}

    ls_eq   = (1 + pd.concat(ls_returns).dropna()).cumprod() * 100
    long_eq = (1 + pd.concat(long_returns).dropna()).cumprod() * 100
    return {"ls": ls_eq.dropna(), "long_only": long_eq.dropna()}


# ── Additional backtest strategies ────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def backtest_mean_reversion(
    tickers: list[str],
    rsi_period: int = 14,
    rsi_buy: float = 30,
    rsi_sell: float = 70,
    max_hold: int = 20,
) -> dict:
    """
    Buy when RSI drops below rsi_buy, sell when RSI rises above rsi_sell
    or after max_hold days. Equal-weight across all triggered positions.
    """
    prices = fetch_prices(tickers, period="5y")
    prices = prices.dropna(axis=1, how="all").ffill()

    daily_portfolio: list[pd.Series] = []

    for ticker in prices.columns:
        col = prices[ticker].dropna()
        if len(col) < rsi_period + 10:
            continue
        rsi   = _calc_rsi(col, rsi_period)
        in_position = False
        hold_days   = 0
        entry_price = 0.0
        trade_rets: list[float] = [0.0] * len(col)

        for i in range(rsi_period + 1, len(col)):
            if not in_position:
                if rsi.iloc[i] < rsi_buy:
                    in_position  = True
                    hold_days    = 0
                    entry_price  = col.iloc[i]
            else:
                hold_days += 1
                pnl = col.iloc[i] / col.iloc[i - 1] - 1
                trade_rets[i] = pnl
                if rsi.iloc[i] > rsi_sell or hold_days >= max_hold:
                    in_position = False

        s = pd.Series(trade_rets, index=col.index)
        daily_portfolio.append(s)

    if not daily_portfolio:
        return {"equity": pd.Series(dtype=float), "trades": 0}

    combined = pd.concat(daily_portfolio, axis=1).mean(axis=1).dropna()
    equity   = (1 + combined).cumprod() * 100
    return {"equity": equity.dropna(), "label": "Mean Reversion (RSI)"}


def backtest_sma_cross(
    tickers: list[str],
    fast: int = 50,
    slow: int = 200,
) -> dict:
    """
    Golden Cross / Death Cross: buy when fast SMA crosses above slow SMA,
    sell when it crosses back below.
    """
    prices = fetch_prices(tickers, period="5y")
    prices = prices.dropna(axis=1, how="all").ffill()

    daily_portfolio: list[pd.Series] = []

    for ticker in prices.columns:
        col = prices[ticker].dropna()
        if len(col) < slow + 10:
            continue
        sma_fast = col.rolling(fast).mean()
        sma_slow = col.rolling(slow).mean()
        signal   = (sma_fast > sma_slow).astype(int)
        # 1 when in position, 0 when out — shift by 1 to avoid look-ahead
        position = signal.shift(1).fillna(0)
        daily_ret = col.pct_change() * position
        daily_portfolio.append(daily_ret)

    if not daily_portfolio:
        return {"equity": pd.Series(dtype=float)}

    combined = pd.concat(daily_portfolio, axis=1).mean(axis=1).dropna()
    equity   = (1 + combined).cumprod() * 100
    return {"equity": equity.dropna(), "label": f"SMA Cross ({fast}/{slow})"}


def backtest_buy_hold(tickers: list[str]) -> dict:
    """Equal-weight buy & hold — the simplest benchmark."""
    prices = fetch_prices(tickers, period="5y")
    prices = prices.dropna(axis=1, how="all").ffill()
    combined = prices.pct_change().dropna().mean(axis=1)
    equity   = (1 + combined).cumprod() * 100
    return {"equity": equity.dropna(), "label": "Buy & Hold (Equal Weight)"}


def backtest_spy_benchmark() -> dict:
    """SPY as the market benchmark."""
    col = fetch_prices(["SPY"], period="5y")["SPY"].dropna()
    equity = (1 + col.pct_change().dropna()).cumprod() * 100
    return {"equity": equity.dropna(), "label": "SPY Benchmark"}


def equity_stats(equity: pd.Series, capital: float = 100) -> dict:
    """Compute standard performance statistics from an equity curve."""
    eq = equity.dropna()
    if len(eq) < 5:
        return {}
    s, e       = eq.iloc[0], eq.iloc[-1]
    total_ret  = (e - s) / s * 100 if s else 0
    ann_ret    = ((e / s) ** (252 / len(eq)) - 1) * 100 if s else 0
    dr         = eq.pct_change().dropna()
    sharpe     = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() else 0
    max_dd     = ((eq / eq.cummax()) - 1).min() * 100
    win_rate   = (dr > 0).mean() * 100
    return {
        "total_ret": round(total_ret, 1),
        "ann_ret":   round(ann_ret, 1),
        "sharpe":    round(sharpe, 2),
        "max_dd":    round(max_dd, 1),
        "win_rate":  round(win_rate, 1),
        "final":     round(capital * (1 + total_ret / 100), 2),
        "equity":    eq,
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
