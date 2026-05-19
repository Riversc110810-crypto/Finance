# Personal Trading Dashboard

A fully personal trading dashboard built on Quality Momentum strategy with Monte Carlo simulation.

## Run

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Pages

| Page | What it does |
|---|---|
| Overview | Portfolio value, P&L, open positions, watchlist snapshot |
| Strategy Signals | Momentum + quality scores with BUY/HOLD/AVOID signals |
| Backtest | 2-year equity curve, Sharpe ratio, max drawdown |
| Monte Carlo | 1000-simulation probability cone for your $250 |
| Trade Log | Log buys/sells, track open positions, see realised P&L |
| Position Sizer | Risk-based share sizing so you never blow up a trade |
| Settings | Watchlist, capital, Alpaca API keys |

## Strategy

**Quality Momentum** — the same factor used by AQR Capital and Renaissance:
- Rank stocks by 12-month return minus 1-month return (avoids short-term reversal)
- Filter by quality (debt/equity < 100%, ROE > 10%)
- Hold top 3 stocks, rebalance monthly
- Size positions so each trade risks max 2% of capital

## Your Starting Point

- Capital: $250
- Mode: Paper trading (no real money until strategy is proven)
- Broker: Alpaca (free API, fractional shares, no minimums)
