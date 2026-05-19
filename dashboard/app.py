import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import yfinance as yf

from strategy import rank_stocks, backtest_momentum, monte_carlo, position_size

DATA_FILE = Path(__file__).parent / "data" / "profile.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_profile() -> dict:
    with open(DATA_FILE) as f:
        return json.load(f)


def save_profile(profile: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(profile, f, indent=2)


def current_price(ticker: str) -> float:
    try:
        return yf.Ticker(ticker).fast_info["last_price"]
    except Exception:
        return 0.0


def portfolio_value(profile: dict) -> float:
    cash = profile["current_cash"]
    holdings_value = 0.0
    for trade in profile["trades"]:
        if trade.get("status") == "open":
            price = current_price(trade["ticker"])
            holdings_value += price * trade["shares"]
    return cash + holdings_value


def open_positions(profile: dict) -> list:
    return [t for t in profile["trades"] if t.get("status") == "open"]


def closed_trades(profile: dict) -> list:
    return [t for t in profile["trades"] if t.get("status") == "closed"]


def total_pnl(profile: dict) -> float:
    pnl = 0.0
    for t in closed_trades(profile):
        pnl += t.get("pnl", 0.0)
    for t in open_positions(profile):
        price = current_price(t["ticker"])
        pnl += (price - t["entry_price"]) * t["shares"]
    return round(pnl, 2)


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="My Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #00d4aa;
  }
  .positive { color: #00d4aa; font-weight: bold; }
  .negative { color: #ff4b4b; font-weight: bold; }
  .signal-buy  { background: #003d2e; color: #00d4aa; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
  .signal-hold { background: #2e2a00; color: #ffd700; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
  .signal-avoid{ background: #3d0000; color: #ff4b4b; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
  h1, h2, h3 { color: #e8eaf6; }
</style>
""", unsafe_allow_html=True)


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Trading Dashboard")
    st.caption("Quality Momentum Strategy")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview", "Strategy Signals", "Backtest", "Monte Carlo", "Trade Log", "Position Sizer", "Settings"],
        label_visibility="collapsed",
    )

    st.divider()
    profile = load_profile()
    port_val = portfolio_value(profile)
    pnl = total_pnl(profile)
    pnl_color = "positive" if pnl >= 0 else "negative"
    pnl_sign = "+" if pnl >= 0 else ""

    st.markdown(f"**Portfolio Value**")
    st.markdown(f"### ${port_val:,.2f}")
    st.markdown(f'<span class="{pnl_color}">P&L: {pnl_sign}${pnl:.2f}</span>', unsafe_allow_html=True)
    st.caption(f"Cash: ${profile['current_cash']:,.2f}")
    st.caption(f"Mode: {'📄 Paper' if profile['alpaca_paper'] else '💰 Live'}")


profile = load_profile()


# ── Overview ─────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Portfolio Overview")

    port_val = portfolio_value(profile)
    pnl = total_pnl(profile)
    start_cap = profile["starting_capital"]
    total_return_pct = ((port_val - start_cap) / start_cap) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${port_val:,.2f}", f"${pnl:+.2f}")
    c2.metric("Total Return", f"{total_return_pct:.2f}%")
    c3.metric("Cash Available", f"${profile['current_cash']:,.2f}")
    c4.metric("Open Positions", len(open_positions(profile)))

    st.divider()

    positions = open_positions(profile)
    if positions:
        st.subheader("Open Positions")
        rows = []
        for p in positions:
            price = current_price(p["ticker"])
            pnl_pos = (price - p["entry_price"]) * p["shares"]
            pnl_pct = ((price - p["entry_price"]) / p["entry_price"]) * 100
            rows.append({
                "Ticker": p["ticker"],
                "Shares": p["shares"],
                "Entry": f"${p['entry_price']:.2f}",
                "Current": f"${price:.2f}",
                "P&L $": f"${pnl_pos:+.2f}",
                "P&L %": f"{pnl_pct:+.2f}%",
                "Value": f"${price * p['shares']:.2f}",
                "Date": p.get("date", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions. Use **Strategy Signals** to find trades.")

    st.subheader("Watchlist Snapshot")
    tickers = profile["watchlist"]
    with st.spinner("Fetching prices..."):
        snap_rows = []
        for t in tickers:
            try:
                info = yf.Ticker(t).fast_info
                price = info.get("last_price", 0)
                prev  = info.get("previous_close", price)
                chg   = price - prev
                chg_pct = (chg / prev * 100) if prev else 0
                snap_rows.append({"Ticker": t, "Price": f"${price:.2f}",
                                   "Change": f"${chg:+.2f}", "Change %": f"{chg_pct:+.2f}%"})
            except Exception:
                snap_rows.append({"Ticker": t, "Price": "—", "Change": "—", "Change %": "—"})
    st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)


# ── Strategy Signals ─────────────────────────────────────────────────────────

elif page == "Strategy Signals":
    st.title("Strategy Signals")
    st.caption("Quality Momentum — 70% 12-1M momentum + 30% quality score")

    tickers = profile["watchlist"]
    with st.spinner("Ranking stocks..."):
        df = rank_stocks(tickers)

    def color_signal(val):
        if val == "BUY":
            return "background-color: #003d2e; color: #00d4aa"
        elif val == "AVOID":
            return "background-color: #3d0000; color: #ff4b4b"
        return "background-color: #2e2a00; color: #ffd700"

    styled = df[["Signal", "Combined Score", "1M Return %", "3M Return %", "12M Return %", "Quality Score"]].style.applymap(
        color_signal, subset=["Signal"]
    )
    st.dataframe(styled, use_container_width=True)

    st.divider()
    st.subheader("Score Breakdown")
    fig = px.bar(
        df.reset_index(),
        x="index", y="Combined Score",
        color="Signal",
        color_discrete_map={"BUY": "#00d4aa", "HOLD": "#ffd700", "AVOID": "#ff4b4b"},
        labels={"index": "Ticker"},
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Return Heatmap")
    heatmap_df = df[["1M Return %", "3M Return %", "12M Return %"]].T
    fig2 = px.imshow(
        heatmap_df,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        aspect="auto",
    )
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117")
    st.plotly_chart(fig2, use_container_width=True)


# ── Backtest ─────────────────────────────────────────────────────────────────

elif page == "Backtest":
    st.title("Strategy Backtest")
    st.caption("2-year backtest: hold top N momentum stocks, rebalance monthly")

    tickers = profile["watchlist"]
    top_n = st.slider("Top N stocks to hold", 1, min(5, len(tickers)), 3)
    capital = profile["starting_capital"]

    with st.spinner("Running backtest..."):
        equity = backtest_momentum(tickers, top_n=top_n)

    if equity.empty:
        st.warning("Not enough data to run backtest.")
    else:
        start_val = equity.iloc[0]
        end_val = equity.iloc[-1]
        total_ret = (end_val - start_val) / start_val * 100
        daily_rets = equity.pct_change().dropna()
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
        max_dd = ((equity / equity.cummax()) - 1).min() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{total_ret:.1f}%")
        c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c3.metric("Max Drawdown", f"{max_dd:.1f}%")
        c4.metric("Final $250 → ", f"${250 * (1 + total_ret/100):.2f}")

        scaled = equity / equity.iloc[0] * capital

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index, y=scaled,
            fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
            line=dict(color="#00d4aa", width=2),
            name="Strategy",
        ))
        fig.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="Starting Capital")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            title="Portfolio Equity Curve ($250 starting capital)",
            yaxis_title="Portfolio Value ($)", xaxis_title="Date",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drawdown Chart")
        dd = ((equity / equity.cummax()) - 1) * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dd.index, y=dd,
            fill="tozeroy", fillcolor="rgba(255,75,75,0.15)",
            line=dict(color="#ff4b4b", width=1.5),
            name="Drawdown %",
        ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            yaxis_title="Drawdown %", xaxis_title="Date",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Monte Carlo ───────────────────────────────────────────────────────────────

elif page == "Monte Carlo":
    st.title("Monte Carlo Simulation")
    st.caption("Probabilistic 1-year projection based on backtest statistics")

    tickers = profile["watchlist"]
    capital = st.number_input("Starting Capital ($)", value=float(profile["starting_capital"]), step=10.0)
    n_sims = st.slider("Number of Simulations", 200, 2000, 1000, step=200)
    days = st.slider("Days Forward", 63, 504, 252)

    with st.spinner("Running simulations..."):
        equity = backtest_momentum(tickers, top_n=3)
        if equity.empty:
            st.warning("Not enough data.")
            st.stop()
        results = monte_carlo(equity, n_simulations=n_sims, days_forward=days, capital=capital)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pessimistic (P10)", f"${results['p10']:,.2f}")
    c2.metric("Median (P50)", f"${results['p50']:,.2f}")
    c3.metric("Optimistic (P90)", f"${results['p90']:,.2f}")
    c4.metric("Prob. of Profit", f"{results['prob_profit']:.1f}%")

    fig = go.Figure()
    sims = results["simulations"]
    show_every = max(1, n_sims // 100)
    x = list(range(days))

    for i in range(0, n_sims, show_every):
        fig.add_trace(go.Scatter(
            x=x, y=sims[:, i],
            line=dict(width=0.4, color="rgba(0,212,170,0.15)"),
            showlegend=False, hoverinfo="skip",
        ))

    for pct, col, name in [(10, "#ff4b4b", "P10"), (50, "#ffffff", "P50 Median"), (90, "#00d4aa", "P90")]:
        vals = np.percentile(sims, pct, axis=1)
        fig.add_trace(go.Scatter(
            x=x, y=vals,
            line=dict(width=2.5, color=col),
            name=name,
        ))

    fig.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="Starting Capital")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=f"Monte Carlo — {n_sims} Simulations over {days} Days",
        yaxis_title="Portfolio Value ($)", xaxis_title="Trading Day",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final Value Distribution")
    final_vals = sims[-1, :]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=final_vals, nbinsx=60,
        marker_color="#00d4aa", opacity=0.75,
        name="Final Value",
    ))
    fig2.add_vline(x=capital, line_dash="dash", line_color="white", annotation_text="Starting Capital")
    fig2.add_vline(x=results["p50"], line_dash="dot", line_color="#ffd700", annotation_text="Median")
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Final Portfolio Value ($)", yaxis_title="Count",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Trade Log ────────────────────────────────────────────────────────────────

elif page == "Trade Log":
    st.title("Trade Log")

    profile = load_profile()

    with st.expander("Log a New Trade"):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker").upper()
        action = c2.selectbox("Action", ["BUY", "SELL"])
        shares = c3.number_input("Shares", min_value=0.0001, step=0.0001, format="%.4f")
        price = st.number_input("Price ($)", min_value=0.01, step=0.01)
        note = st.text_input("Note (optional)")

        if st.button("Log Trade", type="primary"):
            if ticker and shares and price:
                trade_value = shares * price
                if action == "BUY":
                    if trade_value > profile["current_cash"]:
                        st.error(f"Insufficient cash. Available: ${profile['current_cash']:.2f}")
                    else:
                        profile["current_cash"] -= trade_value
                        profile["trades"].append({
                            "id": len(profile["trades"]) + 1,
                            "ticker": ticker,
                            "action": action,
                            "shares": shares,
                            "entry_price": price,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "status": "open",
                            "note": note,
                        })
                        save_profile(profile)
                        st.success(f"Logged BUY {shares} {ticker} @ ${price:.2f}")
                        st.rerun()
                elif action == "SELL":
                    open_pos = [t for t in profile["trades"]
                                if t.get("status") == "open" and t["ticker"] == ticker]
                    if not open_pos:
                        st.error(f"No open position for {ticker}")
                    else:
                        pos = open_pos[0]
                        pnl_trade = (price - pos["entry_price"]) * shares
                        pos["status"] = "closed"
                        pos["exit_price"] = price
                        pos["exit_date"] = datetime.now().strftime("%Y-%m-%d")
                        pos["pnl"] = round(pnl_trade, 2)
                        profile["current_cash"] += shares * price
                        save_profile(profile)
                        sign = "+" if pnl_trade >= 0 else ""
                        st.success(f"Closed {ticker} — P&L: {sign}${pnl_trade:.2f}")
                        st.rerun()
            else:
                st.warning("Fill in all fields.")

    st.divider()

    trades = profile["trades"]
    if trades:
        df = pd.DataFrame(trades)
        open_df = df[df["status"] == "open"] if "status" in df.columns else pd.DataFrame()
        closed_df = df[df["status"] == "closed"] if "status" in df.columns else pd.DataFrame()

        st.subheader(f"Open Positions ({len(open_df)})")
        if not open_df.empty:
            st.dataframe(open_df, use_container_width=True, hide_index=True)

        st.subheader(f"Closed Trades ({len(closed_df)})")
        if not closed_df.empty:
            st.dataframe(closed_df, use_container_width=True, hide_index=True)
            total = closed_df["pnl"].sum() if "pnl" in closed_df.columns else 0
            st.metric("Total Realised P&L", f"${total:+.2f}")
    else:
        st.info("No trades logged yet.")


# ── Position Sizer ────────────────────────────────────────────────────────────

elif page == "Position Sizer":
    st.title("Position Sizer")
    st.caption("Risk-based sizing — never risk more than your set % per trade")

    profile = load_profile()
    capital = st.number_input("Available Capital ($)", value=float(profile["current_cash"]), step=1.0)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, float(profile["risk_per_trade_pct"]), step=0.5)
    ticker_ps = st.text_input("Ticker (for live price)", "AAPL").upper()

    c1, c2 = st.columns(2)
    entry = c1.number_input("Entry Price ($)", value=0.0, step=0.01)
    stop = c2.number_input("Stop Loss Price ($)", value=0.0, step=0.01)

    if ticker_ps and entry == 0.0:
        with st.spinner("Fetching price..."):
            live = current_price(ticker_ps)
        if live:
            st.info(f"Live price for {ticker_ps}: ${live:.2f}")
            entry = live

    if entry > 0 and stop > 0 and entry != stop:
        sizing = position_size(capital, risk_pct, entry, stop)
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Shares to Buy", f"{sizing['shares']:.4f}")
        c2.metric("Capital at Risk", f"${sizing['risk_dollars']:.2f}")
        c3.metric("Position Value", f"${sizing['position_value']:.2f}")

        pct_of_port = (sizing["position_value"] / capital) * 100
        st.caption(f"Position is {pct_of_port:.1f}% of your capital")

        if sizing["position_value"] > capital:
            st.error("Position value exceeds available capital. Widen your stop or reduce risk %.")
        elif pct_of_port > 50:
            st.warning("This position is over 50% of your capital — consider a tighter stop.")
    else:
        st.info("Enter an entry price and stop loss to calculate sizing.")


# ── Settings ─────────────────────────────────────────────────────────────────

elif page == "Settings":
    st.title("Settings")
    profile = load_profile()

    st.subheader("Profile")
    name = st.text_input("Dashboard Name", profile["name"])
    start_cap = st.number_input("Starting Capital ($)", value=float(profile["starting_capital"]), step=10.0)
    risk_pct = st.slider("Default Risk per Trade (%)", 0.5, 5.0, float(profile["risk_per_trade_pct"]), step=0.5)

    st.subheader("Watchlist")
    watchlist_raw = st.text_input(
        "Tickers (comma separated)",
        ", ".join(profile["watchlist"])
    )

    st.subheader("Alpaca API (Paper Trading)")
    paper_mode = st.checkbox("Paper Trading Mode", value=profile["alpaca_paper"])
    api_key = st.text_input("API Key", value=profile["alpaca_api_key"], type="password")
    secret_key = st.text_input("Secret Key", value=profile["alpaca_secret_key"], type="password")

    if st.button("Save Settings", type="primary"):
        profile["name"] = name
        profile["starting_capital"] = start_cap
        profile["risk_per_trade_pct"] = risk_pct
        profile["watchlist"] = [t.strip().upper() for t in watchlist_raw.split(",") if t.strip()]
        profile["alpaca_paper"] = paper_mode
        profile["alpaca_api_key"] = api_key
        profile["alpaca_secret_key"] = secret_key
        save_profile(profile)
        st.success("Settings saved!")
        st.rerun()

    st.divider()
    st.subheader("Danger Zone")
    if st.button("Reset All Trades (keep capital)", type="secondary"):
        profile["trades"] = []
        profile["current_cash"] = profile["starting_capital"]
        save_profile(profile)
        st.success("Trades cleared.")
        st.rerun()
