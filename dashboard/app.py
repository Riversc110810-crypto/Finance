import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import yfinance as yf

from strategy import rank_stocks, backtest_momentum, monte_carlo, position_size
from universe import UNIVERSE, ALL_TICKERS, DEFAULT_CATEGORIES, tickers_for
import alpaca_client as ac

DATA_FILE = Path(__file__).parent / "data" / "profile.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_profile() -> dict:
    with open(DATA_FILE) as f:
        return json.load(f)


def save_profile(p: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(p, f, indent=2)


def current_price(ticker: str) -> float:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def open_positions(profile: dict) -> list:
    return [t for t in profile["trades"] if t.get("status") == "open"]


def closed_trades(profile: dict) -> list:
    return [t for t in profile["trades"] if t.get("status") == "closed"]


def local_portfolio_value(profile: dict) -> float:
    cash = profile["current_cash"]
    for t in open_positions(profile):
        cash += current_price(t["ticker"]) * t["shares"]
    return cash


def total_pnl(profile: dict) -> float:
    pnl = sum(t.get("pnl", 0.0) for t in closed_trades(profile))
    for t in open_positions(profile):
        pnl += (current_price(t["ticker"]) - t["entry_price"]) * t["shares"]
    return round(pnl, 2)


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .positive { color: #00d4aa; font-weight: bold; }
  .negative { color: #ff4b4b; font-weight: bold; }
  .tag-connected { background:#003d2e; color:#00d4aa; padding:3px 10px;
                   border-radius:12px; font-size:0.8em; font-weight:bold; }
  .tag-paper     { background:#1a2a3a; color:#60aaff; padding:3px 10px;
                   border-radius:12px; font-size:0.8em; font-weight:bold; }
  .tag-offline   { background:#2e1a1a; color:#ff6b6b; padding:3px 10px;
                   border-radius:12px; font-size:0.8em; font-weight:bold; }
</style>
""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Trading Dashboard")
    st.caption("Quality Momentum Strategy")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview", "Strategy Signals", "Backtest", "Monte Carlo",
         "Trade Log", "Position Sizer", "Settings"],
        label_visibility="collapsed",
    )

    st.divider()
    profile = load_profile()

    # try Alpaca first, fall back to local
    alpaca_acct = None
    if profile["alpaca_api_key"]:
        alpaca_acct = ac.get_account(
            profile["alpaca_api_key"], profile["alpaca_secret_key"], profile["alpaca_paper"]
        )

    if alpaca_acct:
        port_val = alpaca_acct.portfolio_value
        cash_val = alpaca_acct.cash
        pnl = total_pnl(profile)
        mode_tag = '<span class="tag-paper">📄 Paper</span>' if alpaca_acct.paper else '<span class="tag-connected">💰 Live</span>'
        st.markdown(f"**Portfolio Value** {mode_tag}", unsafe_allow_html=True)
        st.markdown(f"### ${port_val:,.2f}")
        pnl_cls = "positive" if pnl >= 0 else "negative"
        st.markdown(f'<span class="{pnl_cls}">Logged P&L: ${pnl:+.2f}</span>', unsafe_allow_html=True)
        st.caption(f"Cash: ${cash_val:,.2f} | BP: ${alpaca_acct.buying_power:,.2f}")
    else:
        port_val = local_portfolio_value(profile)
        pnl = total_pnl(profile)
        pnl_cls = "positive" if pnl >= 0 else "negative"
        st.markdown('**Portfolio Value** <span class="tag-offline">⚠ Offline</span>', unsafe_allow_html=True)
        st.markdown(f"### ${port_val:,.2f}")
        st.markdown(f'<span class="{pnl_cls}">P&L: ${pnl:+.2f}</span>', unsafe_allow_html=True)
        st.caption(f"Cash: ${profile['current_cash']:,.2f}")

profile = load_profile()


# ── Overview ──────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Portfolio Overview")

    alpaca_acct = None
    if profile["alpaca_api_key"]:
        with st.spinner("Connecting to Alpaca..."):
            alpaca_acct = ac.get_account(
                profile["alpaca_api_key"], profile["alpaca_secret_key"], profile["alpaca_paper"]
            )

    if alpaca_acct:
        st.success(f"Connected to Alpaca {'Paper' if alpaca_acct.paper else 'Live'} Trading")
        c1, c2, c3, c4 = st.columns(4)
        start = profile["starting_capital"]
        ret_pct = (alpaca_acct.portfolio_value - start) / start * 100
        c1.metric("Portfolio Value",  f"${alpaca_acct.portfolio_value:,.2f}")
        c2.metric("Total Return",     f"{ret_pct:.2f}%")
        c3.metric("Cash",             f"${alpaca_acct.cash:,.2f}")
        c4.metric("Buying Power",     f"${alpaca_acct.buying_power:,.2f}")

        st.divider()
        st.subheader("Alpaca Positions")
        positions = ac.get_positions(
            profile["alpaca_api_key"], profile["alpaca_secret_key"], profile["alpaca_paper"]
        )
        if positions:
            rows = [{"Ticker": p.ticker, "Qty": p.qty,
                     "Avg Entry": f"${p.avg_entry:.2f}",
                     "Current": f"${p.current_price:.2f}",
                     "Market Value": f"${p.market_value:.2f}",
                     "Unreal. P&L": f"${p.unrealized_pnl:+.2f}",
                     "P&L %": f"{p.unrealized_pnl_pct:+.2f}%"} for p in positions]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions in Alpaca account.")
    else:
        pnl = total_pnl(profile)
        start = profile["starting_capital"]
        port_val = local_portfolio_value(profile)
        ret_pct = (port_val - start) / start * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", f"${port_val:,.2f}", f"${pnl:+.2f}")
        c2.metric("Total Return",    f"{ret_pct:.2f}%")
        c3.metric("Cash Available",  f"${profile['current_cash']:,.2f}")
        c4.metric("Open Positions",  len(open_positions(profile)))

        if not profile["alpaca_api_key"]:
            st.info("Connect Alpaca in **Settings** to see live account data and place orders.")

        positions = open_positions(profile)
        if positions:
            st.divider()
            st.subheader("Manual Positions")
            rows = []
            for p in positions:
                price = current_price(p["ticker"])
                pnl_p = (price - p["entry_price"]) * p["shares"]
                pnl_pct = ((price - p["entry_price"]) / p["entry_price"]) * 100
                rows.append({"Ticker": p["ticker"], "Shares": p["shares"],
                             "Entry": f"${p['entry_price']:.2f}", "Current": f"${price:.2f}",
                             "P&L $": f"${pnl_p:+.2f}", "P&L %": f"{pnl_pct:+.2f}%",
                             "Value": f"${price * p['shares']:.2f}", "Date": p.get("date", "—")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Watchlist Snapshot")
    tickers = profile.get("watchlist", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
    with st.spinner("Fetching prices..."):
        rows = []
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="5d")
                if len(hist) >= 2:
                    price = float(hist["Close"].iloc[-1])
                    prev  = float(hist["Close"].iloc[-2])
                elif len(hist) == 1:
                    price = prev = float(hist["Close"].iloc[-1])
                else:
                    raise ValueError
                chg = price - prev
                chg_pct = chg / prev * 100 if prev else 0
                rows.append({"Ticker": t, "Price": f"${price:.2f}",
                             "Change": f"${chg:+.2f}", "Change %": f"{chg_pct:+.2f}%"})
            except Exception:
                rows.append({"Ticker": t, "Price": "—", "Change": "—", "Change %": "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Strategy Signals ──────────────────────────────────────────────────────────

elif page == "Strategy Signals":
    st.title("Strategy Signals")
    st.caption("Quality Momentum — 70% 12-1M momentum + 30% quality. Select universe categories below.")

    all_cats = list(UNIVERSE.keys())
    saved_cats = profile.get("selected_categories", DEFAULT_CATEGORIES)

    selected_cats = st.multiselect(
        "Universe categories to screen",
        options=all_cats,
        default=saved_cats,
        help="Select which groups of stocks/ETFs to rank"
    )

    if not selected_cats:
        st.warning("Select at least one category.")
        st.stop()

    tickers = tickers_for(selected_cats)
    st.caption(f"Screening {len(tickers)} instruments across {len(selected_cats)} categories")

    top_n_show = st.slider("Show top N results", 10, min(80, len(tickers)), 30)

    with st.spinner(f"Downloading data and ranking {len(tickers)} tickers..."):
        df = rank_stocks(tickers)

    df_show = df.head(top_n_show)

    def color_signal(val):
        if val == "BUY":
            return "background-color: #003d2e; color: #00d4aa"
        elif val == "AVOID":
            return "background-color: #3d0000; color: #ff4b4b"
        return "background-color: #2e2a00; color: #ffd700"

    buy_count  = (df["Signal"] == "BUY").sum()
    hold_count = (df["Signal"] == "HOLD").sum()
    avoid_count= (df["Signal"] == "AVOID").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("BUY signals",   buy_count)
    c2.metric("HOLD signals",  hold_count)
    c3.metric("AVOID signals", avoid_count)

    styled = df_show[["Signal", "Combined Score", "1M Return %", "3M Return %", "12M Return %", "Quality Score"]].style.map(
        color_signal, subset=["Signal"]
    )
    st.dataframe(styled, use_container_width=True)

    st.divider()
    st.subheader("Score Breakdown — Top Results")
    df_reset = df_show.reset_index()
    ticker_col = df_reset.columns[0]
    fig = px.bar(
        df_reset, x=ticker_col, y="Combined Score",
        color="Signal",
        color_discrete_map={"BUY": "#00d4aa", "HOLD": "#ffd700", "AVOID": "#ff4b4b"},
        labels={ticker_col: "Ticker"},
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Return Heatmap")
    ret_cols = [c for c in ["1M Return %", "3M Return %", "12M Return %"] if c in df_show.columns]
    heatmap_df = df_show[ret_cols].dropna(axis=1, how="all").T
    if not heatmap_df.empty:
        fig2 = px.imshow(
            heatmap_df,
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            aspect="auto", text_auto=".1f",
        )
        fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                           height=200 + len(ret_cols) * 40)
        st.plotly_chart(fig2, use_container_width=True)

    # save selected categories back to profile
    if selected_cats != profile.get("selected_categories"):
        profile["selected_categories"] = selected_cats
        save_profile(profile)


# ── Backtest ──────────────────────────────────────────────────────────────────

elif page == "Backtest":
    st.title("Strategy Backtest")
    st.caption("Hold top N momentum stocks from selected universe, rebalance monthly — 5 year window")

    all_cats = list(UNIVERSE.keys())
    saved_cats = profile.get("selected_categories", DEFAULT_CATEGORIES)

    col_a, col_b = st.columns([3, 1])
    selected_cats = col_a.multiselect(
        "Universe categories", options=all_cats, default=saved_cats
    )
    top_n = col_b.number_input("Hold top N", min_value=1, max_value=20, value=5)

    if not selected_cats:
        st.warning("Select at least one category.")
        st.stop()

    tickers = tickers_for(selected_cats)
    capital = profile["starting_capital"]
    st.caption(f"Backtesting across {len(tickers)} instruments")

    with st.spinner("Running backtest (downloading 5 years of data)..."):
        equity = backtest_momentum(tickers, top_n=int(top_n))

    if equity.empty or len(equity) < 5:
        st.warning("Not enough data. Try adding more categories or check your internet connection.")
        st.stop()

    equity = equity.dropna()
    start_val  = equity.iloc[0]
    end_val    = equity.iloc[-1]
    total_ret  = (end_val - start_val) / start_val * 100 if start_val else 0.0
    daily_rets = equity.pct_change().dropna()
    sharpe     = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() else 0.0
    max_dd     = ((equity / equity.cummax()) - 1).min() * 100
    ann_ret    = ((end_val / start_val) ** (252 / len(equity)) - 1) * 100 if start_val else 0.0
    final_val  = capital * (1 + total_ret / 100)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return",     f"{total_ret:.1f}%")
    c2.metric("Ann. Return",      f"{ann_ret:.1f}%")
    c3.metric("Sharpe Ratio",     f"{sharpe:.2f}")
    c4.metric("Max Drawdown",     f"{max_dd:.1f}%")
    c5.metric(f"${capital:.0f} → ", f"${final_val:,.2f}")

    scaled = equity / equity.iloc[0] * capital
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=scaled,
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        line=dict(color="#00d4aa", width=2), name="Strategy",
    ))
    fig.add_hline(y=capital, line_dash="dash", line_color="gray",
                  annotation_text=f"Starting Capital ${capital:.0f}")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=f"Equity Curve — Top {top_n} Momentum, {len(selected_cats)} Categories",
        yaxis_title="Portfolio Value ($)", xaxis_title="Date", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drawdown")
    dd = ((equity / equity.cummax()) - 1) * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dd.index, y=dd,
        fill="tozeroy", fillcolor="rgba(255,75,75,0.12)",
        line=dict(color="#ff4b4b", width=1.5), name="Drawdown %",
    ))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis_title="Drawdown %", height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Rolling Sharpe (1-year window)")
    roll_sharpe = daily_rets.rolling(252).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() else 0
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe,
        line=dict(color="#ffd700", width=1.5), name="Rolling Sharpe",
    ))
    fig3.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Sharpe = 1")
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis_title="Sharpe", height=260,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Monte Carlo ───────────────────────────────────────────────────────────────

elif page == "Monte Carlo":
    st.title("Monte Carlo Simulation")
    st.caption("Probabilistic projection seeded from 5-year backtest statistics")

    all_cats = list(UNIVERSE.keys())
    saved_cats = profile.get("selected_categories", DEFAULT_CATEGORIES)

    c_left, c_right = st.columns([3, 1])
    selected_cats = c_left.multiselect(
        "Universe categories", options=all_cats, default=saved_cats
    )
    top_n = c_right.number_input("Hold top N", min_value=1, max_value=20, value=5)

    if not selected_cats:
        st.warning("Select at least one category.")
        st.stop()

    tickers = tickers_for(selected_cats)

    c1, c2, c3 = st.columns(3)
    capital = c1.number_input("Starting Capital ($)", value=float(profile["starting_capital"]), step=10.0)
    n_sims  = c2.slider("Simulations", 500, 5000, 2000, step=500)
    days    = c3.slider("Days Forward", 63, 756, 252, step=63,
                        help="252 = 1yr, 504 = 2yr, 756 = 3yr")

    st.caption(f"Running {n_sims} simulations over {days} days across {len(tickers)} instruments")

    with st.spinner("Running backtest then Monte Carlo..."):
        equity = backtest_momentum(tickers, top_n=int(top_n))
        if equity.empty:
            st.warning("Not enough data.")
            st.stop()
        results = monte_carlo(equity, n_simulations=n_sims, days_forward=days, capital=capital)

    daily_mu    = results["mu"] * 252 * 100
    daily_sigma = results["sigma"] * np.sqrt(252) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pessimistic (P10)", f"${results['p10']:,.2f}")
    c2.metric("Median (P50)",      f"${results['p50']:,.2f}")
    c3.metric("Optimistic (P90)", f"${results['p90']:,.2f}")
    c4.metric("Prob. of Profit",  f"{results['prob_profit']:.1f}%")
    c5.metric("Ann. Vol (est.)",  f"{daily_sigma:.1f}%")

    fig = go.Figure()
    sims = results["simulations"]
    show_every = max(1, n_sims // 150)
    x = list(range(days))

    for i in range(0, n_sims, show_every):
        fig.add_trace(go.Scatter(
            x=x, y=sims[:, i],
            line=dict(width=0.3, color="rgba(0,212,170,0.12)"),
            showlegend=False, hoverinfo="skip",
        ))

    band_colors = {"P10": "#ff4b4b", "P25": "#ff9900", "P50 Median": "#ffffff",
                   "P75": "#60aaff", "P90": "#00d4aa"}
    for pct, name in [(10, "P10"), (25, "P25"), (50, "P50 Median"), (75, "P75"), (90, "P90")]:
        vals = np.percentile(sims, pct, axis=1)
        fig.add_trace(go.Scatter(
            x=x, y=vals,
            line=dict(width=2, color=band_colors[name]),
            name=name,
        ))

    fig.add_hline(y=capital, line_dash="dash", line_color="gray",
                  annotation_text=f"Starting Capital ${capital:.0f}")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=f"Monte Carlo — {n_sims:,} Simulations × {days} Days",
        yaxis_title="Portfolio Value ($)", xaxis_title="Trading Day", height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final Value Distribution")
    final_vals = sims[-1, :]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=final_vals, nbinsx=80,
        marker_color="#00d4aa", opacity=0.7, name="Final Value",
    ))
    fig2.add_vline(x=capital,          line_dash="dash", line_color="white",  annotation_text="Start")
    fig2.add_vline(x=results["p10"],   line_dash="dot",  line_color="#ff4b4b",annotation_text="P10")
    fig2.add_vline(x=results["p50"],   line_dash="dot",  line_color="#ffd700", annotation_text="P50")
    fig2.add_vline(x=results["p90"],   line_dash="dot",  line_color="#00d4aa",annotation_text="P90")
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Final Portfolio Value ($)", yaxis_title="Count", height=320,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Return Range by Percentile")
    pct_table = []
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(final_vals, pct)
        ret = (val - capital) / capital * 100
        pct_table.append({"Percentile": f"P{pct}", "Final Value": f"${val:,.2f}",
                          "Return": f"{ret:+.1f}%", "Multiple": f"{val/capital:.2f}x"})
    st.dataframe(pd.DataFrame(pct_table), use_container_width=True, hide_index=True)


# ── Trade Log ─────────────────────────────────────────────────────────────────

elif page == "Trade Log":
    st.title("Trade Log")
    profile = load_profile()

    alpaca_connected = bool(profile["alpaca_api_key"])
    if alpaca_connected:
        st.success("Alpaca connected — orders can be placed directly from here.")
    else:
        st.info("Alpaca not connected. Trades logged manually. Connect in **Settings**.")

    with st.expander("Log / Place a New Trade", expanded=not bool(profile["trades"])):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker").upper().strip()
        action = c2.selectbox("Action", ["BUY", "SELL"])
        order_type = c3.selectbox("Order Type", ["Dollar Amount", "Shares"])

        if order_type == "Dollar Amount":
            dollar_amt = st.number_input("Dollar Amount ($)", min_value=1.0, step=1.0, value=50.0)
            shares_input = None
        else:
            shares_input = st.number_input("Shares", min_value=0.0001, step=0.0001, format="%.4f")
            dollar_amt = None

        price_input = st.number_input("Price ($ — for manual log; 0 = fetch live)", min_value=0.0, step=0.01)
        note = st.text_input("Note (optional)")

        col_log, col_alpaca = st.columns(2)

        # Manual log button
        if col_log.button("Log Trade (manual)", type="secondary"):
            if not ticker:
                st.warning("Enter a ticker.")
            else:
                price = price_input if price_input > 0 else current_price(ticker)
                shares = shares_input if shares_input else round(dollar_amt / price, 4) if price else 0
                trade_value = shares * price

                if action == "BUY":
                    if trade_value > profile["current_cash"]:
                        st.error(f"Insufficient cash. Available: ${profile['current_cash']:.2f}")
                    else:
                        profile["current_cash"] -= trade_value
                        profile["trades"].append({
                            "id": len(profile["trades"]) + 1,
                            "ticker": ticker, "action": action,
                            "shares": shares, "entry_price": price,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "status": "open", "note": note, "source": "manual",
                        })
                        save_profile(profile)
                        st.success(f"Logged BUY {shares:.4f} {ticker} @ ${price:.2f}")
                        st.rerun()
                else:
                    open_pos = [t for t in profile["trades"]
                                if t.get("status") == "open" and t["ticker"] == ticker]
                    if not open_pos:
                        st.error(f"No open manual position for {ticker}")
                    else:
                        pos = open_pos[0]
                        pnl_t = (price - pos["entry_price"]) * shares
                        pos.update({"status": "closed", "exit_price": price,
                                    "exit_date": datetime.now().strftime("%Y-%m-%d"),
                                    "pnl": round(pnl_t, 2)})
                        profile["current_cash"] += shares * price
                        save_profile(profile)
                        st.success(f"Closed {ticker} — P&L: ${pnl_t:+.2f}")
                        st.rerun()

        # Alpaca order button
        if alpaca_connected:
            if col_alpaca.button("Place via Alpaca", type="primary"):
                if not ticker:
                    st.warning("Enter a ticker.")
                else:
                    with st.spinner("Sending order to Alpaca..."):
                        if order_type == "Dollar Amount":
                            result = ac.place_market_order(
                                profile["alpaca_api_key"], profile["alpaca_secret_key"],
                                profile["alpaca_paper"], ticker,
                                qty=0, side=action.lower(), notional=dollar_amt,
                            )
                        else:
                            result = ac.place_market_order(
                                profile["alpaca_api_key"], profile["alpaca_secret_key"],
                                profile["alpaca_paper"], ticker,
                                qty=shares_input, side=action.lower(),
                            )
                    if result["success"]:
                        st.success(f"Order placed! ID: {result['order_id']}")
                        # also log locally
                        price = price_input if price_input > 0 else current_price(ticker)
                        shares = shares_input if shares_input else round(dollar_amt / price, 4) if price else 0
                        profile["trades"].append({
                            "id": len(profile["trades"]) + 1,
                            "ticker": ticker, "action": action,
                            "shares": shares, "entry_price": price,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "status": "open", "note": note,
                            "source": "alpaca", "alpaca_order_id": result["order_id"],
                        })
                        save_profile(profile)
                        st.rerun()
                    else:
                        st.error(f"Order failed: {result['error']}")

    st.divider()
    trades = profile["trades"]
    if trades:
        df_trades = pd.DataFrame(trades)
        open_df   = df_trades[df_trades["status"] == "open"]
        closed_df = df_trades[df_trades["status"] == "closed"]

        st.subheader(f"Open Positions ({len(open_df)})")
        if not open_df.empty:
            st.dataframe(open_df, use_container_width=True, hide_index=True)

        st.subheader(f"Closed Trades ({len(closed_df)})")
        if not closed_df.empty:
            st.dataframe(closed_df, use_container_width=True, hide_index=True)
            if "pnl" in closed_df.columns:
                total_realised = closed_df["pnl"].sum()
                st.metric("Total Realised P&L", f"${total_realised:+.2f}")
    else:
        st.info("No trades logged yet.")


# ── Position Sizer ────────────────────────────────────────────────────────────

elif page == "Position Sizer":
    st.title("Position Sizer")
    st.caption("Risk-based sizing — never risk more than your set % per trade")

    profile = load_profile()
    capital  = st.number_input("Available Capital ($)", value=float(profile["current_cash"]), step=1.0)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, float(profile["risk_per_trade_pct"]), step=0.5)

    ticker_ps = st.text_input("Ticker", "AAPL").upper()

    c1, c2 = st.columns(2)
    entry = c1.number_input("Entry Price ($)", value=0.0, step=0.01)
    stop  = c2.number_input("Stop Loss Price ($)", value=0.0, step=0.01)

    if ticker_ps and entry == 0.0:
        with st.spinner("Fetching live price..."):
            live = current_price(ticker_ps)
        if live:
            st.info(f"Live price for {ticker_ps}: **${live:.2f}**")
            entry = live

    if entry > 0 and stop > 0 and entry != stop:
        sizing = position_size(capital, risk_pct, entry, stop)
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shares to Buy",  f"{sizing['shares']:.4f}")
        c2.metric("Capital at Risk",f"${sizing['risk_dollars']:.2f}")
        c3.metric("Position Value", f"${sizing['position_value']:.2f}")
        pct_port = sizing["position_value"] / capital * 100
        c4.metric("% of Capital",   f"{pct_port:.1f}%")

        if sizing["position_value"] > capital:
            st.error("Position value exceeds available capital. Widen your stop or reduce risk %.")
        elif pct_port > 50:
            st.warning("Over 50% of capital in one position — consider tightening stop loss.")

        dollar_risk = capital * risk_pct / 100
        st.caption(
            f"Max dollar risk per trade at {risk_pct}%: **${dollar_risk:.2f}**  |  "
            f"Stop distance: **${abs(entry - stop):.2f}** ({abs(entry-stop)/entry*100:.1f}%)"
        )
    else:
        st.info("Enter entry and stop loss prices to calculate position size.")


# ── Settings ──────────────────────────────────────────────────────────────────

elif page == "Settings":
    st.title("Settings")
    profile = load_profile()

    st.subheader("Profile")
    name      = st.text_input("Dashboard Name", profile["name"])
    start_cap = st.number_input("Starting Capital ($)", value=float(profile["starting_capital"]), step=10.0)
    risk_pct  = st.slider("Default Risk per Trade (%)", 0.5, 5.0,
                          float(profile["risk_per_trade_pct"]), step=0.5)

    st.subheader("Personal Watchlist (Overview page)")
    watchlist_raw = st.text_input(
        "Tickers (comma separated)", ", ".join(profile.get("watchlist", []))
    )

    st.subheader("Default Universe Categories")
    st.caption("These categories pre-select on Strategy Signals, Backtest, and Monte Carlo pages.")
    all_cats = list(UNIVERSE.keys())
    saved_cats = profile.get("selected_categories", DEFAULT_CATEGORIES)
    selected_cats = st.multiselect("Default categories", options=all_cats, default=saved_cats)

    st.divider()
    st.subheader("🦙 Alpaca API")

    # Step-by-step setup guide
    with st.expander("How to set up Alpaca (takes 2 minutes)", expanded=not bool(profile["alpaca_api_key"])):
        st.markdown("""
**Step 1 — Create a free account**
Go to [alpaca.markets](https://alpaca.markets) → click **Get Started** → sign up (email + password, no credit card needed).

**Step 2 — Go to Paper Trading**
In the dashboard, make sure you are in **Paper Trading** mode (toggle in the top-left corner).
Paper trading uses fake money — safe to learn with.

**Step 3 — Generate API Keys**
- Click your account name (top right) → **API Keys**
- Click **Generate New Key**
- Copy both the **API Key ID** and the **Secret Key** — the secret is only shown once

**Step 4 — Paste them below**
Enter both keys in the fields below, keep **Paper Trading Mode** checked, and save.

**Step 5 — When ready for real money**
- Fund your Alpaca brokerage account (minimum $0 — fractional shares supported)
- Uncheck Paper Trading Mode
- Replace keys with your live trading keys
        """)

    paper_mode  = st.checkbox("Paper Trading Mode", value=profile["alpaca_paper"])
    api_key     = st.text_input("API Key ID", value=profile["alpaca_api_key"], type="password",
                                placeholder="PKXXXXXXXXXXXXXXXXXXXXXXXX")
    secret_key  = st.text_input("Secret Key", value=profile["alpaca_secret_key"], type="password",
                                placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    # Test connection
    if api_key and secret_key:
        if st.button("Test Connection"):
            with st.spinner("Connecting..."):
                acct = ac.get_account(api_key, secret_key, paper_mode)
            if acct:
                mode = "Paper" if acct.paper else "Live"
                st.success(f"Connected! {mode} account — Equity: ${acct.equity:,.2f} | Cash: ${acct.cash:,.2f}")
            else:
                st.error("Connection failed. Check your API keys and make sure the account is active.")

    st.divider()
    if st.button("Save Settings", type="primary"):
        profile["name"]               = name
        profile["starting_capital"]   = start_cap
        profile["risk_per_trade_pct"] = risk_pct
        profile["watchlist"]          = [t.strip().upper() for t in watchlist_raw.split(",") if t.strip()]
        profile["selected_categories"]= selected_cats
        profile["alpaca_paper"]       = paper_mode
        profile["alpaca_api_key"]     = api_key
        profile["alpaca_secret_key"]  = secret_key
        save_profile(profile)
        st.success("Settings saved!")
        st.rerun()

    st.divider()
    st.subheader("Danger Zone")
    if st.button("Reset All Manual Trades (keep capital)", type="secondary"):
        profile["trades"] = []
        profile["current_cash"] = profile["starting_capital"]
        save_profile(profile)
        st.success("Trades cleared.")
        st.rerun()
