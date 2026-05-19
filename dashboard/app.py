import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import yfinance as yf

from auth import is_logged_in, login_page, logout, hash_password
from strategy import (rank_stocks, backtest_momentum, monte_carlo, position_size,
                      ls_score, generate_pairs, backtest_long_short,
                      backtest_mean_reversion, backtest_sma_cross,
                      backtest_buy_hold, backtest_spy_benchmark, equity_stats,
                      fetch_prices)
from universe import (UNIVERSE, DEFAULT_CATEGORIES, tickers_for,
                      LS_SECTORS, INVERSE_ETFS, ls_tickers_for)
import alpaca_client as ac
import research as res
import ai_chat as ai

DATA_FILE = Path(__file__).parent / "data" / "profile.json"

# ── Profile I/O ───────────────────────────────────────────────────────────────

def load_profile() -> dict:
    with open(DATA_FILE) as f:
        return json.load(f)

def save_profile(p: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(p, f, indent=2)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def cached_price(ticker: str) -> float:
    try:
        h = yf.Ticker(ticker).history(period="5d")
        return float(h["Close"].iloc[-1]) if not h.empty else 0.0
    except Exception:
        return 0.0

def open_positions(p): return [t for t in p["trades"] if t.get("status") == "open"]
def closed_trades(p):  return [t for t in p["trades"] if t.get("status") == "closed"]

def local_portfolio_value(p):
    return p["current_cash"] + sum(cached_price(t["ticker"]) * t["shares"] for t in open_positions(p))

def total_pnl(p):
    pnl = sum(t.get("pnl", 0.0) for t in closed_trades(p))
    pnl += sum((cached_price(t["ticker"]) - t["entry_price"]) * t["shares"] for t in open_positions(p))
    return round(pnl, 2)

# ── Global page config ────────────────────────────────────────────────────────

st.set_page_config(page_title="Trading Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0a0e1a; border-right: 1px solid #1e2640; }
  .block-container { padding-top: 1.5rem; }
  .nav-section { color: #4a5580; font-size: 0.72rem; font-weight: 700;
                 letter-spacing: 0.12em; text-transform: uppercase;
                 padding: 1rem 0 0.3rem 0; }
  .kpi-val  { font-size: 1.6rem; font-weight: 800; color: #e8eaf6; font-family: monospace; }
  .kpi-lbl  { font-size: 0.72rem; color: #6b7db3; text-transform: uppercase; letter-spacing: .05em; }
  .up   { color: #00d4aa; font-weight: 700; }
  .down { color: #ff4b4b; font-weight: 700; }
  .chip { display:inline-block; padding:2px 9px; border-radius:20px;
          font-size:0.75rem; font-weight:700; }
  .chip-buy  { background:#003d2e; color:#00d4aa; }
  .chip-sell { background:#3d0000; color:#ff4b4b; }
  .chip-hold { background:#2e2a00; color:#ffd700; }
  .chip-paper{ background:#1a2a3a; color:#60aaff; }
  .chat-user { background:#1e2640; border-radius:12px 12px 2px 12px;
               padding:10px 14px; margin:4px 0; }
  .chat-ai   { background:#0d1524; border:1px solid #1e2640;
               border-radius:12px 12px 12px 2px; padding:10px 14px; margin:4px 0; }
  .divider   { border-top: 1px solid #1e2640; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────

profile = load_profile()

if not is_logged_in():
    login_page(profile)
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

NAV = {
    "PORTFOLIO":  ["Overview", "Trade Log", "Position Sizer"],
    "RESEARCH":   ["Company Search", "Market Research", "AI Assistant"],
    "STRATEGY":   ["Strategy Signals", "L/S Pairs", "Strategy Overview"],
    "BACKTESTING":["Backtesting", "Monte Carlo"],
    "SYSTEM":     ["Settings"],
}

with st.sidebar:
    st.markdown("### 📈 Trading Dashboard")
    st.caption(f"Welcome, {st.session_state.get('username','trader')}")
    st.divider()

    # Portfolio mini-card
    alpaca_acct = None
    if profile.get("alpaca_api_key"):
        alpaca_acct = ac.get_account(profile["alpaca_api_key"],
                                     profile["alpaca_secret_key"],
                                     profile["alpaca_paper"])
    port_val = alpaca_acct.portfolio_value if alpaca_acct else local_portfolio_value(profile)
    pnl      = total_pnl(profile)
    pnl_cls  = "up" if pnl >= 0 else "down"
    mode_tag = ('<span class="chip chip-paper">📄 Paper</span>'
                if (not alpaca_acct or alpaca_acct.paper)
                else '<span class="chip chip-buy">💰 Live</span>')
    st.markdown(
        f'<div class="kpi-lbl">Portfolio {mode_tag}</div>'
        f'<div class="kpi-val">${port_val:,.2f}</div>'
        f'<span class="{pnl_cls}">{"+" if pnl>=0 else ""}${pnl:.2f} P&L</span>',
        unsafe_allow_html=True
    )
    st.divider()

    # Navigation
    all_pages = [p for pages in NAV.values() for p in pages]
    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"

    for section, pages in NAV.items():
        st.markdown(f'<div class="nav-section">{section}</div>', unsafe_allow_html=True)
        for p in pages:
            active = st.session_state["page"] == p
            if st.button(p, key=f"nav_{p}",
                         type="primary" if active else "secondary",
                         use_container_width=True):
                st.session_state["page"] = p
                st.rerun()

    st.divider()
    if st.button("🚪 Sign Out", use_container_width=True):
        logout()

page = st.session_state["page"]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Overview ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Portfolio Overview")

    if alpaca_acct:
        st.success(f"Alpaca {'Paper' if alpaca_acct.paper else 'Live'} — connected")
        c1,c2,c3,c4 = st.columns(4)
        start    = profile["starting_capital"]
        ret_pct  = (alpaca_acct.portfolio_value - start) / start * 100
        c1.metric("Portfolio Value",  f"${alpaca_acct.portfolio_value:,.2f}")
        c2.metric("Total Return",     f"{ret_pct:.2f}%")
        c3.metric("Cash",             f"${alpaca_acct.cash:,.2f}")
        c4.metric("Buying Power",     f"${alpaca_acct.buying_power:,.2f}")
        positions = ac.get_positions(profile["alpaca_api_key"],
                                     profile["alpaca_secret_key"],
                                     profile["alpaca_paper"])
        if positions:
            st.subheader("Alpaca Positions")
            rows = [{"Ticker": p.ticker, "Qty": p.qty,
                     "Entry": f"${p.avg_entry:.2f}", "Current": f"${p.current_price:.2f}",
                     "Value": f"${p.market_value:.2f}",
                     "P&L": f"${p.unrealized_pnl:+.2f}", "P&L%": f"{p.unrealized_pnl_pct:+.2f}%"}
                    for p in positions]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        pnl      = total_pnl(profile)
        port_val = local_portfolio_value(profile)
        start    = profile["starting_capital"]
        ret_pct  = (port_val - start) / start * 100
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Portfolio Value", f"${port_val:,.2f}", f"${pnl:+.2f}")
        c2.metric("Total Return",    f"{ret_pct:.2f}%")
        c3.metric("Cash Available",  f"${profile['current_cash']:,.2f}")
        c4.metric("Open Positions",  len(open_positions(profile)))
        if not profile.get("alpaca_api_key"):
            st.info("Connect Alpaca in **Settings** to see live account data.")

    positions = open_positions(profile)
    if positions:
        st.divider()
        st.subheader("Open Positions")
        rows = []
        for pos in positions:
            price   = cached_price(pos["ticker"])
            pnl_p   = (price - pos["entry_price"]) * pos["shares"]
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
            rows.append({"Ticker": pos["ticker"], "Shares": pos["shares"],
                         "Entry": f"${pos['entry_price']:.2f}", "Current": f"${price:.2f}",
                         "P&L $": f"${pnl_p:+.2f}", "P&L %": f"{pnl_pct:+.2f}%",
                         "Value": f"${price*pos['shares']:.2f}", "Date": pos.get("date","—")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Watchlist")
    tickers = profile.get("watchlist", ["SPY","QQQ","AAPL","MSFT","NVDA"])
    with st.spinner("Fetching prices..."):
        rows = []
        for t in tickers:
            try:
                hist  = yf.Ticker(t).history(period="5d")
                price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
                prev  = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
                chg   = price - prev
                chg_p = chg / prev * 100 if prev else 0
                rows.append({"Ticker": t, "Price": f"${price:.2f}",
                             "Chg $": f"${chg:+.2f}", "Chg %": f"{chg_p:+.2f}%"})
            except Exception:
                rows.append({"Ticker": t, "Price": "—", "Chg $": "—", "Chg %": "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Company Search ────────────────────────────────────────────────────────────
elif page == "Company Search":
    st.title("Company Research")

    ticker_input = st.text_input("Search ticker", placeholder="AAPL, NVDA, MSFT...",
                                 help="Enter any US stock ticker").upper().strip()

    if not ticker_input:
        st.info("Enter a ticker symbol above to pull full company research.")
        st.stop()

    with st.spinner(f"Loading {ticker_input}..."):
        info     = res.get_company_info(ticker_input)
        hist_1y  = res.get_price_history(ticker_input, "1y")
        news     = res.get_news(ticker_input)

    if not info or not info.get("regularMarketPrice") and hist_1y.empty:
        st.error(f"Could not find data for **{ticker_input}**. Check the ticker symbol.")
        st.stop()

    name    = info.get("longName", ticker_input)
    sector  = info.get("sector", "N/A")
    industry= info.get("industry", "N/A")

    # Header
    st.markdown(f"## {name}")
    st.caption(f"{sector} › {industry}  ·  {ticker_input}")

    # Price chart timeframe selector
    tf_map = {"1W":"5d","1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y"}
    tf = st.radio("Timeframe", list(tf_map.keys()), index=4, horizontal=True)

    hist = res.get_price_history(ticker_input, tf_map[tf])
    if not hist.empty:
        price = float(hist["Close"].iloc[-1])
        prev  = float(hist["Close"].iloc[0])
        total_chg_pct = (price - prev) / prev * 100

        fig = go.Figure()
        color = "#00d4aa" if total_chg_pct >= 0 else "#ff4b4b"
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"],
            fill="tozeroy", fillcolor=f"rgba({'0,212,170' if total_chg_pct>=0 else '255,75,75'},0.08)",
            line=dict(color=color, width=2), name="Price",
        ))
        if "Volume" in hist.columns:
            fig.add_trace(go.Bar(
                x=hist.index, y=hist["Volume"],
                marker_color="rgba(100,120,200,0.3)",
                name="Volume", yaxis="y2",
            ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            title=f"{ticker_input} — {tf} ({total_chg_pct:+.2f}%)",
            yaxis_title="Price ($)", height=380,
            yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Key metrics
    tab_metrics, tab_financials, tab_news, tab_ratings = st.tabs(
        ["📊 Key Metrics", "📋 Financials", "📰 News", "⭐ Analyst Ratings"]
    )

    with tab_metrics:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Market Cap",     res.fmt_large(info.get("marketCap")))
        c2.metric("P/E (TTM)",      res.fmt_num(info.get("trailingPE")))
        c3.metric("Forward P/E",    res.fmt_num(info.get("forwardPE")))
        c4.metric("Price/Book",     res.fmt_num(info.get("priceToBook")))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("EV/EBITDA",      res.fmt_num(info.get("enterpriseToEbitda")))
        c2.metric("Profit Margin",  res.fmt_pct(info.get("profitMargins")))
        c3.metric("ROE",            res.fmt_pct(info.get("returnOnEquity")))
        c4.metric("Debt/Equity",    res.fmt_num(info.get("debtToEquity")))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Beta",           res.fmt_num(info.get("beta")))
        c2.metric("52W High",       f"${info.get('fiftyTwoWeekHigh',0):.2f}"
                                    if info.get("fiftyTwoWeekHigh") else "N/A")
        c3.metric("52W Low",        f"${info.get('fiftyTwoWeekLow',0):.2f}"
                                    if info.get("fiftyTwoWeekLow") else "N/A")
        c4.metric("Analyst Target", f"${info.get('targetMeanPrice',0):.2f}"
                                    if info.get("targetMeanPrice") else "N/A")

        st.divider()
        if info.get("longBusinessSummary"):
            st.subheader("Business Summary")
            st.write(info["longBusinessSummary"])

    with tab_financials:
        with st.spinner("Loading financial statements..."):
            fins = res.get_financials(ticker_input)

        def show_statement(df, title):
            if df is None or df.empty:
                st.info(f"No {title} data available.")
                return
            # Show last 4 periods
            df_show = df.iloc[:, :4]
            df_show.columns = [str(c)[:10] for c in df_show.columns]
            df_show = df_show.apply(pd.to_numeric, errors="coerce") / 1e9
            st.caption(f"*Figures in $B*")
            st.dataframe(df_show.style.format("{:.2f}", na_rep="—"),
                         use_container_width=True)

        st.subheader("Income Statement")
        show_statement(fins.get("income"), "income statement")
        st.subheader("Balance Sheet")
        show_statement(fins.get("balance"), "balance sheet")
        st.subheader("Cash Flow")
        show_statement(fins.get("cashflow"), "cash flow")

    with tab_news:
        if not news:
            st.info("No recent news found.")
        else:
            for item in news:
                title   = item.get("title", "")
                pub     = item.get("publisher", "")
                url     = item.get("link", "#")
                ts      = item.get("providerPublishTime", 0)
                date_str= datetime.fromtimestamp(ts).strftime("%b %d, %Y") if ts else ""
                st.markdown(f"**[{title}]({url})**  \n*{pub} · {date_str}*")
                st.divider()

    with tab_ratings:
        with st.spinner("Loading analyst ratings..."):
            recs = res.get_analyst_ratings(ticker_input)
        rec_mean = info.get("recommendationMean")
        n_analysts = info.get("numberOfAnalystOpinions")
        rec_key    = info.get("recommendationKey","").replace("_"," ").title()
        if rec_mean:
            c1,c2,c3 = st.columns(3)
            c1.metric("Consensus",      rec_key or "N/A")
            c2.metric("Mean Score",     f"{rec_mean:.1f} / 5.0")
            c3.metric("# Analysts",     str(n_analysts) if n_analysts else "N/A")
        if not recs.empty:
            st.subheader("Recent Recommendations")
            st.dataframe(recs, use_container_width=True)
        else:
            st.info("No analyst rating history available.")


# ── Market Research ───────────────────────────────────────────────────────────
elif page == "Market Research":
    st.title("Market Research")

    # Macro snapshot
    st.subheader("Macro Snapshot")
    with st.spinner("Loading macro data..."):
        macro = res.get_macro_snapshot()

    if macro:
        cols = st.columns(min(len(macro), 4))
        for i, (name, d) in enumerate(macro.items()):
            col = cols[i % 4]
            chg_cls = "up" if d["chg"] >= 0 else "down"
            col.metric(name, f"{d['price']:.2f}",
                       delta=f"{d['chg']:+.2f}%")
            if (i + 1) % 4 == 0 and i < len(macro) - 1:
                cols = st.columns(4)

    st.divider()

    # Sector performance heatmap
    st.subheader("Sector Performance")
    with st.spinner("Loading sector data..."):
        sect_df = res.get_sector_performance()

    if not sect_df.empty:
        # Heatmap
        heat_cols = ["1D %","1W %","1M %","3M %","6M %","1Y %"]
        heat_data = sect_df.set_index("Sector")[heat_cols]
        fig = px.imshow(
            heat_data,
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            text_auto=".1f", aspect="auto",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                          height=380, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sect_df, use_container_width=True, hide_index=True)

    st.divider()

    # Correlation matrix
    st.subheader("Asset Correlation Matrix")
    corr_tickers = ["SPY","QQQ","IWM","TLT","GLD","USO","^VIX","UUP"]
    st.caption(f"Tickers: {', '.join(corr_tickers)} — 1 year daily returns")
    with st.spinner("Computing correlations..."):
        corr = res.get_correlation_matrix(corr_tickers, "1y")

    if not corr.empty:
        fig2 = px.imshow(
            corr, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            text_auto=".2f", zmin=-1, zmax=1,
        )
        fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Top movers
    st.subheader("Top Movers Today")
    movers_tickers = ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA",
                      "JPM","BAC","GS","XOM","CVX","LLY","UNH","COST"]
    with st.spinner("Fetching movers..."):
        movers = []
        try:
            data = yf.download(movers_tickers, period="5d", auto_adjust=True, progress=False)
            closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
            for t in movers_tickers:
                if t in closes.columns and len(closes[t].dropna()) >= 2:
                    c = closes[t].dropna()
                    chg = (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100
                    movers.append({"Ticker": t, "Price": f"${c.iloc[-1]:.2f}",
                                   "1D %": round(chg, 2)})
        except Exception:
            pass
    if movers:
        mv_df = pd.DataFrame(movers).sort_values("1D %", ascending=False)
        fig3  = px.bar(mv_df, x="Ticker", y="1D %",
                       color="1D %", color_continuous_scale="RdYlGn",
                       color_continuous_midpoint=0)
        fig3.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                           plot_bgcolor="#0e1117", height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)


# ── AI Assistant ──────────────────────────────────────────────────────────────
elif page == "AI Assistant":
    st.title("AI Research Assistant")
    st.caption("Powered by Claude — ask about stocks, strategies, or your dashboard signals")

    api_key = profile.get("anthropic_api_key", "")
    client  = ai.get_client(api_key)

    if not client:
        st.warning(
            "**Anthropic API key required.** Go to **Settings → AI Assistant** to add yours.  \n"
            "Get a free key at [console.anthropic.com](https://console.anthropic.com)"
        )
        st.stop()

    ai.init_history()

    # Suggested prompts
    with st.expander("Suggested questions", expanded=not bool(st.session_state["chat_history"])):
        suggestions = [
            "Explain the Long/Short Equity strategy in simple terms",
            "What does a Sharpe ratio of 1.3 mean?",
            "What's the risk of shorting a stock?",
            "How do I interpret a Monte Carlo simulation?",
            "What is a P/E ratio and when is it useful?",
            "Explain momentum investing vs value investing",
        ]
        cols = st.columns(3)
        for i, s in enumerate(suggestions):
            if cols[i % 3].button(s, use_container_width=True):
                ai.add_message("user", s)
                with st.spinner("Thinking..."):
                    reply = ai.chat(client, st.session_state["chat_history"])
                ai.add_message("assistant", reply)
                st.rerun()

    # Chat history
    for msg in st.session_state["chat_history"]:
        css_class = "chat-user" if msg["role"] == "user" else "chat-ai"
        prefix    = "**You:** " if msg["role"] == "user" else "**Claude:** "
        st.markdown(f'<div class="{css_class}">{prefix}{msg["content"]}</div>',
                    unsafe_allow_html=True)

    # Input
    st.divider()
    c1, c2 = st.columns([5, 1])
    user_input = c1.text_input("Ask anything...", key="chat_input",
                               label_visibility="collapsed",
                               placeholder="e.g. Which sector is showing the strongest momentum right now?")
    if c2.button("Send", type="primary", use_container_width=True):
        if user_input.strip():
            ai.add_message("user", user_input.strip())
            with st.spinner("Thinking..."):
                reply = ai.chat(client, st.session_state["chat_history"])
            ai.add_message("assistant", reply)
            st.rerun()

    if st.session_state["chat_history"]:
        if st.button("Clear conversation", type="secondary"):
            ai.clear_history()
            st.rerun()


# ── Strategy Signals ──────────────────────────────────────────────────────────
elif page == "Strategy Signals":
    st.title("Strategy Signals")
    st.caption("Quality Momentum — 70% 12-1M momentum + 30% quality score")

    all_cats = list(UNIVERSE.keys())
    saved    = profile.get("selected_categories", DEFAULT_CATEGORIES)
    selected = st.multiselect("Universe categories", all_cats, default=saved)
    if not selected:
        st.warning("Select at least one category.")
        st.stop()

    tickers   = tickers_for(selected)
    top_n_show= st.slider("Show top N", 10, min(80, len(tickers)), 30)
    st.caption(f"Screening **{len(tickers)}** instruments across **{len(selected)}** categories")

    with st.spinner("Ranking stocks..."):
        df = rank_stocks(tickers)

    df_show = df.head(top_n_show)
    buy_c   = (df["Signal"] == "BUY").sum()
    hold_c  = (df["Signal"] == "HOLD").sum()
    avoid_c = (df["Signal"] == "AVOID").sum()
    c1,c2,c3 = st.columns(3)
    c1.metric("BUY signals",   buy_c)
    c2.metric("HOLD signals",  hold_c)
    c3.metric("AVOID signals", avoid_c)

    def color_sig(v):
        if v == "BUY":   return "background:#003d2e;color:#00d4aa"
        if v == "AVOID": return "background:#3d0000;color:#ff4b4b"
        return "background:#2e2a00;color:#ffd700"

    styled = df_show[["Signal","Combined Score","1M Return %","3M Return %","12M Return %","Quality Score"]].style.map(color_sig, subset=["Signal"])
    st.dataframe(styled, use_container_width=True)

    st.divider()
    df_r      = df_show.reset_index()
    ticker_col= df_r.columns[0]
    fig = px.bar(df_r, x=ticker_col, y="Combined Score", color="Signal",
                 color_discrete_map={"BUY":"#00d4aa","HOLD":"#ffd700","AVOID":"#ff4b4b"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      xaxis_tickangle=-45, height=350)
    st.plotly_chart(fig, use_container_width=True)

    ret_cols  = [c for c in ["1M Return %","3M Return %","12M Return %"] if c in df_show.columns]
    hmap_data = df_show[ret_cols].dropna(axis=1, how="all").T
    if not hmap_data.empty:
        fig2 = px.imshow(hmap_data, color_continuous_scale="RdYlGn",
                         color_continuous_midpoint=0, text_auto=".1f", aspect="auto")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", height=220)
        st.plotly_chart(fig2, use_container_width=True)

    if selected != profile.get("selected_categories"):
        profile["selected_categories"] = selected
        save_profile(profile)


# ── L/S Pairs ─────────────────────────────────────────────────────────────────
elif page == "L/S Pairs":
    st.title("Long / Short Pairs")
    st.caption("Sector-neutral: LONG the leader, SHORT the laggard — you only need to be right about the spread, not the market.")
    st.info("**Margin account required for real shorting.** Use the Inverse ETF column as a no-margin proxy.", icon="ℹ️")

    all_ls   = list(LS_SECTORS.keys())
    selected = st.multiselect("Sectors", all_ls, default=all_ls[:6])
    if not selected:
        st.warning("Select at least one sector.")
        st.stop()

    sector_map  = {s: LS_SECTORS[s] for s in selected}
    all_tickers = ls_tickers_for(selected)

    with st.spinner(f"Scoring {len(all_tickers)} stocks..."):
        prices = fetch_prices(all_tickers, period="2y")
        prices.index = pd.DatetimeIndex(prices.index)
        pairs  = generate_pairs(sector_map, prices)

    if not pairs:
        st.warning("Not enough data."); st.stop()

    avg_sp3m  = np.mean([p["spread_3m"] for p in pairs])
    pos_pairs = sum(1 for p in pairs if p["spread_3m"] > 0)
    c1,c2,c3 = st.columns(3)
    c1.metric("Pairs", len(pairs))
    c2.metric("Positive Spread (3M)", f"{pos_pairs}/{len(pairs)}")
    c3.metric("Avg 3M Spread", f"{avg_sp3m:+.1f}%")

    st.divider()
    table = []
    for p in pairs:
        inv  = INVERSE_ETFS.get(p["sector"],"—")
        icon = "🟢" if p["spread_3m"] > 5 else ("🟡" if p["spread_3m"] > 0 else "🔴")
        table.append({"Sector": p["sector"], "LONG": p["long"], "SHORT": p["short"],
                      "Inverse ETF": inv,
                      "Long Score": round(p["long_score"],3),
                      "Short Score": round(p["short_score"],3),
                      "1M Spread": f"{p['spread_1m']:+.1f}%",
                      "3M Spread": f"{icon} {p['spread_3m']:+.1f}%",
                      "Long 12M": f"{p['long_12m']:+.0f}%",
                      "Short 12M": f"{p['short_12m']:+.0f}%"})
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Pair Deep-Dive")
    labels = [f"{p['long']} vs {p['short']}  ({p['sector']})" for p in pairs]
    sel    = st.selectbox("Select pair", labels)
    pr     = pairs[labels.index(sel)]

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"### 🟢 LONG — {pr['long']}")
        st.metric("1M", f"{pr['long_1m']:+.1f}%")
        st.metric("3M", f"{pr['long_3m']:+.1f}%")
        st.metric("12M",f"{pr['long_12m']:+.1f}%")
        st.metric("Score", round(pr["long_score"],3))
    with c2:
        st.markdown(f"### 🔴 SHORT — {pr['short']}")
        st.metric("1M", f"{pr['short_1m']:+.1f}%")
        st.metric("3M", f"{pr['short_3m']:+.1f}%")
        st.metric("12M",f"{pr['short_12m']:+.1f}%")
        st.metric("Score", round(pr["short_score"],3))

    curve = pr["spread_curve"]
    if not curve.empty:
        color = "#00d4aa" if curve.iloc[-1] >= 100 else "#ff4b4b"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.index, y=curve, fill="tozeroy",
                                 fillcolor=f"rgba({'0,212,170' if color=='#00d4aa' else '255,75,75'},0.1)",
                                 line=dict(color=color, width=2)))
        fig.add_hline(y=100, line_dash="dash", line_color="gray")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117", height=300, title="Spread Curve (Long − Short)")
        st.plotly_chart(fig, use_container_width=True)

    if pr["long"] in prices.columns and pr["short"] in prices.columns:
        norm = prices[[pr["long"], pr["short"]]].dropna()
        norm = norm / norm.iloc[0] * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=norm.index, y=norm[pr["long"]],
                                  line=dict(color="#00d4aa",width=2), name=f"LONG {pr['long']}"))
        fig2.add_trace(go.Scatter(x=norm.index, y=norm[pr["short"]],
                                  line=dict(color="#ff4b4b",width=2), name=f"SHORT {pr['short']}"))
        fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                           plot_bgcolor="#0e1117", height=300, title="Price History (Normalised to 100)")
        st.plotly_chart(fig2, use_container_width=True)

    inv_etf = INVERSE_ETFS.get(pr["sector"])
    tab1,tab2 = st.tabs(["Option A — Margin Account","Option B — No Margin (ETF proxy)"])
    with tab1:
        st.markdown(f"""
| Leg | Action | Ticker | Allocation |
|---|---|---|---|
| Long  | BUY          | **{pr['long']}**  | $125 (50%) |
| Short | SELL SHORT   | **{pr['short']}** | $125 (50%) |

**Stop:** Close short if it moves >15% against you.
        """)
    with tab2:
        if inv_etf:
            st.markdown(f"""
| Leg | Action | Ticker | Allocation |
|---|---|---|---|
| Long        | BUY | **{pr['long']}** | $175 (70%) |
| Short proxy | BUY | **{inv_etf}** *(inverse ETF)* | $75 (30%) |
            """)
        else:
            st.info("No inverse ETF for this sector. Use SH (Short S&P 500) as broad hedge.")


# ── Strategy Overview ─────────────────────────────────────────────────────────
elif page == "Strategy Overview":
    st.title("Strategy Overview")
    st.caption("Live comparison of all available strategies — same capital, same universe")

    capital  = profile["starting_capital"]
    all_cats = list(UNIVERSE.keys())
    saved    = profile.get("selected_categories", DEFAULT_CATEGORIES)
    selected = st.multiselect("Universe", all_cats, default=saved, key="so_cats")
    if not selected: st.stop()
    tickers  = tickers_for(selected)

    with st.spinner("Running all strategies for comparison..."):
        results = {}
        mom_eq = backtest_momentum(tickers, top_n=5)
        if not mom_eq.empty:
            results["Momentum (Top 5)"] = equity_stats(mom_eq, capital)
        bh_r = backtest_buy_hold(tickers)
        if not bh_r["equity"].empty:
            results["Buy & Hold"] = equity_stats(bh_r["equity"], capital)
        spy_r = backtest_spy_benchmark()
        if not spy_r["equity"].empty:
            results["SPY Benchmark"] = equity_stats(spy_r["equity"], capital)
        mr_r = backtest_mean_reversion(tickers[:20])
        if not mr_r["equity"].empty:
            results["Mean Reversion (RSI)"] = equity_stats(mr_r["equity"], capital)
        sma_r = backtest_sma_cross(tickers[:20])
        if not sma_r["equity"].empty:
            results["SMA Cross (50/200)"] = equity_stats(sma_r["equity"], capital)

    if not results:
        st.warning("Not enough data."); st.stop()

    # Stats comparison table
    comp_rows = []
    for name, s in results.items():
        comp_rows.append({"Strategy": name,
                          "Total Return": f"{s['total_ret']:+.1f}%",
                          "Ann. Return":  f"{s['ann_ret']:+.1f}%",
                          "Sharpe":       f"{s['sharpe']:.2f}",
                          "Max DD":       f"{s['max_dd']:.1f}%",
                          "Win Rate":     f"{s['win_rate']:.1f}%",
                          f"${capital:.0f}→": f"${s['final']:,.2f}"})
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Equity curves overlay
    colors = ["#00d4aa","#60aaff","#ffd700","#ff9900","#ff4b4b","#cc77ff"]
    fig    = go.Figure()
    for i, (name, s) in enumerate(results.items()):
        eq     = s["equity"]
        scaled = eq / eq.iloc[0] * capital
        fig.add_trace(go.Scatter(x=scaled.index, y=scaled,
                                 line=dict(color=colors[i % len(colors)], width=2),
                                 name=name))
    fig.add_hline(y=capital, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      title="All Strategies — Equity Curve Overlay",
                      yaxis_title="Portfolio Value ($)", height=460)
    st.plotly_chart(fig, use_container_width=True)

    # Best strategy callout
    best = max(results.items(), key=lambda x: x[1]["sharpe"])
    st.success(f"**Best risk-adjusted strategy:** {best[0]}  —  Sharpe {best[1]['sharpe']:.2f}, "
               f"Ann. Return {best[1]['ann_ret']:.1f}%")


# ── Backtesting ───────────────────────────────────────────────────────────────
elif page == "Backtesting":
    st.title("Backtesting")

    strategy_mode = st.radio("Strategy", [
        "Momentum (Long Only)",
        "Long/Short (Market Neutral)",
        "Mean Reversion (RSI)",
        "SMA Golden Cross",
        "Multi-Strategy Comparison",
    ], horizontal=False)

    capital  = profile["starting_capital"]
    all_cats = list(UNIVERSE.keys())
    saved    = profile.get("selected_categories", DEFAULT_CATEGORIES)

    def show_equity(eq_series, title, capital, color="#00d4aa", compare=None):
        s    = equity_stats(eq_series, capital)
        if not s: return
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Return", f"{s['total_ret']:+.1f}%")
        c2.metric("Ann. Return",  f"{s['ann_ret']:+.1f}%")
        c3.metric("Sharpe",       f"{s['sharpe']:.2f}")
        c4.metric("Max Drawdown", f"{s['max_dd']:.1f}%")
        c5.metric(f"${capital:.0f}→", f"${s['final']:,.2f}")

        scaled = eq_series / eq_series.iloc[0] * capital
        fig    = go.Figure()
        fig.add_trace(go.Scatter(x=scaled.index, y=scaled,
                                 fill="tozeroy", fillcolor=f"rgba({'0,212,170' if color=='#00d4aa' else '255,75,75'},0.08)",
                                 line=dict(color=color, width=2), name=title))
        if compare:
            for c_name, c_eq in compare.items():
                c_sc = c_eq / c_eq.iloc[0] * capital
                fig.add_trace(go.Scatter(x=c_sc.index, y=c_sc,
                                         line=dict(width=1.5, dash="dot"), name=c_name))
        fig.add_hline(y=capital, line_dash="dash", line_color="gray")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117", title=title,
                          yaxis_title="Portfolio Value ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        dd   = ((eq_series / eq_series.cummax()) - 1) * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                                  fillcolor="rgba(255,75,75,0.1)",
                                  line=dict(color="#ff4b4b", width=1.5)))
        fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                           plot_bgcolor="#0e1117", yaxis_title="Drawdown %", height=240)
        st.plotly_chart(fig2, use_container_width=True)

    if strategy_mode == "Momentum (Long Only)":
        col_a,col_b = st.columns([3,1])
        cats   = col_a.multiselect("Universe", all_cats, default=saved)
        top_n  = col_b.number_input("Hold top N", 1, 20, 5)
        if not cats: st.stop()
        tickers = tickers_for(cats)
        st.caption(f"{len(tickers)} instruments")
        with st.spinner("Running backtest..."):
            eq = backtest_momentum(tickers, top_n=int(top_n))
        if eq.empty: st.warning("Not enough data."); st.stop()
        # benchmark comparison
        spy_eq = backtest_spy_benchmark()["equity"]
        show_equity(eq.dropna(), f"Momentum Top {top_n}", capital,
                    compare={"SPY": spy_eq})

    elif strategy_mode == "Long/Short (Market Neutral)":
        col_a,col_b = st.columns([3,1])
        sects  = col_a.multiselect("Sectors", list(LS_SECTORS.keys()), default=list(LS_SECTORS.keys())[:6])
        top_n  = col_b.number_input("N per side", 1, 5, 2)
        if not sects: st.stop()
        sector_map = {s: LS_SECTORS[s] for s in sects}
        st.caption(f"{len(ls_tickers_for(sects))} stocks across {len(sects)} sectors")
        with st.spinner("Running L/S backtest..."):
            res_ls = backtest_long_short(sector_map, top_n=int(top_n))
        ls_eq   = res_ls["ls"]
        long_eq = res_ls["long_only"]
        if ls_eq.empty: st.warning("Not enough data."); st.stop()
        st.subheader("Long/Short (Market Neutral) vs Long Only")
        show_equity(ls_eq.dropna(), "Long/Short", capital,
                    compare={"Long Only": long_eq, "SPY": backtest_spy_benchmark()["equity"]})

    elif strategy_mode == "Mean Reversion (RSI)":
        col_a,col_b,col_c = st.columns(3)
        cats      = col_a.multiselect("Universe", all_cats, default=saved, key="mr_cats")
        rsi_buy   = col_b.slider("RSI Buy threshold", 20, 40, 30)
        rsi_sell  = col_c.slider("RSI Sell threshold", 60, 80, 70)
        if not cats: st.stop()
        tickers   = tickers_for(cats)[:25]
        st.caption(f"Scanning {len(tickers)} stocks (top 25 of universe)")
        with st.spinner("Running mean reversion backtest..."):
            mr = backtest_mean_reversion(tickers, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
        if mr["equity"].empty: st.warning("Not enough data."); st.stop()
        show_equity(mr["equity"], f"Mean Reversion RSI({rsi_buy}/{rsi_sell})", capital,
                    compare={"SPY": backtest_spy_benchmark()["equity"]})

    elif strategy_mode == "SMA Golden Cross":
        col_a,col_b,col_c = st.columns(3)
        cats  = col_a.multiselect("Universe", all_cats, default=saved, key="sma_cats")
        fast  = col_b.selectbox("Fast SMA", [20,50,100], index=1)
        slow  = col_c.selectbox("Slow SMA", [100,150,200], index=2)
        if not cats: st.stop()
        tickers = tickers_for(cats)[:25]
        st.caption(f"Scanning {len(tickers)} stocks (top 25)")
        with st.spinner("Running SMA cross backtest..."):
            sma = backtest_sma_cross(tickers, fast=fast, slow=slow)
        if sma["equity"].empty: st.warning("Not enough data."); st.stop()
        show_equity(sma["equity"], f"SMA Cross ({fast}/{slow})", capital,
                    compare={"SPY": backtest_spy_benchmark()["equity"]})

    elif strategy_mode == "Multi-Strategy Comparison":
        cats    = st.multiselect("Universe", all_cats, default=saved, key="mc_cats")
        if not cats: st.stop()
        tickers = tickers_for(cats)
        st.caption(f"Comparing all strategies on {len(tickers)} instruments")
        with st.spinner("Running all strategies..."):
            equities = {}
            eq = backtest_momentum(tickers, top_n=5)
            if not eq.empty: equities["Momentum"]       = eq.dropna()
            bh = backtest_buy_hold(tickers)
            if not bh["equity"].empty: equities["Buy & Hold"]  = bh["equity"]
            spy= backtest_spy_benchmark()
            if not spy["equity"].empty: equities["SPY"]         = spy["equity"]
            mr = backtest_mean_reversion(tickers[:20])
            if not mr["equity"].empty: equities["Mean Reversion"]= mr["equity"]
            sm = backtest_sma_cross(tickers[:20])
            if not sm["equity"].empty: equities["SMA Cross"]    = sm["equity"]

        colors = ["#00d4aa","#60aaff","#ffd700","#ff9900","#cc77ff"]
        fig    = go.Figure()
        for i,(name,eq) in enumerate(equities.items()):
            sc = eq / eq.iloc[0] * capital
            fig.add_trace(go.Scatter(x=sc.index, y=sc,
                                     line=dict(color=colors[i%len(colors)],width=2), name=name))
        fig.add_hline(y=capital, line_dash="dash", line_color="gray")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117", title="Strategy Comparison",
                          yaxis_title="Portfolio Value ($)", height=460)
        st.plotly_chart(fig, use_container_width=True)

        rows = []
        for name,eq in equities.items():
            s = equity_stats(eq, capital)
            rows.append({"Strategy": name,
                         "Total Ret": f"{s['total_ret']:+.1f}%",
                         "Ann. Ret":  f"{s['ann_ret']:+.1f}%",
                         "Sharpe":    f"{s['sharpe']:.2f}",
                         "Max DD":    f"{s['max_dd']:.1f}%",
                         f"${capital:.0f}→": f"${s['final']:,.2f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Monte Carlo ───────────────────────────────────────────────────────────────
elif page == "Monte Carlo":
    st.title("Monte Carlo Simulation")
    st.caption("Probabilistic projection seeded from 5-year backtest statistics")

    all_cats = list(UNIVERSE.keys())
    saved    = profile.get("selected_categories", DEFAULT_CATEGORIES)
    c_left,c_right = st.columns([3,1])
    selected = c_left.multiselect("Universe", all_cats, default=saved)
    top_n    = c_right.number_input("Hold top N", 1, 20, 5)
    if not selected: st.stop()
    tickers  = tickers_for(selected)

    c1,c2,c3 = st.columns(3)
    capital  = c1.number_input("Starting Capital ($)", value=float(profile["starting_capital"]), step=10.0)
    n_sims   = c2.slider("Simulations", 500, 5000, 2000, step=500)
    days     = c3.slider("Days Forward", 63, 756, 252, step=63)

    with st.spinner("Running backtest then Monte Carlo..."):
        equity = backtest_momentum(tickers, top_n=int(top_n))
        if equity.empty: st.warning("Not enough data."); st.stop()
        results = monte_carlo(equity, n_simulations=n_sims, days_forward=days, capital=capital)

    ann_vol = results["sigma"] * np.sqrt(252) * 100
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Pessimistic (P10)", f"${results['p10']:,.2f}")
    c2.metric("Median (P50)",      f"${results['p50']:,.2f}")
    c3.metric("Optimistic (P90)", f"${results['p90']:,.2f}")
    c4.metric("Prob. of Profit",  f"{results['prob_profit']:.1f}%")
    c5.metric("Ann. Volatility",  f"{ann_vol:.1f}%")

    sims = results["simulations"]
    x    = list(range(days))
    fig  = go.Figure()
    show_every = max(1, n_sims // 150)
    for i in range(0, n_sims, show_every):
        fig.add_trace(go.Scatter(x=x, y=sims[:,i],
                                 line=dict(width=0.3, color="rgba(0,212,170,0.1)"),
                                 showlegend=False, hoverinfo="skip"))
    band_c = {10:"#ff4b4b", 25:"#ff9900", 50:"#ffffff", 75:"#60aaff", 90:"#00d4aa"}
    band_n = {10:"P10", 25:"P25", 50:"P50 Median", 75:"P75", 90:"P90"}
    for pct in [10,25,50,75,90]:
        vals = np.percentile(sims, pct, axis=1)
        fig.add_trace(go.Scatter(x=x, y=vals,
                                 line=dict(width=2, color=band_c[pct]), name=band_n[pct]))
    fig.add_hline(y=capital, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      title=f"Monte Carlo — {n_sims:,} Simulations × {days} Days",
                      yaxis_title="Portfolio Value ($)", height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final Value Distribution")
    final_vals = sims[-1,:]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=final_vals, nbinsx=80,
                                marker_color="#00d4aa", opacity=0.7))
    for pct, col, label in [(10,"#ff4b4b","P10"),(50,"#ffd700","P50"),(90,"#00d4aa","P90")]:
        fig2.add_vline(x=np.percentile(final_vals,pct), line_dash="dot",
                       line_color=col, annotation_text=label)
    fig2.add_vline(x=capital, line_dash="dash", line_color="white", annotation_text="Start")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                       plot_bgcolor="#0e1117", xaxis_title="Final Value ($)", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    rows = [{"Percentile": f"P{p}",
             "Final Value": f"${np.percentile(final_vals,p):,.2f}",
             "Return": f"{(np.percentile(final_vals,p)-capital)/capital*100:+.1f}%",
             "Multiple": f"{np.percentile(final_vals,p)/capital:.2f}x"}
            for p in [5,10,25,50,75,90,95]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Trade Log ─────────────────────────────────────────────────────────────────
elif page == "Trade Log":
    st.title("Trade Log")
    profile = load_profile()
    alpaca_ok = bool(profile.get("alpaca_api_key"))
    if alpaca_ok:
        st.success("Alpaca connected — orders can be placed directly.")
    else:
        st.info("Alpaca not connected. Trades logged manually. Connect in **Settings**.")

    with st.expander("New Trade", expanded=not bool(profile["trades"])):
        c1,c2,c3 = st.columns(3)
        ticker   = c1.text_input("Ticker").upper().strip()
        action   = c2.selectbox("Action", ["BUY","SELL"])
        otype    = c3.selectbox("Order Type", ["Dollar Amount","Shares"])
        if otype == "Dollar Amount":
            dollar_amt = st.number_input("$ Amount", min_value=1.0, step=1.0, value=50.0)
            shares_in  = None
        else:
            shares_in  = st.number_input("Shares", min_value=0.0001, step=0.0001, format="%.4f")
            dollar_amt = None
        price_in = st.number_input("Price (0 = fetch live)", min_value=0.0, step=0.01)
        note     = st.text_input("Note")

        col_log, col_alp = st.columns(2)
        if col_log.button("Log Manually", type="secondary"):
            if not ticker:
                st.warning("Enter a ticker.")
            else:
                price  = price_in if price_in > 0 else cached_price(ticker)
                shares = shares_in if shares_in else (round(dollar_amt / price, 4) if price else 0)
                val    = shares * price
                if action == "BUY":
                    if val > profile["current_cash"]:
                        st.error(f"Insufficient cash. Available: ${profile['current_cash']:.2f}")
                    else:
                        profile["current_cash"] -= val
                        profile["trades"].append({
                            "id": len(profile["trades"]) + 1, "ticker": ticker,
                            "action": action, "shares": shares, "entry_price": price,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "status": "open", "note": note, "source": "manual"
                        })
                        save_profile(profile); st.success(f"Logged BUY {shares:.4f} {ticker}"); st.rerun()
                else:
                    ops = [t for t in profile["trades"] if t.get("status")=="open" and t["ticker"]==ticker]
                    if not ops:
                        st.error(f"No open position for {ticker}")
                    else:
                        pos = ops[0]
                        pnl_t = (price - pos["entry_price"]) * shares
                        pos.update({"status":"closed","exit_price":price,
                                    "exit_date":datetime.now().strftime("%Y-%m-%d"),"pnl":round(pnl_t,2)})
                        profile["current_cash"] += shares * price
                        save_profile(profile); st.success(f"Closed {ticker} P&L: ${pnl_t:+.2f}"); st.rerun()

        if alpaca_ok:
            if col_alp.button("Place via Alpaca", type="primary"):
                if not ticker:
                    st.warning("Enter a ticker.")
                else:
                    with st.spinner("Sending to Alpaca..."):
                        if otype == "Dollar Amount":
                            r = ac.place_market_order(profile["alpaca_api_key"],
                                                      profile["alpaca_secret_key"],
                                                      profile["alpaca_paper"],
                                                      ticker, 0, action.lower(), notional=dollar_amt)
                        else:
                            r = ac.place_market_order(profile["alpaca_api_key"],
                                                      profile["alpaca_secret_key"],
                                                      profile["alpaca_paper"],
                                                      ticker, shares_in, action.lower())
                    if r["success"]:
                        st.success(f"Order placed! ID: {r['order_id']}")
                        price  = price_in if price_in > 0 else cached_price(ticker)
                        shares = shares_in if shares_in else (round(dollar_amt / price, 4) if price else 0)
                        profile["trades"].append({
                            "id": len(profile["trades"]) + 1, "ticker": ticker,
                            "action": action, "shares": shares, "entry_price": price,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "status": "open", "note": note,
                            "source": "alpaca", "alpaca_order_id": r["order_id"]
                        })
                        save_profile(profile); st.rerun()
                    else:
                        st.error(f"Order failed: {r['error']}")

    st.divider()
    trades = profile["trades"]
    if trades:
        df_t  = pd.DataFrame(trades)
        open_df   = df_t[df_t["status"] == "open"]
        closed_df = df_t[df_t["status"] == "closed"]
        st.subheader(f"Open Positions ({len(open_df)})")
        if not open_df.empty:
            st.dataframe(open_df, use_container_width=True, hide_index=True)
        st.subheader(f"Closed Trades ({len(closed_df)})")
        if not closed_df.empty:
            st.dataframe(closed_df, use_container_width=True, hide_index=True)
            if "pnl" in closed_df.columns:
                st.metric("Total Realised P&L", f"${closed_df['pnl'].sum():+.2f}")
    else:
        st.info("No trades logged yet.")


# ── Position Sizer ────────────────────────────────────────────────────────────
elif page == "Position Sizer":
    st.title("Position Sizer")
    st.caption("Risk-based sizing — never risk more than your set % per trade")

    profile   = load_profile()
    capital   = st.number_input("Available Capital ($)", value=float(profile["current_cash"]), step=1.0)
    risk_pct  = st.slider("Risk per Trade (%)", 0.5, 5.0, float(profile["risk_per_trade_pct"]), step=0.5)
    ticker_ps = st.text_input("Ticker", "AAPL").upper()

    c1,c2 = st.columns(2)
    entry = c1.number_input("Entry Price ($)", value=0.0, step=0.01)
    stop  = c2.number_input("Stop Loss Price ($)", value=0.0, step=0.01)

    if ticker_ps and entry == 0.0:
        with st.spinner("Fetching price..."):
            live = cached_price(ticker_ps)
        if live:
            st.info(f"Live price for **{ticker_ps}**: ${live:.2f}")
            entry = live

    if entry > 0 and stop > 0 and entry != stop:
        sz = position_size(capital, risk_pct, entry, stop)
        st.divider()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Shares to Buy",  f"{sz['shares']:.4f}")
        c2.metric("Capital at Risk",f"${sz['risk_dollars']:.2f}")
        c3.metric("Position Value", f"${sz['position_value']:.2f}")
        c4.metric("% of Capital",   f"{sz['position_value']/capital*100:.1f}%")

        if sz["position_value"] > capital:
            st.error("Position value exceeds available capital.")
        elif sz["position_value"] / capital > 0.5:
            st.warning("Over 50% of capital in one trade — consider a tighter stop.")
        st.caption(f"Max $ risk at {risk_pct}%: **${capital*risk_pct/100:.2f}**  |  "
                   f"Stop distance: **${abs(entry-stop):.2f}** ({abs(entry-stop)/entry*100:.1f}%)")
    else:
        st.info("Enter entry price and stop loss to calculate.")


# ── Settings ──────────────────────────────────────────────────────────────────
elif page == "Settings":
    st.title("Settings")
    profile = load_profile()

    tab_profile, tab_trading, tab_alpaca, tab_ai, tab_security = st.tabs(
        ["Profile", "Trading", "Alpaca API", "AI Assistant", "Security"]
    )

    with tab_profile:
        st.subheader("Profile")
        name      = st.text_input("Dashboard Name", profile["name"])
        start_cap = st.number_input("Starting Capital ($)", value=float(profile["starting_capital"]), step=10.0)

        st.subheader("Personal Watchlist")
        watchlist_raw = st.text_input("Tickers (comma separated)", ", ".join(profile.get("watchlist",[])))

        st.subheader("Default Universe Categories")
        all_cats = list(UNIVERSE.keys())
        saved    = profile.get("selected_categories", DEFAULT_CATEGORIES)
        sel_cats = st.multiselect("Default categories", all_cats, default=saved)

        if st.button("Save Profile", type="primary"):
            profile["name"]               = name
            profile["starting_capital"]   = start_cap
            profile["watchlist"]          = [t.strip().upper() for t in watchlist_raw.split(",") if t.strip()]
            profile["selected_categories"]= sel_cats
            save_profile(profile)
            st.success("Saved!")
            st.rerun()

    with tab_trading:
        st.subheader("Risk Management")
        risk_pct = st.slider("Default Risk per Trade (%)", 0.5, 5.0,
                             float(profile["risk_per_trade_pct"]), step=0.5)
        if st.button("Save Trading Settings", type="primary"):
            profile["risk_per_trade_pct"] = risk_pct
            save_profile(profile)
            st.success("Saved!")
        st.divider()
        if st.button("Reset All Manual Trades", type="secondary"):
            profile["trades"]       = []
            profile["current_cash"] = profile["starting_capital"]
            save_profile(profile)
            st.success("Trades cleared.")
            st.rerun()

    with tab_alpaca:
        st.subheader("🦙 Alpaca API")
        with st.expander("Setup guide (2 minutes)", expanded=not bool(profile.get("alpaca_api_key"))):
            st.markdown("""
**Step 1** — Sign up at [alpaca.markets](https://alpaca.markets) (free, no credit card)
**Step 2** — Switch to **Paper Trading** mode (top-left toggle in their dashboard)
**Step 3** — Click your name → **API Keys** → **Generate New Key**
**Step 4** — Copy both keys *(secret shown only once)*
**Step 5** — Paste below and click **Test Connection**
**Step 6** — When ready for real money: fund account → uncheck Paper Mode → use live keys
            """)
        paper_mode = st.checkbox("Paper Trading Mode", value=profile.get("alpaca_paper", True))
        api_key    = st.text_input("API Key ID", value=profile.get("alpaca_api_key",""),
                                   type="password", placeholder="PKXXXXXXXXXXXXXXXXXXXXXXXX")
        secret_key = st.text_input("Secret Key", value=profile.get("alpaca_secret_key",""),
                                   type="password", placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        if api_key and secret_key:
            if st.button("Test Connection"):
                with st.spinner("Connecting..."):
                    acct = ac.get_account(api_key, secret_key, paper_mode)
                if acct:
                    st.success(f"Connected! {'Paper' if acct.paper else 'Live'} — "
                               f"Equity: ${acct.equity:,.2f} | Cash: ${acct.cash:,.2f}")
                else:
                    st.error("Connection failed. Check your keys.")
        if st.button("Save Alpaca Settings", type="primary"):
            profile["alpaca_paper"]      = paper_mode
            profile["alpaca_api_key"]    = api_key
            profile["alpaca_secret_key"] = secret_key
            save_profile(profile)
            st.success("Saved!")

    with tab_ai:
        st.subheader("🤖 AI Assistant")
        with st.expander("How to get a free Anthropic API key", expanded=not bool(profile.get("anthropic_api_key"))):
            st.markdown("""
**Step 1** — Go to [console.anthropic.com](https://console.anthropic.com)
**Step 2** — Sign up with your email
**Step 3** — Click **API Keys** → **Create Key**
**Step 4** — Copy the key and paste it below

New accounts get **$5 free credits** — enough for thousands of chat messages.
The dashboard uses Claude Haiku (fast + cheap — ~$0.001 per message).
            """)
        ai_key = st.text_input("Anthropic API Key", value=profile.get("anthropic_api_key",""),
                               type="password", placeholder="sk-ant-...")
        if ai_key:
            if st.button("Test AI Connection"):
                with st.spinner("Testing..."):
                    client = ai.get_client(ai_key)
                    reply  = ai.chat(client, [{"role":"user","content":"Say 'connected' and nothing else."}])
                if "connected" in reply.lower():
                    st.success("AI Assistant connected!")
                else:
                    st.error(f"Connection issue: {reply}")
        if st.button("Save AI Settings", type="primary"):
            profile["anthropic_api_key"] = ai_key
            save_profile(profile)
            st.success("Saved!")

    with tab_security:
        st.subheader("Login Credentials")
        current_user = st.text_input("Username", value=profile.get("username","trader"))
        new_pass     = st.text_input("New Password", type="password",
                                     placeholder="Leave blank to keep current")
        conf_pass    = st.text_input("Confirm Password", type="password")
        if st.button("Update Credentials", type="primary"):
            if new_pass and new_pass != conf_pass:
                st.error("Passwords don't match.")
            else:
                profile["username"] = current_user
                if new_pass:
                    profile["password_hash"] = hash_password(new_pass)
                save_profile(profile)
                st.success("Credentials updated! You'll need to log in again.")
                logout()
