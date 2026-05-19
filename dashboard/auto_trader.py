"""
Auto-trader: runs the Momentum strategy and rebalances via Alpaca.
Designed to be called from GitHub Actions on a schedule.
Reads credentials from environment variables (GitHub Secrets).
"""
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_FILE  = Path(__file__).parent / "data" / "profile.json"
LOG_FILE   = Path(__file__).parent / "data" / "auto_trade_log.json"

TOP_N          = 5      # number of stocks to hold at once
MIN_ORDER_USD  = 1.0    # Alpaca minimum notional
RESERVE_PCT    = 0.02   # keep 2% cash as buffer


# ── helpers ───────────────────────────────────────────────────────────────────

def load_profile() -> dict:
    with open(DATA_FILE) as f:
        return json.load(f)


def load_log() -> list:
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def save_log(entries: list):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def is_market_open() -> bool:
    """Check via Alpaca clock endpoint."""
    try:
        from alpaca.trading.client import TradingClient
        key    = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        paper  = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        client = TradingClient(key, secret, paper=paper)
        clock  = client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        log.warning(f"Could not check market clock: {e}")
        return False


def get_momentum_top(n: int) -> list[str]:
    """Return top-N tickers by 12-1 month momentum from a broad universe."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from universe import UNIVERSE, DEFAULT_CATEGORIES

    tickers = []
    for cat in DEFAULT_CATEGORIES:
        tickers.extend(UNIVERSE.get(cat, []))
    tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    log.info(f"Fetching prices for {len(tickers)} tickers …")
    data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.ffill().dropna(axis=1, how="all")

    if prices.shape[0] < 252:
        log.warning("Not enough history for momentum — aborting.")
        return []

    ret_12m = prices.pct_change(252).iloc[-1]
    ret_1m  = prices.pct_change(21).iloc[-1]
    scores  = (ret_12m - ret_1m).dropna().sort_values(ascending=False)
    top     = scores.head(n).index.tolist()
    log.info(f"Top {n} momentum: {top}")
    return top


def get_current_positions(client) -> dict[str, float]:
    """Return {ticker: market_value} for all open positions."""
    try:
        positions = client.get_all_positions()
        return {str(p.symbol): float(p.market_value) for p in positions}
    except Exception as e:
        log.error(f"get_positions error: {e}")
        return {}


def close_position(client, ticker: str, paper: bool) -> dict:
    """Close entire position for a ticker."""
    try:
        from alpaca.trading.requests import ClosePositionRequest
        client.close_position(ticker)
        log.info(f"SELL all {ticker}")
        return {"success": True, "error": ""}
    except Exception as e:
        log.error(f"close_position {ticker}: {e}")
        return {"success": False, "error": str(e)}


def buy_notional(client, ticker: str, dollars: float) -> dict:
    """Buy $dollars worth of ticker at market."""
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        req = MarketOrderRequest(
            symbol=ticker,
            notional=round(dollars, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        log.info(f"BUY ${dollars:.2f} of {ticker} — order {order.id}")
        return {"success": True, "order_id": str(order.id), "error": ""}
    except Exception as e:
        log.error(f"buy_notional {ticker}: {e}")
        return {"success": False, "order_id": "", "error": str(e)}


# ── main rebalance ─────────────────────────────────────────────────────────────

def rebalance():
    # ── credentials ──────────────────────────────────────────────────────────
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    paper      = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

    if not api_key or not secret_key:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Aborting.")
        return

    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key, secret_key, paper=paper)

    # ── market open check ─────────────────────────────────────────────────────
    clock = client.get_clock()
    if not clock.is_open:
        log.info("Market is closed — nothing to do.")
        return

    # ── account info ──────────────────────────────────────────────────────────
    acct        = client.get_account()
    equity      = float(acct.equity)
    cash        = float(acct.cash)
    log.info(f"Account equity=${equity:.2f}  cash=${cash:.2f}")

    # ── get target tickers ────────────────────────────────────────────────────
    targets = get_momentum_top(TOP_N)
    if not targets:
        log.error("Could not determine momentum targets. Aborting.")
        return

    # ── current positions ─────────────────────────────────────────────────────
    current = get_current_positions(client)
    log.info(f"Current positions: {list(current.keys())}")

    now      = datetime.now(timezone.utc).isoformat()
    log_entries = load_log()
    run_log  = {"run_at": now, "equity": equity, "targets": targets, "actions": []}

    # ── close positions not in target ─────────────────────────────────────────
    for ticker, mkt_val in current.items():
        if ticker not in targets:
            result = close_position(client, ticker, paper)
            run_log["actions"].append({
                "action": "SELL", "ticker": ticker,
                "value": round(mkt_val, 2), **result
            })

    # ── reload cash after sells (give it a moment via re-fetch) ───────────────
    acct  = client.get_account()
    cash  = float(acct.cash)

    # ── buy targets not already held ─────────────────────────────────────────
    to_buy     = [t for t in targets if t not in current]
    investable = cash * (1 - RESERVE_PCT)
    if to_buy and investable > MIN_ORDER_USD:
        per_stock = investable / len(to_buy)
        for ticker in to_buy:
            if per_stock < MIN_ORDER_USD:
                log.warning(f"Skipping {ticker} — per_stock ${per_stock:.2f} below minimum.")
                continue
            result = buy_notional(client, ticker, per_stock)
            run_log["actions"].append({
                "action": "BUY", "ticker": ticker,
                "notional": round(per_stock, 2), **result
            })
    else:
        log.info("No new buys needed — already holding all targets or insufficient cash.")

    log.info(f"Run complete. Actions: {len(run_log['actions'])}")
    log_entries.append(run_log)
    save_log(log_entries)


if __name__ == "__main__":
    rebalance()
