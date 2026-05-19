"""
Alpaca API wrapper — handles both paper and live trading modes.
Uses alpaca-py (the official modern SDK).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlpacaAccount:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    status: str
    paper: bool


@dataclass
class AlpacaPosition:
    ticker: str
    qty: float
    avg_entry: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


def get_client(api_key: str, secret_key: str, paper: bool = True):
    """Return a TradingClient or None if credentials are missing."""
    if not api_key or not secret_key:
        return None
    try:
        from alpaca.trading.client import TradingClient
        return TradingClient(api_key, secret_key, paper=paper)
    except Exception:
        return None


def get_account(api_key: str, secret_key: str, paper: bool = True) -> Optional[AlpacaAccount]:
    client = get_client(api_key, secret_key, paper)
    if not client:
        return None
    try:
        acct = client.get_account()
        return AlpacaAccount(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            portfolio_value=float(acct.portfolio_value),
            status=str(acct.status),
            paper=paper,
        )
    except Exception:
        return None


def get_positions(api_key: str, secret_key: str, paper: bool = True) -> list[AlpacaPosition]:
    client = get_client(api_key, secret_key, paper)
    if not client:
        return []
    try:
        positions = client.get_all_positions()
        result = []
        for p in positions:
            result.append(AlpacaPosition(
                ticker=str(p.symbol),
                qty=float(p.qty),
                avg_entry=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_pct=float(p.unrealized_plpc) * 100,
            ))
        return result
    except Exception:
        return []


def place_market_order(
    api_key: str,
    secret_key: str,
    paper: bool,
    ticker: str,
    qty: float,
    side: str,  # "buy" or "sell"
    notional: Optional[float] = None,
) -> dict:
    """
    Place a market order. Use qty for whole/fractional shares,
    or notional for dollar-amount orders (e.g. buy $50 of AAPL).
    Returns {"success": bool, "order_id": str, "error": str}.
    """
    client = get_client(api_key, secret_key, paper)
    if not client:
        return {"success": False, "order_id": "", "error": "No API credentials configured."}
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        if notional:
            req = MarketOrderRequest(
                symbol=ticker,
                notional=round(notional, 2),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

        order = client.submit_order(req)
        return {"success": True, "order_id": str(order.id), "error": ""}
    except Exception as e:
        return {"success": False, "order_id": "", "error": str(e)}


def cancel_all_orders(api_key: str, secret_key: str, paper: bool) -> bool:
    client = get_client(api_key, secret_key, paper)
    if not client:
        return False
    try:
        client.cancel_orders()
        return True
    except Exception:
        return False
