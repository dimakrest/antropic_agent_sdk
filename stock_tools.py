"""
Stock Tools Module

MCP tool functions for fetching stock data from the Technical Analysis API
and calculating position sizes.
"""

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

TECHNICAL_ANALYSIS_API_URL = os.getenv(
    "TECHNICAL_ANALYSIS_API_URL", "http://localhost:8093"
)


async def get_stock_data(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d"
) -> dict[str, Any]:
    """
    Fetch stock data with technical analysis from API.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        period: Time period - "1mo", "3mo", "6mo", "1y" (default: "3mo")
        interval: Data interval - "1d", "1wk" (default: "1d")

    Returns:
        dict: Technical analysis data including price, volume, moving averages,
              momentum indicators, volatility metrics, trend indicators,
              support/resistance levels, and recent candles.
              Returns {"error": "..."} on failure.
    """
    if not symbol or not isinstance(symbol, str):
        return {"error": "Symbol must be a non-empty string"}

    symbol = symbol.upper().strip()

    url = f"{TECHNICAL_ANALYSIS_API_URL}/api/v1/stocks/{symbol}/analysis"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params={"period": period, "interval": interval}
            )

            if response.status_code == 404:
                return {"error": f"Symbol not found: {symbol}"}
            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}"}

            return response.json()
    except httpx.ConnectError:
        return {"error": f"Technical Analysis API unavailable at {TECHNICAL_ANALYSIS_API_URL}"}


def calculate_position_size(
    account_size: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float
) -> dict[str, Any]:
    """
    Calculate position size based on account risk parameters.

    Pure math calculator - agent provides the entry and stop prices.

    Args:
        account_size: Total account value in dollars
        risk_percent: Risk per trade as percentage (e.g., 2.0 for 2%)
        entry_price: Agent-determined entry price
        stop_loss_price: Agent-determined stop-loss price

    Returns:
        dict: Position sizing calculations including max risk dollars,
              risk per share, position size in shares, position value,
              and percent of account. Returns {"error": "..."} on invalid input.
    """
    if account_size <= 0:
        return {"error": "Account size must be positive"}

    if risk_percent <= 0:
        return {"error": "Risk percent must be positive"}

    if entry_price <= 0:
        return {"error": "Entry price must be positive"}

    if stop_loss_price <= 0:
        return {"error": "Stop loss price must be positive"}

    max_risk_dollars = account_size * (risk_percent / 100)
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share == 0:
        return {"error": "Entry price and stop loss cannot be the same"}

    position_size_shares = int(max_risk_dollars / risk_per_share)
    position_value = position_size_shares * entry_price
    percent_of_account = (position_value / account_size) * 100

    return {
        "inputs": {
            "account_size": account_size,
            "risk_percent": risk_percent,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price
        },
        "calculations": {
            "max_risk_dollars": round(max_risk_dollars, 2),
            "risk_per_share": round(risk_per_share, 2),
            "position_size_shares": position_size_shares,
            "position_value": round(position_value, 2),
            "percent_of_account": round(percent_of_account, 2)
        }
    }
