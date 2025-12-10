"""
Stock Tools Module

MCP tool functions for fetching stock data from the Technical Analysis API
and calculating position sizes.
"""

import json
import os
import httpx
from claude_agent_sdk import tool, create_sdk_mcp_server

# Configuration - must be set via environment variable
TRADING_API_BASE_URL = os.getenv("TRADING_API_BASE_URL")
if not TRADING_API_BASE_URL:
    raise ValueError("TRADING_API_BASE_URL environment variable must be set")


async def _get_stock_data_impl(args: dict) -> dict:
    """
    Internal implementation of get_stock_data.

    Fetch stock data with technical analysis from API.
    Returns MCP-formatted content response.
    """
    symbol = args["symbol"]
    period = args["period"]
    interval = args["interval"]

    # Endpoint format: {base_url}/stocks/analysis/{symbol}
    url = f"{TRADING_API_BASE_URL}/stocks/analysis/{symbol}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params={"period": period, "interval": interval}
            )

            if response.status_code == 404:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: Symbol '{symbol}' not found. This could mean: (1) the symbol is misspelled, (2) it's not a US-listed stock (NYSE/NASDAQ), or (3) the company may have been delisted. Please verify the ticker symbol with the user."
                    }]
                }
            if response.status_code != 200:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: API returned status {response.status_code}. The Trading Analyst API may be experiencing issues. Try again or inform the user that market data is temporarily unavailable."
                    }]
                }

            data = response.json()
            # Format as readable text for the agent
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(data, indent=2)
                }]
            }
    except httpx.ConnectError:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: Cannot connect to Trading Analyst API at {TRADING_API_BASE_URL}. The API service is not running. Inform the user that stock data is currently unavailable and they should ensure the Trading Analyst API is started."
            }]
        }


# MCP Tool wrapper for the get_stock_data function
get_stock_data_tool = tool(
    name="get_stock_data",
    description="""Fetch technical analysis data for a stock from the Trading Analyst API.

RETURNS: Current price, moving averages (SMA 20/50/200, EMA 12/26), momentum indicators (RSI 14/7, MACD line/signal/histogram, Stochastic K/D, CCI with signal), volatility metrics (ATR 14/7, Bollinger Bands upper/middle/lower/width), trend indicators (ADX, +DI/-DI), support/resistance levels (3 each with pivot point), volume statistics (current, avg 20/50, ratio vs avg).

USE THIS TOOL: For any stock analysis request. Never estimate or assume data values - always call this tool to get actual market data.

LIMITATIONS: US stocks only (NYSE, NASDAQ). No fundamental data (earnings, P/E, revenue). Some indicators require sufficient data points (SMA-200 needs ~200 days of history, may return null with short periods).""",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol in uppercase (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft, 'NVDA' for NVIDIA, 'TSLA' for Tesla)"
            },
            "period": {
                "type": "string",
                "description": "Historical time period for analysis: '1mo', '3mo', '6mo', or '1y'",
                "enum": ["1mo", "3mo", "6mo", "1y"]
            },
            "interval": {
                "type": "string",
                "description": "Data granularity: '15m' (15-minute intraday), '1d' (daily), or '1wk' (weekly)",
                "enum": ["15m", "1d", "1wk"]
            }
        },
        "required": ["symbol", "period", "interval"],
        "additionalProperties": False
    }
)(_get_stock_data_impl)


# Export the raw async function for testing and direct usage
async def get_stock_data(args: dict) -> dict:
    """
    Fetch stock data with technical analysis from API.

    This is a thin wrapper around _get_stock_data_impl for backwards
    compatibility and testing purposes.

    Args:
        args: Dictionary containing:
            - symbol: Stock ticker symbol (required)
            - period: Time period "1mo", "3mo", "6mo", "1y" (required)
            - interval: Data interval "15m", "1d", "1wk" (required)

    Returns:
        MCP-formatted content response with technical analysis data.
    """
    return await _get_stock_data_impl(args)


def calculate_position_size(
    account_size: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float
) -> dict:
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


# Create MCP server with stock tools
stock_tools_server = create_sdk_mcp_server(
    name="stock_analysis",
    version="1.0.0",
    tools=[get_stock_data_tool]
)
