"""
Stock Tools Module

MCP tool functions for fetching stock data from the Technical Analysis API
and calculating position sizes.
"""

import json
import os
from typing import Optional
import httpx
from claude_agent_sdk import tool, create_sdk_mcp_server

# Get API base URL from environment (required)
TRADING_API_BASE_URL = os.environ.get("TRADING_API_BASE_URL")


def create_stock_tools_server(analysis_date: Optional[str] = None):
    """
    Factory function to create MCP server with bound analysis_date.

    Args:
        analysis_date: Optional date string (YYYY-MM-DD) for historical analysis.
                      If None, uses current date (API default behavior).

    Returns:
        McpSdkServerConfig with the date bound via closure.
    """

    async def _get_stock_data_impl(args: dict) -> dict:
        """
        Internal implementation of get_stock_data.

        Fetch stock data with technical analysis from API.
        Returns MCP-formatted content response.

        Note: analysis_date is captured from outer scope (closure).
        """
        symbol = args["symbol"]
        period = args.get("period", "3mo")
        interval = args.get("interval", "1d")

        if not TRADING_API_BASE_URL:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: TRADING_API_BASE_URL environment variable not set"
                }]
            }

        url = f"{TRADING_API_BASE_URL}/stocks/analysis/{symbol}"

        # Build params - inject analysis_date from closure if provided
        params = {"period": period, "interval": interval}
        if analysis_date:
            params["analysis_date"] = analysis_date

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)

                if response.status_code == 404:
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"Error: Symbol not found: {symbol}"
                        }]
                    }
                if response.status_code != 200:
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"Error: API error: {response.status_code}"
                        }]
                    }

                data = response.json()
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
                    "text": f"Error: Technical Analysis API unavailable at {TRADING_API_BASE_URL}"
                }]
            }

    # Create tool with closure-bound implementation
    get_stock_data_tool = tool(
        name="get_stock_data",
        description="Fetch comprehensive technical analysis data for a stock including price, volume, moving averages, momentum indicators (RSI, MACD), volatility (ATR, Bollinger Bands), trend indicators (ADX), and support/resistance levels.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA')"
                },
                "period": {
                    "type": "string",
                    "description": "Time period for analysis",
                    "enum": ["1mo", "3mo", "6mo", "1y"],
                    "default": "3mo"
                },
                "interval": {
                    "type": "string",
                    "description": "Data interval",
                    "enum": ["1d", "1wk"],
                    "default": "1d"
                }
            },
            "required": ["symbol"]
        }
    )(_get_stock_data_impl)

    return create_sdk_mcp_server(
        name="stock_analysis",
        version="1.0.0",
        tools=[get_stock_data_tool]
    )


# Default server for backwards compatibility (no date binding)
stock_tools_server = create_stock_tools_server()


# Export the raw async function for testing and direct usage
# Note: This does NOT support analysis_date - use create_stock_tools_server() for that
async def get_stock_data(args: dict) -> dict:
    """
    Fetch stock data with technical analysis from API.

    This is a thin wrapper for backwards compatibility and testing purposes.
    NOTE: Does not support analysis_date parameter - use create_stock_tools_server() for that.

    Args:
        args: Dictionary containing:
            - symbol: Stock ticker symbol (required)
            - period: Time period "1mo", "3mo", "6mo", "1y" (default: "3mo")
            - interval: Data interval "1d", "1wk" (default: "1d")

    Returns:
        MCP-formatted content response with technical analysis data.
    """
    symbol = args["symbol"]
    period = args.get("period", "3mo")
    interval = args.get("interval", "1d")

    if not TRADING_API_BASE_URL:
        return {
            "content": [{
                "type": "text",
                "text": "Error: TRADING_API_BASE_URL environment variable not set"
            }]
        }

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
                        "text": f"Error: Symbol not found: {symbol}"
                    }]
                }
            if response.status_code != 200:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: API error: {response.status_code}"
                    }]
                }

            data = response.json()
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
                "text": f"Error: Technical Analysis API unavailable at {TRADING_API_BASE_URL}"
            }]
        }


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
