"""
Stock Tools - Custom MCP tools for stock price data.
"""

import os

import httpx
from claude_agent_sdk import tool, create_sdk_mcp_server
from typing import Any

# Stock Prices API configuration
STOCK_API_BASE_URL = os.getenv("STOCK_API_BASE_URL", "http://localhost:8093")
STOCK_API_TIMEOUT = 30.0  # seconds


@tool(
    "get_stock_prices",
    "Fetch historical stock price data (OHLCV candles) for analysis. Returns open, high, low, close, and volume for each period.",
    {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
            },
            "interval": {
                "type": "string",
                "enum": ["1d", "1wk", "1mo"],
                "default": "1d",
                "description": "Candle interval: 1d (daily), 1wk (weekly), 1mo (monthly)"
            },
            "period": {
                "type": "string",
                "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
                "description": "Quick time range. Use this OR start_date/end_date, not both."
            },
            "start_date": {
                "type": "string",
                "description": "Custom start date in YYYY-MM-DD format. Use with end_date."
            },
            "end_date": {
                "type": "string",
                "description": "Custom end date in YYYY-MM-DD format. Use with start_date."
            },
            "force_refresh": {
                "type": "boolean",
                "default": False,
                "description": "Bypass cache and fetch fresh data from source"
            }
        },
        "required": ["symbol"]
    }
)
async def get_stock_prices(args: dict[str, Any]) -> dict[str, Any]:
    """
    Fetch stock prices from the internal Stock Prices API.

    Returns OHLCV (Open, High, Low, Close, Volume) candle data.
    """
    symbol = args["symbol"].upper().strip()
    interval = args.get("interval", "1d")
    period = args.get("period")
    start_date = args.get("start_date")
    end_date = args.get("end_date")
    force_refresh = args.get("force_refresh", False)

    # Validate: either period OR date range, not both
    if period and (start_date or end_date):
        return {
            "content": [{
                "type": "text",
                "text": "Error: Use either 'period' OR 'start_date'/'end_date', not both."
            }],
            "isError": True
        }

    # Validate: if custom date range, both dates required
    if (start_date and not end_date) or (end_date and not start_date):
        return {
            "content": [{
                "type": "text",
                "text": "Error: Both 'start_date' and 'end_date' are required for custom date range."
            }],
            "isError": True
        }

    # Validate: must have either period or date range
    if not period and not start_date:
        return {
            "content": [{
                "type": "text",
                "text": "Error: Must specify either 'period' or 'start_date'/'end_date'."
            }],
            "isError": True
        }

    # Build query parameters
    params = {"interval": interval}
    if period:
        params["period"] = period
    if start_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    if force_refresh:
        params["force_refresh"] = "true"

    # Make API request
    url = f"{STOCK_API_BASE_URL}/api/v1/stocks/{symbol}/prices"

    try:
        async with httpx.AsyncClient(timeout=STOCK_API_TIMEOUT) as client:
            response = await client.get(url, params=params)

            if response.status_code == 404:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: Stock symbol '{symbol}' not found."
                    }],
                    "isError": True
                }

            if response.status_code != 200:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: API returned status {response.status_code}: {response.text}"
                    }],
                    "isError": True
                }

            data = response.json()

            # Format response for agent consumption
            candles = data.get("data", [])
            if not candles:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"No price data available for {symbol} with the specified parameters."
                    }]
                }

            # Build formatted response
            result_lines = [
                f"Stock: {data.get('symbol', symbol)}",
                f"Interval: {data.get('interval', interval)}",
                f"Data points: {len(candles)}",
                "",
                "Date       | Open     | High     | Low      | Close    | Volume",
                "-" * 70
            ]

            for candle in candles:
                result_lines.append(
                    f"{candle['date']} | "
                    f"{candle['open']:>8.2f} | "
                    f"{candle['high']:>8.2f} | "
                    f"{candle['low']:>8.2f} | "
                    f"{candle['close']:>8.2f} | "
                    f"{candle['volume']:>10,}"
                )

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(result_lines)
                }]
            }

    except httpx.TimeoutException:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: Request to Stock Prices API timed out after {STOCK_API_TIMEOUT}s."
            }],
            "isError": True
        }
    except httpx.ConnectError:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: Could not connect to Stock Prices API at {STOCK_API_BASE_URL}. Is the service running?"
            }],
            "isError": True
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: Unexpected error fetching stock prices: {str(e)}"
            }],
            "isError": True
        }


# Create the MCP server with stock tools
stock_tools_server = create_sdk_mcp_server(
    name="stock-tools",
    version="1.0.0",
    tools=[get_stock_prices]
)
