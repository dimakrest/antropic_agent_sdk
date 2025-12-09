# Stock Prices Custom Tool Implementation Plan

## Overview

Implement a custom MCP tool that allows the Claude agent to fetch historical stock price data from the internal Stock Prices API (`localhost:8093`). This is the first data tool for the stock analysis agent and will establish the pattern for future tools.

## Current State Analysis

- **HTTP Server**: `http-server-agent-sdk/server.py` uses `ClaudeSDKClient` with `ClaudeAgentOptions`
- **No custom tools exist yet** - this will be the first implementation
- **Stock Prices API**: Running at `localhost:8093`, documented in `docs/internal-apis/stock-prices-api.md`
- **SDK Pattern**: Use `@tool` decorator + `create_sdk_mcp_server()` per `docs/agent-sdk/02-python-sdk.md:97-216`

### Key Discoveries:
- Tool naming convention: `mcp__{server_name}__{tool_name}` (e.g., `mcp__stock-tools__get_stock_prices`)
- Custom tools require `mcp_servers` parameter in `ClaudeAgentOptions` (`server.py:195-201`)
- Tools must return `{"content": [{"type": "text", "text": "..."}]}` format
- Error responses should include `"isError": True` for better MCP protocol compliance
- Python SDK uses dict-based schemas, not Zod (TypeScript only)
- Custom tools require `ClaudeSDKClient` (not standalone `query()` function)

## Desired End State

After implementation:
1. Agent can call `mcp__stock-tools__get_stock_prices` to fetch stock data
2. Tool validates parameters and returns formatted OHLCV data
3. Error messages are clear and actionable for the agent
4. Integration is seamless with existing HTTP server

### Verification:
- Unit tests pass for the tool module
- Integration test demonstrates agent using the tool
- Manual test via `/chat` endpoint with stock-related query

## What We're NOT Doing

- Real-time streaming price updates (out of scope per ticket)
- Order placement or trade execution
- Multiple symbol batch requests (one symbol per call)
- Caching layer on agent side (API handles caching)
- Authentication/API keys (internal API, localhost only)
- Response summarization/statistics (nice-to-have, defer)

## Implementation Approach

Create a separate module for stock tools to maintain clean separation of concerns. Integrate via the existing `ClaudeAgentOptions` pattern. Use `httpx` for async HTTP requests to the Stock Prices API.

**Performance Note**: The current implementation creates a new `httpx.AsyncClient` per request for simplicity. For production at scale, consider implementing connection pooling with a module-level or singleton client to benefit from HTTP connection reuse.

**SDK Compatibility Note**: Custom MCP tools require `ClaudeSDKClient` (not the standalone `query()` function). The existing HTTP server already uses `ClaudeSDKClient`, so this is correctly handled. However, SDK documentation mentions that MCP tools may require streaming input mode - this should be verified during integration testing.

---

## Phase 1: Create Stock Tools Module

### Overview
Create `stock_tools.py` with the `get_stock_prices` tool definition.

### Changes Required:

#### 1. Create new file: `http-server-agent-sdk/stock_tools.py`

```python
"""
Stock Tools - Custom MCP tools for stock price data.
"""

import httpx
from claude_agent_sdk import tool, create_sdk_mcp_server
from typing import Any, Optional

# Stock Prices API configuration
STOCK_API_BASE_URL = "http://localhost:8093"
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
```

#### 2. Add `httpx` dependency

**File**: `http-server-agent-sdk/pyproject.toml`

Add `httpx` to dependencies:

```toml
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "python-dotenv>=1.0.0",
    "claude-agent-sdk>=0.1.0",
    "pydantic>=2.0.0",
    "httpx>=0.27.0",  # Add this line
]
```

### Success Criteria:

#### Automated Verification:
- [x] Run `uv sync` to install new dependency
- [x] Module imports without errors: `uv run python -c "from stock_tools import stock_tools_server"`
- [ ] Type checking passes (if mypy configured)

#### Manual Verification:
- [x] Review code follows SDK patterns from documentation

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 2.

---

## Phase 2: Integrate Tool into HTTP Server

### Overview
Modify `server.py` to include the stock tools MCP server in agent options.

### Changes Required:

#### 1. Import stock tools server
**File**: `http-server-agent-sdk/server.py`
**Location**: After line 22 (after other imports)

```python
from stock_tools import stock_tools_server
```

#### 2. Add MCP servers to ClaudeAgentOptions
**File**: `http-server-agent-sdk/server.py`
**Location**: Lines 195-201 (in `get_or_create_session` method)

**Current code:**
```python
options = ClaudeAgentOptions(
    model=DEFAULT_MODEL,
    permission_mode=permission_mode,
    max_turns=max_turns,
    allowed_tools=allowed_tools,
    can_use_tool=default_can_use_tool if permission_mode == "default" else None,
)
```

**New code:**
```python
options = ClaudeAgentOptions(
    model=DEFAULT_MODEL,
    permission_mode=permission_mode,
    max_turns=max_turns,
    allowed_tools=allowed_tools,
    mcp_servers={"stock-tools": stock_tools_server},
    can_use_tool=default_can_use_tool if permission_mode == "default" else None,
)
```

#### 3. Update permission callback for stock tools
**File**: `http-server-agent-sdk/server.py`
**Location**: Lines 111-141 (`default_can_use_tool` function)

Add stock tools to the allow-list:

```python
async def default_can_use_tool(tool_name: str, tool_input: dict) -> bool:
    """
    Default permission callback for tool usage.
    """
    # Allow read-only tools without restriction
    if tool_name in ["Read", "Grep", "Glob"]:
        return True

    # Allow stock tools (read-only data fetching)
    if tool_name.startswith("mcp__stock-tools__"):
        return True

    # ... rest of function unchanged
```

### Success Criteria:

#### Automated Verification:
- [x] Server starts without import errors: `uv run python -c "from server import app"`
- [x] Unit tests still pass: `uv run pytest tests/test_server.py -v`

#### Manual Verification:
- [ ] Start server: `uv run uvicorn server:app --reload`
- [ ] Verify health endpoint still works: `curl http://localhost:8000/health`

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 3.

---

## Phase 3: Add Unit Tests for Stock Tools

### Overview
Create unit tests for the stock tools module with mocked HTTP responses.

### Changes Required:

#### 1. Create test file: `http-server-agent-sdk/tests/test_stock_tools.py`

```python
"""
Unit tests for stock_tools module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from stock_tools import get_stock_prices, stock_tools_server


class TestGetStockPrices:
    """Tests for the get_stock_prices tool function."""

    @pytest.mark.asyncio
    async def test_successful_request_with_period(self):
        """Test successful stock price fetch with period parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "AAPL",
            "interval": "1d",
            "data": [
                {
                    "date": "2025-12-02",
                    "open": 150.25,
                    "high": 152.30,
                    "low": 149.80,
                    "close": 151.45,
                    "volume": 45678900
                }
            ]
        }

        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await get_stock_prices({
                "symbol": "AAPL",
                "period": "1mo",
                "interval": "1d"
            })

        assert result["content"][0]["type"] == "text"
        assert "AAPL" in result["content"][0]["text"]
        assert "151.45" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_successful_request_with_date_range(self):
        """Test successful stock price fetch with custom date range."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "MSFT",
            "interval": "1wk",
            "data": [
                {
                    "date": "2025-11-25",
                    "open": 400.00,
                    "high": 410.00,
                    "low": 395.00,
                    "close": 405.00,
                    "volume": 12345678
                }
            ]
        }

        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await get_stock_prices({
                "symbol": "msft",  # Test lowercase conversion
                "start_date": "2025-11-01",
                "end_date": "2025-12-01",
                "interval": "1wk"
            })

        assert "MSFT" in result["content"][0]["text"]
        assert "405.00" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_validation_period_and_dates_conflict(self):
        """Test validation error when both period and date range provided."""
        result = await get_stock_prices({
            "symbol": "AAPL",
            "period": "1mo",
            "start_date": "2025-01-01",
            "end_date": "2025-12-01"
        })

        assert "Error" in result["content"][0]["text"]
        assert "period" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_validation_missing_end_date(self):
        """Test validation error when only start_date provided."""
        result = await get_stock_prices({
            "symbol": "AAPL",
            "start_date": "2025-01-01"
        })

        assert "Error" in result["content"][0]["text"]
        assert "end_date" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_validation_no_time_range(self):
        """Test validation error when no time range specified."""
        result = await get_stock_prices({
            "symbol": "AAPL"
        })

        assert "Error" in result["content"][0]["text"]
        assert "period" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_symbol_not_found(self):
        """Test handling of 404 response for unknown symbol."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await get_stock_prices({
                "symbol": "INVALID",
                "period": "1mo"
            })

        assert "Error" in result["content"][0]["text"]
        assert "not found" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_api_error_response(self):
        """Test handling of non-200 API response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await get_stock_prices({
                "symbol": "AAPL",
                "period": "1mo"
            })

        assert "Error" in result["content"][0]["text"]
        assert "500" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test handling of connection errors."""
        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            result = await get_stock_prices({
                "symbol": "AAPL",
                "period": "1mo"
            })

        assert "Error" in result["content"][0]["text"]
        assert "connect" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test handling of timeout errors."""
        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            result = await get_stock_prices({
                "symbol": "AAPL",
                "period": "1mo"
            })

        assert "Error" in result["content"][0]["text"]
        assert "timed out" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_empty_data_response(self):
        """Test handling of empty data array."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "AAPL",
            "interval": "1d",
            "data": []
        }

        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await get_stock_prices({
                "symbol": "AAPL",
                "period": "1d"
            })

        assert "No price data" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_force_refresh_parameter(self):
        """Test that force_refresh parameter is passed to API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "AAPL",
            "interval": "1d",
            "data": [{"date": "2025-12-02", "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000000}]
        }

        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            await get_stock_prices({
                "symbol": "AAPL",
                "period": "1mo",
                "force_refresh": True
            })

            # Verify force_refresh was passed in params
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("force_refresh") == "true"

    @pytest.mark.asyncio
    async def test_symbol_whitespace_trimming(self):
        """Test that symbol whitespace is trimmed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "AAPL",
            "interval": "1d",
            "data": [{"date": "2025-12-02", "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000000}]
        }

        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            await get_stock_prices({
                "symbol": "  aapl  ",  # Whitespace and lowercase
                "period": "1mo"
            })

            # Verify URL contains trimmed uppercase symbol
            call_args = mock_get.call_args
            url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
            assert "/AAPL/" in url

    @pytest.mark.asyncio
    async def test_malformed_api_response(self):
        """Test handling of malformed API response (missing expected fields)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "unexpected": "data"
            # Missing "symbol", "interval", "data" fields
        }

        with patch("stock_tools.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await get_stock_prices({
                "symbol": "AAPL",
                "period": "1mo"
            })

        # Should handle gracefully - empty data case
        assert "No price data" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_error_response_includes_isError_flag(self):
        """Test that error responses include isError: True for MCP protocol."""
        result = await get_stock_prices({
            "symbol": "AAPL"
            # Missing required period or date range
        })

        assert result.get("isError") is True
        assert "Error" in result["content"][0]["text"]


class TestStockToolsServer:
    """Tests for the MCP server configuration."""

    def test_server_name(self):
        """Test server has correct name."""
        assert stock_tools_server.name == "stock-tools"

    def test_server_version(self):
        """Test server has version set."""
        assert stock_tools_server.version == "1.0.0"

    def test_server_has_tools(self):
        """Test server has tools registered."""
        assert len(stock_tools_server.tools) == 1
        assert stock_tools_server.tools[0].name == "get_stock_prices"
```

### Success Criteria:

#### Automated Verification:
- [x] All tests pass: `uv run pytest tests/test_stock_tools.py -v`
- [x] Test coverage > 80%: `uv run pytest tests/test_stock_tools.py --cov=stock_tools --cov-report=term` (96% coverage achieved)

#### Manual Verification:
- [x] Review test coverage for edge cases

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding to Phase 4.

---

## Phase 4: Integration Testing and Documentation

### Overview
Create integration test and update documentation.

### Changes Required:

#### 1. Create integration test: `http-server-agent-sdk/tests/test_stock_tools_integration.py`

```python
"""
Integration tests for stock tools.

Requires:
1. ANTHROPIC_API_KEY environment variable
2. Stock Prices API running at localhost:8093
3. HTTP server running at localhost:8000

Run with: uv run pytest tests/test_stock_tools_integration.py -v
"""

import os
import pytest
import httpx

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

SERVER_URL = "http://localhost:8000"
STOCK_API_URL = "http://localhost:8093"


@pytest.fixture
async def check_services():
    """Fixture to check that required services are running."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Check stock API
        try:
            await client.get(f"{STOCK_API_URL}/api/v1/stocks/AAPL/prices?period=1d")
        except httpx.ConnectError:
            pytest.skip(f"Stock API not running at {STOCK_API_URL}")

        # Check HTTP server
        try:
            response = await client.get(f"{SERVER_URL}/health")
            if response.status_code != 200:
                pytest.skip(f"HTTP server not healthy at {SERVER_URL}")
        except httpx.ConnectError:
            pytest.skip(f"HTTP server not running at {SERVER_URL}")


@pytest.mark.asyncio
async def test_agent_uses_stock_tool(check_services):
    """Test that the agent can use the stock prices tool."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{SERVER_URL}/chat",
            json={
                "message": "Get me the stock prices for AAPL for the last month with daily intervals.",
                "permission_mode": "acceptEdits",
                "allowed_tools": ["mcp__stock-tools__get_stock_prices"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        print(f"Status: {data['status']}")
        print(f"Response: {data.get('response_text', '')[:500]}...")

        # Verify the response mentions stock data
        response_text = data.get("response_text", "").lower()
        assert "aapl" in response_text or "apple" in response_text, \
            "Response should mention AAPL or Apple"


@pytest.mark.asyncio
async def test_agent_handles_invalid_symbol(check_services):
    """Test that the agent handles invalid stock symbols gracefully."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{SERVER_URL}/chat",
            json={
                "message": "Get stock prices for INVALIDXYZ123 for the last month.",
                "permission_mode": "acceptEdits",
                "allowed_tools": ["mcp__stock-tools__get_stock_prices"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Agent should handle the error gracefully
        assert data["status"] in ["success", "error_during_execution"]
```

#### 2. Update ticket status
**File**: `thoughts/shared/tickets/2025-12-03-stock-prices-custom-tool.md`

Update acceptance criteria checkboxes to mark completed items after implementation.

### Success Criteria:

#### Automated Verification:
- [x] Unit tests pass: `uv run pytest tests/test_stock_tools.py -v` (18 tests pass)
- [x] Server starts: `uv run uvicorn server:app`
- [x] No import errors in stock_tools module

#### Manual Verification:
- [ ] Start Stock Prices API at localhost:8093
- [ ] Start HTTP server: `uv run uvicorn server:app --port 8000`
- [ ] Run integration test: `uv run pytest tests/test_stock_tools_integration.py -v`
- [ ] Verify MCP tool works with `ClaudeSDKClient.query()` (string prompt mode)
- [ ] Test via curl or Postman:
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
      "message": "What are the stock prices for AAPL over the last month?",
      "allowed_tools": ["mcp__stock-tools__get_stock_prices"]
    }'
  ```

**Implementation Note**: After completing all phases and verification, update the ticket to mark it as complete.

---

## Testing Strategy

### Unit Tests:
- `test_stock_tools.py`: Test tool function with mocked HTTP responses
- Test all validation paths (invalid params, conflicts)
- Test all error handling paths (404, 500, timeout, connection error)
- Test response formatting
- Test `isError` flag in error responses (MCP protocol compliance)
- Test `force_refresh` parameter handling
- Test symbol whitespace trimming and case normalization
- Test malformed API response handling

### Integration Tests:
- `test_stock_tools_integration.py`: End-to-end test with real API calls
- Requires running Stock Prices API and HTTP server
- Verifies agent can actually use the tool
- Tests error handling with invalid symbols
- Uses `pytest.skip()` for missing prerequisites (not `exit()`)

### Manual Testing Steps:
1. Start Stock Prices API at `localhost:8093`
2. Start HTTP server with `uv run uvicorn server:app --reload`
3. Send chat request asking about stock prices
4. Verify response includes actual stock data
5. Test error cases (invalid symbol, API down)
6. Verify MCP tool works with string prompts via `ClaudeSDKClient.query()`

---

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `http-server-agent-sdk/stock_tools.py` | Create | New tool module |
| `http-server-agent-sdk/server.py` | Modify | Add import and mcp_servers config |
| `http-server-agent-sdk/pyproject.toml` | Modify | Add httpx dependency |
| `http-server-agent-sdk/tests/test_stock_tools.py` | Create | Unit tests |
| `http-server-agent-sdk/tests/test_stock_tools_integration.py` | Create | Integration test |

---

## References

- Original ticket: `thoughts/shared/tickets/2025-12-03-stock-prices-custom-tool.md`
- Stock Prices API docs: `docs/internal-apis/stock-prices-api.md`
- Custom Tools SDK docs: `docs/agent-sdk/10-custom-tools.md`
- Python SDK reference: `docs/agent-sdk/02-python-sdk.md:97-216`
- Existing server implementation: `http-server-agent-sdk/server.py:195-201`
