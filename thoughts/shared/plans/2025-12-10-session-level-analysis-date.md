# Session-Level Analysis Date Parameter Implementation Plan

## Overview

Add an optional `analysis_date` parameter to the `/analyze` endpoint that is transparently passed to the downstream Trading API without the agent/model needing to know about or manage it. This uses a factory function pattern to bind the date via closure when creating the MCP server.

## Current State Analysis

### Key Files:
- `server.py:178-185` - `AnalyzeRequest` model (only accepts `stock`)
- `server.py:763-916` - `/analyze` endpoint (stateless, creates temp SDK client)
- `stock_tools.py:17-76` - `_get_stock_data_impl()` makes HTTP call to Trading API
- `stock_tools.py:37-44` - Constructs URL with `period`, `interval` params
- `stock_tools.py:188-193` - Creates static `stock_tools_server` MCP server

### Current Flow:
1. HTTP POST `/analyze` with `{"stock": "AAPL"}`
2. Creates `ClaudeSDKClient` with static `stock_tools_server`
3. Agent calls `mcp__stock_analysis__get_stock_data` tool
4. Tool makes GET to `/stocks/analysis/AAPL?period=3mo&interval=1d`

### Key Discoveries:
- The `/analyze` endpoint is stateless - each request creates/destroys a temp SDK client (`server.py:814-815`)
- The MCP server is created once at module load (`stock_tools.py:188-193`)
- No existing mechanism for passing request-scoped context to tools
- Claude Agent SDK does NOT have built-in `data_parameters` or context injection (confirmed by SDK expert research)

## Desired End State

After implementation:
1. `/analyze` accepts optional `analysis_date` parameter: `{"stock": "AAPL", "analysis_date": "2024-06-15"}`
2. When `analysis_date` is provided, the tool adds it to the API call: `?period=3mo&interval=1d&analysis_date=2024-06-15`
3. When `analysis_date` is NOT provided, behavior is unchanged (no date param added)
4. The agent never sees `analysis_date` - it's injected transparently via closure

### Verification:
- Unit tests pass with mocked responses for both with/without date scenarios
- Manual test: Call `/analyze` with `analysis_date` and verify the downstream API receives the parameter

## What We're NOT Doing

- Date validation beyond basic format (downstream API handles validation)
- Date range queries or multiple dates
- Caching based on analysis date
- Adding `analysis_date` to the tool's input schema (agent must NOT see it)
- Modifying the `/chat` endpoint (out of scope per ticket)

## Implementation Approach

Use the **Factory Function with Closure** pattern:
1. Create `create_stock_tools_server(analysis_date)` factory function
2. Move tool implementation inside factory so it captures `analysis_date` via closure
3. Modify `/analyze` endpoint to create fresh MCP server per request when date is provided
4. Maintain backwards compatibility with default `stock_tools_server` export

This approach was validated by the agent-sdk-expert as the recommended pattern since the Claude Agent SDK lacks built-in context injection.

## Phase 1: Modify stock_tools.py

### Overview
Refactor `stock_tools.py` to use a factory function pattern that creates MCP servers with `analysis_date` bound via closure.

### Changes Required:

#### 1. Create Factory Function
**File**: `http-server-agent-sdk/stock_tools.py`
**Changes**: Add `create_stock_tools_server()` factory function

```python
from typing import Optional

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
```

#### 2. Keep Existing Exports for Testing
**File**: `http-server-agent-sdk/stock_tools.py`
**Changes**: Keep `get_stock_data()` function for testing but note it doesn't support `analysis_date`

The existing `get_stock_data()` wrapper function should remain for backwards compatibility in tests, but add a docstring note that it doesn't support `analysis_date`.

### Success Criteria:

#### Automated Verification:
- [ ] All existing tests pass: `uv run pytest tests/test_server.py -v`
- [ ] No import errors when running server: `uv run python -c "from stock_tools import create_stock_tools_server, stock_tools_server"`

#### Manual Verification:
- [ ] Code review confirms factory function correctly captures `analysis_date` via closure

---

## Phase 2: Modify server.py

### Overview
Update the `/analyze` endpoint to accept `analysis_date` and use the factory function.

### Changes Required:

#### 1. Update AnalyzeRequest Model
**File**: `http-server-agent-sdk/server.py`
**Changes**: Add `analysis_date` field to `AnalyzeRequest`

```python
class AnalyzeRequest(BaseModel):
    """Request model for swing trading analysis"""
    stock: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')"
    )
    analysis_date: Optional[str] = Field(
        None,
        description="Historical analysis date (YYYY-MM-DD). Omit for current date.",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
```

#### 2. Update Import Statement
**File**: `http-server-agent-sdk/server.py`
**Changes**: Import factory function

```python
from stock_tools import create_stock_tools_server
```

(Remove `stock_tools_server` from import since we'll create it dynamically)

#### 3. Update /analyze Endpoint
**File**: `http-server-agent-sdk/server.py`
**Changes**: Use factory function to create MCP server with analysis_date

In the `analyze_stock()` function, replace:
```python
options = ClaudeAgentOptions(
    ...
    mcp_servers={"stock_analysis": stock_tools_server},
    ...
)
```

With:
```python
# Create MCP server with analysis_date bound (or None for current date)
stock_server = create_stock_tools_server(analysis_date=request.analysis_date)

options = ClaudeAgentOptions(
    ...
    mcp_servers={"stock_analysis": stock_server},
    ...
)
```

### Success Criteria:

#### Automated Verification:
- [ ] All existing tests pass: `uv run pytest tests/test_server.py -v`
- [ ] Server starts without errors: `TRADING_API_BASE_URL=http://localhost:8131/api/v1 uv run uvicorn server:app --port 8000`

#### Manual Verification:
- [ ] `/analyze` endpoint accepts requests without `analysis_date` (existing behavior)
- [ ] `/analyze` endpoint accepts requests with `analysis_date` parameter

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 3: Add Unit Tests

### Overview
Add comprehensive unit tests for the new `analysis_date` functionality.

### Changes Required:

#### 1. Add Test for analysis_date Parameter
**File**: `http-server-agent-sdk/tests/test_server.py`
**Changes**: Add tests for analysis_date scenarios

```python
def test_analyze_with_analysis_date(test_client, mock_sdk_client_for_analyze):
    """Test /analyze with analysis_date parameter"""
    with patch('server.ClaudeSDKClient', return_value=mock_sdk_client_for_analyze):
        with patch('server.create_stock_tools_server') as mock_factory:
            mock_factory.return_value = MagicMock()  # Return mock MCP server

            response = test_client.post("/analyze", json={
                "stock": "TEST",
                "analysis_date": "2024-06-15"
            })

            assert response.status_code == 200
            # Verify factory was called with the date
            mock_factory.assert_called_once_with(analysis_date="2024-06-15")


def test_analyze_without_analysis_date(test_client, mock_sdk_client_for_analyze):
    """Test /analyze without analysis_date (default behavior)"""
    with patch('server.ClaudeSDKClient', return_value=mock_sdk_client_for_analyze):
        with patch('server.create_stock_tools_server') as mock_factory:
            mock_factory.return_value = MagicMock()

            response = test_client.post("/analyze", json={
                "stock": "TEST"
            })

            assert response.status_code == 200
            # Verify factory was called with None
            mock_factory.assert_called_once_with(analysis_date=None)


def test_analyze_invalid_date_format(test_client):
    """Test /analyze rejects invalid date format"""
    response = test_client.post("/analyze", json={
        "stock": "TEST",
        "analysis_date": "2024/06/15"  # Wrong format
    })
    assert response.status_code == 422  # Validation error


def test_analyze_invalid_date_format_no_dashes(test_client):
    """Test /analyze rejects date without dashes"""
    response = test_client.post("/analyze", json={
        "stock": "TEST",
        "analysis_date": "20240615"
    })
    assert response.status_code == 422
```

#### 2. Add Test for Factory Function
**File**: `http-server-agent-sdk/tests/test_server.py`
**Changes**: Add test to verify factory creates proper closure

```python
@pytest.mark.asyncio
async def test_create_stock_tools_server_with_date():
    """Test factory function binds analysis_date correctly"""
    from stock_tools import create_stock_tools_server

    # Create server with date
    server = create_stock_tools_server(analysis_date="2024-06-15")

    # Verify server was created (basic sanity check)
    assert server is not None
    assert server.name == "stock_analysis"


@pytest.mark.asyncio
async def test_create_stock_tools_server_without_date():
    """Test factory function works without date"""
    from stock_tools import create_stock_tools_server

    server = create_stock_tools_server()
    assert server is not None
    assert server.name == "stock_analysis"
```

### Success Criteria:

#### Automated Verification:
- [ ] All tests pass: `uv run pytest tests/test_server.py -v`
- [ ] Test coverage includes new scenarios

#### Manual Verification:
- [ ] Code review confirms test coverage is adequate

---

## Phase 4: Update Documentation

### Overview
Update CLAUDE.md to document the new parameter.

### Changes Required:

#### 1. Update CLAUDE.md
**File**: `http-server-agent-sdk/CLAUDE.md`
**Changes**: Document `analysis_date` parameter in External API section

Add to the "Stock Analysis Endpoint" section:

```markdown
### /analyze Endpoint Parameters

**`POST /analyze`** - Analyze a stock for swing trading

**Request Body**:
- `stock` (required): Stock ticker symbol (e.g., "AAPL")
- `analysis_date` (optional): Historical analysis date in YYYY-MM-DD format.
  - If provided, analysis uses data as of that date
  - If omitted, uses current date (default behavior)
  - The agent does NOT see this parameter - it's injected transparently into the API call

**Example**:
```json
{
  "stock": "AAPL",
  "analysis_date": "2024-06-15"
}
```
```

### Success Criteria:

#### Automated Verification:
- [ ] Documentation renders correctly (no markdown errors)

#### Manual Verification:
- [ ] Documentation accurately describes the feature

---

## Testing Strategy

### Unit Tests:
- Test `/analyze` with `analysis_date` parameter calls factory with date
- Test `/analyze` without `analysis_date` calls factory with `None`
- Test invalid date format returns 422 validation error
- Test factory function creates valid MCP server

### Integration Tests (Manual):
1. Start server with `TRADING_API_BASE_URL` pointing to real/mock Trading API
2. Call `/analyze` with `analysis_date` parameter
3. Verify downstream API receives `analysis_date` query parameter
4. Call `/analyze` without `analysis_date`
5. Verify downstream API does NOT receive `analysis_date` parameter

### Manual Testing Steps:
1. Start the server: `TRADING_API_BASE_URL=http://localhost:8131/api/v1 uv run uvicorn server:app --port 8000`
2. Test without date: `curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"stock": "AAPL"}'`
3. Test with date: `curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"stock": "AAPL", "analysis_date": "2024-06-15"}'`
4. Verify responses are valid and downstream API logs show correct parameters

## Performance Considerations

- **MCP Server Creation**: Creating a new MCP server per request has minimal overhead (microseconds)
- **Existing Pattern**: The `/analyze` endpoint already creates/destroys a `ClaudeSDKClient` per request, so this is consistent
- **Memory**: `McpSdkServerConfig` objects are lightweight and garbage collected after request

## Migration Notes

- **Backwards Compatible**: Existing clients that don't send `analysis_date` continue to work unchanged
- **No Database Changes**: No migrations needed
- **No Breaking Changes**: All existing behavior preserved

## References

- Original ticket: `thoughts/shared/tickets/2025-12-10-session-level-analysis-date.md`
- SDK Expert research: Confirmed factory pattern is recommended (no built-in context injection)
- Current implementation: `server.py:763-916`, `stock_tools.py:17-76`
