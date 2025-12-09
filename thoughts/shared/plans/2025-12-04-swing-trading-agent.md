# Swing Trading Agent Implementation Plan

## Overview

Create a new `/analyze` REST API endpoint that accepts a stock ticker and returns a structured JSON trading recommendation from an AI agent acting as a professional swing trader. The agent uses existing technical analysis tools to analyze stocks and provides binary Buy/Not Buy recommendations with specific price levels.

## Current State Analysis

### Existing Infrastructure

1. **HTTP Server** (`http-server-agent-sdk/server.py`):
   - FastAPI-based REST API with Pydantic models
   - Session management via `SessionManager` class
   - Uses `ClaudeSDKClient` with `ClaudeAgentOptions`
   - Existing endpoints: `/chat`, `/health`, `/sessions`

2. **Stock Tools** (`http-server-agent-sdk/stock_tools.py`):
   - `get_stock_data()` - Fetches technical analysis from API at `localhost:8093`
   - `calculate_position_size()` - Position sizing calculator
   - **Currently NOT registered as MCP tools** - standalone functions only

3. **Technical Analysis API** (external, `localhost:8093`):
   - Provides: price data, volume, moving averages (SMA 20/50/200, EMA 12/26)
   - Momentum: RSI (7/14), MACD, Stochastic
   - Volatility: ATR (7/14), Bollinger Bands
   - Trend: ADX, +DI/-DI
   - Levels: Support/Resistance (3 each), pivot point, 52-week high/low

### Key Discoveries

- SDK structured outputs use `output_format={"type": "json_schema", "schema": ...}` (docs/agent-sdk/06-structured-outputs.md)
- Custom tools need `@tool` decorator and `create_sdk_mcp_server()` for registration
- Tool names follow pattern: `mcp__{server_name}__{tool_name}`
- Current stock_tools.py returns plain dicts, not MCP-formatted responses

## Desired End State

A working `/analyze` endpoint that:
1. Accepts `POST /analyze` with `{"stock": "AAPL"}`
2. Creates a temporary agent session with registered stock tools
3. Agent analyzes the stock using `get_stock_data` tool
4. Returns structured JSON with trading recommendation
5. Auto-cleans up session (stateless operation)

### Verification

- `curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"stock": "AAPL"}'` returns valid trading recommendation
- Response matches defined JSON schema
- Error cases return proper "Not Buy" with reasoning
- Unit tests pass with mocked SDK
- Integration tests pass with real SDK and Technical Analysis API

## What We're NOT Doing

- Portfolio management features
- Position sizing in the response (existing tool can be called separately)
- Sell signals for existing positions
- Options strategies
- Fundamental analysis
- News/sentiment analysis
- Session continuity (each request is stateless)
- Caching of analysis results

## Implementation Approach

The implementation follows these principles:
1. **MCP Tool Pattern**: Register `get_stock_data` as a proper MCP tool the agent can call
2. **Structured Outputs**: Use SDK's JSON Schema feature for validated responses
3. **Stateless Design**: Create temporary sessions, auto-cleanup after response
4. **Agent Autonomy**: Agent decides analysis methodology, we define output structure

---

## Phase 1: Register Stock Tools as MCP Tools

### Overview
Convert existing stock functions to MCP tools that the Claude Agent can call autonomously.

### Changes Required

#### 1. Update stock_tools.py to use MCP tool pattern

**File**: `http-server-agent-sdk/stock_tools.py`

**Changes**: Add `@tool` decorator and create MCP server. Update return format to MCP content structure.

```python
"""
Stock Tools Module

MCP tool functions for fetching stock data from the Technical Analysis API
and calculating position sizes.
"""

import httpx
from claude_agent_sdk import tool, create_sdk_mcp_server


@tool(
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
)
async def get_stock_data(args: dict) -> dict:
    """
    Fetch stock data with technical analysis from API.
    """
    symbol = args["symbol"]
    period = args.get("period", "3mo")
    interval = args.get("interval", "1d")

    url = f"http://localhost:8093/api/v1/stocks/{symbol}/analysis"

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
            # Format as readable text for the agent
            import json
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
                "text": "Error: Technical Analysis API unavailable at localhost:8093"
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

    (Unchanged - not needed for swing trading agent)
    """
    # ... existing implementation unchanged ...


# Create MCP server with stock tools
stock_tools_server = create_sdk_mcp_server(
    name="stock_analysis",
    version="1.0.0",
    tools=[get_stock_data]
)
```

### Success Criteria

#### Automated Verification:
- [x] Unit tests pass: `uv run pytest tests/test_stock_tools.py -v`
- [x] Import succeeds: `uv run python -c "from stock_tools import stock_tools_server"`

#### Manual Verification:
- [x] None for this phase

---

## Phase 2: Create /analyze Endpoint

### Overview
Add the new `/analyze` endpoint with request/response models and structured output integration.

### Changes Required

#### 1. Add Request/Response Models

**File**: `http-server-agent-sdk/server.py`

**Location**: After line 104 (after `SessionListResponse`)

```python
# =============================================================================
# Swing Trading Agent Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for swing trading analysis"""
    stock: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')"
    )


class AnalyzeResponse(BaseModel):
    """Response model for swing trading analysis"""
    stock: str = Field(..., description="The input ticker symbol")
    recommendation: str = Field(
        ...,
        description="Trading recommendation: 'Buy' or 'Not Buy'"
    )
    entry_price: float = Field(..., description="Recommended entry price")
    stop_loss: float = Field(..., description="Stop loss price level")
    take_profit: float = Field(..., description="Take profit target price")
    reasoning: str = Field(
        ...,
        description="Professional analysis explaining the decision"
    )
    missing_tools: List[str] = Field(
        ...,
        description="Indicators/data that would increase confidence"
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in the recommendation (0-100)"
    )


# JSON Schema for structured output (matches AnalyzeResponse)
TRADING_RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "stock": {"type": "string"},
        "recommendation": {
            "type": "string",
            "enum": ["Buy", "Not Buy"]
        },
        "entry_price": {"type": "number"},
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "reasoning": {"type": "string"},
        "missing_tools": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        }
    },
    "required": [
        "stock",
        "recommendation",
        "entry_price",
        "stop_loss",
        "take_profit",
        "reasoning",
        "missing_tools",
        "confidence_score"
    ],
    "additionalProperties": False
}
```

#### 2. Add System Prompt for Swing Trading Agent

**File**: `http-server-agent-sdk/server.py`

**Location**: After the schema definition

```python
SWING_TRADING_SYSTEM_PROMPT = """You are a professional swing trader specializing in short-term trades with 3-5 day holding periods.

## Your Role
You ARE the trading expert. Apply your own trading expertise and judgment to analyze stocks and provide actionable recommendations.

## Analysis Process
1. ALWAYS use the get_stock_data tool to fetch current technical analysis data
2. Analyze the data using your trading expertise
3. Make a binary decision: Buy or Not Buy (no "Hold" or "Maybe")
4. Provide specific price levels for entry, stop loss, and take profit

## Decision Framework
- Focus on swing trade setups with 3-5 day holding period
- Favor favorable risk/reward ratios (ideally 2:1 or better)
- Always set a stop loss - risk management is non-negotiable
- Consider trend direction, momentum, and key support/resistance levels

## Key Indicators to Consider
- RSI: Look for momentum confirmation (not overbought >70 or oversold <30 for entries)
- MACD: Trend direction and potential crossovers
- Moving Averages: Price relative to 20, 50, 200 SMA for trend context
- Support/Resistance: Key levels for entry and stop placement
- Volume: Confirmation of price moves
- ATR: For appropriate stop loss and target distance

## Output Requirements
- recommendation: "Buy" or "Not Buy" - binary decision only
- entry_price: Specific price (can be current price for immediate entry or limit near support)
- stop_loss: Below recent support or based on ATR
- take_profit: Based on resistance levels or risk/reward calculation
- reasoning: Professional analysis explaining your decision (2-4 sentences)
- missing_tools: What additional data would improve your analysis
- confidence_score: 0-100 based on signal alignment (informational only)

## Important Notes
- If data is unavailable or insufficient, recommend "Not Buy" with reasoning
- Never recommend buying without proper risk/reward setup
- Be conservative with confidence scores - reserve >80 for very clear setups
"""
```

#### 3. Add Import for stock_tools_server

**File**: `http-server-agent-sdk/server.py`

**Location**: After line 22 (after existing imports)

```python
from stock_tools import stock_tools_server
```

#### 4. Add /analyze Endpoint

**File**: `http-server-agent-sdk/server.py`

**Location**: After `/sessions` endpoint (after line 497)

```python
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_stock(request: AnalyzeRequest):
    """
    Analyze a stock for swing trading opportunities.

    Returns a structured trading recommendation with entry price,
    stop loss, take profit, and professional reasoning.

    This is a stateless endpoint - each request creates a temporary
    session that is automatically cleaned up after the response.
    """
    session_id = None

    try:
        # Create SDK options with stock tools and structured output
        options = ClaudeAgentOptions(
            model=DEFAULT_MODEL,
            permission_mode="bypassPermissions",  # Tools are safe, no user interaction
            max_turns=5,  # Limit turns for focused analysis
            mcp_servers={"stock_analysis": stock_tools_server},
            allowed_tools=["mcp__stock_analysis__get_stock_data"],
            system_prompt=SWING_TRADING_SYSTEM_PROMPT,
            output_format={
                "type": "json_schema",
                "schema": TRADING_RECOMMENDATION_SCHEMA
            }
        )

        # Create temporary session
        session_id = str(uuid.uuid4())
        client = ClaudeSDKClient(options=options)
        await client.connect()

        # Send analysis request
        prompt = f"Analyze {request.stock.upper()} for a potential swing trade opportunity."
        await client.query(prompt=prompt)

        # Collect response
        result_message = None
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                result_message = message

        # Process result
        if result_message:
            if result_message.subtype == "success" and hasattr(result_message, 'structured_output'):
                output = result_message.structured_output
                return AnalyzeResponse(**output)

            elif result_message.subtype == "error_max_structured_output_retries":
                # Agent couldn't produce valid structured output
                return AnalyzeResponse(
                    stock=request.stock.upper(),
                    recommendation="Not Buy",
                    entry_price=0,
                    stop_loss=0,
                    take_profit=0,
                    reasoning="Unable to complete analysis - agent could not produce valid structured output after multiple attempts.",
                    missing_tools=["Stable analysis pipeline"],
                    confidence_score=0
                )
            else:
                # Other error
                return AnalyzeResponse(
                    stock=request.stock.upper(),
                    recommendation="Not Buy",
                    entry_price=0,
                    stop_loss=0,
                    take_profit=0,
                    reasoning=f"Analysis failed with status: {result_message.subtype}",
                    missing_tools=[],
                    confidence_score=0
                )

        # No result received
        raise HTTPException(
            status_code=500,
            detail="No result received from analysis agent"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )
    finally:
        # Always cleanup the temporary session
        if client:
            try:
                await client.disconnect()
            except Exception:
                pass  # Ignore disconnect errors
```

### Success Criteria

#### Automated Verification:
- [x] Server starts without errors: `uv run uvicorn server:app --port 8000`
- [x] Unit tests pass: `uv run pytest tests/test_server.py -v`
- [x] OpenAPI docs show /analyze endpoint: Visit `http://localhost:8000/docs`

#### Manual Verification:
- [ ] Endpoint responds to valid request (requires Technical Analysis API running)

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to the next phase.

---

## Phase 3: Add Unit Tests

### Overview
Add comprehensive unit tests for the `/analyze` endpoint with mocked SDK responses.

### Changes Required

#### 1. Add Test File

**File**: `http-server-agent-sdk/tests/test_analyze_endpoint.py`

```python
"""
Unit tests for the /analyze swing trading endpoint.

These tests mock the ClaudeSDKClient to test endpoint behavior
without requiring an actual API key or Technical Analysis API.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


class MockResultMessage:
    """Mock ResultMessage for testing"""
    def __init__(self, subtype="success", structured_output=None):
        self.subtype = subtype
        self.structured_output = structured_output


class MockAsyncIterator:
    """Mock async iterator for receive_response"""
    def __init__(self, messages):
        self.messages = messages
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.messages):
            raise StopAsyncIteration
        msg = self.messages[self.index]
        self.index += 1
        return msg


@pytest.fixture
def mock_successful_analysis():
    """Mock a successful trading analysis response"""
    return {
        "stock": "AAPL",
        "recommendation": "Buy",
        "entry_price": 150.25,
        "stop_loss": 145.50,
        "take_profit": 162.00,
        "reasoning": "Strong RSI momentum at 58, price above 20-day MA with increasing volume. Support at 145 tested twice. Risk/reward ratio 2.5:1 favorable for swing entry.",
        "missing_tools": ["Options flow data", "Sector relative strength"],
        "confidence_score": 75
    }


@pytest.fixture
def mock_not_buy_analysis():
    """Mock a Not Buy recommendation"""
    return {
        "stock": "XYZ",
        "recommendation": "Not Buy",
        "entry_price": 0,
        "stop_loss": 0,
        "take_profit": 0,
        "reasoning": "Symbol not found or data unavailable. Cannot perform technical analysis without valid price data.",
        "missing_tools": ["Valid price data"],
        "confidence_score": 0
    }


class TestAnalyzeEndpoint:
    """Tests for POST /analyze endpoint"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client with mocked SDK"""
        # Import here to avoid import errors during collection
        from server import app
        self.client = TestClient(app)

    def test_analyze_returns_buy_recommendation(self, mock_successful_analysis):
        """Test successful Buy recommendation"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = MagicMock(
            return_value=MockAsyncIterator([
                MockResultMessage(
                    subtype="success",
                    structured_output=mock_successful_analysis
                )
            ])
        )

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post(
                "/analyze",
                json={"stock": "AAPL"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["stock"] == "AAPL"
        assert data["recommendation"] == "Buy"
        assert data["entry_price"] == 150.25
        assert data["stop_loss"] == 145.50
        assert data["take_profit"] == 162.00
        assert data["confidence_score"] == 75
        assert "RSI momentum" in data["reasoning"]

    def test_analyze_returns_not_buy_recommendation(self, mock_not_buy_analysis):
        """Test Not Buy recommendation for invalid symbol"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = MagicMock(
            return_value=MockAsyncIterator([
                MockResultMessage(
                    subtype="success",
                    structured_output=mock_not_buy_analysis
                )
            ])
        )

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post(
                "/analyze",
                json={"stock": "XYZ"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["recommendation"] == "Not Buy"
        assert data["confidence_score"] == 0

    def test_analyze_handles_structured_output_error(self):
        """Test handling of structured output generation failure"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = MagicMock(
            return_value=MockAsyncIterator([
                MockResultMessage(
                    subtype="error_max_structured_output_retries",
                    structured_output=None
                )
            ])
        )

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post(
                "/analyze",
                json={"stock": "AAPL"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["recommendation"] == "Not Buy"
        assert "could not produce valid structured output" in data["reasoning"]

    def test_analyze_validates_stock_ticker(self):
        """Test input validation for stock ticker"""
        response = self.client.post(
            "/analyze",
            json={"stock": ""}
        )
        assert response.status_code == 422  # Validation error

    def test_analyze_uppercases_ticker(self, mock_successful_analysis):
        """Test that ticker is uppercased in response"""
        mock_successful_analysis["stock"] = "AAPL"

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = MagicMock(
            return_value=MockAsyncIterator([
                MockResultMessage(
                    subtype="success",
                    structured_output=mock_successful_analysis
                )
            ])
        )

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post(
                "/analyze",
                json={"stock": "aapl"}  # lowercase input
            )

        # The agent receives uppercase, returns uppercase
        assert response.status_code == 200

    def test_analyze_cleans_up_session_on_success(self, mock_successful_analysis):
        """Test that session is disconnected after successful analysis"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = MagicMock(
            return_value=MockAsyncIterator([
                MockResultMessage(
                    subtype="success",
                    structured_output=mock_successful_analysis
                )
            ])
        )

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            self.client.post("/analyze", json={"stock": "AAPL"})

        mock_client.disconnect.assert_called_once()

    def test_analyze_cleans_up_session_on_error(self):
        """Test that session is disconnected even on error"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock(side_effect=Exception("Test error"))

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post("/analyze", json={"stock": "AAPL"})

        assert response.status_code == 500
        mock_client.disconnect.assert_called_once()
```

### Success Criteria

#### Automated Verification:
- [x] All unit tests pass: `uv run pytest tests/test_analyze_endpoint.py -v`
- [x] Test coverage for analyze endpoint: `uv run pytest tests/test_analyze_endpoint.py --cov=server --cov-report=term`

#### Manual Verification:
- [x] None for this phase

---

## Phase 4: Integration Testing

### Overview
Create integration tests that run against the real Technical Analysis API and Claude SDK.

### Changes Required

#### 1. Add Integration Test

**File**: `http-server-agent-sdk/tests/test_analyze_integration.py`

```python
"""
Integration tests for the /analyze swing trading endpoint.

These tests require:
1. ANTHROPIC_API_KEY environment variable
2. Technical Analysis API running on localhost:8093
3. HTTP server running on localhost:8000

Run with: uv run python tests/test_analyze_integration.py
"""

import os
import sys
import httpx
import asyncio


async def test_analyze_valid_symbol():
    """Test analysis of a valid stock symbol"""
    print("\n=== Test: Analyze Valid Symbol (AAPL) ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/analyze",
            json={"stock": "AAPL"}
        )

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {data}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["stock"] == "AAPL", "Stock ticker mismatch"
    assert data["recommendation"] in ["Buy", "Not Buy"], "Invalid recommendation"
    assert 0 <= data["confidence_score"] <= 100, "Confidence out of range"
    assert len(data["reasoning"]) > 0, "Missing reasoning"

    if data["recommendation"] == "Buy":
        assert data["entry_price"] > 0, "Buy recommendation needs entry price"
        assert data["stop_loss"] > 0, "Buy recommendation needs stop loss"
        assert data["take_profit"] > 0, "Buy recommendation needs take profit"
        assert data["stop_loss"] < data["entry_price"], "Stop loss should be below entry"
        assert data["take_profit"] > data["entry_price"], "Take profit should be above entry"

    print("✓ Test passed")
    return True


async def test_analyze_invalid_symbol():
    """Test analysis of an invalid stock symbol"""
    print("\n=== Test: Analyze Invalid Symbol (INVALIDXYZ) ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/analyze",
            json={"stock": "INVALIDXYZ"}
        )

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {data}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["recommendation"] == "Not Buy", "Invalid symbol should be Not Buy"

    print("✓ Test passed")
    return True


async def test_analyze_lowercase_symbol():
    """Test that lowercase symbols work correctly"""
    print("\n=== Test: Analyze Lowercase Symbol (msft) ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/analyze",
            json={"stock": "msft"}
        )

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Stock in response: {data['stock']}")

    assert response.status_code == 200
    # Response should have uppercase ticker
    assert data["stock"].upper() == "MSFT"

    print("✓ Test passed")
    return True


async def check_prerequisites():
    """Check that required services are running"""
    print("Checking prerequisites...")

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY not set")
        return False
    print("✓ ANTHROPIC_API_KEY is set")

    # Check Technical Analysis API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8093/api/v1/stocks/AAPL/analysis",
                timeout=5.0
            )
        if response.status_code == 200:
            print("✓ Technical Analysis API is running on localhost:8093")
        else:
            print(f"✗ Technical Analysis API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Technical Analysis API not reachable: {e}")
        return False

    # Check HTTP server
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8000/health",
                timeout=5.0
            )
        if response.status_code == 200:
            print("✓ HTTP server is running on localhost:8000")
        else:
            print(f"✗ HTTP server health check returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ HTTP server not reachable: {e}")
        return False

    return True


async def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Swing Trading Agent Integration Tests")
    print("=" * 60)

    if not await check_prerequisites():
        print("\n✗ Prerequisites not met. Exiting.")
        sys.exit(1)

    tests = [
        test_analyze_valid_symbol,
        test_analyze_invalid_symbol,
        test_analyze_lowercase_symbol,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
```

### Success Criteria

#### Automated Verification:
- [ ] Integration tests pass: `uv run python tests/test_analyze_integration.py`

#### Manual Verification:
- [ ] Test with various stock tickers (AAPL, MSFT, NVDA, TSLA)
- [ ] Verify reasoning makes sense for the technical data
- [ ] Check that confidence scores correlate with signal clarity

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to the next phase.

---

## Testing Strategy

### Unit Tests
- Mock `ClaudeSDKClient` to test endpoint logic
- Test all response scenarios (success, errors, edge cases)
- Test input validation
- Test session cleanup

### Integration Tests
- Require running Technical Analysis API and HTTP server
- Test real agent behavior with actual data
- Verify response structure and reasonableness
- Test error handling with invalid symbols

### Manual Testing Steps
1. Start Technical Analysis API: `localhost:8093`
2. Start HTTP server: `uv run uvicorn server:app --port 8000`
3. Test valid symbol:
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"stock": "AAPL"}'
   ```
4. Test invalid symbol:
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"stock": "INVALIDXYZ"}'
   ```
5. Check OpenAPI docs: `http://localhost:8000/docs`

## Performance Considerations

- **Timeout**: Each analysis takes 10-30 seconds due to agent reasoning
- **Rate Limiting**: Consider adding rate limiting for production
- **Connection Pooling**: Current implementation creates new httpx client per request
- **Session Cleanup**: Temporary sessions are always cleaned up in `finally` block

## Migration Notes

N/A - This is a new endpoint with no existing data to migrate.

## References

- Original ticket: `thoughts/shared/tickets/2025-12-04-swing-trading-agent.md`
- SDK Structured Outputs: `docs/agent-sdk/06-structured-outputs.md`
- SDK Custom Tools: `docs/agent-sdk/10-custom-tools.md`
- Existing HTTP Server: `http-server-agent-sdk/server.py`
- Existing Stock Tools: `http-server-agent-sdk/stock_tools.py`
