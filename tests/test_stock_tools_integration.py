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
