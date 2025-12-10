"""
Unit tests for the /analyze swing trading endpoint.

These tests mock the ClaudeSDKClient to test endpoint behavior
without requiring an actual API key or Technical Analysis API.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app
from claude_agent_sdk import ResultMessage


def create_result_message(subtype="success", structured_output=None):
    """Create a ResultMessage with structured output for testing"""
    msg = ResultMessage(
        subtype=subtype,
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="test-session"
    )
    # Add structured_output attribute
    if structured_output is not None:
        msg.structured_output = structured_output
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
        self.client = TestClient(app)

    def test_analyze_returns_buy_recommendation(self, mock_successful_analysis):
        """Test successful Buy recommendation"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            yield create_result_message(
                subtype="success",
                structured_output=mock_successful_analysis
            )

        mock_client.receive_response = mock_receive_response

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

        async def mock_receive_response():
            yield create_result_message(
                subtype="success",
                structured_output=mock_not_buy_analysis
            )

        mock_client.receive_response = mock_receive_response

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

        async def mock_receive_response():
            yield create_result_message(
                subtype="error_max_structured_output_retries",
                structured_output=None
            )

        mock_client.receive_response = mock_receive_response

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

    def test_analyze_validates_stock_ticker_too_long(self):
        """Test input validation for stock ticker that's too long"""
        response = self.client.post(
            "/analyze",
            json={"stock": "VERYLONGTICKER123"}
        )
        assert response.status_code == 422  # Validation error

    def test_analyze_uppercases_ticker(self, mock_successful_analysis):
        """Test that ticker is uppercased in prompt"""
        mock_successful_analysis["stock"] = "AAPL"

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            yield create_result_message(
                subtype="success",
                structured_output=mock_successful_analysis
            )

        mock_client.receive_response = mock_receive_response

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post(
                "/analyze",
                json={"stock": "aapl"}  # lowercase input
            )

        # Verify the query was called with uppercased ticker
        mock_client.query.assert_called_once()
        call_args = mock_client.query.call_args
        assert "AAPL" in call_args.kwargs["prompt"]

        # The agent receives uppercase, returns uppercase
        assert response.status_code == 200

    def test_analyze_cleans_up_session_on_success(self, mock_successful_analysis):
        """Test that session is disconnected after successful analysis"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            yield create_result_message(
                subtype="success",
                structured_output=mock_successful_analysis
            )

        mock_client.receive_response = mock_receive_response

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

    def test_analyze_handles_generic_error_result(self):
        """Test handling of generic error result from agent"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            yield create_result_message(
                subtype="error_during_execution",
                structured_output=None
            )

        mock_client.receive_response = mock_receive_response

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post(
                "/analyze",
                json={"stock": "AAPL"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["recommendation"] == "Not Buy"
        assert "error_during_execution" in data["reasoning"]

    def test_analyze_handles_no_result(self):
        """Test handling when no result message is received"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            # Empty async generator - no messages
            return
            yield  # Make it a generator

        mock_client.receive_response = mock_receive_response

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post(
                "/analyze",
                json={"stock": "AAPL"}
            )

        assert response.status_code == 500
        assert "No result received" in response.json()["detail"]

    def test_analyze_missing_stock_field(self):
        """Test request with missing stock field"""
        response = self.client.post(
            "/analyze",
            json={}
        )
        assert response.status_code == 422

    def test_analyze_response_schema_validation(self, mock_successful_analysis):
        """Test that response matches the expected schema"""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            yield create_result_message(
                subtype="success",
                structured_output=mock_successful_analysis
            )

        mock_client.receive_response = mock_receive_response

        with patch('server.ClaudeSDKClient', return_value=mock_client):
            response = self.client.post(
                "/analyze",
                json={"stock": "AAPL"}
            )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields are present
        assert "stock" in data
        assert "recommendation" in data
        assert "entry_price" in data
        assert "stop_loss" in data
        assert "take_profit" in data
        assert "reasoning" in data
        assert "missing_tools" in data
        assert "confidence_score" in data

        # Verify types
        assert isinstance(data["stock"], str)
        assert isinstance(data["recommendation"], str)
        assert isinstance(data["entry_price"], (int, float))
        assert isinstance(data["stop_loss"], (int, float))
        assert isinstance(data["take_profit"], (int, float))
        assert isinstance(data["reasoning"], str)
        assert isinstance(data["missing_tools"], list)
        assert isinstance(data["confidence_score"], int)
