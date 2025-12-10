"""
Tests for Stock Tools Module

Run unit tests: uv run pytest tests/test_stock_tools.py -v -m "not integration"
Run all tests:  uv run pytest tests/test_stock_tools.py -v
"""

import pytest
import httpx
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_tools import get_stock_data, calculate_position_size, stock_tools_server


# =============================================================================
# Position Size Calculator Tests
# =============================================================================

class TestCalculatePositionSize:
    """Tests for calculate_position_size function"""

    def test_basic_calculation(self):
        """Standard position size calculation"""
        result = calculate_position_size(
            account_size=100000,
            risk_percent=2,
            entry_price=234.50,
            stop_loss_price=225.50
        )

        assert result["inputs"]["account_size"] == 100000
        assert result["calculations"]["max_risk_dollars"] == 2000.00
        assert result["calculations"]["risk_per_share"] == 9.00
        assert result["calculations"]["position_size_shares"] == 222

    def test_small_account(self):
        """Small account position sizing"""
        result = calculate_position_size(
            account_size=10000,
            risk_percent=1,
            entry_price=500.00,
            stop_loss_price=490.00
        )

        assert result["calculations"]["max_risk_dollars"] == 100.00
        assert result["calculations"]["position_size_shares"] == 10

    def test_zero_risk_error(self):
        """Entry equals stop loss should return error"""
        result = calculate_position_size(
            account_size=100000,
            risk_percent=2,
            entry_price=234.50,
            stop_loss_price=234.50
        )

        assert "error" in result
        assert "same" in result["error"].lower()

    def test_short_position(self):
        """Short position (stop above entry)"""
        result = calculate_position_size(
            account_size=100000,
            risk_percent=2,
            entry_price=234.50,
            stop_loss_price=244.50
        )

        assert result["calculations"]["risk_per_share"] == 10.00
        assert result["calculations"]["position_size_shares"] == 200

    def test_negative_account_size(self):
        """Negative account size should return error"""
        result = calculate_position_size(
            account_size=-10000,
            risk_percent=2,
            entry_price=100.00,
            stop_loss_price=95.00
        )

        assert "error" in result
        assert "account size" in result["error"].lower()

    def test_zero_account_size(self):
        """Zero account size should return error"""
        result = calculate_position_size(
            account_size=0,
            risk_percent=2,
            entry_price=100.00,
            stop_loss_price=95.00
        )

        assert "error" in result

    def test_negative_risk_percent(self):
        """Negative risk percent should return error"""
        result = calculate_position_size(
            account_size=100000,
            risk_percent=-2,
            entry_price=100.00,
            stop_loss_price=95.00
        )

        assert "error" in result
        assert "risk percent" in result["error"].lower()

    def test_negative_entry_price(self):
        """Negative entry price should return error"""
        result = calculate_position_size(
            account_size=100000,
            risk_percent=2,
            entry_price=-100.00,
            stop_loss_price=95.00
        )

        assert "error" in result
        assert "entry price" in result["error"].lower()

    def test_negative_stop_loss(self):
        """Negative stop loss should return error"""
        result = calculate_position_size(
            account_size=100000,
            risk_percent=2,
            entry_price=100.00,
            stop_loss_price=-95.00
        )

        assert "error" in result
        assert "stop loss" in result["error"].lower()

    def test_percent_of_account(self):
        """Verify percent of account calculation"""
        result = calculate_position_size(
            account_size=100000,
            risk_percent=2,
            entry_price=234.50,
            stop_loss_price=225.50
        )

        # 222 shares * $234.50 = $52,059
        # 52,059 / 100,000 = 52.06%
        assert result["calculations"]["percent_of_account"] == 52.06

    def test_position_value(self):
        """Verify position value calculation"""
        result = calculate_position_size(
            account_size=100000,
            risk_percent=2,
            entry_price=234.50,
            stop_loss_price=225.50
        )

        expected_value = 222 * 234.50  # 52,059
        assert result["calculations"]["position_value"] == round(expected_value, 2)


# =============================================================================
# MCP Server Tests
# =============================================================================

class TestStockToolsServer:
    """Tests for the MCP server configuration"""

    def test_server_created(self):
        """Verify MCP server is properly configured"""
        assert stock_tools_server is not None
        assert stock_tools_server["type"] == "sdk"
        assert stock_tools_server["name"] == "stock_analysis"


# =============================================================================
# Stock Data Tool Tests (Unit - Mocked)
# =============================================================================

@pytest.fixture
def mock_analysis_api_response():
    """Mock response from Technical Analysis API"""
    return {
        "symbol": "AAPL",
        "period": "3mo",
        "interval": "1d",
        "price": {
            "current": 234.50,
            "open": 233.00,
            "high": 236.80,
            "low": 232.50,
            "previous_close": 232.00,
            "change": 2.50,
            "change_percent": 1.08
        },
        "volume": {
            "current": 58000000,
            "avg_20d": 50000000,
            "avg_50d": 48000000,
            "ratio_vs_20d_avg": 1.16
        },
        "moving_averages": {
            "sma_20": 230.00,
            "sma_50": 225.00,
            "sma_200": 210.00,
            "ema_12": 232.00,
            "ema_26": 228.00
        },
        "momentum": {
            "rsi_14": 58.5,
            "rsi_7": 62.3,
            "macd_line": 2.30,
            "macd_signal": 1.80,
            "macd_histogram": 0.50,
            "stochastic_k": 75.2,
            "stochastic_d": 70.8
        },
        "volatility": {
            "atr_14": 4.50,
            "atr_7": 5.20,
            "bollinger_upper": 245.00,
            "bollinger_middle": 230.00,
            "bollinger_lower": 215.00,
            "bollinger_width": 13.04
        },
        "trend": {
            "adx_14": 28.5,
            "plus_di": 32.1,
            "minus_di": 18.4
        },
        "levels": {
            "support": [225.00, 220.00, 210.00],
            "resistance": [240.00, 250.00, 260.00],
            "pivot_point": 234.10,
            "52_week_high": 260.00,
            "52_week_low": 180.00
        },
        "recent_candles": [
            {
                "date": "2025-12-03",
                "open": 233.0,
                "high": 236.8,
                "low": 232.5,
                "close": 234.5,
                "volume": 58000000
            }
        ]
    }


class TestGetStockDataUnit:
    """Unit tests for get_stock_data MCP tool with mocked API"""

    @pytest.mark.asyncio
    async def test_response_structure(self, mock_analysis_api_response, mocker):
        """Verify response structure matches MCP content format"""
        mock_response = mocker.Mock()
        mock_response.json.return_value = mock_analysis_api_response
        mock_response.status_code = 200

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        # Use args dict pattern for MCP tool
        result = await get_stock_data({"symbol": "AAPL", "period": "3mo"})

        # Verify MCP content format
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"

        # Parse the JSON content and verify structure
        import json
        data = json.loads(result["content"][0]["text"])
        assert "symbol" in data
        assert "price" in data
        assert "momentum" in data
        assert "moving_averages" in data
        assert "volatility" in data
        assert "trend" in data
        assert "levels" in data
        assert "volume" in data

    @pytest.mark.asyncio
    async def test_api_unavailable(self, mocker):
        """Handle API unavailable gracefully"""
        mock_client = mocker.AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await get_stock_data({"symbol": "AAPL"})

        # Verify MCP error format
        assert "content" in result
        assert "unavailable" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_invalid_symbol(self, mocker):
        """Handle invalid symbol"""
        mock_response = mocker.Mock()
        mock_response.status_code = 404

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await get_stock_data({"symbol": "INVALID123"})

        assert "content" in result
        assert "not found" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_api_error_status(self, mocker):
        """Handle non-200/404 API errors"""
        mock_response = mocker.Mock()
        mock_response.status_code = 500

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await get_stock_data({"symbol": "AAPL"})

        assert "content" in result
        assert "500" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_default_parameters(self, mock_analysis_api_response, mocker):
        """Verify default parameters are passed correctly"""
        mock_response = mocker.Mock()
        mock_response.json.return_value = mock_analysis_api_response
        mock_response.status_code = 200

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        await get_stock_data({"symbol": "AAPL"})

        # Verify the call was made with default params
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args.kwargs["params"]["period"] == "3mo"
        assert call_args.kwargs["params"]["interval"] == "1d"

    @pytest.mark.asyncio
    async def test_custom_parameters(self, mock_analysis_api_response, mocker):
        """Verify custom parameters are passed correctly"""
        mock_response = mocker.Mock()
        mock_response.json.return_value = mock_analysis_api_response
        mock_response.status_code = 200

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        await get_stock_data({"symbol": "MSFT", "period": "6mo", "interval": "1wk"})

        call_args = mock_client.get.call_args
        # URL format: /api/v1/stocks/analysis/{symbol}
        assert "analysis/MSFT" in call_args.args[0]
        assert call_args.kwargs["params"]["period"] == "6mo"
        assert call_args.kwargs["params"]["interval"] == "1wk"


# =============================================================================
# Integration Tests (Require Technical Analysis API on localhost:8093)
# =============================================================================

@pytest.mark.integration
class TestGetStockDataIntegration:
    """Integration tests - require Technical Analysis API on localhost:8093"""

    @pytest.mark.asyncio
    async def test_real_api_aapl(self):
        """Integration test with real API for AAPL"""
        import json
        result = await get_stock_data({"symbol": "AAPL", "period": "1mo"})

        # Check for API unavailable error
        if "unavailable" in result["content"][0]["text"].lower():
            pytest.skip("Technical Analysis API not available")

        # Parse the JSON response
        data = json.loads(result["content"][0]["text"])

        assert data["symbol"] == "AAPL"
        assert data["current_price"] > 0
        # RSI can be None or a number
        if data["momentum"]["rsi_14"] is not None:
            assert 0 <= data["momentum"]["rsi_14"] <= 100
        assert data["moving_averages"]["sma_20"] is not None

    @pytest.mark.asyncio
    async def test_real_api_invalid_symbol(self):
        """Integration test - invalid symbol handling"""
        result = await get_stock_data({"symbol": "THISSYMBOLSHOULDNOTEXIST12345"})

        # Check for API unavailable error
        if "unavailable" in result["content"][0]["text"].lower():
            pytest.skip("Technical Analysis API not available")

        # Either "error" or "not found" indicates invalid symbol handling
        text_lower = result["content"][0]["text"].lower()
        assert "error" in text_lower or "not found" in text_lower

    @pytest.mark.asyncio
    async def test_real_api_different_periods(self):
        """Integration test - verify different periods work"""
        import json
        result = await get_stock_data({"symbol": "MSFT", "period": "6mo", "interval": "1wk"})

        # Check for API unavailable error
        if "unavailable" in result["content"][0]["text"].lower():
            pytest.skip("Technical Analysis API not available")

        # Parse the JSON response
        data = json.loads(result["content"][0]["text"])

        assert data["symbol"] == "MSFT"
        assert data["period"] == "6mo"
        assert data["interval"] == "1wk"
