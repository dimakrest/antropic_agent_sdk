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

from stock_tools import get_stock_data, calculate_position_size


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
    """Unit tests for get_stock_data with mocked API"""

    @pytest.mark.asyncio
    async def test_response_structure(self, mock_analysis_api_response, mocker):
        """Verify response structure matches spec"""
        mock_response = mocker.Mock()
        mock_response.json.return_value = mock_analysis_api_response
        mock_response.status_code = 200

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await get_stock_data("AAPL", period="3mo")

        assert "symbol" in result
        assert "price" in result
        assert "momentum" in result
        assert "moving_averages" in result
        assert "volatility" in result
        assert "trend" in result
        assert "levels" in result
        assert "volume" in result

    @pytest.mark.asyncio
    async def test_api_unavailable(self, mocker):
        """Handle API unavailable gracefully"""
        mock_client = mocker.AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await get_stock_data("AAPL")

        assert "error" in result
        assert "unavailable" in result["error"].lower()

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

        result = await get_stock_data("INVALID123")

        assert "error" in result
        assert "not found" in result["error"].lower()

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

        result = await get_stock_data("AAPL")

        assert "error" in result
        assert "500" in result["error"]

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

        await get_stock_data("AAPL")

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

        await get_stock_data("MSFT", period="6mo", interval="1wk")

        call_args = mock_client.get.call_args
        assert "MSFT" in call_args.args[0]
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
        result = await get_stock_data("AAPL", period="1mo")

        # Skip if API is unavailable
        if "error" in result and "unavailable" in result["error"]:
            pytest.skip("Technical Analysis API not available")

        assert result["symbol"] == "AAPL"
        assert result["price"]["current"] > 0
        assert 0 <= result["momentum"]["rsi_14"] <= 100
        assert result["moving_averages"]["sma_20"] is not None

    @pytest.mark.asyncio
    async def test_real_api_invalid_symbol(self):
        """Integration test - invalid symbol handling"""
        result = await get_stock_data("THISSYMBOLSHOULDNOTEXIST12345")

        # Skip if API is unavailable
        if "error" in result and "unavailable" in result["error"]:
            pytest.skip("Technical Analysis API not available")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_real_api_different_periods(self):
        """Integration test - verify different periods work"""
        result = await get_stock_data("MSFT", period="6mo", interval="1wk")

        # Skip if API is unavailable
        if "error" in result and "unavailable" in result["error"]:
            pytest.skip("Technical Analysis API not available")

        assert result["symbol"] == "MSFT"
        assert result["period"] == "6mo"
        assert result["interval"] == "1wk"
