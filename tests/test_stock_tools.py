"""
Unit tests for stock_tools module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from stock_tools import get_stock_prices, stock_tools_server

# Get the underlying handler function from the decorated tool
get_stock_prices_handler = get_stock_prices.handler


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

            result = await get_stock_prices_handler({
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

            result = await get_stock_prices_handler({
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
        result = await get_stock_prices_handler({
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
        result = await get_stock_prices_handler({
            "symbol": "AAPL",
            "start_date": "2025-01-01"
        })

        assert "Error" in result["content"][0]["text"]
        assert "end_date" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_validation_no_time_range(self):
        """Test validation error when no time range specified."""
        result = await get_stock_prices_handler({
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

            result = await get_stock_prices_handler({
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

            result = await get_stock_prices_handler({
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

            result = await get_stock_prices_handler({
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

            result = await get_stock_prices_handler({
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

            result = await get_stock_prices_handler({
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

            await get_stock_prices_handler({
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

            await get_stock_prices_handler({
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

            result = await get_stock_prices_handler({
                "symbol": "AAPL",
                "period": "1mo"
            })

        # Should handle gracefully - empty data case
        assert "No price data" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_error_response_includes_isError_flag(self):
        """Test that error responses include isError: True for MCP protocol."""
        result = await get_stock_prices_handler({
            "symbol": "AAPL"
            # Missing required period or date range
        })

        assert result.get("isError") is True
        assert "Error" in result["content"][0]["text"]


class TestStockToolsServer:
    """Tests for the MCP server configuration."""

    def test_server_name(self):
        """Test server has correct name."""
        assert stock_tools_server["name"] == "stock-tools"

    def test_server_type(self):
        """Test server has correct type."""
        assert stock_tools_server["type"] == "sdk"

    def test_server_has_instance(self):
        """Test server has MCP instance."""
        assert "instance" in stock_tools_server
        assert stock_tools_server["instance"] is not None

    def test_tool_name(self):
        """Test tool has correct name."""
        assert get_stock_prices.name == "get_stock_prices"
