"""
Pytest configuration for http-server-agent-sdk tests.

This file is automatically loaded by pytest before any test modules.
It sets up required environment variables for tests to run.
"""

import os

# Set TRADING_API_BASE_URL before any test modules import stock_tools
# This must happen at conftest load time (before module imports)
os.environ.setdefault("TRADING_API_BASE_URL", "http://localhost:8131/api/v1")
