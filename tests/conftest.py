"""
Pytest configuration and fixtures.

This file sets up environment variables needed for testing before
any test modules are imported.
"""
import os

# Set required environment variables for testing
# Must be done before any test imports (stock_tools.py validates at import)
os.environ.setdefault("TRADING_API_BASE_URL", "http://localhost:8131/api/v1")
