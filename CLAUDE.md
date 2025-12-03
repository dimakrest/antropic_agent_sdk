# CLAUDE.md

Instructions for Claude Code when working in this project.

## Package Manager

**CRITICAL**: This project uses `uv` as the package manager.

- **ALWAYS** use `uv run` to execute Python commands
- **NEVER** manually activate the venv with `source .venv/bin/activate`

### Common Commands

```bash
# Run the server
uv run python server.py

# Run with uvicorn (with reload)
uv run uvicorn server:app --reload

# Run tests
uv run pytest tests/test_server.py -v

# Run tests with coverage
uv run pytest tests/test_server.py --cov=server --cov-report=term

# Run integration tests
uv run python tests/integration_test.py

# Install/sync dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>
```

## Project Structure

```
http-server-agent-sdk/
├── server.py              # Main FastAPI application
├── stock_tools.py         # Custom MCP tools for stock price data
├── tests/
│   ├── test_server.py     # Unit tests (mocked SDK)
│   ├── test_stock_tools.py # Unit tests for stock tools
│   ├── test_stock_tools_integration.py # Integration tests for stock tools
│   └── integration_test.py # Integration tests (real SDK)
├── pyproject.toml         # Project config and dependencies
├── uv.lock                # Locked dependencies
└── .env.example           # Environment variable template
```

## Testing

- Unit tests use mocked SDK client - no API key needed
- Integration tests require a running server and Stock Prices API
- The Claude Agent SDK automatically picks up the API key from the environment (no `.env` file needed when running from Claude Code)
- Always run unit tests after making changes: `uv run pytest tests/ -v`

## Stock Tools

The server includes custom MCP tools for fetching stock price data:

### Available Tools

- `mcp__stock-tools__get_stock_prices` - Fetch historical OHLCV data for a stock symbol

### Testing Stock Tools

1. Ensure Stock Prices API is running at `localhost:8093`
2. Start the HTTP server: `uv run uvicorn server:app --port 8000`
3. Test with curl:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Get me the stock prices for AAPL for the last month with daily intervals.",
    "permission_mode": "acceptEdits",
    "allowed_tools": ["mcp__stock-tools__get_stock_prices"]
  }'
```

## Key Implementation Details

- `SessionManager` manages `ClaudeSDKClient` instances per HTTP session
- Sessions auto-expire after 1 hour (configurable via `SESSION_TIMEOUT_HOURS`)
- Permission callback (`default_can_use_tool`) is only active in `default` permission mode
- Providing a non-existent `session_id` returns 404 (not a silent new session)
