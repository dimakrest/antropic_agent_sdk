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
├── tests/
│   ├── test_server.py     # Unit tests (mocked SDK)
│   └── integration_test.py # Integration tests (real SDK)
├── pyproject.toml         # Project config and dependencies
├── uv.lock                # Locked dependencies
└── .env.example           # Environment variable template
```

## Testing

- Unit tests use mocked SDK client - no API key needed
- Integration tests require `ANTHROPIC_API_KEY` and a running server
- Always run unit tests after making changes: `uv run pytest tests/test_server.py -v`

## Key Implementation Details

- `SessionManager` manages `ClaudeSDKClient` instances per HTTP session
- Sessions auto-expire after 1 hour (configurable via `SESSION_TIMEOUT_HOURS`)
- Permission callback (`default_can_use_tool`) is only active in `default` permission mode
- Providing a non-existent `session_id` returns 404 (not a silent new session)
