# CLAUDE.md

Instructions for Claude Code when working in this project.

## Package Manager

**CRITICAL**: This project uses `uv` as the package manager.

- **ALWAYS** use `uv run` to execute Python commands
- **NEVER** manually activate the venv with `source .venv/bin/activate`

### Common Commands

```bash
# Run the server (ALWAYS use port 8001)
uv run uvicorn server:app --port 8001

# Run with reload (development)
uv run uvicorn server:app --port 8001 --reload

# Run tests
uv run pytest tests/test_server.py -v

# Run tests with coverage
uv run pytest tests/test_server.py --cov=server --cov-report=term

# Run integration tests (requires server on port 8001 + Trading API on port 8131)
uv run python tests/test_analyze_integration.py

# Install/sync dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>
```

## Server Port

**CRITICAL**: This server MUST run on **port 8001** (not 8000).

Port 8000 is typically used by Docker on macOS. Always start the server with `--port 8001`.

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

## Response Metadata Standard

**CRITICAL**: All agent endpoints that interact with the Claude SDK MUST include a `metadata` field in their response.

The `metadata` field must contain:
- `model`: The model used for the response (e.g., "claude-sonnet-4-5-20250929")
- `system_prompt`: The system prompt used (if available)
- `user_prompt`: The user prompt sent to the agent
- `usage`: Token usage information (see below)
- `tool_calls`: List of tool calls with parameters and counts

### Usage Structure

The `usage` field contains token counts with prompt caching support:
- `input_tokens`: Non-cached input tokens (after last cache breakpoint)
- `output_tokens`: Number of output tokens generated
- `cache_creation_input_tokens`: Tokens used to create cache entries
- `cache_read_input_tokens`: Tokens read from cache (reduces cost)
- `total_input_tokens`: **True total** = input + cache_creation + cache_read
- `total_cost_usd`: Total cost in USD (may be null)

**Important**: With prompt caching, `input_tokens` only represents non-cached tokens. Always use `total_input_tokens` for accurate token counts.

### Tool Call Structure

Each tool call in `tool_calls` contains:
- `name`: Tool name (e.g., "Read", "Bash", "mcp__stock_analysis__get_stock_data")
- `input`: Dictionary of parameters passed to the tool
- `count`: Number of times this exact call (same name + params) was made

Tool calls are deduplicated by (name, input) combination. If the same tool is called with different parameters, each unique combination appears as a separate entry.

### Implementation Pattern

Use the `_build_metadata()` helper function to construct metadata:

```python
import json

# Initialize tracking variables at start of message processing
tool_calls_raw: Dict[str, Dict[str, Any]] = {}  # key = hash of (name, input)
model_used: Optional[str] = None
user_prompt = request.message  # Capture user prompt

# During message iteration
async for message in client.receive_response():
    if isinstance(message, AssistantMessage):
        if model_used is None:
            model_used = message.model
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                # Create unique key from tool name + sorted params
                key = json.dumps({"name": block.name, "input": block.input}, sort_keys=True)
                if key in tool_calls_raw:
                    tool_calls_raw[key]["count"] += 1
                else:
                    tool_calls_raw[key] = {"name": block.name, "input": block.input, "count": 1}
    if isinstance(message, ResultMessage):
        result_message = message

# Convert to list for metadata
tool_calls = list(tool_calls_raw.values())

# Build metadata for response
metadata = _build_metadata(
    model_used,
    tool_calls,
    result_message,
    system_prompt=SYSTEM_PROMPT,  # if applicable
    user_prompt=user_prompt
)
return YourResponse(..., metadata=metadata)
```

### Example Response

```json
{
  "metadata": {
    "model": "claude-sonnet-4-5-20250929",
    "system_prompt": "You are a professional swing trader...",
    "user_prompt": "Analyze AAPL for swing trading opportunities",
    "usage": {
      "input_tokens": 10,
      "output_tokens": 1001,
      "cache_creation_input_tokens": 16260,
      "cache_read_input_tokens": 30392,
      "total_input_tokens": 46662,
      "total_cost_usd": null
    },
    "tool_calls": [
      {"name": "mcp__stock_analysis__get_stock_data", "input": {"symbol": "AAPL"}, "count": 1},
      {"name": "Read", "input": {"path": "/data/analysis.txt"}, "count": 2}
    ]
  }
}
```

## /analyze Endpoint

**`POST /analyze`** - Analyze a stock for swing trading opportunities

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `stock` | string | Yes | Stock ticker symbol (e.g., "AAPL", "MSFT") |
| `analysis_date` | string | No | Historical analysis date in `YYYY-MM-DD` format |

### analysis_date Behavior

- **If provided**: Analysis uses data as of that date (passed to downstream Trading API)
- **If omitted**: Uses current date (default behavior)
- **Transparency**: The agent does NOT see this parameter - it's injected transparently via closure in the MCP tool

### Example Requests

```json
// Current date analysis (default)
{"stock": "AAPL"}

// Historical analysis
{"stock": "AAPL", "analysis_date": "2024-06-15"}
```

### Date Format Validation

The `analysis_date` must match the pattern `YYYY-MM-DD`. Invalid formats return HTTP 422:
- `2024/06/15` - Invalid (wrong separator)
- `20240615` - Invalid (no separators)
- `2024-06-15` - Valid
