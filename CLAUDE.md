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

## Response Metadata Standard

**CRITICAL**: All agent endpoints that interact with the Claude SDK MUST include a `metadata` field in their response.

The `metadata` field must contain:
- `model`: The model used for the response (e.g., "claude-sonnet-4-5-20250929")
- `system_prompt`: The system prompt used (if available)
- `user_prompt`: The user prompt sent to the agent
- `usage`: Token counts with `input_tokens` and `output_tokens`
- `tool_calls`: List of tool calls with parameters and counts

### Tool Call Structure

Each tool call in `tool_calls` contains:
- `name`: Tool name (e.g., "Read", "Bash")
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
    "usage": {"input_tokens": 100, "output_tokens": 50},
    "tool_calls": []
  }
}
```

