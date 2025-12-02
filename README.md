# Claude Agent HTTP Server

HTTP server interface for the Claude Agent SDK, enabling HTTP-based interactions with Claude agents.

## Features

- **REST API**: Simple HTTP interface for Claude Agent SDK
- **Multi-turn conversations**: Maintains context across requests
- **Session management**: Create, manage, and terminate sessions
- **Tool support**: Agent can read/write files, run commands
- **Configurable permissions**: Control agent's file system access
- **Auto-generated docs**: Interactive API documentation at /docs

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or create project directory
cd http-server-agent-sdk

# Install dependencies (creates .venv automatically)
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run Server

```bash
# Start server
uv run python server.py

# Or with uvicorn directly:
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: **http://localhost:8000**

## API Usage

### Interactive Documentation

Open http://localhost:8000/docs for interactive API documentation with try-it-out functionality.

### Example: Simple Query

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'
```

Response:
```json
{
  "session_id": "8f7e6d5c-4b3a-2f1e-9d8c-7b6a5e4d3c2b",
  "status": "success",
  "response_text": "Python is a high-level programming language...",
  "conversation_turns": 1
}
```

### Example: Multi-turn Conversation

```bash
# First message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about FastAPI"}' > response1.json

# Extract session_id from response
SESSION_ID=$(cat response1.json | jq -r .session_id)

# Continue conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION_ID\", \"message\": \"Show me a code example\"}"
```

### Example: With File Operations

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a Python hello world script in hello.py",
    "permission_mode": "acceptEdits"
  }'
```

### Example: Delete Session

```bash
curl -X DELETE http://localhost:8000/sessions/$SESSION_ID
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and server status |
| `/chat` | POST | Send message to agent |
| `/sessions` | GET | List active sessions |
| `/sessions/{id}` | DELETE | Terminate a session |
| `/sessions/{id}/interrupt` | POST | Interrupt running agent |
| `/docs` | GET | Interactive API documentation |

## Configuration

Environment variables in `.env`:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional
HOST=0.0.0.0
PORT=8000
SESSION_TIMEOUT_HOURS=1
MAX_TURNS=10
DEFAULT_PERMISSION_MODE=default  # default, acceptEdits, bypassPermissions
```

## Permission Modes

- **`default`**: Prompts for approval on sensitive operations (recommended)
- **`acceptEdits`**: Auto-approves file edits, still prompts for dangerous commands
- **`bypassPermissions`**: No prompts (use with caution)

## Testing

### Unit Tests (Mocked SDK)

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/test_server.py -v

# With coverage
uv run pytest tests/test_server.py --cov=server --cov-report=html
```

### Integration Tests (Real SDK)

```bash
# Terminal 1: Start server
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python server.py

# Terminal 2: Run integration tests
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python tests/integration_test.py
```

## Python Client Example

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000", timeout=120)

# Start conversation
response = client.post("/chat", json={
    "message": "Help me write a Python function"
})

session_id = response.json()["session_id"]
print(response.json()["response_text"])

# Continue conversation
response = client.post("/chat", json={
    "session_id": session_id,
    "message": "Now add error handling to that function"
})

print(response.json()["response_text"])

# Clean up
client.delete(f"/sessions/{session_id}")
```

## Architecture

- **FastAPI**: Async web framework
- **ClaudeSDKClient**: Maintains persistent SDK connections
- **Session Management**: In-memory storage with TTL cleanup
- **Single Message Mode**: HTTP request = one message to agent

## Troubleshooting

**Server won't start:**
- Check ANTHROPIC_API_KEY is set: `echo $ANTHROPIC_API_KEY`
- Verify Python version: `python --version` (need 3.10+)
- Check port 8000 is available: `lsof -i :8000`

**Agent responses are slow:**
- Normal for complex queries (can take 30-60 seconds)
- Increase timeout in client: `timeout=120`

**Session not found error:**
- Sessions expire after 1 hour of inactivity
- Check `/sessions` endpoint to see active sessions

## License

MIT

## Links

- [Claude Agent SDK Documentation](https://platform.claude.com/docs/en/agent-sdk/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Anthropic API](https://console.anthropic.com/)
