# Enhanced Agent Response Metadata Implementation Plan

## Overview

Add metadata to the `/chat` endpoint response to expose tools used, token counts, and model information. This enhances transparency for debugging, cost tracking, and understanding agent behavior.

## Current State Analysis

The `ChatResponse` model (`server.py:74-81`) currently returns:
- `session_id`, `status`, `response_text`, `error`, `conversation_turns`

The SDK provides additional data that is NOT exposed:
- `AssistantMessage.model` - Model used for the response
- `AssistantMessage.content` contains `ToolUseBlock` with tool `name` field
- `ResultMessage.usage` - Dict with token usage including `input_tokens`, `output_tokens`

### Key Discoveries:
- `ResultMessage.usage` structure verified: `{'input_tokens': int, 'output_tokens': int, 'cache_creation_input_tokens': int, 'cache_read_input_tokens': int, ...}`
- `ToolUseBlock` has `name` field we can collect during message iteration
- `AssistantMessage.model` contains model ID string (e.g., `"claude-sonnet-4-5-20250929"`)

## Desired End State

The `/chat` endpoint will return an additional `metadata` field containing:
```json
{
  "session_id": "uuid",
  "status": "success",
  "response_text": "...",
  "error": null,
  "conversation_turns": 1,
  "metadata": {
    "model": "claude-sonnet-4-5-20250929",
    "usage": {
      "input_tokens": 100,
      "output_tokens": 50
    },
    "tools_used": {
      "Read": 2,
      "Bash": 1
    }
  }
}
```

### Verification:
- Existing API consumers continue to work (new field is additive)
- `metadata` field is present in all successful and error responses
- `usage` contains `input_tokens` and `output_tokens` (other fields omitted for simplicity)
- `tools_used` shows count per tool name
- `model` shows the actual model used

## What We're NOT Doing

- Per-tool execution metrics (latency, success/failure) - out of scope
- Cost calculation or billing integration - out of scope
- Historical token usage tracking across sessions - out of scope
- Token usage alerts or limits - out of scope
- Exposing all usage fields (cache_creation_input_tokens, etc.) - keep simple with just input/output tokens

## Implementation Approach

Single-phase implementation with:
1. Add new Pydantic models for metadata structure
2. Update message iteration to collect tool usage and model info
3. Update ChatResponse to include metadata
4. Update tests to verify new fields

## Phase 1: Add Response Metadata

### Overview
Add metadata models and update ChatResponse to include tools used, token counts, and model information.

### Changes Required:

#### 1. Add Metadata Models
**File**: `server.py`
**Location**: After `ChatResponse` model (around line 81)

Add new Pydantic models:

```python
class UsageInfo(BaseModel):
    """Token usage information"""
    input_tokens: int = Field(..., description="Number of input tokens consumed")
    output_tokens: int = Field(..., description="Number of output tokens generated")


class ResponseMetadata(BaseModel):
    """Metadata about the agent response"""
    model: Optional[str] = Field(None, description="Model used for the response")
    usage: Optional[UsageInfo] = Field(None, description="Token usage statistics")
    tools_used: Dict[str, int] = Field(
        default_factory=dict,
        description="Tools used during execution with count per tool"
    )
```

#### 2. Update ChatResponse Model
**File**: `server.py`
**Location**: Lines 74-81

Add metadata field:

```python
class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    session_id: str = Field(..., description="Session ID for continuing conversation")
    status: str = Field(..., description="Result status: 'success' or error type")
    response_text: Optional[str] = Field(None, description="Agent's text response")
    error: Optional[str] = Field(None, description="Error message if status != success")
    conversation_turns: int = Field(0, description="Number of turns in this response")
    metadata: Optional[ResponseMetadata] = Field(None, description="Response metadata including model, usage, and tools")
```

#### 3. Update Import Statement
**File**: `server.py`
**Location**: Line 22

Add `ToolUseBlock` to the imports:

```python
from claude_agent_sdk.types import TextBlock, ToolUseBlock
```

#### 4. Update Chat Endpoint Message Iteration
**File**: `server.py`
**Location**: Lines 470-490 (inside `chat` function)

Update to collect metadata during iteration:

```python
# Collect response messages
response_text_parts = []
conversation_turns = 0
tools_used: Dict[str, int] = {}  # Track tool usage counts
model_used: Optional[str] = None  # Track model from AssistantMessage

# Iterate through messages
# IMPORTANT: Never use `break` - causes asyncio cleanup issues per SDK docs
# Let the iteration complete naturally, use flags to track state
result_message = None
async for message in client.receive_response():
    # Count conversation turns for assistant messages
    if isinstance(message, AssistantMessage):
        conversation_turns += 1

        # Capture model from first AssistantMessage
        if model_used is None:
            model_used = message.model

        # Extract text content and track tool usage
        for block in message.content:
            if isinstance(block, TextBlock):
                response_text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                # Count tool usage
                tool_name = block.name
                tools_used[tool_name] = tools_used.get(tool_name, 0) + 1

    # Track result message (don't break - let iteration complete)
    if isinstance(message, ResultMessage):
        result_message = message
```

#### 5. Update Response Construction
**File**: `server.py`
**Location**: Lines 492-518 (result processing)

Add helper function before the chat endpoint and update response construction:

```python
def _build_metadata(
    model_used: Optional[str],
    tools_used: Dict[str, int],
    result_message: Optional[ResultMessage]
) -> ResponseMetadata:
    """Build response metadata from collected data"""
    usage_info = None
    if result_message and result_message.usage:
        usage_info = UsageInfo(
            input_tokens=result_message.usage.get("input_tokens", 0),
            output_tokens=result_message.usage.get("output_tokens", 0)
        )

    return ResponseMetadata(
        model=model_used,
        usage=usage_info,
        tools_used=tools_used
    )
```

Update all `ChatResponse` returns to include metadata:

```python
# Success case (around line 494-500)
if result_message.subtype == "success":
    return ChatResponse(
        session_id=session_id,
        status="success",
        response_text="\n".join(response_text_parts),
        conversation_turns=conversation_turns,
        metadata=_build_metadata(model_used, tools_used, result_message)
    )

# Error during execution case (around line 501-508)
elif result_message.subtype == "error_during_execution":
    return ChatResponse(
        session_id=session_id,
        status="error_during_execution",
        error="Agent encountered an error during execution",
        response_text="\n".join(response_text_parts),
        conversation_turns=conversation_turns,
        metadata=_build_metadata(model_used, tools_used, result_message)
    )

# Generic fallback case (around line 509-518)
else:
    error_msg = f"Agent returned non-success result: {result_message.subtype}"
    return ChatResponse(
        session_id=session_id,
        status=result_message.subtype or "error_unknown",
        error=error_msg,
        response_text="\n".join(response_text_parts),
        conversation_turns=conversation_turns,
        metadata=_build_metadata(model_used, tools_used, result_message)
    )
```

Also update the cancelled case (around line 526-533):

```python
except asyncio.CancelledError:
    # Client disconnected or interrupted
    return ChatResponse(
        session_id=session_id,
        status="cancelled",
        error="Request cancelled",
        conversation_turns=0,
        metadata=ResponseMetadata(tools_used={})  # No metadata available for cancelled requests
    )
```

#### 6. Update Unit Tests
**File**: `tests/test_server.py`

Add test imports:
```python
from claude_agent_sdk.types import TextBlock, ToolUseBlock
```

Update mock to include tool usage:
```python
@pytest.fixture
def mock_sdk_client_with_tools():
    """Create mock ClaudeSDKClient with tool usage"""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.query = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.interrupt = AsyncMock()

    async def mock_receive_response():
        # Simulate assistant message with tool use
        assistant_msg = AssistantMessage(
            content=[
                ToolUseBlock(id="tool_1", name="Read", input={"path": "/test"}),
                ToolUseBlock(id="tool_2", name="Read", input={"path": "/test2"}),
                ToolUseBlock(id="tool_3", name="Bash", input={"command": "ls"}),
                TextBlock(text="This is a test response from Claude")
            ],
            model="claude-sonnet-4-5-20250929"
        )
        yield assistant_msg

        # Simulate result message with usage
        result_msg = ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id="test-session",
            usage={
                "input_tokens": 150,
                "output_tokens": 75
            }
        )
        yield result_msg

    mock.receive_response = mock_receive_response
    return mock
```

Add new test for metadata:
```python
def test_chat_response_includes_metadata(test_client, mock_sdk_client_with_tools):
    """Test that chat response includes metadata with model, usage, and tools"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client_with_tools):
        response = test_client.post("/chat", json={
            "message": "Hello Claude"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify metadata structure
        assert "metadata" in data
        metadata = data["metadata"]

        # Verify model
        assert metadata["model"] == "claude-sonnet-4-5-20250929"

        # Verify usage
        assert "usage" in metadata
        assert metadata["usage"]["input_tokens"] == 150
        assert metadata["usage"]["output_tokens"] == 75

        # Verify tools_used with counts
        assert "tools_used" in metadata
        assert metadata["tools_used"]["Read"] == 2
        assert metadata["tools_used"]["Bash"] == 1
```

Add test for backward compatibility:
```python
def test_chat_response_backward_compatible(test_client, mock_sdk_client):
    """Test that existing response fields still work"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        response = test_client.post("/chat", json={
            "message": "Hello Claude"
        })

        assert response.status_code == 200
        data = response.json()

        # All original fields should still be present
        assert "session_id" in data
        assert "status" in data
        assert "response_text" in data
        assert "conversation_turns" in data
        # error is optional, may be null
```

Update existing mock fixture to include usage (for metadata test compatibility):
```python
@pytest.fixture
def mock_sdk_client():
    """Create mock ClaudeSDKClient"""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.query = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.interrupt = AsyncMock()

    async def mock_receive_response():
        assistant_msg = AssistantMessage(
            content=[TextBlock(text="This is a test response from Claude")],
            model="claude-sonnet-4-5-20250929"
        )
        yield assistant_msg

        result_msg = ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id="test-session",
            usage={
                "input_tokens": 100,
                "output_tokens": 50
            }
        )
        yield result_msg

    mock.receive_response = mock_receive_response
    return mock
```

### Success Criteria:

#### Automated Verification:
- [x] All unit tests pass: `uv run pytest tests/test_server.py -v`
- [x] New metadata test passes
- [x] Backward compatibility test passes
- [ ] No type errors or linting issues

#### Manual Verification:
- [ ] Start server and send a chat request via curl or API docs
- [ ] Verify response includes `metadata` field with expected structure
- [ ] Verify `tools_used` shows correct counts when agent uses tools
- [ ] Verify existing API clients are not broken

---

## Testing Strategy

### Unit Tests:
- Test metadata is present in successful responses
- Test metadata is present in error responses
- Test tools_used correctly counts multiple uses of same tool
- Test backward compatibility (original fields unchanged)
- Test metadata when no tools are used (empty dict)
- Test metadata when usage is None (graceful handling)

### Integration Tests:
- Send real request and verify metadata in response
- Verify token counts are reasonable values
- Verify model matches configured model

### Manual Testing Steps:
1. Start server: `uv run uvicorn server:app --port 8000`
2. Send request: `curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "List files in the current directory"}'`
3. Verify response contains `metadata` with `model`, `usage`, and `tools_used`
4. Check that `tools_used` shows tool counts (e.g., `{"Bash": 1}` if Bash was used)

## References

- Original ticket: `thoughts/shared/tickets/2025-12-06-agent-response-metadata.md`
- SDK message types verified via introspection (AssistantMessage, ResultMessage, ToolUseBlock)
- Usage dict structure verified: `{input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens, server_tool_use, service_tier, cache_creation}`
