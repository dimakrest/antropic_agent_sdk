"""
Unit tests for HTTP Server with mocked Claude SDK
Run with: pytest tests/test_server.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, SessionManager, session_manager
from claude_agent_sdk import AssistantMessage, ResultMessage
from claude_agent_sdk.types import TextBlock, ToolUseBlock


@pytest.fixture
def test_client():
    """Create FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def mock_sdk_client():
    """Create mock ClaudeSDKClient"""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.query = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.interrupt = AsyncMock()

    # Mock successful response using actual SDK types
    async def mock_receive_response():
        # Simulate assistant message with proper SDK types
        assistant_msg = AssistantMessage(
            content=[TextBlock(text="This is a test response from Claude")],
            model="claude-sonnet-4-5-20250929"
        )
        yield assistant_msg

        # Simulate result message with proper SDK types and usage
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


@pytest.fixture(autouse=True)
def reset_session_manager():
    """Reset session manager state before each test"""
    session_manager.sessions.clear()
    session_manager.session_activity.clear()
    yield
    session_manager.sessions.clear()
    session_manager.session_activity.clear()


def test_health_check(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_sessions" in data
    assert "sdk_ready" in data
    # Verify capacity info is included
    assert "analyze_capacity" in data
    assert "analyze_available" in data
    assert data["analyze_capacity"] > 0
    assert data["analyze_available"] <= data["analyze_capacity"]


def test_root_endpoint(test_client):
    """Test root endpoint returns API info"""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert data["message"] == "Claude Agent HTTP Server"


def test_chat_new_session(test_client, mock_sdk_client):
    """Test creating new session and sending message"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        response = test_client.post("/chat", json={
            "message": "Hello Claude"
        })

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "success"
        assert "response_text" in data


def test_chat_continue_session(test_client, mock_sdk_client):
    """Test continuing existing session"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        # First message
        resp1 = test_client.post("/chat", json={
            "message": "First message"
        })
        assert resp1.status_code == 200
        session_id = resp1.json()["session_id"]

        # Second message in same session
        resp2 = test_client.post("/chat", json={
            "session_id": session_id,
            "message": "Second message"
        })

        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id


def test_chat_validation_missing_message(test_client):
    """Test request validation for missing message"""
    response = test_client.post("/chat", json={})
    assert response.status_code == 422  # Validation error


def test_chat_validation_empty_message(test_client):
    """Test request validation for empty message"""
    response = test_client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_delete_session_not_found(test_client):
    """Test deleting non-existent session"""
    response = test_client.delete("/sessions/invalid-session-id")
    assert response.status_code == 404


def test_delete_session_success(test_client, mock_sdk_client):
    """Test successful session deletion"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        # Create session
        resp = test_client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Delete session
        del_resp = test_client.delete(f"/sessions/{session_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "terminated"


def test_list_sessions_empty(test_client):
    """Test listing sessions when none exist"""
    response = test_client.get("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert data["active_sessions"] == 0
    assert len(data["sessions"]) == 0


def test_list_sessions_with_active(test_client, mock_sdk_client):
    """Test listing sessions with active sessions"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        # Create session
        resp = test_client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # List sessions
        list_resp = test_client.get("/sessions")
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert data["active_sessions"] >= 1
        assert any(s["session_id"] == session_id for s in data["sessions"])


def test_interrupt_session_not_found(test_client):
    """Test interrupting non-existent session"""
    response = test_client.post("/sessions/invalid-id/interrupt")
    assert response.status_code == 404


def test_interrupt_session_success(test_client, mock_sdk_client):
    """Test successful session interrupt"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        # Create session
        resp = test_client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Interrupt session
        int_resp = test_client.post(f"/sessions/{session_id}/interrupt")
        assert int_resp.status_code == 200
        assert int_resp.json()["status"] == "interrupted"


def test_chat_with_permission_mode(test_client, mock_sdk_client):
    """Test chat with custom permission mode"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        response = test_client.post("/chat", json={
            "message": "Hello Claude",
            "permission_mode": "acceptEdits"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


def test_chat_with_max_turns(test_client, mock_sdk_client):
    """Test chat with custom max_turns"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        response = test_client.post("/chat", json={
            "message": "Hello Claude",
            "max_turns": 5
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


def test_chat_with_invalid_max_turns(test_client):
    """Test chat with invalid max_turns (too high)"""
    response = test_client.post("/chat", json={
        "message": "Hello Claude",
        "max_turns": 100  # Over limit of 50
    })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_session_manager_get_or_create_new():
    """Test SessionManager creates new session"""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()

    manager = SessionManager(client_factory=lambda **kwargs: mock_client)

    session_id, client = await manager.get_or_create_session(
        session_id=None,
        permission_mode="default",
        max_turns=10
    )

    assert session_id is not None
    assert len(session_id) == 36  # UUID format
    assert client == mock_client
    assert session_id in manager.sessions


@pytest.mark.asyncio
async def test_session_manager_get_existing():
    """Test SessionManager returns existing session"""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()

    manager = SessionManager(client_factory=lambda **kwargs: mock_client)

    # Create first session
    session_id1, _ = await manager.get_or_create_session(
        session_id=None,
        permission_mode="default",
        max_turns=10
    )

    # Get same session
    session_id2, client2 = await manager.get_or_create_session(
        session_id=session_id1,
        permission_mode="default",
        max_turns=10
    )

    assert session_id1 == session_id2
    assert client2 == mock_client


@pytest.mark.asyncio
async def test_session_manager_delete():
    """Test SessionManager deletes session"""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()

    manager = SessionManager(client_factory=lambda **kwargs: mock_client)

    # Create session
    session_id, _ = await manager.get_or_create_session(
        session_id=None,
        permission_mode="default",
        max_turns=10
    )

    # Delete session
    result = await manager.delete_session(session_id)

    assert result is True
    assert session_id not in manager.sessions


@pytest.mark.asyncio
async def test_session_manager_delete_nonexistent():
    """Test SessionManager handles deleting non-existent session"""
    manager = SessionManager()

    result = await manager.delete_session("nonexistent-session")

    assert result is False


@pytest.mark.asyncio
async def test_session_cleanup_logic():
    """Test session cleanup identifies expired sessions"""
    manager = SessionManager()
    manager.session_timeout = timedelta(seconds=1)

    # Create mock session
    mock_client = AsyncMock()
    mock_client.disconnect = AsyncMock()
    session_id = "test-session"
    manager.sessions[session_id] = mock_client
    manager.session_activity[session_id] = datetime.now() - timedelta(seconds=2)

    # Check expired logic
    now = datetime.now()
    expired = [
        sid for sid, last_active in manager.session_activity.items()
        if now - last_active > manager.session_timeout
    ]

    assert session_id in expired


def test_chat_error_during_execution(test_client, mock_sdk_client):
    """Test handling error_during_execution result"""
    # Modify mock to return error using proper SDK types
    async def mock_receive_response_error():
        result_msg = ResultMessage(
            subtype="error_during_execution",
            duration_ms=100,
            duration_api_ms=50,
            is_error=True,
            num_turns=0,
            session_id="test-session"
        )
        yield result_msg

    mock_sdk_client.receive_response = mock_receive_response_error

    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        response = test_client.post("/chat", json={
            "message": "Hello Claude"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error_during_execution"
        assert data["error"] is not None


def test_chat_invalid_session_id_returns_404(test_client):
    """Test that providing a non-existent session_id returns 404"""
    response = test_client.post("/chat", json={
        "session_id": "non-existent-session-id",
        "message": "Hello Claude"
    })

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_chat_with_allowed_tools(test_client, mock_sdk_client):
    """Test chat with allowed_tools parameter"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        response = test_client.post("/chat", json={
            "message": "Hello Claude",
            "allowed_tools": ["Read", "Write", "Edit"]
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


@pytest.mark.asyncio
async def test_session_manager_require_existing_raises_error():
    """Test SessionManager raises error for non-existent session when require_existing=True"""
    manager = SessionManager()

    with pytest.raises(ValueError) as exc_info:
        await manager.get_or_create_session(
            session_id="non-existent-session",
            permission_mode="default",
            max_turns=10,
            require_existing=True
        )

    assert "not found" in str(exc_info.value).lower()


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

        # Verify user_prompt
        assert metadata["user_prompt"] == "Hello Claude"

        # Verify usage
        assert "usage" in metadata
        assert metadata["usage"]["input_tokens"] == 150
        assert metadata["usage"]["output_tokens"] == 75

        # Verify tool_calls with params and counts
        assert "tool_calls" in metadata
        tool_calls = metadata["tool_calls"]
        assert len(tool_calls) == 3  # Read /test, Read /test2, Bash ls

        # Find specific tool calls
        read_calls = [tc for tc in tool_calls if tc["name"] == "Read"]
        bash_calls = [tc for tc in tool_calls if tc["name"] == "Bash"]
        assert len(read_calls) == 2  # Two different paths
        assert len(bash_calls) == 1
        assert bash_calls[0]["input"] == {"command": "ls"}
        assert bash_calls[0]["count"] == 1


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


def test_chat_response_metadata_without_tools(test_client, mock_sdk_client):
    """Test that metadata is present even when no tools are used"""
    with patch('server.session_manager.client_factory', return_value=mock_sdk_client):
        response = test_client.post("/chat", json={
            "message": "Hello Claude"
        })

        assert response.status_code == 200
        data = response.json()

        # Metadata should be present
        assert "metadata" in data
        metadata = data["metadata"]

        # Model should be set
        assert metadata["model"] == "claude-sonnet-4-5-20250929"

        # Usage should be present
        assert metadata["usage"]["input_tokens"] == 100
        assert metadata["usage"]["output_tokens"] == 50

        # tool_calls should be empty list when no tools were used
        assert metadata["tool_calls"] == []


@pytest.fixture
def mock_sdk_client_for_analyze():
    """Create mock ClaudeSDKClient for /analyze endpoint with structured output"""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.query = AsyncMock()
    mock.disconnect = AsyncMock()

    async def mock_receive_response():
        # Simulate assistant message with MCP tool use
        assistant_msg = AssistantMessage(
            content=[
                ToolUseBlock(id="tool_1", name="mcp__stock_analysis__get_stock_data", input={"symbol": "TEST"}),
                TextBlock(text="Analyzing stock data...")
            ],
            model="claude-sonnet-4-5-20250929"
        )
        yield assistant_msg

        # Simulate result message with structured output
        result_msg = ResultMessage(
            subtype="success",
            duration_ms=200,
            duration_api_ms=150,
            is_error=False,
            num_turns=2,
            session_id="test-analyze-session",
            usage={
                "input_tokens": 500,
                "output_tokens": 200
            }
        )
        # Add structured output
        result_msg.structured_output = {
            "stock": "TEST",
            "recommendation": "Buy",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "reasoning": "Test analysis",
            "missing_tools": [],
            "confidence_score": 75
        }
        yield result_msg

    mock.receive_response = mock_receive_response
    return mock


def test_analyze_response_includes_metadata(test_client, mock_sdk_client_for_analyze):
    """Test that /analyze response includes metadata with model, usage, and tools"""
    with patch('server.ClaudeSDKClient', return_value=mock_sdk_client_for_analyze):
        response = test_client.post("/analyze", json={
            "stock": "TEST"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["recommendation"] == "Buy"

        # Verify metadata structure
        assert "metadata" in data
        metadata = data["metadata"]

        # Verify model
        assert metadata["model"] == "claude-sonnet-4-5-20250929"

        # Verify usage
        assert "usage" in metadata
        assert metadata["usage"]["input_tokens"] == 500
        assert metadata["usage"]["output_tokens"] == 200

        # Verify tool_calls includes MCP tool with params
        assert "tool_calls" in metadata
        tool_calls = metadata["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "mcp__stock_analysis__get_stock_data"
        assert tool_calls[0]["input"] == {"symbol": "TEST"}
        assert tool_calls[0]["count"] == 1

        # Verify system_prompt and user_prompt are present
        assert "system_prompt" in metadata
        assert metadata["system_prompt"] is not None  # Should have swing trading prompt
        assert "user_prompt" in metadata
        assert "TEST" in metadata["user_prompt"]  # User prompt should contain the stock symbol


def test_analyze_with_analysis_date(test_client, mock_sdk_client_for_analyze):
    """Test /analyze with analysis_date parameter"""
    with patch('server.ClaudeSDKClient', return_value=mock_sdk_client_for_analyze):
        with patch('server.create_stock_tools_server') as mock_factory:
            mock_factory.return_value = MagicMock()  # Return mock MCP server

            response = test_client.post("/analyze", json={
                "stock": "TEST",
                "analysis_date": "2024-06-15"
            })

            assert response.status_code == 200
            # Verify factory was called with the date
            mock_factory.assert_called_once_with(analysis_date="2024-06-15")


def test_analyze_without_analysis_date(test_client, mock_sdk_client_for_analyze):
    """Test /analyze without analysis_date (default behavior)"""
    with patch('server.ClaudeSDKClient', return_value=mock_sdk_client_for_analyze):
        with patch('server.create_stock_tools_server') as mock_factory:
            mock_factory.return_value = MagicMock()

            response = test_client.post("/analyze", json={
                "stock": "TEST"
            })

            assert response.status_code == 200
            # Verify factory was called with None
            mock_factory.assert_called_once_with(analysis_date=None)


def test_analyze_invalid_date_format(test_client):
    """Test /analyze rejects invalid date format"""
    response = test_client.post("/analyze", json={
        "stock": "TEST",
        "analysis_date": "2024/06/15"  # Wrong format
    })
    assert response.status_code == 422  # Validation error


def test_analyze_invalid_date_format_no_dashes(test_client):
    """Test /analyze rejects date without dashes"""
    response = test_client.post("/analyze", json={
        "stock": "TEST",
        "analysis_date": "20240615"
    })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_stock_tools_server_with_date():
    """Test factory function binds analysis_date correctly"""
    from stock_tools import create_stock_tools_server

    # Create server with date
    server = create_stock_tools_server(analysis_date="2024-06-15")

    # Verify server was created (basic sanity check)
    assert server is not None
    assert server["name"] == "stock_analysis"


@pytest.mark.asyncio
async def test_create_stock_tools_server_without_date():
    """Test factory function works without date"""
    from stock_tools import create_stock_tools_server

    server = create_stock_tools_server()
    assert server is not None
    assert server["name"] == "stock_analysis"
