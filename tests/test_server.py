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
from claude_agent_sdk.types import TextBlock


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

        # Simulate result message with proper SDK types
        result_msg = ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id="test-session"
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
