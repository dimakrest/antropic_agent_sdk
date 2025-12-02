"""
HTTP Server Interface for Claude Agent SDK
A FastAPI-based REST API that exposes the Claude Agent SDK.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import asyncio
import uuid
import os
from dotenv import load_dotenv
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
)
from claude_agent_sdk.types import TextBlock

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SESSION_TIMEOUT = timedelta(hours=int(os.getenv("SESSION_TIMEOUT_HOURS", "1")))
DEFAULT_PERMISSION_MODE = os.getenv("DEFAULT_PERMISSION_MODE", "default")
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    active_sessions: int
    sdk_ready: bool = True


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID to continue conversation. Omit to start new session."
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Message to send to the agent"
    )
    permission_mode: Optional[str] = Field(
        None,
        description="Permission mode: 'default', 'acceptEdits', or 'bypassPermissions'"
    )
    max_turns: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Maximum conversation turns (1-50)"
    )
    allowed_tools: Optional[List[str]] = Field(
        None,
        description="List of tools the agent can use"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    session_id: str = Field(..., description="Session ID for continuing conversation")
    status: str = Field(..., description="Result status: 'success' or error type")
    response_text: Optional[str] = Field(None, description="Agent's text response")
    error: Optional[str] = Field(None, description="Error message if status != success")
    conversation_turns: int = Field(0, description="Number of turns in this response")


class SessionDeleteResponse(BaseModel):
    """Response model for session deletion"""
    status: str
    session_id: str


class SessionInterruptResponse(BaseModel):
    """Response model for session interrupt"""
    status: str
    session_id: str


class SessionInfo(BaseModel):
    """Information about an active session"""
    session_id: str
    last_activity: str
    age_seconds: int


class SessionListResponse(BaseModel):
    """Response model for session list"""
    active_sessions: int
    sessions: List[SessionInfo]


# =============================================================================
# Permission Callback (Optional)
# =============================================================================

async def default_can_use_tool(tool_name: str, tool_input: dict) -> bool:
    """
    Default permission callback for tool usage.

    In 'default' permission mode, this callback is consulted before allowing
    tool usage. Return True to allow, False to deny.

    For an HTTP server, we implement a simple allow-list approach.
    """
    # Allow read-only tools without restriction
    if tool_name in ["Read", "Grep", "Glob"]:
        return True

    # Allow write tools (user specified acceptEdits or bypassPermissions)
    # Note: This is called only in 'default' mode
    if tool_name in ["Write", "Edit"]:
        # In HTTP context, we can't prompt user, so deny by default
        # Users should use 'acceptEdits' mode for file operations
        return False

    # Allow bash for safe commands
    if tool_name == "Bash":
        # Basic safety check - deny dangerous commands
        command = tool_input.get("command", "")
        dangerous = ["rm -rf", "dd if=", "> /dev/", "mkfs", "format"]
        if any(cmd in command.lower() for cmd in dangerous):
            return False
        return True

    # Deny unknown tools by default
    return False


# =============================================================================
# Session Manager
# =============================================================================

class SessionManager:
    """Manages Claude SDK client instances and session lifecycle"""

    def __init__(self, client_factory=None):
        self.sessions: Dict[str, ClaudeSDKClient] = {}
        self.session_activity: Dict[str, datetime] = {}
        self.session_timeout = SESSION_TIMEOUT
        self.client_factory = client_factory or ClaudeSDKClient

    async def get_or_create_session(
        self,
        session_id: Optional[str],
        permission_mode: str,
        max_turns: int,
        allowed_tools: Optional[List[str]] = None,
        require_existing: bool = False,
    ) -> tuple[str, ClaudeSDKClient]:
        """
        Get existing session or create new one.

        Args:
            session_id: Optional session ID to continue
            permission_mode: Permission mode for the session
            max_turns: Maximum conversation turns
            allowed_tools: List of tools the agent can use
            require_existing: If True, raises error when session_id not found

        Returns:
            Tuple of (session_id, client)

        Raises:
            ValueError: If require_existing=True and session_id not found
        """

        # Return existing session if valid
        if session_id and session_id in self.sessions:
            self.session_activity[session_id] = datetime.now()
            return session_id, self.sessions[session_id]

        # If session_id was provided but not found, and we require existing
        if session_id and require_existing:
            raise ValueError(f"Session not found: {session_id}")

        # Create new session
        new_session_id = str(uuid.uuid4())

        # Configure SDK options with allowed_tools if provided
        options = ClaudeAgentOptions(
            model=DEFAULT_MODEL,
            permission_mode=permission_mode,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
            can_use_tool=default_can_use_tool if permission_mode == "default" else None,
        )

        # Create and connect SDK client
        client = self.client_factory(options=options)
        await client.connect()

        # Store session
        self.sessions[new_session_id] = client
        self.session_activity[new_session_id] = datetime.now()

        return new_session_id, client

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and disconnect client"""
        if session_id not in self.sessions:
            return False

        try:
            await self.sessions[session_id].disconnect()
        except Exception:
            pass  # Ignore disconnect errors
        finally:
            del self.sessions[session_id]
            if session_id in self.session_activity:
                del self.session_activity[session_id]

        return True

    async def cleanup_idle_sessions(self):
        """Background task to clean up idle sessions"""
        try:
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes

                now = datetime.now()
                expired = [
                    sid for sid, last_active in self.session_activity.items()
                    if now - last_active > self.session_timeout
                ]

                for sid in expired:
                    try:
                        await self.delete_session(sid)
                        print(f"Cleaned up idle session: {sid}")
                    except Exception as e:
                        print(f"Error cleaning up session {sid}: {e}")
        except asyncio.CancelledError:
            print("Session cleanup task cancelled")
            raise
        except Exception as e:
            print(f"Unexpected error in cleanup task: {e}")

    def get_active_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)


# Initialize global session manager
session_manager = SessionManager()


# =============================================================================
# Application Lifecycle (using modern lifespan approach)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    # Startup
    cleanup_task = asyncio.create_task(session_manager.cleanup_idle_sessions())
    print(f"Server started - Session timeout: {SESSION_TIMEOUT}")
    print(f"API docs: http://localhost:{os.getenv('PORT', '8000')}/docs")

    yield

    # Shutdown
    print("Shutting down - cleaning up sessions...")

    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Clean up all sessions
    for session_id in list(session_manager.sessions.keys()):
        await session_manager.delete_session(session_id)


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="Claude Agent HTTP Server",
    description="REST API interface for the Claude Agent SDK",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        active_sessions=session_manager.get_active_session_count(),
        sdk_ready=ANTHROPIC_API_KEY is not None
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Claude Agent HTTP Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to Claude and get a response.

    Creates a new session if session_id is not provided.
    Continues existing conversation if session_id is provided.
    Returns 404 if session_id is provided but not found.
    """
    try:
        # Get or create session
        # If session_id is provided, require it to exist (returns 404 if not found)
        session_id, client = await session_manager.get_or_create_session(
            session_id=request.session_id,
            permission_mode=request.permission_mode or DEFAULT_PERMISSION_MODE,
            max_turns=request.max_turns or MAX_TURNS,
            allowed_tools=request.allowed_tools,
            require_existing=request.session_id is not None,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        # Send query to SDK
        await client.query(prompt=request.message)

        # Collect response messages
        response_text_parts = []
        conversation_turns = 0

        # Iterate through messages
        # IMPORTANT: Never use `break` - causes asyncio cleanup issues per SDK docs
        # Let the iteration complete naturally, use flags to track state
        result_message = None
        async for message in client.receive_response():
            # Count conversation turns for assistant messages
            if isinstance(message, AssistantMessage):
                conversation_turns += 1

                # Extract text content from assistant messages
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text_parts.append(block.text)

            # Track result message (don't break - let iteration complete)
            if isinstance(message, ResultMessage):
                result_message = message

        # Process result with explicit error handling
        if result_message:
            if result_message.subtype == "success":
                return ChatResponse(
                    session_id=session_id,
                    status="success",
                    response_text="\n".join(response_text_parts),
                    conversation_turns=conversation_turns
                )
            elif result_message.subtype == "error_during_execution":
                return ChatResponse(
                    session_id=session_id,
                    status="error_during_execution",
                    error="Agent encountered an error during execution",
                    response_text="\n".join(response_text_parts),
                    conversation_turns=conversation_turns
                )
            else:
                # Generic fallback for any non-success subtype
                error_msg = f"Agent returned non-success result: {result_message.subtype}"
                return ChatResponse(
                    session_id=session_id,
                    status=result_message.subtype or "error_unknown",
                    error=error_msg,
                    response_text="\n".join(response_text_parts),
                    conversation_turns=conversation_turns
                )

        # No result received (shouldn't happen)
        raise HTTPException(
            status_code=500,
            detail="No result received from agent"
        )

    except asyncio.CancelledError:
        # Client disconnected or interrupted
        return ChatResponse(
            session_id=session_id,
            status="cancelled",
            error="Request cancelled",
            conversation_turns=0
        )

    except Exception as e:
        # Unexpected error - clean up session
        await session_manager.delete_session(session_id)

        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    """
    Explicitly terminate a conversation session.

    Disconnects the SDK client and removes the session from storage.
    """
    deleted = await session_manager.delete_session(session_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    return SessionDeleteResponse(
        status="terminated",
        session_id=session_id
    )


@app.post("/sessions/{session_id}/interrupt", response_model=SessionInterruptResponse)
async def interrupt_session(session_id: str):
    """
    Interrupt a long-running agent execution.

    Only works in streaming mode. Sends interrupt signal to the SDK client.
    """
    if session_id not in session_manager.sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    client = session_manager.sessions[session_id]

    try:
        await client.interrupt()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to interrupt session: {str(e)}"
        )

    return SessionInterruptResponse(
        status="interrupted",
        session_id=session_id
    )


@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """
    List all active sessions with their metadata.

    Useful for debugging and monitoring.
    """
    now = datetime.now()
    sessions_info = []

    for session_id, last_activity in session_manager.session_activity.items():
        age = (now - last_activity).total_seconds()
        sessions_info.append(SessionInfo(
            session_id=session_id,
            last_activity=last_activity.isoformat(),
            age_seconds=int(age)
        ))

    return SessionListResponse(
        active_sessions=len(sessions_info),
        sessions=sessions_info
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
