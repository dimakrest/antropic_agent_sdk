"""
HTTP Server Interface for Claude Agent SDK
A FastAPI-based REST API that exposes the Claude Agent SDK.
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
)
from claude_agent_sdk.types import TextBlock, ToolUseBlock
from stock_tools import create_stock_tools_server

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SESSION_TIMEOUT = timedelta(hours=int(os.getenv("SESSION_TIMEOUT_HOURS", "1")))
DEFAULT_PERMISSION_MODE = os.getenv("DEFAULT_PERMISSION_MODE", "default")
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "5"))
ANALYSIS_QUEUE_TIMEOUT = float(os.getenv("ANALYSIS_QUEUE_TIMEOUT", "15.0"))


# =============================================================================
# Concurrency Control
# =============================================================================

# Semaphore to limit concurrent /analyze requests
# Initialized at module level, shared across all requests
_analyze_semaphore: asyncio.Semaphore | None = None


def get_analyze_semaphore() -> asyncio.Semaphore:
    """Get or create the analyze semaphore (lazy initialization)."""
    global _analyze_semaphore
    if _analyze_semaphore is None:
        _analyze_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
    return _analyze_semaphore


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    active_sessions: int
    sdk_ready: bool = True
    analyze_capacity: int = Field(..., description="Max concurrent /analyze requests")
    analyze_available: int = Field(..., description="Available /analyze slots")


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


class UsageInfo(BaseModel):
    """Token usage information.

    Note: With prompt caching, `input_tokens` only represents non-cached tokens.
    Use `total_input_tokens` for the true total input token count.

    Formula: total_input_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
    """
    input_tokens: int = Field(..., description="Non-cached input tokens (after last cache breakpoint)")
    output_tokens: int = Field(..., description="Number of output tokens generated")
    cache_creation_input_tokens: int = Field(
        0, description="Tokens used to create cache entries"
    )
    cache_read_input_tokens: int = Field(
        0, description="Tokens read from cache (reduces cost)"
    )
    total_input_tokens: int = Field(
        ..., description="Total input tokens (input + cache_creation + cache_read)"
    )
    total_cost_usd: Optional[float] = Field(
        None, description="Total cost in USD (authoritative)"
    )


class ToolCall(BaseModel):
    """Single tool call with parameters"""
    name: str = Field(..., description="Tool name")
    input: Dict[str, Any] = Field(..., description="Tool parameters")
    count: int = Field(1, description="Number of times this exact call was made")


class ResponseMetadata(BaseModel):
    """Metadata about the agent response"""
    model: Optional[str] = Field(None, description="Model used for the response")
    system_prompt: Optional[str] = Field(None, description="System prompt used")
    user_prompt: Optional[str] = Field(None, description="User prompt sent")
    usage: Optional[UsageInfo] = Field(None, description="Token usage statistics")
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Tools called with parameters and count"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    session_id: str = Field(..., description="Session ID for continuing conversation")
    status: str = Field(..., description="Result status: 'success' or error type")
    response_text: Optional[str] = Field(None, description="Agent's text response")
    error: Optional[str] = Field(None, description="Error message if status != success")
    conversation_turns: int = Field(0, description="Number of turns in this response")
    metadata: Optional[ResponseMetadata] = Field(None, description="Response metadata including model, usage, and tools")


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
# Swing Trading Agent Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for swing trading analysis"""
    stock: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')"
    )
    analysis_date: Optional[str] = Field(
        None,
        description="Historical analysis date (YYYY-MM-DD). Omit for current date.",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )


class AnalyzeResponse(BaseModel):
    """Response model for swing trading analysis"""
    stock: str = Field(..., description="The input ticker symbol")
    recommendation: str = Field(
        ...,
        description="Trading recommendation: 'Buy' or 'Not Buy'"
    )
    entry_price: float = Field(..., description="Recommended entry price")
    stop_loss: float = Field(..., description="Stop loss price level")
    take_profit: float = Field(..., description="Take profit target price")
    reasoning: str = Field(
        ...,
        description="Professional analysis explaining the decision"
    )
    missing_tools: List[str] = Field(
        ...,
        description="Indicators/data that would increase confidence"
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in the recommendation (0-100)"
    )
    metadata: Optional[ResponseMetadata] = Field(
        None,
        description="Response metadata including model, usage, and tools"
    )


class ServiceUnavailableResponse(BaseModel):
    """Response when server is at capacity"""
    detail: str = Field(..., description="Error message")
    retry_after: int = Field(..., description="Seconds to wait before retrying")


# JSON Schema for structured output (matches AnalyzeResponse)
TRADING_RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "stock": {"type": "string"},
        "recommendation": {
            "type": "string",
            "enum": ["Buy", "Not Buy"]
        },
        "entry_price": {"type": "number"},
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "reasoning": {"type": "string"},
        "missing_tools": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        }
    },
    "required": [
        "stock",
        "recommendation",
        "entry_price",
        "stop_loss",
        "take_profit",
        "reasoning",
        "missing_tools",
        "confidence_score"
    ],
    "additionalProperties": False
}


SWING_TRADING_SYSTEM_PROMPT = """You are a professional swing trader specializing in short-term trades with 3-5 day holding periods.

## Your Role
You ARE the trading expert. Apply your own trading expertise and judgment to analyze stocks and provide actionable recommendations.

## Analysis Process
1. ALWAYS use the get_stock_data tool to fetch current technical analysis data
2. Analyze the data using your trading expertise
3. Make a binary decision: Buy or Not Buy (no "Hold" or "Maybe")
4. Provide specific price levels for entry, stop loss, and take profit

## Decision Framework
- Focus on swing trade setups with 3-5 day holding period
- Favor favorable risk/reward ratios (ideally 2:1 or better)
- Always set a stop loss - risk management is non-negotiable
- Consider trend direction, momentum, and key support/resistance levels

## Key Indicators to Consider
- RSI: Look for momentum confirmation (not overbought >70 or oversold <30 for entries)
- MACD: Trend direction and potential crossovers
- Moving Averages: Price relative to 20, 50, 200 SMA for trend context
- Support/Resistance: Key levels for entry and stop placement
- Volume: Confirmation of price moves
- ATR: For appropriate stop loss and target distance

## Output Requirements
- recommendation: "Buy" or "Not Buy" - binary decision only
- entry_price: Specific price (can be current price for immediate entry or limit near support)
- stop_loss: Below recent support or based on ATR
- take_profit: Based on resistance levels or risk/reward calculation
- reasoning: Professional analysis explaining your decision (2-4 sentences)
- missing_tools: What additional data would improve your analysis
- confidence_score: 0-100 based on signal alignment (informational only)

## Important Notes
- If data is unavailable or insufficient, recommend "Not Buy" with reasoning
- Never recommend buying without proper risk/reward setup
- Be conservative with confidence scores - reserve >80 for very clear setups
"""


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
    semaphore = get_analyze_semaphore()
    # Semaphore._value gives current available count
    available = semaphore._value if hasattr(semaphore, '_value') else MAX_CONCURRENT_ANALYSES

    return HealthResponse(
        status="healthy",
        active_sessions=session_manager.get_active_session_count(),
        sdk_ready=ANTHROPIC_API_KEY is not None,
        analyze_capacity=MAX_CONCURRENT_ANALYSES,
        analyze_available=available
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


def _build_metadata(
    model_used: Optional[str],
    tool_calls: List[Dict[str, Any]],
    result_message: Optional[ResultMessage],
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None
) -> ResponseMetadata:
    """Build response metadata from collected data"""
    usage_info = None
    if result_message and result_message.usage:
        usage = result_message.usage
        # Debug logging to verify token data
        print(f"[DEBUG] ResultMessage.usage: {usage}")

        # Extract individual token fields
        input_tokens = usage.get("input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)

        # Calculate TRUE total input tokens
        # With caching: input_tokens only = non-cached portion
        total_input = input_tokens + cache_creation + cache_read

        usage_info = UsageInfo(
            input_tokens=input_tokens,
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            total_input_tokens=total_input,
            total_cost_usd=usage.get("total_cost_usd"),
        )

    return ResponseMetadata(
        model=model_used,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        usage=usage_info,
        tool_calls=[ToolCall(**tc) for tc in tool_calls]
    )


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
        tool_calls_raw: Dict[str, Dict[str, Any]] = {}  # Track tool calls with params
        model_used: Optional[str] = None  # Track model from AssistantMessage
        user_prompt = request.message  # Capture user prompt

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
                        # Create unique key from tool name + sorted params
                        key = json.dumps({"name": block.name, "input": block.input}, sort_keys=True)
                        if key in tool_calls_raw:
                            tool_calls_raw[key]["count"] += 1
                        else:
                            tool_calls_raw[key] = {"name": block.name, "input": block.input, "count": 1}

            # Track result message (don't break - let iteration complete)
            if isinstance(message, ResultMessage):
                result_message = message

        # Convert tool calls to list
        tool_calls = list(tool_calls_raw.values())

        # Process result with explicit error handling
        if result_message:
            if result_message.subtype == "success":
                return ChatResponse(
                    session_id=session_id,
                    status="success",
                    response_text="\n".join(response_text_parts),
                    conversation_turns=conversation_turns,
                    metadata=_build_metadata(model_used, tool_calls, result_message, user_prompt=user_prompt)
                )
            elif result_message.subtype == "error_during_execution":
                return ChatResponse(
                    session_id=session_id,
                    status="error_during_execution",
                    error="Agent encountered an error during execution",
                    response_text="\n".join(response_text_parts),
                    conversation_turns=conversation_turns,
                    metadata=_build_metadata(model_used, tool_calls, result_message, user_prompt=user_prompt)
                )
            else:
                # Generic fallback for any non-success subtype
                error_msg = f"Agent returned non-success result: {result_message.subtype}"
                return ChatResponse(
                    session_id=session_id,
                    status=result_message.subtype or "error_unknown",
                    error=error_msg,
                    response_text="\n".join(response_text_parts),
                    conversation_turns=conversation_turns,
                    metadata=_build_metadata(model_used, tool_calls, result_message, user_prompt=user_prompt)
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
            conversation_turns=0,
            metadata=ResponseMetadata(tool_calls=[])  # No metadata available for cancelled requests
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


@app.post("/analyze", response_model=AnalyzeResponse, responses={
    429: {"description": "Rate limit exceeded (Claude API)"},
    503: {"model": ServiceUnavailableResponse, "description": "Server at capacity"}
})
async def analyze_stock(request: AnalyzeRequest):
    """
    Analyze a stock for swing trading opportunities.

    Returns a structured trading recommendation with entry price,
    stop loss, take profit, and professional reasoning.

    This is a stateless endpoint - each request creates a temporary
    session that is automatically cleaned up after the response.

    Concurrency is limited to MAX_CONCURRENT_ANALYSES (default: 5).
    Requests exceeding capacity wait up to ANALYSIS_QUEUE_TIMEOUT seconds.
    """
    semaphore = get_analyze_semaphore()

    # Try to acquire semaphore with timeout
    try:
        await asyncio.wait_for(
            semaphore.acquire(),
            timeout=ANALYSIS_QUEUE_TIMEOUT
        )
    except asyncio.TimeoutError:
        # Server at capacity, return 503
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Please retry later.",
            headers={"Retry-After": str(int(ANALYSIS_QUEUE_TIMEOUT))}
        )

    client = None

    try:
        # Create MCP server with analysis_date bound (or None for current date)
        stock_server = create_stock_tools_server(analysis_date=request.analysis_date)

        # Create SDK options with stock tools and structured output
        options = ClaudeAgentOptions(
            model=DEFAULT_MODEL,
            permission_mode="bypassPermissions",  # Tools are safe, no user interaction
            max_turns=5,  # Limit turns for focused analysis
            mcp_servers={"stock_analysis": stock_server},
            allowed_tools=["mcp__stock_analysis__get_stock_data"],
            system_prompt=SWING_TRADING_SYSTEM_PROMPT,
            output_format={
                "type": "json_schema",
                "schema": TRADING_RECOMMENDATION_SCHEMA
            }
        )

        # Create temporary client
        client = ClaudeSDKClient(options=options)
        await client.connect()

        # Send analysis request
        user_prompt = f"Analyze {request.stock.upper()} for a potential swing trade opportunity."
        await client.query(prompt=user_prompt)

        # Collect response and metadata
        result_message = None
        tool_calls_raw: Dict[str, Dict[str, Any]] = {}  # Track tool calls with params
        model_used: Optional[str] = None  # Track model from AssistantMessage

        async for message in client.receive_response():
            # Collect metadata from assistant messages
            if isinstance(message, AssistantMessage):
                # Capture model from first AssistantMessage
                if model_used is None:
                    model_used = message.model

                # Track tool usage with params
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        key = json.dumps({"name": block.name, "input": block.input}, sort_keys=True)
                        if key in tool_calls_raw:
                            tool_calls_raw[key]["count"] += 1
                        else:
                            tool_calls_raw[key] = {"name": block.name, "input": block.input, "count": 1}

            if isinstance(message, ResultMessage):
                result_message = message

        # Convert tool calls to list
        tool_calls = list(tool_calls_raw.values())

        # Process result
        if result_message:
            metadata = _build_metadata(
                model_used, tool_calls, result_message,
                system_prompt=SWING_TRADING_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )

            if result_message.subtype == "success" and hasattr(result_message, 'structured_output'):
                output = result_message.structured_output
                return AnalyzeResponse(**output, metadata=metadata)

            elif result_message.subtype == "error_max_structured_output_retries":
                # Agent couldn't produce valid structured output
                return AnalyzeResponse(
                    stock=request.stock.upper(),
                    recommendation="Not Buy",
                    entry_price=0,
                    stop_loss=0,
                    take_profit=0,
                    reasoning="Unable to complete analysis - agent could not produce valid structured output after multiple attempts.",
                    missing_tools=["Stable analysis pipeline"],
                    confidence_score=0,
                    metadata=metadata
                )
            else:
                # Other error
                return AnalyzeResponse(
                    stock=request.stock.upper(),
                    recommendation="Not Buy",
                    entry_price=0,
                    stop_loss=0,
                    take_profit=0,
                    reasoning=f"Analysis failed with status: {result_message.subtype}",
                    missing_tools=[],
                    confidence_score=0,
                    metadata=metadata
                )

        # No result received
        raise HTTPException(
            status_code=500,
            detail="No result received from analysis agent"
        )

    except HTTPException:
        raise
    except Exception as e:
        # Handle Claude API rate limits (propagate as 429)
        error_str = str(e).lower()
        if "429" in error_str or "rate_limit" in error_str or "too many requests" in error_str:
            raise HTTPException(
                status_code=429,
                detail="Claude API rate limit exceeded. Please retry later.",
                headers={"Retry-After": "60"}
            )
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )
    finally:
        # Always cleanup the temporary session
        if client:
            try:
                await client.disconnect()
            except Exception:
                pass  # Ignore disconnect errors
        # Release semaphore
        semaphore.release()


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
