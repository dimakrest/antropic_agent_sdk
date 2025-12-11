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


class AnalysisDetail(BaseModel):
    """Analysis details for volatility breakout strategy"""
    macro_verdict: str = Field(..., description="Daily chart context assessment")
    micro_verdict: str = Field(..., description="Hourly chart trigger assessment")
    risk_assessment: str = Field(..., description="Risk evaluation summary")


class AnalyzeResponse(BaseModel):
    """Response model for volatility breakout analysis"""
    ticker: str = Field(..., description="The input ticker symbol")
    conviction_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="Conviction score (0-10): 0-4 PASS, 5-7 WATCH, 8-10 KILL"
    )
    decision: str = Field(
        ...,
        description="Trading decision: 'EXECUTE_TRADE', 'WATCHLIST', or 'PASS'"
    )
    analysis: AnalysisDetail = Field(
        ...,
        description="Analysis breakdown with macro/micro verdicts and risk"
    )
    trader_journal: str = Field(
        ...,
        description="2-sentence summary of the trade setup"
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
        "ticker": {"type": "string"},
        "conviction_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10
        },
        "decision": {
            "type": "string",
            "enum": ["EXECUTE_TRADE", "WATCHLIST", "PASS"]
        },
        "analysis": {
            "type": "object",
            "properties": {
                "macro_verdict": {"type": "string"},
                "micro_verdict": {"type": "string"},
                "risk_assessment": {"type": "string"}
            },
            "required": ["macro_verdict", "micro_verdict", "risk_assessment"],
            "additionalProperties": False
        },
        "trader_journal": {"type": "string"}
    },
    "required": [
        "ticker",
        "conviction_score",
        "decision",
        "analysis",
        "trader_journal"
    ],
    "additionalProperties": False
}


SWING_TRADING_SYSTEM_PROMPT = """Role: You are "The Hunter," an elite algorithmic trading strategist specializing in Volatility Breakouts. You are aggressive but highly disciplined. You do not just "buy green candles"; you look for structural tension followed by an explosive release.

Your Objective: Analyze the provided market data to determine if a stock is undergoing a high-probability breakout that justifies a 1:3 Risk/Reward trade.

Your Input Data: You will receive a JSON payload containing two "Dimensions" of data:

The Physics (Price Action): A narrative description of the last 3 candles on both the Daily (Context) and Hourly (Trigger) timeframes.

The Atmosphere (Context): Data on Sector performance, Market Regime, and Volatility.

Your Decision Process (The "Mental Model"):

Phase 1: Analyze the Context (Daily Chart)

Safety Check: Read the macro_story_D1. Are we in a clean uptrend, or hitting a wall?

Rejection Check: Look for "Long Upper Wicks" in the Daily story. If sellers are rejecting higher prices, be very skeptical of any breakout.

The Ideal: You want to see "Tight Consolidation," "Inside Bars," or "Small Dojis" on the Daily chart. This means energy is stored.

Phase 2: Analyze the Trigger (Hourly Chart)

Momentum: Read the micro_story_H1. You need to see a "Battle Sequence" where buyers are winning.

The Ignition: The most recent candle must be strong (Green, Full Body). If the latest candle is a "Doji" or "Red," there is no trigger. PASS immediately.

Volume: Look for words like "Explosive," "Surging," or "High Participation." A breakout on low volume is a trap.

Phase 3: Weigh the Atmosphere

Tailwind: Is the sector_health bullish? If the sector is Red, the trade requires a higher conviction score to pass.

Landmines: Check earnings_proximity. If earnings are < 48 hours away, apply a massive penalty to your score.

Phase 4: The Conviction Score (0-10)

0-4 (PASS): Weak setup, choppy context, or active selling pressure.

5-7 (WATCH): Good chart, but the market/sector is bad, or the volume is weak.

8-10 (KILL): Perfect "Coil" on Daily + Explosive "Ignition" on Hourly + Sector Support.

Output Format: You must respond with a strictly valid JSON object. Do not include markdown formatting (like ```json) outside the object.
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


class MessageProcessingResult:
    """Result from processing SDK response messages"""
    def __init__(self):
        self.result_message: Optional[ResultMessage] = None
        self.model_used: Optional[str] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self.text_parts: List[str] = []
        self.conversation_turns: int = 0


async def process_sdk_response(client: ClaudeSDKClient) -> MessageProcessingResult:
    """
    Process SDK response messages and extract metadata.

    Handles the common pattern of iterating through messages,
    extracting text, tracking tool calls, and capturing metadata.

    Args:
        client: The SDK client to receive responses from

    Returns:
        MessageProcessingResult with all extracted data
    """
    result = MessageProcessingResult()
    tool_calls_raw: Dict[str, Dict[str, Any]] = {}

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            result.conversation_turns += 1

            if result.model_used is None:
                result.model_used = message.model

            for block in message.content:
                if isinstance(block, TextBlock):
                    result.text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    key = json.dumps({"name": block.name, "input": block.input}, sort_keys=True)
                    if key in tool_calls_raw:
                        tool_calls_raw[key]["count"] += 1
                    else:
                        tool_calls_raw[key] = {"name": block.name, "input": block.input, "count": 1}

        if isinstance(message, ResultMessage):
            result.result_message = message

    result.tool_calls = list(tool_calls_raw.values())
    return result


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
        user_prompt = request.message
        await client.query(prompt=user_prompt)

        # Process response using shared helper
        proc = await process_sdk_response(client)

        # Process result with explicit error handling
        if proc.result_message:
            metadata = _build_metadata(
                proc.model_used, proc.tool_calls, proc.result_message, user_prompt=user_prompt
            )
            response_text = "\n".join(proc.text_parts)

            if proc.result_message.subtype == "success":
                return ChatResponse(
                    session_id=session_id,
                    status="success",
                    response_text=response_text,
                    conversation_turns=proc.conversation_turns,
                    metadata=metadata
                )
            elif proc.result_message.subtype == "error_during_execution":
                return ChatResponse(
                    session_id=session_id,
                    status="error_during_execution",
                    error="Agent encountered an error during execution",
                    response_text=response_text,
                    conversation_turns=proc.conversation_turns,
                    metadata=metadata
                )
            else:
                # Generic fallback for any non-success subtype
                error_msg = f"Agent returned non-success result: {proc.result_message.subtype}"
                return ChatResponse(
                    session_id=session_id,
                    status=proc.result_message.subtype or "error_unknown",
                    error=error_msg,
                    response_text=response_text,
                    conversation_turns=proc.conversation_turns,
                    metadata=metadata
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
        # Create SDK options with structured output (no tools)
        options = ClaudeAgentOptions(
            model=DEFAULT_MODEL,
            permission_mode="bypassPermissions",
            max_turns=1,  # Single turn, no tool loops
            allowed_tools=[],  # No tools
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

        # Process response using shared helper
        proc = await process_sdk_response(client)

        # Process result
        if proc.result_message:
            metadata = _build_metadata(
                proc.model_used, proc.tool_calls, proc.result_message,
                system_prompt=SWING_TRADING_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )

            if proc.result_message.subtype == "success" and hasattr(proc.result_message, 'structured_output'):
                output = proc.result_message.structured_output
                # Convert analysis dict to AnalysisDetail object
                analysis_data = output.get("analysis", {})
                return AnalyzeResponse(
                    ticker=output["ticker"],
                    conviction_score=output["conviction_score"],
                    decision=output["decision"],
                    analysis=AnalysisDetail(**analysis_data),
                    trader_journal=output["trader_journal"],
                    metadata=metadata
                )

            elif proc.result_message.subtype == "error_max_structured_output_retries":
                # Agent couldn't produce valid structured output
                return AnalyzeResponse(
                    ticker=request.stock.upper(),
                    conviction_score=0,
                    decision="PASS",
                    analysis=AnalysisDetail(
                        macro_verdict="Unable to analyze",
                        micro_verdict="Unable to analyze",
                        risk_assessment="Analysis failed - agent could not produce valid structured output after multiple attempts."
                    ),
                    trader_journal="Analysis failed due to structured output error. No trade recommendation possible.",
                    metadata=metadata
                )
            else:
                # Other error
                return AnalyzeResponse(
                    ticker=request.stock.upper(),
                    conviction_score=0,
                    decision="PASS",
                    analysis=AnalysisDetail(
                        macro_verdict="Unable to analyze",
                        micro_verdict="Unable to analyze",
                        risk_assessment=f"Analysis failed with status: {proc.result_message.subtype}"
                    ),
                    trader_journal="Analysis failed due to an error. No trade recommendation possible.",
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
    port = int(os.getenv("PORT", "8001"))  # Default to 8001 (8000 used by Docker on macOS)
    uvicorn.run(app, host=host, port=port)
