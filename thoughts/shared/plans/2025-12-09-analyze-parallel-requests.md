# Scale /analyze Endpoint for Parallel Requests - Implementation Plan

## Overview

Add concurrency control to the `/analyze` endpoint to handle parallel requests efficiently without resource exhaustion.

## Current State Analysis

### `/analyze` endpoint (`server.py:730-851`)
- Creates new `ClaudeSDKClient` per request with no concurrency control
- No limit on concurrent Claude API calls
- Each request independently connects/disconnects

### Key Discovery
- No existing concurrency controls - unlimited parallel requests can overwhelm Claude API

## Desired End State

After implementation:
1. Maximum N concurrent `/analyze` requests execute simultaneously (N configurable, default 5)
2. Requests exceeding capacity wait up to 15 seconds, then return 503 with Retry-After
3. Health endpoint reports current load status

### Verification
- Run load test with 20 concurrent requests
- Verify only 5 execute simultaneously
- Verify requests 6-20 queue and complete or timeout appropriately
- Verify 503 responses include proper headers

## What We're NOT Doing

- HTTP connection pooling (minor optimization, can add later if needed)
- Distributed request queuing (Redis, RabbitMQ)
- Horizontal scaling across multiple servers
- Request prioritization or user-based quotas
- Caching of analysis results
- Rate limiting (can be added later if needed)

## Implementation

### Overview
Add `asyncio.Semaphore` to limit concurrent `/analyze` requests. Requests wait up to 15 seconds for a slot, then return 503 Service Unavailable.

### Changes Required

#### 1. Configuration (`server.py` - after line 34)
**File**: `http-server-agent-sdk/server.py`
**Changes**: Add concurrency configuration constants

```python
# After line 34 (DEFAULT_MODEL = ...)
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "5"))
ANALYSIS_QUEUE_TIMEOUT = float(os.getenv("ANALYSIS_QUEUE_TIMEOUT", "15.0"))
```

#### 2. Semaphore initialization (`server.py` - after configuration section)
**File**: `http-server-agent-sdk/server.py`
**Changes**: Add global semaphore for /analyze endpoint

```python
# After the configuration section, before Request/Response Models
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
```

#### 3. Response model for capacity errors (`server.py` - after AnalyzeResponse)
**File**: `http-server-agent-sdk/server.py`
**Changes**: Add error response model

```python
class ServiceUnavailableResponse(BaseModel):
    """Response when server is at capacity"""
    detail: str = Field(..., description="Error message")
    retry_after: int = Field(..., description="Seconds to wait before retrying")
```

#### 4. Update `/analyze` endpoint (`server.py` - lines 730-851)
**File**: `http-server-agent-sdk/server.py`
**Changes**: Wrap endpoint logic with semaphore and timeout

```python
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
        acquired = await asyncio.wait_for(
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

    try:
        # Existing implementation (lines 741-851 - client creation through finally block)
        client = None
        # ... rest of existing implementation ...
    except Exception as e:
        # Handle Claude API rate limits (propagate as 429)
        error_str = str(e).lower()
        if "429" in error_str or "rate_limit" in error_str or "too many requests" in error_str:
            raise HTTPException(
                status_code=429,
                detail="Claude API rate limit exceeded. Please retry later.",
                headers={"Retry-After": "60"}
            )
        raise  # Re-raise other exceptions
    finally:
        semaphore.release()
```

#### 5. Update health endpoint (`server.py` - lines 472-479)
**File**: `http-server-agent-sdk/server.py`
**Changes**: Add concurrency info to health response

```python
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    active_sessions: int
    sdk_ready: bool = True
    # Add new fields
    analyze_capacity: int = Field(..., description="Max concurrent /analyze requests")
    analyze_available: int = Field(..., description="Available /analyze slots")


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
```

### Success Criteria

#### Automated Verification:
- [x] Unit tests pass: `uv run pytest tests/test_server.py -v`
- [ ] New test: concurrent requests beyond limit receive 503
- [ ] New test: requests within limit succeed
- [x] New test: health endpoint reports capacity correctly

#### Manual Verification:
- [x] Start server and send 10 concurrent `/analyze` requests
- [x] Verify only 5 execute simultaneously (check server logs)
- [x] Verify requests 6-10 wait and either complete or timeout with 503
- [x] Verify 503 response includes `Retry-After: 15` header

---

## Testing Strategy

### Unit Tests

Add to `tests/test_server.py`:

```python
async def test_analyze_concurrent_limit():
    """Test that concurrent requests beyond limit return 503"""
    pass

async def test_analyze_within_capacity():
    """Test that requests within capacity succeed"""
    pass

async def test_health_reports_capacity():
    """Test health endpoint includes capacity info"""
    pass
```

### Load Testing

Create `tests/load_test.py`:

```python
"""
Load test for /analyze endpoint.
Run with: uv run python tests/load_test.py
"""
import asyncio
import httpx
import time

async def analyze_stock(client: httpx.AsyncClient, stock: str, request_id: int):
    """Make single analyze request"""
    start = time.time()
    try:
        response = await client.post(
            "http://localhost:8000/analyze",
            json={"stock": stock}
        )
        elapsed = time.time() - start
        return {
            "request_id": request_id,
            "status": response.status_code,
            "elapsed": elapsed
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e)
        }

async def run_load_test(num_requests: int = 20):
    """Run concurrent requests"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            analyze_stock(client, "AAPL", i)
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)

    # Analyze results
    successes = [r for r in results if r["status"] == 200]
    capacity_errors = [r for r in results if r["status"] == 503]

    print(f"\n=== Load Test Results ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful (200): {len(successes)}")
    print(f"Capacity exceeded (503): {len(capacity_errors)}")

if __name__ == "__main__":
    asyncio.run(run_load_test(20))
```

### Manual Testing Steps

1. Start server: `TRADING_API_BASE_URL=http://localhost:8131/api/v1 uv run uvicorn server:app --port 8000`
2. Check health: `curl http://localhost:8000/health`
3. Run load test: `uv run python tests/load_test.py`
4. Verify capacity limiting works

---

## Environment Variables Summary

New environment variables added:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_ANALYSES` | `5` | Maximum concurrent /analyze requests |
| `ANALYSIS_QUEUE_TIMEOUT` | `15.0` | Seconds to wait for capacity before 503 |

---

## Production Configuration Notes

### Claude API Rate Limits

The Anthropic API has organization-level rate limits (all concurrent clients share the same quota):

| Tier | Requests/min | Notes |
|------|--------------|-------|
| Tier 1 | 50 RPM | Default for new accounts |
| Tier 2 | 1,000 RPM | After initial usage |
| Tier 3+ | 2,000+ RPM | Higher tiers |

With 5 concurrent requests averaging 30s each → ~10 RPM (safe for Tier 1).
If requests complete faster, may need to reduce `MAX_CONCURRENT_ANALYSES`.

429 errors from Claude API are propagated to clients with `Retry-After: 60` header.

### Multi-worker deployment

For production with multiple workers:

```bash
# Run with multiple workers (each has its own semaphore state)
uvicorn server:app --workers 4 --host 0.0.0.0 --port 8000
```

**Note**: In-memory semaphores are per-process. With 4 workers:
- Effective max concurrent = `MAX_CONCURRENT_ANALYSES * 4`

For strict global limits, would need Redis (out of scope per ticket).

### Recommended production settings

```bash
export MAX_CONCURRENT_ANALYSES=10
export ANALYSIS_QUEUE_TIMEOUT=30.0
```

---

## References

- Original ticket: `thoughts/shared/tickets/2025-12-09-analyze-parallel-requests.md`
- server.py: `/analyze` endpoint at lines 730-851
- FastAPI lifespan: server.py lines 434-457
