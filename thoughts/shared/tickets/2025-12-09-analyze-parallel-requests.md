# Ticket: Scale /analyze Endpoint for Parallel Requests

**Date**: 2025-12-09
**Status**: Open

---

## Problem

The `/analyze` endpoint currently handles requests sequentially without optimization for concurrent usage. As we scale up, multiple users sending analysis requests simultaneously will experience:
- Degraded performance under load
- Potential resource exhaustion (too many concurrent API connections)
- No protection against request floods
- Inefficient resource utilization

### Current Implementation Issues

1. **No concurrency control**: Each request creates a new `ClaudeSDKClient` without limits on concurrent executions
2. **No connection pooling**: The `stock_tools.py` module creates a new `httpx.AsyncClient()` for every tool call to the Trading API
3. **Single worker**: Default uvicorn configuration runs a single worker process
4. **No rate limiting**: No protection against API abuse or request floods
5. **No request queuing**: High load can overwhelm both the Claude API and Trading API

---

## Requirements

### Must Have
1. **Concurrency limiting**: Control the maximum number of concurrent Claude API calls to prevent overwhelming the API and manage costs
2. **HTTP connection pooling**: Reuse connections to the Trading API instead of creating new connections per request
3. **Multi-worker support**: Configuration for running multiple uvicorn workers to utilize multiple CPU cores

### Should Have
1. **Request queuing**: When at capacity, queue incoming requests instead of failing immediately
2. **Rate limiting**: Protect the endpoint from abuse with configurable rate limits
3. **Graceful degradation**: Return meaningful errors when capacity is exceeded

### Nice to Have
1. **Metrics/monitoring**: Expose metrics for concurrent requests, queue depth, response times
2. **Health check enhancement**: Report current load status in health endpoint
3. **Configurable limits**: Environment variables for tuning concurrency and rate limits

---

## Acceptance Criteria

1. [ ] Server can handle N concurrent `/analyze` requests without resource exhaustion (N configurable via env var)
2. [ ] HTTP connections to Trading API are pooled and reused
3. [ ] Documentation includes recommended production configuration (workers, limits)
4. [ ] Requests exceeding capacity receive appropriate HTTP status code (429 or 503) with retry guidance
5. [ ] Existing single-request behavior unchanged (backward compatible)
6. [ ] Load testing demonstrates improved throughput under concurrent load

---

## Out of Scope (YAGNI!)

- Distributed request queuing (Redis, RabbitMQ)
- Horizontal scaling across multiple servers
- Request prioritization or user-based quotas
- Caching of analysis results
- WebSocket streaming for long-running requests
- Authentication/authorization (separate concern)

---

## Technical Context

### Current Architecture (for reference)

```
POST /analyze
    └── Creates new ClaudeSDKClient per request
         └── Connects to Claude API
         └── MCP tool calls create new httpx.AsyncClient per call
              └── Calls Trading API
         └── Disconnects and cleanup
```

### Key Files
- `server.py`: Main FastAPI app, `/analyze` endpoint (lines 730-851)
- `stock_tools.py`: Trading API integration, creates httpx client per call (lines 39-44)

### Environment Variables (current)
- `TRADING_API_BASE_URL`: Trading API endpoint (required)
- `ANTHROPIC_API_KEY`: Claude API key
- `CLAUDE_MODEL`: Model to use (default: claude-sonnet-4-5-20250929)

---

## Success Metrics

- Can handle 10+ concurrent requests without failures
- Response time degradation under load is predictable/linear
- No connection errors to Trading API under load
- Clear error messages when capacity exceeded
