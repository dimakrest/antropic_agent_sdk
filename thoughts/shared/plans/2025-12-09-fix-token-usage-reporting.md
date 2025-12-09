# Fix Token Usage Reporting Implementation Plan

## Overview

The `/analyze` endpoint reports incorrect token usage (Input: 10, Output: 996) when actual usage should be significantly higher (Input: 1,500-5,000+, Output: 500-2,000). This plan addresses the issue by expanding the `UsageInfo` model to capture all token types and calculating the **true total input tokens**.

## Root Cause (Discovered via Expert Review)

The `input_tokens` field with prompt caching **does NOT represent total input tokens**. It only represents tokens **after the last cache breakpoint**. The correct formula is:

```
total_input_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
```

This is a **known issue** documented in [GitHub Issue #112](https://github.com/anthropics/claude-code-sdk-python/issues/112) on the claude-code-sdk-python repository.

## Current State Analysis

**File**: `http-server-agent-sdk/server.py`

### Current Implementation

1. **UsageInfo Model** (lines 75-78):
   - Only captures `input_tokens` and `output_tokens`
   - Missing: `cache_creation_input_tokens`, `cache_read_input_tokens`, `total_input_tokens`, `total_cost_usd`

2. **_build_metadata()** (lines 475-496):
   - Extracts only `input_tokens` and `output_tokens` from `ResultMessage.usage`
   - Does not calculate true total input tokens

### Key Discoveries

- SDK's `input_tokens` = only non-cached tokens (after last cache breakpoint)
- Most tokens are in `cache_creation_input_tokens` or `cache_read_input_tokens`
- The observed "10 input tokens" is correct for the non-cached portion
- System prompt tokens (~1,000+) are in cache tokens, not `input_tokens`

## Desired End State

The API should return accurate, complete token usage with calculated totals:

```json
{
  "metadata": {
    "usage": {
      "input_tokens": 10,
      "output_tokens": 892,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 3500,
      "total_input_tokens": 3510,
      "total_cost_usd": 0.0234
    }
  }
}
```

### Verification

- `total_input_tokens` should reflect actual API usage (system prompt ~1,000+ tokens, tools, messages, tool results)
- `total_input_tokens` should be at minimum 1,000+ for any request with the swing trading system prompt
- `total_cost_usd` should be present and non-zero

## What We're NOT Doing

- **Per-step usage tracking**: Not needed for this fix - `ResultMessage.usage` is already cumulative
- **Custom cost calculation**: We'll use `total_cost_usd` from the SDK as the authoritative metric
- **Breaking API changes**: New fields are additive with defaults

## Implementation Approach

Single-phase change in `server.py`:
1. Expand `UsageInfo` model with cache tokens, `total_input_tokens`, and `total_cost_usd`
2. Update `_build_metadata()` to extract all fields and calculate `total_input_tokens`
3. Add debug logging to verify the fix

## Phase 1: Expand Token Usage Capture

### Overview

Expand `UsageInfo` model and `_build_metadata()` to capture all token types, calculate true total input, and include authoritative cost.

### Changes Required

#### 1. Update UsageInfo Model

**File**: `http-server-agent-sdk/server.py:75-78`

**Current**:
```python
class UsageInfo(BaseModel):
    """Token usage information"""
    input_tokens: int = Field(..., description="Number of input tokens consumed")
    output_tokens: int = Field(..., description="Number of output tokens generated")
```

**New**:
```python
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
```

#### 2. Update _build_metadata() Function

**File**: `http-server-agent-sdk/server.py:475-496`

**Current** (lines 483-488):
```python
    usage_info = None
    if result_message and result_message.usage:
        usage_info = UsageInfo(
            input_tokens=result_message.usage.get("input_tokens", 0),
            output_tokens=result_message.usage.get("output_tokens", 0)
        )
```

**New**:
```python
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
```

#### 3. Verify Optional Import

**File**: `http-server-agent-sdk/server.py:12`

`Optional` is already imported from typing at line 12 - no change needed.

### Success Criteria

#### Automated Verification:
- [x] Unit tests pass: `cd http-server-agent-sdk && uv run pytest tests/test_server.py -v`
- [x] Server starts without errors: `uv run uvicorn server:app --port 8000`

#### Manual Verification:
- [x] Call `/analyze` endpoint with a stock symbol
- [x] Check debug logs show full `ResultMessage.usage` data including cache tokens
- [x] Verify `total_input_tokens` is now a realistic number (1,000+)
- [ ] Verify `total_cost_usd` is populated (SDK returns `null` - not an error, SDK behavior)
- [x] Response shows breakdown: `input_tokens` (small) + cache tokens (large) = `total_input_tokens` (realistic)

---

## Testing Strategy

### Unit Tests

Update existing unit tests to verify:
- `UsageInfo` model accepts new fields with defaults
- `_build_metadata()` correctly calculates `total_input_tokens`

### Integration Tests

Manual testing against the real API:
```bash
# Start server
cd http-server-agent-sdk
TRADING_API_BASE_URL=http://localhost:8131/api/v1 uv run uvicorn server:app --port 8000

# Test analyze endpoint
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"stock": "AAPL"}'
```

Expected debug output:
```
[DEBUG] ResultMessage.usage: {
  'input_tokens': 10,
  'output_tokens': 996,
  'cache_creation_input_tokens': 0,
  'cache_read_input_tokens': 3500,
  'total_cost_usd': 0.0234
}
```

Expected response metadata:
```json
{
  "usage": {
    "input_tokens": 10,
    "output_tokens": 996,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 3500,
    "total_input_tokens": 3510,
    "total_cost_usd": 0.0234
  }
}
```

## References

- Original ticket: `thoughts/shared/tickets/fix-token-usage-reporting.md`
- SDK Cost Tracking: `docs/agent-sdk/14-cost-tracking.md`
- Known Issue: [GitHub Issue #112](https://github.com/anthropics/claude-code-sdk-python/issues/112)
- Claude Prompt Caching: [docs.claude.com](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
