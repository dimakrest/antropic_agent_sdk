# Fix Token Usage Reporting

## Problem

The `/analyze` endpoint reports incorrect token usage:
- **Reported**: Input: 10, Output: 996
- **Expected**: Input: 1,500-5,000+, Output: 500-2,000

10 input tokens is impossibly low given the system prompt alone is ~1,000+ tokens, plus tool definitions and tool results.

## Root Cause (Suspected)

The SDK's `ResultMessage.usage` may be reporting only the final turn's tokens rather than cumulative totals across all turns. According to documentation, `ResultMessage.usage` should contain cumulative totals.

## Requirements

### 1. Capture All Token Types
The `UsageInfo` model should include:
- `input_tokens`
- `output_tokens`
- `cache_creation_input_tokens`
- `cache_read_input_tokens`

### 2. Track Per-Step Usage
Track usage from individual `AssistantMessage` objects throughout the response stream, not just from `ResultMessage`. Deduplicate by message ID to avoid double-counting.

### 3. Report Total Cost
Include `total_cost_usd` from `ResultMessage` as the authoritative cost metric.

### 4. Debug Logging
Add temporary logging to capture full `ResultMessage.usage` dict for investigation.

## Acceptance Criteria

- [ ] Token counts reflect actual API usage (system prompt + tools + messages + tool results)
- [ ] Cache tokens are captured separately
- [ ] `total_cost_usd` is included in metadata
- [ ] Per-step usage is tracked and summed correctly

## Files Affected

- `server.py` - `UsageInfo` model, `_build_metadata()`, `analyze_stock()` endpoint

## References

- SDK Cost Tracking: `docs/agent-sdk/14-cost-tracking.md`
- Known Issue: https://github.com/anthropics/claude-agent-sdk-python/issues/289
