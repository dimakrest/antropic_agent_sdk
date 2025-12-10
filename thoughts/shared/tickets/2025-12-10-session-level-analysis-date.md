# Session-Level Analysis Date Parameter

## Problem Statement

As an API user, I want to be able to provide an `analysis_date` parameter when making stock analysis requests, so that I can get historical analysis data for a specific date rather than only the current date.

Currently, the stock analysis tool calls `GET /api/v1/stocks/analysis/{symbol}` without any date context, meaning all analysis is performed against current/latest data only.

## Requirements

### Functional Requirements

1. **API Input**: The HTTP API should accept an optional `analysis_date` parameter alongside the stock ticker
2. **Session Management**: The `analysis_date` should be managed at the session level, not by the model during tool calling
3. **Tool Behavior**: When the agent calls the stock analysis tool, it should automatically include the `analysis_date` in the downstream API request if one was provided for the session
4. **Optional Parameter**: If `analysis_date` is not provided, the downstream `GET /api/v1/stocks/analysis/{symbol}` request should NOT include any date parameter (current behavior preserved)

### Non-Functional Requirements

1. The model/agent should not need to know about or manage the `analysis_date` - this is session-level configuration
2. The parameter should be transparent to the agent's decision-making process

## Acceptance Criteria

1. [ ] API accepts optional `analysis_date` parameter in the request
2. [ ] When `analysis_date` is provided, it is passed to the stock analysis API call
3. [ ] When `analysis_date` is NOT provided, the stock analysis API call remains unchanged (no date parameter added)
4. [ ] The agent/model does not need to explicitly handle or pass the `analysis_date` - it's injected at the session/tool level
5. [ ] Existing functionality without `analysis_date` continues to work unchanged

## Out of Scope (YAGNI)

- Date validation or format enforcement beyond what the downstream API requires
- Support for multiple dates in a single session
- Date range queries
- Caching based on analysis date
- Any UI/frontend changes
