# Ticket: Stock Prices Custom Tool Integration

**Date**: 2025-12-03
**Status**: Implementation Complete (Pending Manual Verification)

---

## Problem

The stock analysis agent needs access to historical stock price data from our internal Stock Prices API. Currently, there is no tool that allows the agent to fetch this data programmatically.

---

## Requirements

### Must Have
1. Custom tool that fetches stock price data from the internal API (`localhost:8093`)
2. Support for all API parameters:
   - Symbol (path parameter)
   - Interval (`1d`, `1wk`, `1mo`)
   - Period (`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `ytd`, `max`)
   - Custom date range (`start_date`, `end_date`)
3. Type-safe implementation using Zod schemas
4. Proper error handling for API failures

### Should Have
1. `force_refresh` parameter support to bypass cache
2. Clear, structured response format for agent consumption
3. Validation of date formats and parameter combinations

### Nice to Have
1. Response summarization for large datasets (agent-friendly output)
2. Basic statistics in response (e.g., price change %, high/low range)

---

## Acceptance Criteria

1. [x] Agent can request stock prices for any valid ticker symbol
2. [x] Agent can specify different intervals (daily, weekly, monthly)
3. [x] Agent can use quick periods (1mo, 1y, etc.) OR custom date ranges
4. [x] Invalid parameters return helpful error messages
5. [x] Tool integrates with the Agent SDK via `create_sdk_mcp_server` (Python equivalent)
6. [ ] Documentation includes usage examples (pending manual verification)

---

## Technical Notes

### API Reference
See: `docs/internal-apis/stock-prices-api.md`

### Suggested Tool Design

```typescript
tool(
  "get_stock_prices",
  "Fetch historical stock price data (OHLCV candles) for analysis",
  {
    symbol: z.string().describe("Stock ticker symbol (e.g., AAPL, MSFT)"),
    interval: z.enum(["1d", "1wk", "1mo"]).default("1d").describe("Candle interval"),
    period: z.enum(["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"])
      .optional()
      .describe("Quick time range (use this OR start_date/end_date)"),
    start_date: z.string().optional().describe("Custom start date (YYYY-MM-DD)"),
    end_date: z.string().optional().describe("Custom end date (YYYY-MM-DD)"),
    force_refresh: z.boolean().optional().default(false).describe("Bypass cache")
  },
  async (args) => {
    // Implementation here
  }
)
```

### Integration Pattern
- Use `createSdkMcpServer` to create an in-process MCP server
- Register as `stock-tools` server
- Tool name will be: `mcp__stock-tools__get_stock_prices`

---

## Out of Scope (YAGNI!)

- Real-time streaming price updates
- Order placement or trade execution
- Multiple symbol batch requests (one symbol per call for now)
- Caching layer on the agent side (API handles caching)
- Authentication/API keys (internal API, localhost only)

---

## Context

This is the first data tool for the stock analysis agent. The custom tool approach was chosen because:
1. Low latency (in-process execution)
2. Type safety with Zod
3. Simple integration with single codebase
4. Internal API doesn't need external MCP server infrastructure

Future tools (fundamentals, news, etc.) will follow the same pattern.
