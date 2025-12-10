# Ticket: Swing Trading Agent

## Summary

Create an AI agent that behaves as a professional swing trader specializing in short-term trades (3-5 days). The agent analyzes stocks and provides structured buy/no-buy recommendations with entry price, stop loss, take profit, and confidence scoring.

## User Story

As a trader, I want to send a stock ticker to a REST API endpoint and receive a professional swing trading analysis with a clear buy/no-buy recommendation, specific price levels, and reasoning.

## Input

- **Stock ticker** (string): e.g., "AAPL", "MSFT", "NVDA"
- Delivered via REST API endpoint

## Output

Structured JSON response:

```json
{
  "stock": "AAPL",
  "recommendation": "Buy",
  "entry_price": 150.25,
  "stop_loss": 145.50,
  "take_profit": 162.00,
  "reasoning": "Strong RSI momentum at 58, price above 20-day MA with increasing volume. Support at 145 tested twice. Risk/reward ratio 2.5:1 favorable for swing entry.",
  "missing_tools": ["Options flow data", "Sector relative strength", "Earnings calendar proximity"],
  "confidence_score": 75
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `stock` | string | The input ticker symbol |
| `recommendation` | enum | "Buy" or "Not Buy" |
| `entry_price` | number | Recommended entry price - agent decides between current market price or limit order near support based on conditions |
| `stop_loss` | number | Stop loss price level |
| `take_profit` | number | Take profit target price |
| `reasoning` | string | Professional analysis explaining the decision |
| `missing_tools` | array | Indicators/data that would increase confidence |
| `confidence_score` | number | 0-100 confidence in the recommendation (informational only - does not affect Buy/Not Buy decision) |

## Existing Tools Available

The agent will leverage these existing tools:

1. **`get_stock_data()`** - Fetches comprehensive technical analysis:
   - Price data (OHLCV)
   - Moving averages (SMA 20, 50, 200)
   - Momentum indicators (RSI, MACD)
   - Volatility (ATR, Bollinger Bands)
   - Trend indicators (ADX)
   - Support/Resistance levels

2. **`calculate_position_size()`** - Position sizing calculator

## Agent Behavior Requirements

### Trading Philosophy
- Focus: Swing trades with 3-5 day holding period
- Decision: Binary Buy/Not Buy (no "Hold" or "Maybe")
- The agent IS the professional - it applies its own trading expertise and judgment

### High-Level Guidelines
- Always provide stop loss (risk management is non-negotiable)
- Favor favorable risk/reward setups
- Agent determines its own analysis methodology based on available data

### Self-Assessment
The agent must identify what tools/indicators it's missing to do better analysis. This is critical feedback for improving the system over time.

## REST API Specification

### New Dedicated Endpoint
```
POST /analyze
```

This is a NEW dedicated endpoint (not using existing /chat) that returns structured JSON directly.

### Request Body
```json
{
  "stock": "AAPL"
}
```

### Response
Returns the structured trading recommendation (see Output section above)

### Error Responses
```json
{
  "error": "Symbol not found",
  "stock": "INVALID"
}
```

## Success Criteria

1. Agent returns valid structured response for any valid stock ticker
2. Recommendations include concrete price levels (not ranges)
3. Reasoning reflects professional swing trading analysis
4. Missing tools field identifies gaps in available data
5. Confidence score correlates with signal alignment
6. API endpoint is accessible and responds within reasonable time

## Out of Scope

- Portfolio management
- Position sizing recommendations (use existing tool separately)
- Sell signals for existing positions
- Options strategies
- Fundamental analysis
- News/sentiment analysis (unless tools added)

## Dependencies

- Existing HTTP server infrastructure (`http-server-agent-sdk/server.py`)
- Stock tools module (`http-server-agent-sdk/stock_tools.py`)
- Internal stock prices API running on `localhost:8093`
