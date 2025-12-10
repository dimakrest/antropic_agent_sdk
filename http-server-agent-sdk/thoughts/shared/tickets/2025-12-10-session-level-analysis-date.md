# Session-Level Analysis Date Parameter

## Problem

Users cannot specify a historical date for stock analysis. Currently, stock analysis always uses the latest available data. Users need the ability to analyze stocks as of a specific date for backtesting and historical analysis purposes.

## Requirements

1. Users can optionally provide an `analysis_date` parameter alongside the stock ticker when requesting analysis
2. When `analysis_date` is provided, it is automatically included in the Trading API request (`GET /api/v1/stocks/analysis/{symbol}`)
3. When `analysis_date` is NOT provided, the request remains unchanged (no date parameter added)
4. The `analysis_date` is managed at the session level, transparent to the model/tool calling logic

## Acceptance Criteria

- [ ] API accepts optional `analysis_date` parameter in analysis requests
- [ ] When provided, `analysis_date` is passed to the Trading API
- [ ] When not provided, Trading API request has no date parameter
- [ ] The model/agent does not need to be aware of or handle the date parameter
- [ ] Session manages the parameter injection transparently

## Out of Scope (YAGNI)

- Date validation beyond basic format checking
- Date range queries (only single date supported)
- Caching of historical analyses
- UI/frontend changes
