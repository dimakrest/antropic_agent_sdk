# Stock Prices API

Internal API for fetching historical stock price data with configurable intervals and time periods.

## Base URL

```
http://localhost:8093
```

## Endpoints

### Get Stock Prices

Retrieve historical price data (OHLCV candles) for a given stock symbol.

```
GET /api/v1/stocks/{symbol}/prices
```

#### Path Parameters

| Parameter | Type   | Required | Description                          |
|-----------|--------|----------|--------------------------------------|
| symbol    | string | Yes      | Stock ticker symbol (e.g., AAPL, MSFT) |

#### Query Parameters

| Parameter     | Type    | Options                                     | Default | Description                    |
|---------------|---------|---------------------------------------------|---------|--------------------------------|
| interval      | string  | `1d`, `1wk`, `1mo`                          | `1d`    | Candle interval                |
| period        | string  | `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `ytd`, `max` | -       | Quick time range               |
| start_date    | string  | `YYYY-MM-DD`                                | -       | Custom start date              |
| end_date      | string  | `YYYY-MM-DD`                                | -       | Custom end date                |
| force_refresh | boolean | `true`, `false`                             | `false` | Bypass cache, fetch fresh data |

**Note:** Either `period` OR (`start_date` + `end_date`) should be provided, not both.

## Usage Examples

### Daily Candles (Last 3 Months)

```bash
curl "http://localhost:8093/api/v1/stocks/AAPL/prices?period=3mo&interval=1d"
```

### Weekly Candles (Last Year)

```bash
curl "http://localhost:8093/api/v1/stocks/AAPL/prices?period=1y&interval=1wk"
```

### Monthly Candles (5 Years)

```bash
curl "http://localhost:8093/api/v1/stocks/AAPL/prices?period=5y&interval=1mo"
```

### Custom Date Range with Weekly Candles

```bash
curl "http://localhost:8093/api/v1/stocks/AAPL/prices?start_date=2025-06-01&end_date=2025-12-02&interval=1wk"
```

### Force Fresh Data (Bypass Cache)

```bash
curl "http://localhost:8093/api/v1/stocks/AAPL/prices?period=1mo&force_refresh=true"
```

## Response Format

```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "data": [
    {
      "date": "2025-12-02",
      "open": 150.25,
      "high": 152.30,
      "low": 149.80,
      "close": 151.45,
      "volume": 45678900
    }
  ]
}
```

## Interval Selection Guide

| Use Case                        | Recommended Interval | Recommended Period |
|---------------------------------|---------------------|-------------------|
| Day trading / Short-term        | `1d`                | `1mo` - `3mo`     |
| Swing trading                   | `1d` or `1wk`       | `3mo` - `1y`      |
| Long-term investing             | `1wk` or `1mo`      | `1y` - `5y`       |
| Technical analysis (patterns)   | `1d`                | `6mo` - `1y`      |
| Fundamental analysis (trends)   | `1mo`               | `2y` - `5y`       |
