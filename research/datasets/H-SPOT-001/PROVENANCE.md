# Dataset provenance — H-SPOT-001

| Field | Value |
|-------|--------|
| Source | Coinbase Exchange REST `GET /products/BTC-USD/candles` |
| URL | https://api.exchange.coinbase.com/products/BTC-USD/candles |
| Granularity | 86400 (1 day) |
| Retrieved (UTC) | 2026-04-20T04:04:35.718523+00:00 |
| Rows | 950 |
| Span (days) | ~949 |
| First candle (UTC) | 2023-09-14T00:00:00+00:00 |
| Last candle (UTC) | 2026-04-20T00:00:00+00:00 |

Schema: time (epoch s), low, high, open, close, volume (Coinbase native order).
Pagination: initial unbounded request + backward `300`-day `start`/`end` windows.
