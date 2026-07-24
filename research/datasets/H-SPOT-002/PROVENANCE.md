# Dataset provenance — H-SPOT-002

| Field | Value |
|-------|--------|
| Source | Coinbase Exchange REST `GET /products/{BTC-USD,ETH-USD}/candles` |
| Granularity | 86400 (1 day) |
| Retrieved (UTC) | 2026-07-23T23:46:01.603752+00:00 |
| Aligned rows | 950 |
| Span (days) | ~949 |
| First candle (UTC) | 2023-12-17T00:00:00+00:00 |
| Last candle (UTC) | 2026-07-23T00:00:00+00:00 |
| BTC-only rows (dropped) | 0 |
| ETH-only rows (dropped) | 0 |

Schema: time (epoch s), btc_close, eth_close, btc_volume, eth_volume.
Join: inner join on candle timestamp. Pagination: initial unbounded request +
backward `300`-day `start`/`end` windows per product.
