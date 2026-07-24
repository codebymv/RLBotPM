# Dataset provenance — H-PERP-001

| Field | Value |
|-------|--------|
| Primary source (attempted) | Bybit `GET /v5/market/funding/history` — **HTTP 403** from fetch environment (2026-04-19) |
| **Substitution** | OKX public `GET /api/v5/public/funding-rate-history` |
| URL | https://www.okx.com/api/v5/public/funding-rate-history |
| Instrument | `BTC-USDT-SWAP` |
| Retrieved (UTC) | 2026-04-20T03:37:07.369619+00:00 |
| Rows | 288 |
| First timestamp | 2026-01-14T08:00:00+00:00 |
| Last timestamp | 2026-04-20T00:00:00+00:00 |
| Output CSV | `btcusdt_swap_funding_okx.csv` |

## Sign convention

OKX publishes `fundingRate` / `realizedRate` per interval. **Backtest code re-validates**
direction against OKX documentation before interpreting long/short cashflows.

## Reproducibility

```bash
python fetch_funding.py
```
