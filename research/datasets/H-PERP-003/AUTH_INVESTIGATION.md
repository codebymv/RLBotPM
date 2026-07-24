# A2 — Authenticated OKX history depth investigation

> Track A2 of [architecture-audit-03.md](../../architecture-audit-03.md). Goal:
> determine whether OKX API keys unlock deeper historical data than the public
> endpoints used by [fetch_hedged_panel.py](fetch_hedged_panel.py).

## Empirical probe (2026-05-04)

Public `GET /api/v5/public/funding-rate-history` for `BTC-USDT-SWAP` was
walked with `after=<oldest_in_page>` pagination using the same logic as
`fetch_hedged_panel.fetch_funding_all`. The endpoint **stops returning data
after page 3** (limit 100). One-shot probe script:
[`.probe_funding_depth.py`](.probe_funding_depth.py).

| Metric | Value |
|--------|--------|
| Pages walked before empty response | **3** |
| Total funding rows returned | **289** |
| Oldest `fundingTime` returned | `2026-01-28T16:00:00Z` |
| Newest `fundingTime` returned | `2026-05-04T16:00:00Z` |
| Effective span | **~96 calendar days** |

This is consistent with the prior pull (285 rows / 94.7 days) and with the
public OKX docs pattern of advertising "last 3 months" history on most
account/order endpoints.

## Conclusion

**Authentication will not solve D1.** The hard cap is at the
`funding-rate-history` endpoint level, not a rate-limit / auth tier. Even with
read-only API keys the endpoint is not documented to extend history.

For comparison, the candle endpoints used in the same fetcher
(`/market/history-candles`, `/market/history-mark-price-candles`) **do** reach
back to ~July 2023 by repeated `after`-pagination — that part of the data
contract is satisfiable from public REST. The blocker is funding only.

## Implication for Track A

- **A1 (self-capture)** is the only path that closes D1 from the OKX REST
  surface alone — and only over ~9 months of accumulation.
- **A2 (auth)** is **closed as a depth path**. Read-only OKX keys are still
  worth provisioning later for higher rate limits and consistent 5xx behavior,
  but they do not change the timeline.
- **A3 (paid archive)** becomes the only fast path to D1. OKX itself
  distributes historical funding rate data from March 2022 via
  <https://www.okx.com/en-us/historical-data> ("Historical perpetual funding
  rates from March 2022 onwards"). Vendor evaluation continues in A3.

## Reproduce

```bash
python research/datasets/H-PERP-003/.probe_funding_depth.py
```
