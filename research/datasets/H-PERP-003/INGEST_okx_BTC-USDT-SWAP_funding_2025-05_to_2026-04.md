# Ingest provenance — `okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.csv`

| Field | Value |
|-------|--------|
| Ingested (UTC) | 2026-05-05T00:23:15.141272+00:00 |
| Source file | `research\datasets\H-PERP-003\okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.csv` |
| Detected funding-time column | `fundingTime` |
| Detected funding-rate column | `fundingRate` |
| Time units | `ms` |
| External rows parsed | 1095 |
| Rows added to panel | 797 |
| Rows confirmed (already matched) | 298 |
| Rows overwritten | 0 |
| Rows skipped (mismatch, no overwrite) | 0 |
| Panel rows after ingest | 1108 |
| Panel `align_ok` after ingest | 100.00% |
| Span (days) | 369.00 |
| D1 met (≥365d) | True |

## Reproduce

```bash
python research/datasets/H-PERP-003/ingest_external_csv.py research\datasets\H-PERP-003\okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.csv \
    --funding-time-col fundingTime \
    --funding-rate-col fundingRate \
    --ts-units ms
```
