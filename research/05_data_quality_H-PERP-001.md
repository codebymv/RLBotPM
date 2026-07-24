# 05 — Data quality: H-PERP-001 (perpetual funding)

## Coverage

| Metric | Value | Gate (plan) | Pass? |
|--------|-------|-------------|-------|
| Calendar span (OKX `BTC-USDT-SWAP` funding) | **~2026-01-14 → 2026-04-20** (~96 days) | ≥ 12 months | **NO** |
| Event count (`fundingTime` intervals) | **288** | ≥ 200 events | **YES** |
| Source | OKX public REST (Bybit/Binance blocked — see [DATA_SOURCE_SUBSTITUTION.md](datasets/H-PERP-001/DATA_SOURCE_SUBSTITUTION.md)) | — | — |

## Lookahead audit

- Each row uses **only** funding rates and timestamps **published at or before** that interval’s settlement.  
- No future rows merged into past decisions for this dataset.

## Survivorship bias

- Instrument is the **front** BTC USDT linear swap on OKX; no delisting occurred in-window.  
- If OKX truncates public funding history, **older regimes are missing** — biases tests **toward recent** crowding regimes only.

## Phase 3 gate verdict

**NOT MET** for the plan’s **12-month** requirement. Per plan instructions, we **do not** advance H-PERP-001 to Phase 4 as the primary funded research track without a longer pull (VPN/cloud re-fetch) or vendor data.

**Contingency executed:** acquired **H-SPOT-001** daily spot candles (see [05_data_quality_H-SPOT-001.md](05_data_quality_H-SPOT-001.md)) to satisfy Phase 3 for the **#2-ranked** hypothesis.

## Artifacts

- [datasets/H-PERP-001/btcusdt_swap_funding_okx.csv](datasets/H-PERP-001/btcusdt_swap_funding_okx.csv)
- [datasets/H-PERP-001/PROVENANCE.md](datasets/H-PERP-001/PROVENANCE.md)
