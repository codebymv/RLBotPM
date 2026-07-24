# Data source substitution — H-PERP-001

## What happened

On **2026-04-19**, automated download from:

- `https://api.bybit.com/v5/market/funding/history` → **HTTP 403 Forbidden**
- `https://fapi.binance.com/fapi/v1/fundingRate` → **HTTP 451** (restricted location)

The research environment appears **US- or provider-geo restricted** for those hosts.

## Substitution (allowed under Phase 3)

Per **Phase 3 goal** (“Acquire enough historical data”), we substituted the **public OKX** endpoint:

- `GET https://www.okx.com/api/v5/public/funding-rate-history`
- Instrument: `BTC-USDT-SWAP`

**No change** to the pre-registered **P&L mapping** in `06_backtest_design_H-PERP-001.md`:

- Still: short perpetual, per-interval `pnl_i = fundingRate_i × V` once sign is verified against the venue’s definition for that field.

## Action item (optional hardening)

Re-run `fetch_funding.py` from a **non-restricted** network or cloud region and compare OKX vs Bybit funding correlation on overlapping timestamps; document correlation in `05_data_quality_H-PERP-001.md` appendix.
