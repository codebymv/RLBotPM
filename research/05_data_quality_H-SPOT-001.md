# 05 — Data quality: H-SPOT-001 (Coinbase BTC-USD daily)

## Coverage

| Metric | Value | Gate (plan) | Pass? |
|--------|-------|-------------|-------|
| Calendar span | **2023-09-14 → 2026-04-20** (~949 days / ~31 months) | ≥ 12 months | **YES** |
| Bar count | **950** daily candles | ≥ 200 events | **YES** |

> Extended pull uses `start`/`end` pagination on Coinbase (see `fetch_candles.py`).

## Lookahead audit

- Bars are OHLC **as of** each day’s close; strategy uses **close** and **lags** of closes for moving averages — **no same-bar peek** beyond documented rule in `06_backtest_design_H-SPOT-001.md`.

## Survivorship bias

- `BTC-USD` is a continuous spot pair; no corporate-action style survivorship like single-stock delisting.  
- Exchange outage gaps: **none visible** in the pulled window (would appear as missing timestamps — not checked row-by-row in v1).

## Phase 3 gate verdict

**MET** for both **≥12 months** and **≥200 observations** after Coinbase pagination fix (2026-04-20).

## Artifacts

- [datasets/H-SPOT-001/btcusd_daily_coinbase.csv](datasets/H-SPOT-001/btcusd_daily_coinbase.csv)
- [datasets/H-SPOT-001/PROVENANCE.md](datasets/H-SPOT-001/PROVENANCE.md)
