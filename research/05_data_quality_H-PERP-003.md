# 05 — Data quality: H-PERP-003 (OKX hedged panel)

**Design:** [06_backtest_design_H-PERP-003.md](06_backtest_design_H-PERP-003.md)  
**Pull script:** [datasets/H-PERP-003/fetch_hedged_panel.py](datasets/H-PERP-003/fetch_hedged_panel.py)  
**Latest provenance:** [datasets/H-PERP-003/PROVENANCE_PULL.md](datasets/H-PERP-003/PROVENANCE_PULL.md)

---

## 1. Coverage vs contract

| Clause | Requirement | Latest panel |
|--------|-------------|---------------------------|
| **D1** | ≥ 365 calendar days of venue-native funding intervals | **Met** — **369.0** days from first to last `fundingTime` (**1108** rows) after first-party OKX historical funding archive ingest. |
| **D2** | Same venue for funding, mark, spot | **Met** — OKX `BTC-USDT-SWAP` funding + mark 1H, `BTC-USDT` spot 1H. |
| **D3** | Provenance | **Met** — `PROVENANCE_PULL.md`, [provenance.txt](datasets/H-PERP-003/provenance.txt), and [INGEST_okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.md](datasets/H-PERP-003/INGEST_okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.md). |
| **D4** | No calendar week with **>50%** missing intervals after merge | **Met** — `align_ok` **100%** of rows; ISO-week check in [H-PERP-003.py](backtests/H-PERP-003.py) found **0** weeks with majority non-aligned rows. |

**Verdict on Phase 3 gate:** **PASS**. D1/D2/D3/D4 are all satisfied on the first-party OKX archive panel. D4 can still fail on future pulls if alignment degrades; re-run the weekly check after each new dataset.

---

## 2. Alignment (1H bar + skew)

Merge rule (frozen in `06`): unique 1H candle whose half-open interval `[ts, ts + 1h)` contains `fundingTime`; each leg’s skew to the nearer bar boundary must be **≤ 60s**.

On the latest CSV, **100%** of funding rows satisfied both legs (mark and spot closes present, skew within bound). Skew columns are retained in the CSV for spot audits on future pulls.

---

## 3. Candle history depth (API behavior)

Public `GET /api/v5/market/history-mark-price-candles` and `history-candles` return **up to 300** bars per request. Older history is obtained by paging with **`after=<min_open_ts − 1>`** (newest-first payload). The previous **`before=`** walk did **not** extend coverage in practice.

The first-party archive ingest back-filled the required mark/spot 1H closes from OKX candle endpoints for the **369.0d** funding span. The final merged panel has **1108** funding rows and **100%** `align_ok`.

---

## 4. Funding sign (short)

Positive OKX `fundingRate` is treated as **payment to the short** (long pays short), consistent with common USDT linear perp documentation. **Verify** at execution time against current OKX help pages; if the venue’s sign convention differs, flip the mapping once in [H-PERP-003.py](backtests/H-PERP-003.py) and record the change in `provenance.txt`.

---

## 5. Archive ingest that cleared D1

On 2026-05-05 UTC, the D1 blocker was cleared via the OKX historical-data
download endpoint:

```powershell
python research/datasets/H-PERP-003/download_okx_funding_archive.py `
    --start 2025-05 --end 2026-04 `
    --out research/datasets/H-PERP-003/okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.csv

python research/datasets/H-PERP-003/ingest_external_csv.py `
    research/datasets/H-PERP-003/okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.csv `
    --funding-time-col fundingTime `
    --funding-rate-col fundingRate `
    --ts-units ms
```

Ingest payload: `external_rows=1095`, `rows_added=797`,
`rows_confirmed=298`, `rows_total=1108`, `align_ok_pct=100.0`,
`days_span=369.0`, `phase3_d1_met=true`.
