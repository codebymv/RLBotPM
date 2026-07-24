# A3 — Vendor evaluation for H-PERP-003 D1 backfill

> Track A3 of [architecture-audit-03.md](../../architecture-audit-03.md). Goal:
> close the **D1 ≥ 365d** gap on `BTC-USDT-SWAP` funding history that the
> public OKX REST endpoint cannot serve (see
> [AUTH_INVESTIGATION.md](AUTH_INVESTIGATION.md): hard cap ~96d).
>
> Mark and spot 1H closes are NOT a vendor problem — they back-fill from the
> OKX public candle endpoints to ~July 2023 via
> [`fetch_hedged_panel.py`](fetch_hedged_panel.py) pagination. Vendors are only
> needed for the funding-rate column.

## Recommendation

**Use OKX's own free historical-data download as the primary backfill path.**
The CSV format from <https://www.okx.com/en-us/historical-data> covers
perpetual funding rates from **March 2022 onwards** — already past the 365d
gate — at zero cost, with the canonical exchange as the source.

If that download is somehow unavailable for an instrument, fall back to
Tardis.dev, which has a 60-day free trial covering Q2 2022 onwards.

## Comparison

| Vendor | Coverage for `BTC-USDT-SWAP` funding | Cost (one-time D1 backfill) | Format | Notes |
|--------|--------------------------------------|------------------------------|--------|-------|
| **OKX historical-data** (recommended) | March 2022 → present | **$0** | CSV download per asset/month | Canonical source; same publisher as the REST endpoint; no auth needed for the public download. |
| Tardis.dev | 2020-08 → present (Q2 2022 on trial) | $0 (free trial), then per-symbol-month | Normalized CSV / WebSocket replay | Best if you also want trades/book; 60-day trial covers D1 by itself. |
| Kaiko | 2020+ → present | Enterprise / contact sales | API + S3 | Overkill for one venue / one instrument. |
| Amberdata | 2018+ → present | Subscription, free tier limited | REST | Useful if you also need on-chain data. |
| CoinAPI | 2020+ → present | Subscription tiers | REST + flat files | Generic, less crypto-derivatives focused. |

## Action

1. Download `BTC-USDT-SWAP` historical funding CSV from OKX
   <https://www.okx.com/en-us/historical-data>. Pull every month from
   March 2022 to present (or whatever the page exposes as a single archive).
2. Concatenate into one CSV (any column shape; the ingest script auto-detects
   common header names).
3. Run:

   ```bash
   python research/datasets/H-PERP-003/ingest_external_csv.py \
       path/to/okx-funding-archive.csv
   ```

4. Verify the resulting `pull_log.jsonl` shows `phase3_d1_met: true`. The
   script back-fills mark/spot 1H closes from the OKX candle endpoints using
   the same `align_ok` and 60s-skew rules from
   [`06_backtest_design_H-PERP-003.md`](../../06_backtest_design_H-PERP-003.md).

## Why not pay first

The architecture audit's operating rule is to spend money only when free paths
are exhausted. OKX's first-party download is the most authoritative and the
cheapest option; paying a vendor for the same series would be redundant unless
the OKX download disappears or proves to be partial.
