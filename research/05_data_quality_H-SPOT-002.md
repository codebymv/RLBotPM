# 05 — Data quality: H-SPOT-002 (Coinbase BTC–ETH daily)

**Design:** [06_backtest_design_H-SPOT-002.md](06_backtest_design_H-SPOT-002.md)  
**Pull script:** [datasets/H-SPOT-002/fetch_candles.py](datasets/H-SPOT-002/fetch_candles.py)  
**Provenance:** [datasets/H-SPOT-002/PROVENANCE.md](datasets/H-SPOT-002/PROVENANCE.md)

**Motivation (library):** BTC–ETH log-ratio mean reversion after `|z| > 2` on a 30-day rolling z-score — a **new mechanism**, not a retune of H-SPOT-001 dual-SMA windows ([04_hypothesis_library.md](04_hypothesis_library.md) · H-SPOT-001 closed FAIL in [07_backtest_results_H-SPOT-001.md](07_backtest_results_H-SPOT-001.md)).

---

## 1. Coverage vs contract

| Clause | Requirement | Latest panel |
|--------|-------------|---------------------------|
| **D1** | ≥ 730 calendar days of aligned BTC-USD + ETH-USD daily closes | **Met** — **949.0** days from first to last candle (**950** rows) after dual-product Coinbase pull. |
| **D2** | Same venue for both legs | **Met** — Coinbase Exchange public candles for `BTC-USD` and `ETH-USD`. |
| **D3** | Provenance | **Met** — [PROVENANCE.md](datasets/H-SPOT-002/PROVENANCE.md) (pull UTC, script, host, products, align stats). |
| **D4** | No calendar week with **>50%** missing days after align | **Met** — script check in [H-SPOT-002.py](backtests/H-SPOT-002.py) found **0** interior weeks with majority missing days. |

**Verdict on Phase 3 gate:** **PASS**. D1/D2/D3/D4 satisfied on the 2026-07-23 pull. Re-check D4 after any new dataset.

---

## 2. Alignment (dual daily)

Merge rule (frozen in `06`): **inner join** on candle timestamp (UTC epoch seconds). Dropped BTC-only / ETH-only rows: **0** / **0** per provenance.

Window: first candle `2023-12-17` → last `2026-07-23` (UTC day opens). Schema: `time`, `btc_close`, `eth_close`, `btc_volume`, `eth_volume`.

---

## 3. Candle history depth (API behavior)

Public Coinbase `GET /products/{product_id}/candles` with granularity `86400` returns a bounded newest page; older history uses backward `start`/`end` windows of **300** days per product (same pagination pattern as H-SPOT-001). Both legs paginated independently, then joined.

---

## 4. Lookahead / survivorship

- Strategy uses **closes** and a **one-day causal lag** on position (`pos[t] = pos_raw[t-1]`) — no same-bar execution peek beyond the rule frozen in `06`.
- Both pairs are continuous spot products; no equity-style delisting survivorship. Gap risk would show as missing timestamps — D4 is the density gate.

---

## 5. Artifacts

- [datasets/H-SPOT-002/btc_eth_daily_coinbase.csv](datasets/H-SPOT-002/btc_eth_daily_coinbase.csv)
- [datasets/H-SPOT-002/PROVENANCE.md](datasets/H-SPOT-002/PROVENANCE.md)
- [datasets/H-SPOT-002/fetch_candles.py](datasets/H-SPOT-002/fetch_candles.py)
