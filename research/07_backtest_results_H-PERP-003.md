# 07 — Backtest results: H-PERP-003

**Status: PASS (Phase 4)** — calendar depth **D1** is satisfied after the
first-party OKX historical funding archive ingest (**369.0** days, **1108**
funding rows, `align_ok = 1.0`). See
[05_data_quality_H-PERP-003.md](05_data_quality_H-PERP-003.md).

Metrics below are the gate-eligible Phase 4 verdict under the frozen
[06_backtest_design_H-PERP-003.md](06_backtest_design_H-PERP-003.md).

---

## Data window (current)

| Item | Value |
|------|--------|
| Funding rows | 1108 |
| `days_span` | 369.0 |
| First / last `fundingTime` | See [PROVENANCE_PULL.md](datasets/H-PERP-003/PROVENANCE_PULL.md) and [INGEST_okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.md](datasets/H-PERP-003/INGEST_okx_BTC-USDT-SWAP_funding_2025-05_to_2026-04.md) |
| `align_ok` fraction | 1.0 |
| D4 (weeks with >50% non-merge) | 0 bad weeks (script check) |
| Data acquisition | First-party OKX historical funding archive via [download_okx_funding_archive.py](datasets/H-PERP-003/download_okx_funding_archive.py) + daily tail capture via [daily_capture.py](datasets/H-PERP-003/daily_capture.py) |

---

## OOS protocol (reference)

`K = 4` equal-duration segments on the funding timeline; **OOS = segments 2–4** (0-indexed segments 1–3). Interval PnL uses consecutive aligned rows: `pnl_i = V·fundingRate_{i+1} + V·(ln S_{i+1}/S_i − ln F_{i+1}/F_i)` with `V = 100`, per [06_backtest_design_H-PERP-003.md](06_backtest_design_H-PERP-003.md) §3.

---

## Phase 4 metrics ([H-PERP-003_metrics.json](backtests/H-PERP-003_metrics.json))

Regenerate with `python RLBotPM/research/backtests/H-PERP-003.py` after each new CSV. Pre-registered `--bootstrap-trials 500 --seed 42`.

| Field | Phase 4 run (2026-05-05 UTC, 369.0d) |
|-------|---------------------------------|
| `n_oos_intervals` | 831 |
| `sharpe_oos` | 7.9733 (**G1 PASS**) |
| `profit_factor_oos` | 1.8826 (**G2 PASS**) |
| `cum_oos_after_window_fees_usdt` | 2.705304 |
| `cum_oos_2x_window_fees_usdt` | 2.485304 (**G3 PASS**) |
| `g4_segpos` | True; all four segment means positive: `[0.005262, 0.005828, 0.003372, 0.00136]` |
| `g5_concentration` | True |
| `g6_method` | `random_sign_flip` (06 §5 amendment, pre-registered 2026-05-04) |
| `g6_placebo_frac_beats` | 0.0 (**G6 PASS**) |
| `verdict` | **PASS** |

### G6 placebo: degeneracy fixed

The previous `g6_placebo_frac_beats: 1.0` was caused by a permutation-of-multiset comparator. Cumulative sum (and Sharpe) are invariant under any rearrangement of the per-interval PnLs, so the test produced no signal. The pre-registered fix in [06 §5](06_backtest_design_H-PERP-003.md) is a **random sign-flip** placebo: each trial multiplies each interval by an independent ±1, producing a near-symmetric null distribution around 0. The Phase 4 beat-rate is **0.0**, so the realized OOS carry stream clears the placebo gate.

---

## Verdict

**PASS** — `data_contract_ok = true` and every gate `g1..g6_pass` is true.

This triggers Phase 5 paper protocol activation in
[08_paper_protocol_H-PERP-003.md](08_paper_protocol_H-PERP-003.md). It does
**not** authorize live capital; [10_fundability_review_H-PERP-003.md](10_fundability_review_H-PERP-003.md)
remains blocked until the paper protocol completes and the operational risk
items are written for real.

## Phase 4 runner (when D1 met)

Once `pull_log.jsonl` reports `phase3_d1_met: true` (or `ingest_external_csv.py` returns the same), the procedure is:

1. Run `python RLBotPM/research/backtests/H-PERP-003.py` (deterministic, seed 42).
2. Read `H-PERP-003_metrics.json`.
3. Replace the table above with the new values, append the run timestamp.
4. State the verdict:
   - **PASS** ⇔ all of `g1..g6_pass` true AND `data_contract_ok` true → trigger A5.paper-protocol (see [08_paper_protocol_H-PERP-003.md](08_paper_protocol_H-PERP-003.md)).
   - **FAIL** ⇔ `data_contract_ok` true AND any gate false → archive cleanly, update [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md).
   - **INCONCLUSIVE_DATA** ⇔ `data_contract_ok` false → keep capturing, no verdict.
5. Commit with message starting `phase-4(H-PERP-003): <verdict>` so it grep-finds easily later.
