# 10 — Fundability gate review: H-PERP-003

> **Rule:** Any **NO** blocks live capital — **no overrides**.
>
> **Status:** Gate-active after Phase 4 PASS. Items below are pre-filled with
> the evidence currently available; cells marked _PENDING_ flip to YES/NO only
> after the corresponding upstream artifact lands. Capital remains BLOCKED
> because Phase 5 paper evidence and operational risk items are not complete.

| # | Item | YES / NO | Evidence |
|---|------|----------|----------|
| 1 | Phase 1 venue survey evaluated this venue | **YES** | OKX is the canonical venue listed in `02_venue_survey.md` for crypto perps. |
| 2 | Phase 2 hypothesis has documented mechanism | **YES** | `04_hypothesis_library.md` records the funding-carry + delta-hedge structure; `06 §3` formalizes `pnl_i = V·fundingRate + V·(r_S − r_F)`. |
| 3 | Phase 3 data passes lookahead / survivorship audits | **YES** | [05_data_quality_H-PERP-003.md](05_data_quality_H-PERP-003.md) — D1/D2/D3/D4 all met after OKX first-party archive ingest (`days_span=369.0`, `align_ok=100%`). |
| 4 | Phase 4 design pre-registered before results | **YES** | [06_backtest_design_H-PERP-003.md](06_backtest_design_H-PERP-003.md) committed before any `07_*` row; G6 amendment also pre-registered (06 §5 amendment, 2026-05-04). Cross-cutting enforcement: [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md) §pre-registration. |
| 5 | Phase 4 OOS Sharpe ≥ 1.5, ≥100 trades | **YES** | Phase 4: `sharpe_oos=7.9733`, `n_oos_intervals=831`; see [07_backtest_results_H-PERP-003.md](07_backtest_results_H-PERP-003.md). |
| 6 | Survives 2× fee stress (G3) | **YES** | Phase 4: `cum_oos_2x_window_fees_usdt=2.485304` (positive). |
| 7 | Not single-trade / single-regime driven (G4 + G5) | **YES** | Phase 4: all four segment means positive `[0.005262, 0.005828, 0.003372, 0.00136]`; `g5_concentration=true`. |
| 8 | Null / placebo test passes (G6) | **YES** | Phase 4 random sign-flip placebo: `g6_placebo_frac_beats=0.0` with 500 trials, seed 42. |
| 9 | Paper Sharpe within 30% of backtest | **NO** | Phase 5 verifier 2026-07-23 (Sharpe aligned to Phase 4 `sharpe_8h`): paper Sharpe **~4.07** vs Phase 4 OOS **7.97** (~49% relative drift, outside 30% band). Earlier **1.90** figure was a sample-`sqrt(n)` bug — corrected; still **NO**. See [09_paper_results_H-PERP-003.md](09_paper_results_H-PERP-003.md). |
| 10 | Paper ran ≥50 trades or ≥30 days | **YES** | ~79 calendar days, 237 snapshots / 236 closed intervals. |
| 11 | Paper edge ≥50% of backtest after fees | **NO** | Paper PF **1.44** vs Phase 4 OOS PF **1.88** is within 30% PF drift, but Sharpe transfer failed item 9; treat edge transfer as **NO** for promotion. |
| 12 | Capacity ≥5× per-trade $ | **YES** | Per-trade notional is `V = 100 USDT` (06 §3). [shared/config/h_perp_003_risk.yaml](../shared/config/h_perp_003_risk.yaml) `capacity` block: floor `500 USDT` (5×), expected slippage at this notional `~5 bps`, OKX VIP-0 single-account cap on `BTC-USDT-SWAP` is ~$40M face — about 5 orders of magnitude above the floor. Maker default in 06 §3 keeps slippage at zero by design. |
| 13 | Kill-switch & wealth floor documented | **YES** | [shared/config/h_perp_003_risk.yaml](../shared/config/h_perp_003_risk.yaml) `kill_switch` and `wealth_floor` blocks: tracking-error breaker (verifier FAIL ×2 → halt), 12h funding-snapshot staleness breaker, funding-sign-flip breaker (rolling 7d mean ≤ −1 bp → halt), 5 USDT / 5% daily P&L cap, 15% total drawdown cap, account floor 200 USDT (2×), max 5% of liquid net worth, forced unwind in 15 min on floor breach, no auto-resume. |
| 14 | Re-evaluation cadence documented | **YES** | [shared/config/h_perp_003_risk.yaml](../shared/config/h_perp_003_risk.yaml) `re_evaluation` block: scheduled re-run every 90 days on rolling 365d window via `python research/backtests/H-PERP-003.py`. Forced triggers: Sharpe or PF regression > 50% vs last PASS, G6 placebo `frac_beats > 0.05`, weekly `align_ok < 99%`, any OOS segment mean turns negative, or venue mechanism change. Capital pauses BEFORE the re-run starts and stays paused until a new Phase 4 PASS + fundability review is committed. |

## Bottom line

**LIVE CAPITAL: BLOCKED.**

Closed by items **9** and **11** (paper Sharpe / edge transfer failed). Item
**10** is met. Items 12–14 remain documented in
[shared/config/h_perp_003_risk.yaml](../shared/config/h_perp_003_risk.yaml).
Do not reopen live discussion on H-PERP-003 without a new pre-registered
mechanism or regime hypothesis — do not tune the failed paper window.
