# 10 — Fundability gate review: H-SPOT-001

> **Rule:** Any **NO** blocks live capital — **no overrides**.

| # | Item | YES / NO | Evidence |
|---|------|----------|----------|
| 1 | Phase 1 venue survey evaluated this venue | **YES** | [02_venue_survey.md](02_venue_survey.md) (Coinbase ranked #1) |
| 2 | Phase 2 hypothesis has documented mechanism | **YES** | [04_hypothesis_library.md](04_hypothesis_library.md) (momentum lineage + fee cite) |
| 3 | Phase 3 data passes lookahead / survivorship audits | **YES** | [05_data_quality_H-SPOT-001.md](05_data_quality_H-SPOT-001.md) — **~949d / 950 bars** after pagination fix |
| 4 | Phase 4 design pre-registered before results | **YES** | `06_backtest_design_H-SPOT-001.md` timestamp order vs `07_*` |
| 5 | Phase 4 OOS Sharpe ≥ 1.5, ≥100 trades | **NO** | [07_backtest_results_H-SPOT-001.md](07_backtest_results_H-SPOT-001.md) — Sharpe **0.85** (830 OOS days) |
| 6 | Survives 2× fee stress | **YES** | 2× fee cumulative **+** (see metrics JSON) |
| 7 | Not single-trade / single-regime driven | **NO** | G4: only **2/3** OOS segments with mean `r_net` > 0 |
| 8 | Null / placebo test passes | **NO** | G6 failed; definition weak for negative-sum paths (documented in §3 of results) |
| 9 | Paper Sharpe within 30% of backtest | **NO / N/A** | Phase 4 **FAIL** — no Sharpe comparison required for plumbing ([09](09_paper_results_H-SPOT-001.md)) |
|10 | Paper ran ≥50 trades or ≥30 days | **PARTIAL (plumbing OK)** | **524** snapshots; **~3.7d** wall clock — plumbing gate per [08 amendment](08_paper_protocol_H-SPOT-001.md) |
|11 | Paper edge ≥50% of backtest after fees | **NO / N/A** | Same — not applicable once offline FAIL |
|12 | Capacity ≥5× per-trade $ | **N/A** | No live sizing approved |
|13 | Kill-switch & wealth floor documented | **PARTIAL** | Exists for Kalshi live stack; **not** ported to spot in this cycle |
|14 | Re-evaluation cadence documented | **NO** | Not defined for H-SPOT-001 live (hypothesis stopped at FAIL) |

## Bottom line

**LIVE CAPITAL: BLOCKED** — multiple **NO** answers, dominated by **Phase 4 FAIL**. Phase 5 **plumbing** is documented as complete in [09](09_paper_results_H-SPOT-001.md); that does **not** unblock capital.

Return to **Phase 2** ([04_hypothesis_library.md](04_hypothesis_library.md)) to pick a **new** hypothesis; do **not** resize this same SMA rule.
