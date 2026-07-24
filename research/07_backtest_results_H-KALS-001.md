# 07 — Observation results: H-KALS-001

> **Design:** [06_backtest_design_H-KALS-001.md](06_backtest_design_H-KALS-001.md)  
> **Raw log:** [datasets/H-KALS-001/scan_events.jsonl](datasets/H-KALS-001/scan_events.jsonl)  
> **Scanner:** [scanners/kals_001_probability_sum_scan.py](scanners/kals_001_probability_sum_scan.py)

This is an **observation study**, not a profitability backtest. **No trading recommendation.**

---

## 0. Final verdict (steering closure)

| Scope | Verdict |
|--------|---------|
| **Narrow claim** (rule set A detects **MECE** probability-algebra violations you would trade) | **FAIL / PARKED** — single-scan evidence (~88% of “candidates” flagged) refutes a clean MECE interpretation; see §3. |
| **Infrastructure** (pagination, JSONL pipeline, adapter) | **PASS** — G0/G1 satisfied on recorded runs. |
| **Pre-registered G2/G3** (56 scans under **unchanged** rule set A) | **Waived** — not pursued; completing 56 scans on a definition we no longer treat as meaningful would be **science theater** (see [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md); follow-on **H-KALS-001b**). |

The frozen thresholds in `06` are **not** edited; this document records an **early steering closure** on interpretation and effort, not a retroactive change to the pre-registration file.

---

## 1. Scan batch to date

| Field | Value |
|-------|--------|
| Successful full scans logged | **≥1** (multiple demo runs may exist in JSONL) |
| API | Kalshi **demo** (`demo=true`) on recorded lines |
| Example `timestamp` | `2026-04-25T22:48:13.029076+00:00` (first documented line) |
| `markets_fetched` (pagination complete) | **74,990** (representative full pass) |
| `events_partition_candidates` (rule set A) | **42** |
| `violation_count` (same scan) | **37** |

---

## 2. Gates (from `06`) — literal vs steering

| Gate | Literal status | Steering note |
|------|----------------|---------------|
| G0 Pagination complete | **PASS** | |
| G1 API reliability | **PASS** | |
| G2 Existence (≥56 scans, ≥1 violation) | **Not completed** | Waived; see §0. |
| G3 Null absence | **Not completed** | Waived with G2. |

---

## 3. Interpretation (non-gated)

Under **rule set A**, a single full pass already flags **37 / 42** “partition candidate” events (~88%). That rate is **too high** to treat as genuine probability-algebra violations on mutually exclusive outcomes: the same `event_ticker` often bundles **related but not MECE** buckets (e.g. multiple threshold lines, partial ladders, or sibling contracts that do not form one exhaustive partition).

**Conclusion for research steering:** H-KALS-001 **as specified** is useful as a **data plumbing** and **grouping-stress** test, but **not** a clean test of the library’s “pure mechanical sum violation” story. Follow-on **H-KALS-001b** uses a **new** `06_*` and scanner variant — see [06_backtest_design_H-KALS-001b.md](06_backtest_design_H-KALS-001b.md).

---

## 4. Sample violation rows (abbreviated)

Full payloads are in `scan_events.jsonl`. Examples from an early scan:

| `event_ticker` | `kind` | `sum_p` | Note |
|----------------|--------|---------|------|
| KXHIGHLAX-26APR25 | OVER | 1.12 | 3 markets |
| KXPAYROLLS-26APR | OVER | 5.945 | Many co-listed thresholds — likely non-MECE under naive sum |
| KXGOVTCUTS-28 | OVER | 1.99 | Two buckets at ~0.995 YES mid each |

---

## 5. Operator next steps

1. **Do not** schedule 56 additional scans for H-KALS-001 rule set A unless you explicitly reopen that scientific question.
2. Run **H-KALS-001b**: `python research/scanners/kals_001_probability_sum_scan.py --variant 001b --once` (see `06_backtest_design_H-KALS-001b.md`).
3. **Do not** allocate capital from this document.
