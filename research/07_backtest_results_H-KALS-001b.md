# 07 — Observation results: H-KALS-001b

> **Design:** [06_backtest_design_H-KALS-001b.md](06_backtest_design_H-KALS-001b.md)  
> **Raw log:** [datasets/H-KALS-001b/scan_events.jsonl](datasets/H-KALS-001b/scan_events.jsonl)  
> **Scanner:** [scanners/kals_001_probability_sum_scan.py](scanners/kals_001_probability_sum_scan.py) (`--variant 001b`, optional `--repeat N`)

**Status:** **G3 batch complete** (10 successful demo scans). **Steering:** **PARKED** — see [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md). No trading recommendation.

---

## 1. Scan batch

| Field | Value |
|-------|--------|
| Successful scans | **10** |
| API | Kalshi **demo** (`demo=true` on all lines) |
| First `timestamp` | `2026-04-25T23:03:47.887302+00:00` |
| Last `timestamp` | `2026-04-26T00:12:46.641115+00:00` |
| `violation_count` per scan | 7, 7, 7, 6, 6, 7, 7, 7, 7, 6 |
| Sum of `violation_count` over batch | **67** |
| `markets_fetched` range | ~77,048–79,111 (venue listing drift between runs) |

---

## 2. Gates (from `06`)

| Gate | Result | Notes |
|------|--------|--------|
| G0 | **PASS** | Pagination complete on each line. |
| G1 | **PASS** | No scan-level HTTP failure in this batch. |
| G2 | **Descriptive** | Stable small ladder count (6–7 candidates/scan); violations every scan. |
| G3 | **Evaluated** | Pre-registered **FAIL** applies only if **zero** violations **across all 10** scans. Observed **≥ 6 violations every scan** → that FAIL branch **does not apply**. Per `06`, outcome is **“continue logging”** (optional); **not** evidence of a tradeable bundle. |

**Overall:** **Not FAIL (G3)** on the zero-violation null; **not** a Phase-4-style PASS for capital.

---

## 3. Interpretation

Violations remain mostly **UNDER** (`Σp < 0.95`) on **contiguous `between` ladders**, consistent with **incomplete outcome coverage** (ladder is contiguous on the strike axis but not guaranteed exhaustive for the event). **OVER** cases are rarer in this batch but still appear inside multi-leg ladders (e.g. long BTC year strips).

**Toy fields:** Use only as **sanity**, not PnL. Example: `KXBTCY-27JAN0100` often shows `sum_yes_ask_frac` near or above 1.0 while mids sum **UNDER** — asks do not automatically replicate “free money.”

---

## 4. Operator next steps

1. **Default:** no further demo **`--variant 001b`** runs unless you reopen the science question.
2. **Optional:** a bounded **`--live`** batch needs its **own** JSONL path + provenance (Option B in [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md); defaults under `datasets/H-KALS-001b-live/`); do not mix with demo lines.
3. **Do not** allocate capital from this document.
