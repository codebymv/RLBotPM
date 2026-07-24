# 11 — Live vs demo PARKED G3: H-KALS-001b (Option B)

> **Not a Phase-4/5 PASS.** No capital. No retune of B0–B4 / 1.05 / 0.95.  
> **H-PERP-003** remains **FAIL / DO NOT PROMOTE** (out of scope here).  
> **Demo G3** stays **PARKED** ([07_backtest_results_H-KALS-001b.md](07_backtest_results_H-KALS-001b.md)).

**Inputs**

| Stream | Path | Scans used |
|--------|------|------------|
| Demo PARKED G3 | [datasets/H-KALS-001b/scan_events.jsonl](datasets/H-KALS-001b/scan_events.jsonl) | 10 (2026-04-25 → 04-26) |
| Live Option B | [datasets/H-KALS-001b-live/scan_events.jsonl](datasets/H-KALS-001b-live/scan_events.jsonl) | 2 (2026-07-24) |
| Comparator | [scanners/compare_kals_001b_demo_live.py](scanners/compare_kals_001b_demo_live.py) | offline |

---

## 1. Snapshot (as of 2026-07-24)

| Metric | Demo G3 | Live |
|--------|---------|------|
| Successful scans | 10 | 2 |
| `markets_fetched` | ~77k–79k | ~725k–731k |
| Candidates / scan | 6–7 | 32–33 |
| Violations / scan | 6–7 (every scan) | 28–29 (every scan) |
| Violation rate (viol/cand) | **1.00** | **~0.88** |
| Kind mix (batch) | UNDER 61 / OVER 6 | UNDER 49 / OVER 8 |
| Unique violation `event_ticker` | 7 | 16 |
| Exact event overlap | — | **0** |
| Family (`KX…` prefix) overlap | — | **0** |
| Mean legs / violation | ~5.6 | ~19.2 |

Live purity (separate check): 2 live / 0 demo — green ([PROVENANCE.md](datasets/H-KALS-001b-live/PROVENANCE.md)).  
Live G3 freeze helper: **2/10**, `existence_gate=PENDING`, `addendum_ready=false`, `capital_pass=false`.

---

## 2. Overlap vs differences

### Identity (tickers / families)

- **Zero exact `event_ticker` overlap** and **zero series-family overlap** between demo G3 and live.
- Expected given calendar gap (~3 months) and listing drift: demo was weather / politics / `KXBTCY` year strip; live is dominated by crypto ladders (`KXXRP`, `KXDOGEY`, `KXBTC`, `KXNEARY`, …) plus a few political/index strips.
- **Within live:** 15/16 violation events stick across both production scans (only `KXNASDAQ100-26JUL24H1600` appears on scan 2 only) → short-horizon stickiness is high.

### Structure (rule set B phenomenology)

| Pattern | Demo | Live | Match? |
|---------|------|------|--------|
| Violations on every successful scan | yes | yes | yes |
| UNDER-dominated | ~91% | ~86% | yes |
| Contiguous `between` ladders with Σp outside [0.95, 1.05] | yes | yes | yes |
| Candidate count scale | small (6–7) | larger (32–33) | scale differs |
| Ladder depth | short | much longer | scale differs |
| Toy `toy_naive_long_ask_vs_par` often ≥ 0 on UNDER rows | common | common | yes — asks usually kill “free money” |

Live is **not** a copy of the demo event set; it **is** a replication of the **observation**: material UNDER/OVER Σp on contiguous ladders under the frozen rule set, on production host, with larger listing coverage.

---

## 3. Verdict (honest)

| Question | Answer |
|----------|--------|
| Does live show the same *phenomenon* as demo G3? | **Yes** — violations every scan, UNDER-heavy. |
| Does live share the same *names*? | **No** — 0 event / 0 family overlap. |
| Is G3-style 10-scan freeze ready on live? | **No** — only 2 live scans (`06` G3 freeze was 10). |
| Could the zero-violation FAIL branch still fire on live? | **Unlikely given current data** (already 57 violation rows); formal G3 still wants 10 successful live scans if you want parallel bookkeeping. |
| Tradeable / promote / allocate? | **No.** Toy ask fields still argue against naive long-ask edge. |
| Comparator label | `STRUCTURAL_REPLICATION_PROMISING` |

**Bottom line:** Live replication looks **promising enough to keep Option B open** (accumulate more live batches or stop after a short G3-style 10-scan freeze). It does **not** reopen demo PARKED science, does **not** authorize capital, and does **not** touch H-PERP-003.

---

## 4. Offline checklist (before another ~25 min live scan)

Use when deciding whether analysis is blocked without more data:

- [ ] Purity: `python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --audit-only` → ok, live-only. Comparator also audits both paths and exits **3** on mix (forensics: `--allow-impure`).
- [ ] Compare: `python research/scanners/compare_kals_001b_demo_live.py` → label still `STRUCTURAL_*` or document change; `purity.ok=true`.
- [ ] Freeze bookkeeping: report `live_g3_freeze` shows `successful_live_scans` / `target_scans=10`, `existence_gate=PENDING|FAIL|VIOLATIONS_OBSERVED`, `capital_pass=false`. Ready for a short live G3 addendum only when `addendum_ready=true` (never invent PASS).
- [ ] Sticky set: live `stable_events.intersection_size` not collapsing toward 0 across new appends.
- [ ] Kind mix: UNDER still majority; if live flips to ~all OVER or zero violations for several scans, pause and write a note (do **not** retune thresholds).
- [ ] Toy economics sanity: majority of UNDER rows still have `toy_naive_long_ask_vs_par ≥ 0` (asks not free). If that breaks systematically, note it — still not a retune of Σp gates.
- [ ] Count: if pursuing parallel G3 bookkeeping, stop at **10 successful live scans** then write a short live G3 addendum; do not invent a new PASS.

If all boxes hold and you only care about “does production still show the bug-shaped Σp?” — **more scans are optional**, not blocking.

---

## 5. Operator commands

```bash
# Offline compare + purity gate + live G3 freeze status (no network)
python research/scanners/compare_kals_001b_demo_live.py
python research/scanners/compare_kals_001b_demo_live.py --json

# Optional further live append (network + credentials; ~long pagination)
python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --once --require-live-credentials
python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --audit-only
```

Fixtures: [../tests/test_kals_001b_demo_live_compare.py](../tests/test_kals_001b_demo_live_compare.py).
