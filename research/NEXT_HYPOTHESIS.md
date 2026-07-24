# NEXT_HYPOTHESIS — steering order (best judgement)

> One active research track at a time. **No new live capital** until a hypothesis clears Phase 4 + Phase 5 under the rules in `08` / `10`.

## Closed

| ID | Verdict | Notes |
|----|---------|--------|
| H-SPOT-001 | Phase 4 **FAIL** | Dual SMA on Coinbase daily — see `07_backtest_results_H-SPOT-001.md`. Paper plumbing logged in `09_paper_results_H-SPOT-001.md`. **Operational:** keep `RESEARCH_LOG_H_SPOT` unset/false; do not run `kalshi paper-trade` solely to append that JSONL (see `.env.template`). |
| H-PERP-001 | **INCONCLUSIVE / blocked** | Funding depth + US geo on some APIs — see `07_backtest_results_H-PERP-001.md`. Revisit only with 12+ months of clean funding **and** a hedged PnL definition. |
| **H-KALS-001** | **PARKED / narrow FAIL** | Rule set A dominated by **non-MECE** grouping; scan infra validated. [07_backtest_results_H-KALS-001.md](07_backtest_results_H-KALS-001.md). |
| **H-KALS-001b** | **PARKED** | **10** demo scans, **G3** complete — violations **every** scan (67 rows total); zero-violation FAIL branch **not** triggered. Not a tradeable thesis without new execution economics. Artifacts: [06_backtest_design_H-KALS-001b.md](06_backtest_design_H-KALS-001b.md) · [07_backtest_results_H-KALS-001b.md](07_backtest_results_H-KALS-001b.md) · [datasets/H-KALS-001b/scan_events.jsonl](datasets/H-KALS-001b/scan_events.jsonl). **Do not** routine-scan demo unless reopening science or running an explicit **`--live`** study with separate provenance. |
| **H-SPOT-002** | Phase 4 **FAIL** | BTC–ETH log-ratio z-band on Coinbase daily — see [07_backtest_results_H-SPOT-002.md](07_backtest_results_H-SPOT-002.md). Sharpe **≈ −1.33**; G1/G2/G3/G4/G6 fail. **Do not** retune `W` / bands / fees; **no paper**. |

## Active

| ID | Status | Artifacts |
|----|--------|-----------|
| **H-PERP-003** | **Phase 5 FAIL (drift) — DO NOT PROMOTE** | [06](06_backtest_design_H-PERP-003.md) · [05](05_data_quality_H-PERP-003.md) · [datasets/H-PERP-003/](datasets/H-PERP-003/) · [07](07_backtest_results_H-PERP-003.md) · [08](08_paper_protocol_H-PERP-003.md) · [09](09_paper_results_H-PERP-003.md) · [10](10_fundability_review_H-PERP-003.md) · [backtests/H-PERP-003.py](backtests/H-PERP-003.py) |

**Claim:** Short perpetual **plus** spot-side return offset (`pnl_i = V·b_i + V·(r_S − r_F)` per §3) on **OKX**, **≥365d** funding-aligned history — successor to blocked depth on H-PERP-001; **not** the same hypothesis as library **H-PERP-002** (rate decile event study).

**Phase 5 close-out (2026-07-23):** 79d / 237 snapshots; tracking formula
PASS; Sharpe drift FAIL (**~4.07** vs 7.97 on Phase 4 `sharpe_8h` scale;
earlier 1.90 was a verifier `sqrt(n)` bug — corrected, still FAIL). Fleet
dry-run path is the H-PERP-003 paper logger (`shared/config/fleet.yaml` →
`h_perp_003.enabled=true`).
**Next engineering step:** Option A (H-SPOT-002) is closed FAIL; Option B below, or a new library id — do not retune H-PERP-003.

---

## Next (if H-PERP-003 stalls)

### Option A — **H-SPOT-002** — **CLOSED (Phase 4 FAIL)**

Formal close-out 2026-07-23: [05](05_data_quality_H-SPOT-002.md) · [06](06_backtest_design_H-SPOT-002.md) · [07](07_backtest_results_H-SPOT-002.md) · [metrics](backtests/H-SPOT-002_metrics.json). No paper / no retune.

### Option B — **Kalshi `--live` Σp replication** (optional) — **NEXT candidate**

Same rule set B as H-KALS-001b; **no new hypothesis id**. Demo and live logs must not mix.

| Piece | Location |
|-------|----------|
| Scanner | [scanners/kals_001_probability_sum_scan.py](scanners/kals_001_probability_sum_scan.py) — `--variant 001b --live` |
| Live JSONL | [datasets/H-KALS-001b-live/scan_events.jsonl](datasets/H-KALS-001b-live/scan_events.jsonl) (default; created on first append) |
| Provenance | [datasets/H-KALS-001b-live/PROVENANCE.md](datasets/H-KALS-001b-live/PROVENANCE.md) |
| Offline guards | [../tests/test_kals_001_live_sigma_p.py](../tests/test_kals_001_live_sigma_p.py) |
| Demo vs live note | [11_live_vs_demo_G3_H-KALS-001b.md](11_live_vs_demo_G3_H-KALS-001b.md) · [scanners/compare_kals_001b_demo_live.py](scanners/compare_kals_001b_demo_live.py) |
| Offline freeze helper | Same comparator — `live_g3_freeze` block + purity gate (exit 3 if mixed modes) |

**Operator commands (from RLBotPM root):**

```bash
python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --once --require-live-credentials
python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --audit-only
python research/scanners/compare_kals_001b_demo_live.py
```

**Status (2026-07-24):** First production appends landed (2 live scans; purity green). Offline compare vs demo PARKED G3 → **`STRUCTURAL_REPLICATION_PROMISING`** (violations every scan, UNDER-heavy; **0** exact event/family overlap). Comparator now gates on mode purity and reports live G3 freeze bookkeeping (`2/10`, `existence_gate=PENDING`, never invents capital PASS). **Not** a capital PASS; demo G3 stays PARKED. Optional next: more live batches toward a 10-scan G3-style freeze (credentials), or stop — offline analysis is not blocked.

---

## Operating rule

1. Write / freeze **`06_backtest_design_<id>.md`**.
2. Run offline evaluation → **`07_*`** verdict.
3. Only on **PASS**: add paper logger / live hook.
4. Update this file: move row from **Next** to **Closed** with one-line outcome.

### Pre-registration enforcement (architecture-audit-03 cross-cutting)

Every change to a `06_backtest_design_<id>.md` file MUST be timestamped and
land in the repo **before** the next run of the corresponding backtest. Reorder
violation (results first → spec amended to fit) is the single most expensive
research bug we can ship. Concretely:

- Spec edits go in their own commit, separate from any `07_*` results commit.
- The commit message includes the phrase `pre-register:` and lists the gate(s)
  changed.
- The commit lands strictly before the timestamp on the next `07_*` write or
  metrics JSON refresh. CI / reviewer should reject a PR that bundles them in
  one commit.
- The G6 amendment in [06_backtest_design_H-PERP-003.md](06_backtest_design_H-PERP-003.md)
  (2026-05-04, random sign-flip) is the canonical example of how to amend
  without introducing the hindsight failure mode.
