# Dataset provenance — H-KALS-001b-live (Option B)

**Purpose:** Separate append-only JSONL for Kalshi **production** (`--live`) Σp scans under rule set B. Must **not** mix with demo lines in [../H-KALS-001b/scan_events.jsonl](../H-KALS-001b/scan_events.jsonl).

**Design:** [../../06_backtest_design_H-KALS-001b.md](../../06_backtest_design_H-KALS-001b.md) (same rule set B; new venue host only).  
**Steering:** [../../NEXT_HYPOTHESIS.md](../../NEXT_HYPOTHESIS.md) Option B.

## How to append

```bash
# From RLBotPM root; requires network + preferably KALSHI_* credentials
python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --once
# Hard-fail if credentials look unset:
python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --once --require-live-credentials
# Offline purity audit (no network):
python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --audit-only
```

Default output: `scan_events.jsonl` in this directory.

## Recorded runs

| When (UTC) | Scans | Notes |
|------------|-------|--------|
| 2026-07-24T00:14:31Z | 1 | Live append: markets=724748, candidates=32, violations=28; `demo=false`, `api_mode=live`. |
| 2026-07-24T00:23:31Z | 1 | Live append (`--once --require-live-credentials`): markets=730960, candidates=33, violations=29; `demo=false`, `api_mode=live`. Purity audit: ok (2 live / 0 demo). |

## Gates reminder

Do not interpret empty / missing live JSONL as a science FAIL. Demo G3 in [../../07_backtest_results_H-KALS-001b.md](../../07_backtest_results_H-KALS-001b.md) stays **PARKED**; live replication is a **separate** provenance stream.

## Offline compare vs demo G3

After live appends exist, compare violation/candidate phenomenology (no retune, no promote).
The comparator audits demo/live mode purity and reports Option B **live G3 freeze**
bookkeeping (`successful_live_scans` / 10, `existence_gate`, never `capital_pass`).

```bash
python research/scanners/compare_kals_001b_demo_live.py
# exit 3 if JSONL modes are mixed; forensics only:
python research/scanners/compare_kals_001b_demo_live.py --allow-impure
```

Research note: [../../11_live_vs_demo_G3_H-KALS-001b.md](../../11_live_vs_demo_G3_H-KALS-001b.md).

**Freeze honesty:** With 2 successful live scans (2026-07-24), freeze status is
`PENDING` (8 remaining). Do not write a live G3 addendum until
`addendum_ready=true`. Demo PARKED G3 is unchanged.
