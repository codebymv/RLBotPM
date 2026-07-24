# Phase 5b — Wall-clock handoff

The plan requires **≥ 50 paper trades / snapshots OR ≥ 30 calendar days** of paper-mode operation.

## Start signal

| Field | Value |
|-------|--------|
| Recommended start | When you begin continuous `paper-trade` with `RESEARCH_LOG_H_SPOT=true` |
| First log line | Check `bot/logs/paper_research_H-SPOT-001.jsonl` |
| Offline baseline (2026-04-20) | H-SPOT-001 backtest **re-run** on **950** daily bars — see [07_backtest_results_H-SPOT-001.md](07_backtest_results_H-SPOT-001.md) / `backtests/H-SPOT-001_metrics.json` |

## Completion checklist

- [x] ≥ 50 lines in `paper_research_H-SPOT-001.jsonl` (**done** — 524 lines as of 2026-04-23)
- [ ] ≥ 30 days elapsed since first snapshot (**waived for plumbing-only** per [08 amendment](08_paper_protocol_H-SPOT-001.md) because Phase 4 already **FAIL**)
- [x] Summarize in [09_paper_results_H-SPOT-001.md](09_paper_results_H-SPOT-001.md)
- [ ] Update [10_fundability_review_H-SPOT-001.md](10_fundability_review_H-SPOT-001.md) items 9–11 if you want paper rows to read **COMPLETE** for plumbing (optional; **live capital still blocked**)

## Note

Kalshi paper scans (`paper_trades.jsonl`) and research snapshots are **orthogonal** — you can run paper mode primarily for research logging even if Kalshi opens zero positions.
