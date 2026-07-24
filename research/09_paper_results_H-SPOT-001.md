# 09 — Paper results: H-SPOT-001

**Status: PLUMBING COMPLETE (hypothesis remains Phase 4 FAIL)**

This paper run was **never** meant to overturn the offline verdict in [07_backtest_results_H-SPOT-001.md](07_backtest_results_H-SPOT-001.md). It only validates **Coinbase fetch + SMA recompute + JSONL logging** alongside Kalshi paper scans.

---

## 1. Log inventory (as of file on disk)

| Field | Value |
|-------|--------|
| Artifact | [bot/logs/paper_research_H-SPOT-001.jsonl](../bot/logs/paper_research_H-SPOT-001.jsonl) |
| Lines (JSON events) | **524** |
| First `timestamp` | `2026-04-20T03:43:13.220978+00:00` |
| Last `timestamp` | `2026-04-23T19:30:49.711935+00:00` |
| Last `kalshi_scan` | **522** |
| `pos_raw == 1` count | **0** (grep: no `"pos_raw": 1` in file) |

**Interpretation:** `SMA20 > SMA120` never turned true on the live candle window the logger uses, so the **long** branch of H-SPOT-001 was **not** exercised in paper (only “flat” observations).

---

## 2. Protocol compliance

| Gate (see [08](08_paper_protocol_H-SPOT-001.md)) | Result |
|--------------------------------------------------|--------|
| ≥ 50 snapshots | **YES** (524) |
| ≥ 30 calendar days | **NO** (~3.7 days wall span) |
| **Amended rule (2026-04-23):** For hypotheses **already FAIL at Phase 4**, Phase 5 is **plumbing-only**. Completion = **≥50 successful snapshots** + no sustained API outage (see §3 amendment in `08`). | **MET** |

---

## 3. Tracking vs backtest

| Metric | Offline backtest (extended CSV) | Paper | Note |
|--------|----------------------------------|-------|------|
| Verdict | **FAIL** ([07](07_backtest_results_H-SPOT-001.md)) | n/a | Paper does not re-score Sharpe here. |
| `pos_raw` distribution | Mixed in history | **Always 0** in log window | Different question: logger saw a **narrow** live window. |

---

## 4. Recommended operator action

1. **Stop** burning cycles: unset `RESEARCH_LOG_H_SPOT` and/or stop the `kalshi paper-trade` process unless you are actively debugging Kalshi paper itself.
2. **Do not** attach more capital or hope to this SMA rule — follow [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md).

---

## Phase 5 gate (strict vs plumbing)

- **Strict fundability path** (live dollars): still **blocked** — Phase 4 failed; paper does not repair that.
- **Plumbing path** (research infra): **COMPLETE** per amendment in `08`.
