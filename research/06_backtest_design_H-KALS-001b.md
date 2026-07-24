# 06 — Pre-registered design: H-KALS-001b (observation + toy economics)

> **Successor to:** [06_backtest_design_H-KALS-001.md](06_backtest_design_H-KALS-001.md) (H-KALS-001 **parked** — rule set A; see [07_backtest_results_H-KALS-001.md](07_backtest_results_H-KALS-001.md)).  
> **Type:** Read-only **observation study** plus **toy** (non-executable) cost bounds. **No orders.**

---

## 1. Motivation (frozen)

H-KALS-001 grouped many non-MECE contracts under one `event_ticker`. Rule set **B** restricts to **contiguous `between` ladders** so Σp is taken only over bins that plausibly subdivide one interval on the strike axis.

---

## 2. Universe (frozen)

Same as H-KALS-001: **`status=open`**, **all categories**, full **cursor pagination** via [kalshi.py](../bot/src/data/sources/kalshi.py) `list_open_markets_all_pages`.

---

## 3. Grouping (frozen)

| Rule | Value |
|------|--------|
| Primary key | `event_ticker` (non-null only) |
| Sub-universe per event | Markets sharing the same `close_time` |

Within each `(event_ticker, close_time)` bucket, we build **candidate sets** as in §4 (each candidate set is one contiguous ladder).

---

## 4. Rule set B — “contiguous between ladder” (frozen)

### 4.1 Per-market inclusion

A market enters ladder construction **only if**:

| ID | Criterion |
|----|-----------|
| B0 | `strike_type` (lowercased) == **`between`** |
| B1 | `floor_strike` and `cap_strike` both **non-null** |
| B2 | Same `close_time` as siblings in the same evaluation bucket |
| B3 | **Spread:** `yes_ask - yes_bid ≤ 15` cents and `yes_ask > yes_bid` |

### 4.2 Ladder construction

Let `B` be all markets in one `(event_ticker, close_time)` group satisfying B0–B3.

1. If `|B| < 2`, **no** candidate set from this group.
2. Build an undirected graph on `B`: edge between `u` and `v` if  
   `abs(u.cap_strike - v.floor_strike) ≤ tol` **or** `abs(v.cap_strike - u.floor_strike) ≤ tol`,  
   where `tol = max(1e-4, 1e-6 * max(1, |u.cap_strike|, |v.floor_strike|, …))` (float noise).
3. Each **connected component** with `≥ 2` nodes is a **raw component**.
4. For each raw component, **sort** by `floor_strike` ascending. Accept as **one candidate ladder** only if, for every consecutive pair `(m_i, m_{i+1})`,  
   `abs(m_i.cap_strike - m_{i+1}.floor_strike) ≤ tol`.  
   Otherwise **discard** that component (not a single chain on the line).
5. **Evaluate each** accepted ladder separately (one event may yield **0, 1, or many** ladders).

### 4.3 Implied YES probability (frozen)

Same as H-KALS-001:

```
mid_c = (yes_bid + yes_ask) / 2
p = mid_c / 100.0
```

### 4.4 Violation definition (frozen)

For one ladder `L`, let `S = Σ p` over members of `L`.

| Label | Condition |
|-------|-----------|
| **OVER** | `S > 1.05` |
| **UNDER** | `S < 0.95` |

---

## 5. Toy economics (frozen, non-PnL)

These fields are **diagnostic only**; they do **not** assume fills, margin, or settlement rules.

| Quantity | Definition |
|----------|-------------|
| `sum_yes_ask_frac` | `Σ (yes_ask / 100)` over ladder members |
| `sum_yes_bid_frac` | `Σ (yes_bid / 100)` over ladder members |
| `toy_naive_long_ask_vs_par` | `1.0 - sum_yes_ask_frac` (positive ⇒ full YES basket at **asks** is below **1** notional before any fee) |

**Illustrative fee upper bound (not a gate):** assume **1¢ per contract per side** on each leg as a **toy** (not Kalshi’s real schedule). Let `n = |L|`. Rough **round-trip drag** order: `2 * n * 0.01` on the notional scale above — log alongside violations for intuition only.

**No PASS/FAIL** on toy fields in Phase 4 sense; they prevent mistaking Σp alone for edge.

---

## 6. Evidence gates (observation)

| ID | Gate | Meaning |
|----|------|---------|
| G0 | Pagination complete | Same as H-KALS-001 |
| G1 | API reliability | Same as H-KALS-001 |
| G2 | **Descriptive** | Log distribution of `violation_count` per scan and `events_ladders_evaluated` |
| G3 | Optional formal existence test | **Freeze after first 10 successful scans:** if **zero** OVER/UNDER violations across all 10 scans → **FAIL** on “material violations under rule set B” for that window; if **≥ 1** → continue logging (does not imply tradeable edge). |

Verdict labels for G3 only: `PASS`, `FAIL`, `PENDING`, `INCONCLUSIVE_DATA`.

---

## 7. Stop rule

Do not change B0–B4, `tol` policy, or 1.05 / 0.95 thresholds without a **new** hypothesis id and new `06_*`.

---

## 8. Implementation

| Piece | Location |
|-------|----------|
| Scanner | [scanners/kals_001_probability_sum_scan.py](scanners/kals_001_probability_sum_scan.py) — `--variant 001b` |
| Log (demo) | [datasets/H-KALS-001b/scan_events.jsonl](datasets/H-KALS-001b/scan_events.jsonl) |
| Log (live / Option B) | [datasets/H-KALS-001b-live/scan_events.jsonl](datasets/H-KALS-001b-live/scan_events.jsonl) — separate provenance; `--live` |
| Results | [07_backtest_results_H-KALS-001b.md](07_backtest_results_H-KALS-001b.md) |

**Amendment (2026-07-23, artifact paths only — no gate change):** document dedicated `--live` JSONL under `H-KALS-001b-live/` so production Σp replication cannot append into the demo log. B0–B3, thresholds, and G0–G3 unchanged.

---

## 9. No trading

No capital deployment authorization. Execution and fundability remain gated elsewhere.
