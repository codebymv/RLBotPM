# 06 — Pre-registered design: H-KALS-001 (observation study)

> **Type:** Live / snapshot **observation study** — not a PnL backtest. No hypothetical fills, no Kelly, no orders.  
> **Hypothesis (from [04_hypothesis_library.md](04_hypothesis_library.md)):** Across Kalshi markets that form a **single-event partition**, **YES mid-implied probabilities** sometimes violate probability algebra (`Σp` materially above 1 or below 1). Such violations should be **detectable** in read-only snapshots if they exist.

---

## 1. Universe (frozen)

| Rule | Value |
|------|--------|
| Market `status` | **`open`** only |
| Category filter | **None** — all categories returned by the paginated `/markets` listing for `status=open` |
| Pagination | **Mandatory full enumeration** per scan using cursor from the API (see [kalshi.py](../bot/src/data/sources/kalshi.py) `get_markets_page` / `list_open_markets_all_pages`). Partial first-page-only reads **invalidate** that scan for gate G0. |

**Rationale:** Restricting categories risks false negatives on the core question (“do violations exist on the venue at all?”). Breadth trades API volume for coverage.

---

## 2. Grouping key (frozen)

| Rule | Value |
|------|--------|
| Primary key | `event_ticker` on each market (`KalshiMarket.event_ticker`) |
| Null `event_ticker` | **Drop** the market from grouping (do not invent synthetic groups). |

---

## 3. Rule set A — “partition candidate” (frozen)

A set of open markets with the same non-null `event_ticker` is evaluated **only if** all of the following hold:

| ID | Criterion |
|----|-----------|
| P1 | **Count:** `n ≥ 2` markets in the group after other filters. |
| P2 | **Common close:** All members share the same `close_time` (exact equality on parsed `datetime`, including tz). |
| P3 | **Strike typing:** Every member has `strike_type ∈ {"greater", "less", "between"}`. |
| P4 | **Liquidity / spread:** For each member, let `spread_c = yes_ask - yes_bid` (cents, 0–100 scale). **Exclude** any member with `spread_c > 15` or with `yes_ask <= yes_bid`. If **no** members remain, **skip** the event for this scan (no violation test). |
| P5 | **No ambiguous cross-venue structure:** If any member lacks both `floor_strike` and `cap_strike` when `strike_type == "between"`, skip the **entire** event for this scan. (Guards obviously broken API rows; rare.) |

**Epistemic caveat (not a gate):** Rule set A is a **practical** filter for “likely sibling outcome buckets,” not a formal proof of mutual exclusivity. Violations under A are still **economically interesting** as mis-pricing signals subject to manual review. **False positives** (Σp > 1 because buckets were not truly MECE) are a known failure mode — document in `07` if suspected.

---

## 4. Implied YES probability (frozen)

For each included member after P4:

```
mid_c = (yes_bid + yes_ask) / 2    # cents, 0–100
p = mid_c / 100.0                  # unit interval
```

Use fields from `KalshiMarket` after adapter parsing (cents scale).

---

## 5. Violation definition (frozen)

For one event group in one scan, let `S = Σ p_i` over **included** members (post P4).

| Label | Condition |
|-------|-------------|
| **OVER** | `S > 1.05` |
| **UNDER** | `S < 0.95` |
| **OK** | neither |

Either OVER or UNDER counts as **one violation event** for that scan line.

---

## 6. Evidence gates (observation study)

| ID | Gate | Meaning |
|----|------|---------|
| G0 | **Pagination complete** | Every scan used full cursor walk until termination per adapter; log `markets_fetched` count. |
| G1 | **API reliability** | If HTTP / auth failures exceed **5%** of scheduled scan attempts in a rolling window, verdict **INCONCLUSIVE_DATA** (not FAIL). |
| G2 | **Existence (hypothesis-oriented)** | If after **≥ 56** successful full scans (target: ~14 calendar days at 4 scans/day) under rule set A we observe **≥ 1** violation event total → **PASS** on “violations are detectable.” |
| G3 | **Null / absence** | If G0–G1 OK and after **≥ 56** successful scans we observe **0** violations → **FAIL** on the existence claim for this venue window (does not prove impossibility forever). |

**Verdict labels:** `PASS`, `FAIL`, `INCONCLUSIVE_DATA` (G1), `PENDING` (insufficient scans).

---

## 7. Stop rule

Do not change thresholds (`1.05`, `0.95`, spread `15`, `n≥2`) or rule set A without a **new** hypothesis id and new `06_*` document.

---

## 8. Implementation references

| Piece | Location |
|-------|----------|
| Paginated listing | `KalshiAdapter.get_markets_page`, `list_open_markets_all_pages` (default **0.25s** inter-page delay to reduce HTTP 429) in [bot/src/data/sources/kalshi.py](../bot/src/data/sources/kalshi.py) |
| Scanner | [research/scanners/kals_001_probability_sum_scan.py](scanners/kals_001_probability_sum_scan.py) |
| Raw log | [research/datasets/H-KALS-001/scan_events.jsonl](datasets/H-KALS-001/scan_events.jsonl) (append-only) |
| Provenance | [research/datasets/H-KALS-001/provenance.txt](datasets/H-KALS-001/provenance.txt) |

---

## 9. No trading

This document **does not** authorize orders, sizing, or capital deployment. A separate execution and fee model would be required before any trade hypothesis.
