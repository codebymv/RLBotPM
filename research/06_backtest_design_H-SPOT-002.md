# 06 — Pre-Registered Backtest Design: H-SPOT-002

> **Instrument:** Coinbase `BTC-USD` + `ETH-USD` **daily** closes (aligned inner join).  
> **Hypothesis:** BTC–ETH **log-ratio mean reversion** after a **|z| > 2** excursion on a 30-day rolling z-score produces positive **after-fee** daily returns out-of-sample under walk-forward segmentation.  
> **Relationship:** New mechanism — **not** a retune of H-SPOT-001 dual-SMA windows. H-SPOT-001 Phase 4 **FAIL** remains closed ([07_backtest_results_H-SPOT-001.md](07_backtest_results_H-SPOT-001.md)).  
> **Rule:** No window, entry/exit band, fee, or lag changes after this file is used to score results.

**Pre-registered:** 2026-07-23 (before any H-SPOT-002 gate-eligible `07_*` or metrics JSON).

---

## 1. Data contract (must pass before Phase 4 is gate-eligible)

| Clause | Requirement |
|--------|-------------|
| D1 | **≥ 730 calendar days** (~24 months) of **aligned** daily closes for both BTC-USD and ETH-USD after inner join on candle timestamp. |
| D2 | **Same venue** (Coinbase Exchange public candles) for both legs — no cross-venue splicing. |
| D3 | Provenance file in [datasets/H-SPOT-002/](datasets/H-SPOT-002/) listing pull date, script, API host, products. |
| D4 | After align, if any calendar week has **>50%** missing trading days vs a contiguous daily grid → **INCONCLUSIVE_DATA** (do not tune gates). |

**Primary pull (frozen):** Coinbase Exchange REST
`GET /products/{BTC-USD|ETH-USD}/candles` granularity `86400`, paginated like H-SPOT-001.

---

## 2. Signal & position (frozen)

Let `B[t]`, `E[t]` be BTC and ETH **closes** on calendar day `t` (UTC epoch).

| Parameter | Value |
|-----------|--------|
| Spread | `s[t] = ln(B[t]) - ln(E[t])` |
| Z-window | `W = 30` trading days |
| Entry | `Z_ENTER = 2.0` |
| Exit band | `Z_EXIT = 0.5` |
| Warm-up | First `W` bars **excluded** from any P&L (no trades until z-score exists). |

Define rolling mean / population std over the inclusive window `[t-W+1, t]`:

```
μ[t] = mean(s[t-W+1 : t])
σ[t] = pstdev(s[t-W+1 : t])   # population; if σ < 1e-12 → no signal (flat)
z[t] = (s[t] - μ[t]) / σ[t]
```

**Raw position** `pos_raw[t] ∈ {-1, 0, +1}` (stateful band trade):

- Start flat (`0`).
- If flat and `z[t] > Z_ENTER` → `pos_raw[t] = -1` (**short the ratio**: short BTC / long ETH — expect ratio to fall).
- If flat and `z[t] < -Z_ENTER` → `pos_raw[t] = +1` (**long the ratio**: long BTC / short ETH).
**Exit rule (frozen):** leave a non-zero position when `|z[t]| < Z_EXIT` (return to flat).  
**Flip rule (frozen):** if already positioned and `z` crosses the **opposite** entry threshold, flip in one step (do not require flat first).

Explicit state transition:

```
if pos == 0:
    if z > +Z_ENTER: pos = -1
    elif z < -Z_ENTER: pos = +1
elif pos == +1:
    if z > +Z_ENTER: pos = -1
    elif abs(z) < Z_EXIT: pos = 0
elif pos == -1:
    if z < -Z_ENTER: pos = +1
    elif abs(z) < Z_EXIT: pos = 0
```

**Executed position** uses **one-day causal lag:** `pos[t] = pos_raw[t-1]` applied to the return from `t-1` → `t`.

**Daily strategy return (fraction of one-leg notional):**

```
r_ratio[t] = (B[t]/B[t-1] - 1) - (E[t]/E[t-1] - 1)
r[t]       = pos_raw[t-1] * r_ratio[t]
```

(`+1` earns BTC excess over ETH; `-1` earns the opposite.)

---

## 3. Fee model (frozen, conservative retail)

Coinbase Advanced retail taker ≈ **0.60%** per side ([same source as H-SPOT-001](https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/advanced-trade-fees)).

A unit change in ratio position touches **both** legs:

```
fee[t] = 2 * 0.006 * abs(pos_raw[t] - pos_raw[t-1])
```

Net daily return: `r_net[t] = r[t] - fee[t]`.

---

## 4. Walk-forward / OOS (frozen)

After warm-up index `W`, split `[W, n)` into **three equal-length contiguous** blocks. **All three** are OOS (same rationale as H-SPOT-001 — no burn-in quartile that can erase the only active regime).

All metrics are computed on the **concatenation** of the three OOS blocks.

---

## 5. Evidence gates

| ID | Metric | Gate |
|----|--------|------|
| G1 | OOS Sharpe (annualized `sqrt(365) * mean/std` on daily `r_net`) | ≥ 1.5 |
| G2 | Profit factor on daily `r_net` | ≥ 1.4 |
| G3 | 2× fee stress | Double `fee[t]`; cumulative OOS P&L still > 0 |
| G4 | ≥3 of 3 OOS segments have **mean `r_net` > 0** | yes |
| G5 | No single day’s `r_net` contributes >25% of cumulative OOS **gross profit** (sum of positive days) | yes |
| G6 | Placebo: **random sign-flip** of OOS `r_net`; ≤5% of 500 trials (`seed=42`) beat actual cumulative OOS sum | yes |

**G6 method (frozen):** random sign-flip — **not** within-segment permutation (multiset-invariant / degenerate for cum-sum; lesson from H-PERP-003 §5 amendment).

```
for trial in range(500):
    shuf_sum = sum(rng.choice([-1, +1]) * x for x in oos_r_net)
    if shuf_sum >= cum_oos: beat += 1
g6_pass = (beat / 500) <= 0.05
```

**Verdict:** PASS only if **all** gates hold **and** §1 data contract is met. Otherwise FAIL or `INCONCLUSIVE_DATA`.

---

## 6. Stop rule

If any gate fails → **FAIL**. No retuning `W`, `Z_ENTER`, `Z_EXIT`, fee, or lag in this document. Do **not** fall back to retuning H-SPOT-001 SMAs.

---

## 7. Implementation

| Piece | Location |
|-------|----------|
| Dataset + provenance | [datasets/H-SPOT-002/](datasets/H-SPOT-002/) |
| Fetch script | [datasets/H-SPOT-002/fetch_candles.py](datasets/H-SPOT-002/fetch_candles.py) |
| Backtest | [backtests/H-SPOT-002.py](backtests/H-SPOT-002.py) |
| Unit tests (synthetic) | [../tests/test_h_spot_002_signal.py](../tests/test_h_spot_002_signal.py) |
| Results | [07_backtest_results_H-SPOT-002.md](07_backtest_results_H-SPOT-002.md) — Phase 4 **FAIL** |
| Data quality | [05_data_quality_H-SPOT-002.md](05_data_quality_H-SPOT-002.md) |

---

## 8. Live capital / paper

**Blocked** until Phase 4 PASS. No paper logger until then (operating rule in [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md)).
