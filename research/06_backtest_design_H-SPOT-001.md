# 06 — Pre-Registered Backtest Design: H-SPOT-001

> **Instrument:** Coinbase `BTC-USD` **daily** closes from `btcusd_daily_coinbase.csv`.  
> **Hypothesis:** Dual moving-average **trend filter** (slow trend + fast confirmation) produces positive **after-fee** daily returns out-of-sample vs a shuffle placebo, under walk-forward segmentation.

---

## 1. Signal & position (frozen)

Let `C[t]` be the **close** on calendar day `t` (UTC epoch from file).

| Parameter | Value |
|-----------|--------|
| Slow window | `S = 120` trading days |
| Fast window | `F = 20` trading days |
| Warm-up | First `S` bars **excluded entirely** from any P&L (no trades until `SMA120` exists). |

Define:

- `SMAn[t] = mean(C[t-n+1 : t])` inclusive (simple moving average).
- **Raw position** `pos_raw[t] = 1` iff `C[t] > SMA120[t]` **and** `SMA20[t] > SMA120[t]`, else `0`.
- **Executed position** `pos[t] = pos_raw[t]` with **one-day lag**: trade at **next day’s open** is approximated by **same close execution** (acknowledged simplification) → we use **`pos[t] = pos_raw[t-1]`** applied to return from `t-1` to `t` (standard causal lag).

**Daily strategy return (fraction):**

```
r[t] = pos[t-1] * (C[t] / C[t-1] - 1)
```

(`pos[t-1]` is known at close `t-1`; earns `t-1→t` return — still slightly optimistic vs open execution.)

---

## 2. Fee model (frozen, conservative retail)

Coinbase Advanced retail taker ≈ **0.60%** per side at sub–$10k 30d volume ([Coinbase Help — Advanced Trade fees](https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/advanced-trade-fees)).

Whenever `pos[t] != pos[t-1]`, charge **one-sided taker** on notional for the **legs that changed**:

```
fee[t] = 0.006 * abs(pos[t] - pos[t-1])
```

(i.e., **0.6%** of notional per unit change; max 0.006 when flipping 0↔1).

Net daily return: `r_net[t] = r[t] - fee[t]`.

---

## 3. Walk-forward / OOS (frozen)

After warm-up, split `[SLOW, n)` into **three equal-length contiguous index blocks** (not four). **All three** are OOS — there is **no** “drop first 25%” burn-in here, because a dry run showed a 4-segment “drop first quartile” design can **accidentally exclude every day with a non-zero position** on this BTC sample (signal concentrated in month 1 post-warm-up), which would trivialize the test.

All metrics are computed on the **concatenation** of the three OOS blocks.

---

## 4. Evidence gate (same thresholds as plan)

| ID | Metric | Gate |
|----|--------|------|
| G1 | OOS Sharpe (annualized `sqrt(365) * mean/std`) | ≥ 1.5 |
| G2 | Profit factor on **daily** `r_net` treated as trades | ≥ 1.4 |
| G3 | 2× fee stress | Double `fee[t]`; cumulative OOS P&L still > 0 |
| G4 | ≥3 of 3 OOS segments have **mean `r_net` > 0** | yes |
| G5 | No single day’s `r_net` contributes >25% of cumulative OOS profit | yes |
| G6 | Placebo: ≤5% of 500 shuffles of `r_net` within each OOS segment (independently permuted) beat actual cumulative OOS sum | yes |

**Verdict:** PASS only if **all** gates hold.

---

## 5. Stop rule

If any gate fails → **FAIL**. No retuning `F`, `S`, fee, or lag in this document.

---

## 6. Implementation

[backtests/H-SPOT-001.py](backtests/H-SPOT-001.py)
