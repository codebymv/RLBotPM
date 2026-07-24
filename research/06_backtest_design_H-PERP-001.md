# 06 — Pre-Registered Backtest Design: H-PERP-001

> **Rule:** No parameter changes after this file is committed to the research cycle.  
> **Hypothesis:** Short linear BTCUSDT perpetual exposes the trader to **positive funding payments when `fundingRate > 0`** (longs pay shorts on Bybit linear).  
> **v1 scope:** **Funding leg only** — we approximate P&L as `Σ (fundingRate_i × V)` where `V` is a constant **$100 USDT** notional short, ignoring **delta P&L from BTC moves**. This is a deliberate **upper bound on how easy “carry” looks** before hedging; if v1 fails the evidence gate, we do **not** rescue with extra legs in the same pre-registered run.

---

## 1. Instruments & data

| Field | Value |
|-------|--------|
| Perpetual | `BTCUSDT` **linear** (`category=linear`) |
| Funding source | Bybit public `GET /v5/market/funding/history` ([docs](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate)) |
| Spot reference (optional v1.1) | Coinbase `BTC-USD` daily close for narrative cross-check — **not used in v1 P&L** |

---

## 2. Sign convention (frozen)

Per Bybit public documentation: **positive `fundingRate` ⇒ long positions pay short positions.**  
Therefore a **short** earns `+fundingRate × position_value` each funding event **when `fundingRate > 0`**.  
When `fundingRate < 0`, the short **pays** the absolute amount.

**v1 position:** Always **short** `V = 100` USDT notional (conceptually).  
**Per-interval P&L (USDT):**  
`pnl_i = - (fundingRate_i × V)` if shorts **receive** when rate>0…  

Wait: if long pays short, short **receives** `+rate * V` for positive rate.  
Payment **to** short = `+ rate * V`.  
We store `pnl_i = rate_i * V`.

When rate < 0, shorts pay longs: `pnl_i = rate_i * V` still works algebraically (negative rate → negative pnl for short).

So: **`pnl_i = rate_i * V`** always for a **short** under Bybit sign convention.

---

## 3. Fee & slippage model (frozen)

| Item | Value | Citation |
|------|--------|----------|
| Open + close taker fee | **0.11% round-trip** of notional (0.055% × 2) | Approximate retail linear taker; verify live at [Bybit fees](https://www.bybit.com/en-US/help-center/article/Trading-Fee-Structure) |
| Slippage | **0 bps** in v1 (funding is scheduled; not a market order simulation) | Conservative **against** us on execution realism — acknowledged limitation |
| Fee application | Deduct **once** at `t0`: `−0.0011 × V`, and **once** at `t_end`: `−0.0011 × V` | Opening short + closing short |

---

## 4. Walk-forward / OOS protocol (frozen)

No parameter training. Split the merged funding timeline **chronologically** into **K = 4** equal-length contiguous segments (by timestamp).  

- **OOS evaluation:** Segments **#2, #3, #4** (drop the earliest segment as “burn-in” to reduce oldest-data quirks).  
- **Why:** Guarantees ≥3 disjoint OOS windows on long pulls (plan asks ≥3 non-overlapping windows).

**Trade definition:** Each funding timestamp is one **trade observation** for return math.

---

## 5. Metrics & evidence gate (plan thresholds)

Computed on **concatenated OOS segments** (segments 2–4):

| # | Metric | Gate |
|---|--------|------|
| G1 | OOS Sharpe (annualized) | **≥ 1.5** |
| G2 | OOS profit factor (sum wins / abs(sum losses) on per-interval pnl) | **≥ 1.4** |
| G3 | 2× fee stress | Still **> 0** cumulative P&L after doubling fee to **0.22% RT** |
| G4 | Window stability | **≥ 3 of 3** OOS segments have **mean pnl_i > 0** |
| G5 | Concentration | No single interval’s pnl contributes **> 25%** of total OOS profit |
| G6 | Placebo | **≤ 5%** of 500 permutations (circular shuffle of `rate_i` within each segment) achieve ≥ actual cumulative OOS pnl |

**Stop rule:** If **any** gate fails → verdict **FAIL** for v1. No retuning `V`, windows, or fee in this document.

---

## 6. Known limitations (must appear in results)

1. **Delta risk dominates real shorts** — v1 is **not** a tradable live strategy; it isolates the **funding leg only**.  
2. **US persons** may be unable to execute on Bybit — research validity is independent.  
3. **API gaps** / maintenance → missing intervals; we **do not** impute.

---

## 7. Implementation pointer

Code: [research/backtests/H-PERP-001.py](backtests/H-PERP-001.py)  
Data: [research/datasets/H-PERP-001/](datasets/H-PERP-001/)
