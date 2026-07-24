# 06 — Pre-registered design: H-PERP-003 (hedged carry, Phase 3 unblock)

> **Relationship:** **H-PERP-001** is [INCONCLUSIVE](07_backtest_results_H-PERP-001.md) on Phase 3 depth; **H-PERP-002** in [04_hypothesis_library.md](04_hypothesis_library.md) is a **different** claim (rate decile mean-reversion). **H-PERP-003** is the **successor funding thesis** with an explicit **delta-hedged** PnL definition and a **data contract** suitable for US-side research (OKX-first).  
> **Rule:** No fee, window, or formula changes after this file is used to score results.

---

## 1. Data contract (must pass before Phase 4 code is gate-eligible)

| Clause | Requirement |
|--------|-------------|
| D1 | **≥ 365 calendar days** of **8h** (or venue-native) funding intervals for **one** primary perpetual, frozen below. |
| D2 | **Same venue** for funding, perp mark, and spot/index used in §3 (no cross-venue splicing without a new `06_*`). |
| D3 | **Provenance file** in [datasets/H-PERP-003/](datasets/H-PERP-003/) listing pull date, script name, API host, and instrument tickers. |
| D4 | If **any** calendar week has **>50%** missing intervals after merge → **INCONCLUSIVE_DATA** for that pull (fix upstream, do not tune gates). |

**Primary pull (frozen):** OKX **BTC-USDT-SWAP** (swap) public funding + mark price history + **BTC-USDT** spot (or unified index) sampled at **funding boundary timestamps** (aligned within **±60s**; intervals with larger skew are **dropped** from PnL, not interpolated).

**Implementation note (instrument detail, not a gate change):** OKX’s `history-mark-price-candles` endpoint **does not accept** an `8H` bar size. The reference pull uses **1H** mark and spot **closes** on the unique 1h candle whose half-open interval `[ts, ts+1h)` contains each `fundingTime`, with skew-to-boundary **≤ 60s**. Consecutive funding rows (~8h apart) supply `F_i, F_{i+1}` and `S_i, S_{i+1}` for §3 log returns. Public candle endpoints return up to **300** rows per request; older bars are paged with **`after=<oldest_open_ts_in_page − 1>`** (newest-first responses). In testing, **`before=`** did not walk additional history reliably on these routes.

**Rationale:** OKX was already used as a substitute source in the H-PERP-001 cycle when Bybit/Binance were geo-blocked; keeping one venue reduces basis confounding from cross-exchange stitching.

---

## 2. Instruments (frozen)

| Leg | Symbol / series | Role |
|-----|-----------------|------|
| Perpetual | OKX `BTC-USDT-SWAP` | Funding + perp mark `F` |
| Spot | OKx `BTC-USDT` spot **last** or **mark** as documented in provenance | Spot `S` for hedge return |

---

## 3. Hedged bundle PnL (frozen, discrete)

**Notional:** `V = 100` USDT (scale only; same spirit as H-PERP-001).

**Position concept:** **Short 1 unit** perp exposure on BTC notional `V`, **long** spot BTC of notional `V` at each interval boundary so **approximate delta neutrality** in the continuous-time limit; here we use **discrete** 8h windows between consecutive funding events `t_i → t_{i+1}`.

Let `r_{F,i} = ln(F_{i+1}/F_i)`, `r_{S,i} = ln(S_{i+1}/S_i)` using marks/lasts at the timestamps used in D4 alignment.

Let `b_i` be the **USDT funding payment per unit notional to a short** at event `i+1` as published by OKX (sign: positive if short **receives** — **verify once** against OKX docs at pull time and **freeze** the mapping in `provenance.txt`; if ambiguous, **INCONCLUSIVE_DATA**).

**Per-interval PnL (research dollars, not live execution):**

```
pnl_i = V * b_i + V * (r_{S,i} - r_{F,i})
```

Interpretation: **funding** to the short plus **basis / convergence** of spot vs perp over the window (hedge slippage and discrete rebalance error are **not** modeled beyond this difference of logs).

**Fees (frozen, conservative):** Same round-trip idea as H-PERP-001 §3: deduct **0.11% × V** at bundle open and again at bundle close **once each** across the full backtest window edges only (not per 8h), unless you later fork a new id with explicit per-interval fee — **here frozen as two one-time hits** on `V` at `t_0` and `t_T` to avoid double-counting 8h churn.

---

## 4. Walk-forward / OOS (frozen)

Mirror [06_backtest_design_H-PERP-001.md](06_backtest_design_H-PERP-001.md) §4: **K = 4** contiguous time segments by timestamp; **OOS = segments #2–#4**.

---

## 5. Evidence gates (same numeric thresholds as H-PERP-001 §5)

| # | Metric | Gate |
|---|--------|------|
| G1 | OOS Sharpe (annualized) | **≥ 1.5** |
| G2 | OOS profit factor on `pnl_i` | **≥ 1.4** |
| G3 | 2× fee stress on the **two** window-edge fee hits | Cumulative OOS P&L **> 0** |
| G4 | **≥ 3 of 3** OOS segments have **mean pnl_i > 0** | yes |
| G5 | No single interval **> 25%** of cumulative OOS profit | yes |
| G6 | Placebo (random sign-flip): **≤ 5%** of sign-flip shuffles produce cumulative OOS sum ≥ actual | yes |

**Verdict:** **PASS** only if **all** gates hold **and** Phase 3 data contract (§1) is met.

### G6 amendment (pre-registered 2026-05-04, no gate-eligible results yet)

The original wording inherited from H-PERP-001 §5 used "fraction of within-segment shuffles beating cumulative OOS sum". Both the cumulative sum and the annualized Sharpe (mean / std) are **multiset invariants** under any permutation of the per-interval PnLs — within-segment OR across-segment — so any permutation-of-the-multiset placebo is degenerate (frac_beats ≡ 1.0 regardless of edge). A first attempt at fixing this with Sharpe-of-shuffle was also degenerate for the same reason.

The replacement is a **random sign-flip placebo**, which destroys the directional edge while keeping per-interval magnitudes:

```
for trial in range(trials):
    signs = [rng.choice([-1, +1]) for _ in oos_pnls]
    shuf_sum = sum(s * x for s, x in zip(signs, oos_pnls))
    if shuf_sum >= cum_oos_gross:
        beat += 1
g6_pass = (beat / trials) <= 0.05
```

Under the null of no edge, each `shuf_sum` is a sum of independent symmetric ±|x|, so by CLT it is approximately Normal with mean 0 and variance Σx². The threshold (≤ 5% beats) is unchanged from prior versions. `trials = 500`, `seed = 42` — frozen here.

This amendment was made **before** any new gate run. The 94d / 103d exploratory metrics in [07](07_backtest_results_H-PERP-003.md) predate it and are explicitly flagged as exploratory (not gate-eligible due to D1 failure regardless of placebo definition).

---

## 6. Stop rule

Any gate fail → **FAIL** for H-PERP-003. No retuning `V`, fee placement, or segment count in this document.

---

## 7. Implementation (after data exists)

| Piece | Location (planned) |
|-------|----------------------|
| Dataset + provenance | [datasets/H-PERP-003/](datasets/H-PERP-003/) |
| Backtest script | [backtests/H-PERP-003.py](backtests/H-PERP-003.py) (to be written) |
| Results | [07_backtest_results_H-PERP-003.md](07_backtest_results_H-PERP-003.md) |

---

## 8. Live capital

**Blocked** until Phase 4 PASS **and** fundability artifacts exist — unchanged global rule.
