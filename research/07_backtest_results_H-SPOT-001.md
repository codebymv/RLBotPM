# 07 — Backtest results: H-SPOT-001

**Verdict: FAIL** (pre-registered evidence gate — **not all gates pass**)

**Artifacts:** [backtests/H-SPOT-001_metrics.json](backtests/H-SPOT-001_metrics.json), [backtests/H-SPOT-001.py](backtests/H-SPOT-001.py)

> **Re-run 2026-04-20:** Coinbase daily history extended to **950 bars (~31 months)** via `start`/`end` pagination ([`fetch_candles.py`](datasets/H-SPOT-001/fetch_candles.py)). Metrics below reflect the extended CSV.

---

## 1. OOS configuration

- **Bars:** 950 closes; **OOS daily returns:** 830 (post `SLOW=120` warm-up, three equal time blocks).
- **Rule & fees:** unchanged vs [06_backtest_design_H-SPOT-001.md](06_backtest_design_H-SPOT-001.md).

---

## 2. Headline metrics (OOS) — extended sample

| Metric | Value | Gate | Pass? |
|--------|-------|------|-------|
| Sharpe (annualized) | **0.85** | ≥ 1.5 | **NO** |
| Profit factor | **1.18** | ≥ 1.4 | **NO** |
| Cumulative `r_net` | **+67.4%** (fraction of notional) | — | — |
| 2× fee cumulative | **+54.8%** | > 0 | **YES** |
| Segment means (>0) | **2 / 3** | 3 / 3 | **NO** |
| G5 concentration | pass | — | **YES** |
| Placebo (shuffle-in-segment) | **100%** ≥ actual | ≤ 5% | **NO** |

### Interpretation

With ~2.5 years of data, the crude dual-SMA rule is **no longer catastrophically negative** (it was **−11%** OOS on the short 350-day window before pagination). It now shows **positive** cumulative return before extreme gates — but it **still fails** Sharpe, profit-factor, **all-three-segments-positive**, and the placebo gate.

The **placebo** issue remains: with **positive** cumulative OOS, random within-segment shuffles often beat a **specific path** under a naive `sum(shuf) >= sum(actual)` rule — the test needs redesign (e.g. compare Sharpe of shuffles vs actual). Documented as methodology debt; **does not** overturn FAIL because G1/G2/G4 already fail.

---

## 3. Phase 4 gate

**Aggregate: FAIL** — do not deploy live capital; paper logging may continue for infrastructure only.

---

## 4. Next step

Pick a **new** hypothesis in [04_hypothesis_library.md](04_hypothesis_library.md) (or add H-SPOT-002 with different signal / frequency). **Do not** tune `F`/`S` on this same rule to chase gates.
