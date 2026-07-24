# 07 — Backtest results: H-PERP-001

**Verdict: INCONCLUSIVE (insufficient data)** — **not** a PASS or a clean FAIL against the full pre-registered Phase 4 gate because **Phase 3’s 12-month requirement was not met**.

---

## Status

| Check | Result |
|-------|--------|
| Phase 3 ≥ 12 months funding history | **NO** (~96 days OKX pull) |
| Phase 3 ≥ 200 events | **YES** (288 funding intervals) |
| Pre-registered walk-forward on H-PERP-001 | **Not executed** (would be under-powered and non-compliant) |

## Diagnostic only

Script [backtests/H-PERP-001.py](backtests/H-PERP-001.py) computes **all-sample** Sharpe on `pnl_i = fundingRate_i × 100 USDT` for transparency. Output: [backtests/H-PERP-001_metrics.json](backtests/H-PERP-001_metrics.json).

> That Sharpe is **not** gate-eligible OOS — it ignores the walk-forward protocol entirely and mixes regimes.

## Next steps

1. Re-fetch from a **non-geo-blocked** host or paid vendor to get **≥12 months** of funding, **or**
2. Abandon H-PERP-001 in favor of hypotheses that meet Phase 3 on **Coinbase** or **Kalshi archival** data.

---

**Phase 4 gate:** **NOT RUN** (blocked upstream by Phase 3).
