# Architecture Audit 00 — Executive Recommendation

> Scope: both trading bots in RLBotPM: the RL crypto spot bot and the Kalshi prediction-market bot.  
> Purpose: preserve the current architecture findings in numbered audit files so future changes have a clear trail.

---

## Recommendation

Prioritize **Track A: H-PERP-003 hedged perpetual carry**.

This is the only currently active path with a credible, documented market mechanism: perpetual funding exists to keep the perp anchored to spot, and persistent funding premia can be tested as carry. The current implementation is blocked by data depth, not by a falsified economic claim.

Do **not** restart live or paper trading for the existing Kalshi lognormal strategies, H-KALS-001 / 001b, H-SPOT-001, or the current RL model path as a profit-seeking activity. They can be useful as infrastructure tests, but not as capital candidates.

---

## Current state

| Area | Status | Interpretation |
|------|--------|----------------|
| RL crypto spot bot | Not live-ready | No deployable model exists at the configured path; prior paper evidence is too thin and negative. |
| Kalshi prediction bot | Strategy layer falsified / parked | Infrastructure works, but the tested edge hypotheses failed or trapped capital. |
| H-SPOT-001 | Phase 4 FAIL | Positive cumulative return, but failed Sharpe, profit factor, segment stability, and placebo gates. |
| H-PERP-001 | INCONCLUSIVE | Funding depth blocked the one-leg test. |
| H-PERP-003 | INCONCLUSIVE_DATA | Best current candidate; 94.7d clean alignment, needs >=365d. |
| H-KALS-001 / 001b | PARKED | Scanners work, but no executable economics. |

---

## Highest-value next move

Build the H-PERP-003 data path into a reliable, daily append-only research pipeline:

1. Keep the OKX public puller running on a schedule so we accumulate clean aligned rows.
2. Add an authenticated / archive ingest path if deeper OKX history is available.
3. Keep the Phase 3 gate strict: no Phase 4 verdict until >=365d data exists.
4. If H-PERP-003 passes later, then and only then write paper protocol and fundability review files.

This gives us a real chance at a small, mechanical, evidence-backed edge without pretending that the existing bots already found alpha.

---

## Capital rule

No new live capital until a hypothesis clears:

1. `06_backtest_design_<id>.md`
2. Phase 3 data quality
3. `07_backtest_results_<id>.md` with PASS
4. Paper protocol / paper results
5. Fundability review with no blocking `NO`

This rule should apply equally to RL crypto, Kalshi, and any future venue.

---

## Audit series

- `architecture-audit-00.md` — executive recommendation and current state.
- `architecture-audit-01.md` — RL crypto spot bot architecture findings.
- `architecture-audit-02.md` — Kalshi bot architecture findings.
- `architecture-audit-03.md` — recommended operating plan.

