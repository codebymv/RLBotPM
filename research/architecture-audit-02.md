# Architecture Audit 02 — Kalshi Bot

> Bot: Kalshi prediction-market scanner / paper trader / live trader.  
> Main files: `bot/src/data/sources/kalshi.py`, `bot/src/execution/kalshi_client.py`, `bot/src/strategies/kalshi_edges.py`, `bot/src/strategies/paper_trader.py`, `bot/src/strategies/live_trader.py`, `research/01_postmortem.md`, `research/07_backtest_results_H-KALS-001.md`, `research/07_backtest_results_H-KALS-001b.md`.

---

## Verdict

The Kalshi infrastructure is valuable and reusable, but the strategy layer tested so far should remain **off for profit-seeking use**.

The post-mortem conclusion still holds: the code loop was mostly not the problem; the hypotheses it executed were wrong, unproven, or unsuitable for the bankroll.

---

## What is salvageable

The following pieces are worth keeping:

- Kalshi market-data adapter.
- Execution client covering auth, balances, positions, orders, cancellation, and settlement checks.
- JSONL decision / scan logging pattern.
- Paper-trading scaffold.
- Live-trading loop and kill-switch plumbing.
- Candidate logging from the H-KALS research cycle.
- Scanner pagination and provenance pattern.

These are strong reusable pieces for future hypotheses.

---

## Failed or parked strategy families

### 1. Lognormal-vs-strike near-money model

From `research/01_postmortem.md`:

- 8 settled near-money `spot_vs_strike` trades.
- 0 wins.
- `-$1.65` settled PnL.
- Model reported average edge around 16.7%.

**Interpretation:** the reported model edge was not executable edge. It disappeared into bid/ask spread and market microstructure.

### 2. Far-OTM "trivial edge"

Historical wins came from a transient regime where Kalshi strike ladders were far from spot. Current H1 scans found no live candidate satisfying the required distance and time-to-expiry conditions.

**Interpretation:** not a persistent strategy. Do not assume historical 38W / 0L repeats.

### 3. Macro data model

The macro strategy placed long-duration positions that did not settle quickly.

For a tiny bankroll, this is a capital-recycling failure even before model quality is known.

**Interpretation:** inappropriate for the current bankroll and evidence cadence.

### 4. H-KALS-001

Rule set A found many apparent probability-sum violations, but the grouping was not reliably MECE. A single representative scan found an implausibly high violation rate, indicating the rule was mostly measuring grouping artifacts.

**Status:** parked / narrow FAIL.

### 5. H-KALS-001b

Rule set B tightened the grouping to contiguous `between` ladders. It found stable diagnostic violations across 10 scans, but still did not establish executable economics.

**Status:** parked. Useful scanner evidence, not a trade.

---

## Root-cause findings

### 1. Model edge was measured against mid, not executable price

The post-mortem's most important finding: if a model says fair is 40c and market mid is 50c, but bid/ask is 45c / 55c, there may be no executable edge at all.

Future Kalshi hypotheses must define edge using executable bid / ask prices, not mids.

### 2. Capacity and stale markets were underweighted

Several prior orders were in markets with very low volume and open interest.

Future Kalshi hypotheses need explicit minimum activity and spread constraints before any trade simulation.

### 3. Holding period must match bankroll

Macro and long-dated contracts can be intellectually interesting but useless for a small bankroll if capital is trapped for months.

Future Kalshi hypotheses should prefer events with short settlement cycles unless the bankroll is large enough to diversify.

### 4. No mechanism means no trade

"My model says this is mispriced" is not a mechanism.

Future Kalshi hypotheses must explain why the other side is persistently willing to be wrong after fees, spreads, and capacity.

---

## Recommended Kalshi policy

1. Keep the adapters and logging.
2. Keep old detectors disabled for profit-seeking use.
3. Do not resume `kalshi paper-trade` merely to append logs for falsified strategies.
4. Only reopen Kalshi with a new `06_backtest_design_<id>.md`.
5. Require executable-price edge, spread cap, activity floor, holding-period cap, and bankroll fit in every future Kalshi `06`.

---

## Recommended status

**Infrastructure retained; strategies disqualified until new pre-registered hypothesis.**

