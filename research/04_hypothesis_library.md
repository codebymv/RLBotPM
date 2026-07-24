# 04 — Hypothesis Library (Top-2 Venues)

> Format per plan. **H-PERP-001** is ranked #1 for Phase 3–4 (data-rich, mechanical, citation-backed).

---

```
H-PERP-001: A passive short linear-BTC-perpetual position that collects scheduled
funding payments when the published funding rate is positive (longs pay shorts)
produces positive expected P&L after conservative taker fees over 12+ months of
out-of-sample windows, net of large drawdowns from BTC price moves (hedged variant
optional; v1 is funding-only approximation with documented limitation).

Mechanism: Perpetual swaps use periodic funding transfers so that contract prices
re-anchor toward spot; no-arbitrage/replication analysis shows funding is an
integral part of perpetual pricing (not an arbitrary fee). See Ackerer,
Hugonnier, Jermann, "Perpetual Futures Pricing" (2023/2024), arXiv:2310.11771v2
https://arxiv.org/html/2310.11771v2 and NBER working paper w32936
https://www.nber.org/papers/w32936 — funding design keeps perp aligned with spot
in equilibrium; empirical crypto markets exhibit persistent non-zero funding
episodes exploitable only if price risk is managed (v1 documents one-leg bias).

Falsifiable form: Pre-registered rules in 06_backtest_design_H-PERP-001.md —
if walk-forward OOS fails any item in the evidence gate (Sharpe, profit factor,
fee stress, window stability, concentration, placebo), declare FAIL and abandon
v1 without parameter hunting.

Required data: ≥12 months of 8h (or venue-native interval) fundingRate for
BTCUSDT linear from Bybit public API; aligned BTCUSD spot closes for USD PnL
optional in v1.1.

Expected sample size in 12 months: ~365 × 3 ≈ 1,095 funding intervals per year
per symbol (subject to exchange maintenance gaps).

Expected gross edge per trade: order of **0.01%–0.05% of notional per 8h** in
quiet regimes (1–5 bps per interval) when funding is positive — **not** 50%
"model edge"; this is a carry sleeve, not mispricing vs a wrong model.

Capacity: For $40 bankroll, effectively **$0** live on perps until jurisdiction +
margin verified; research capacity is **unbounded** on public data.

Pre-test confidence: **Medium** — mechanism is solid in theory; **one-leg**
short-BTC is dominated by delta P&L, so v1 uses an explicit **funding-only /
delta-approximation** disclaimer and may FAIL quickly — which is a successful
falsification.
```

---

```
H-PERP-002: Extreme positive funding (top decile vs rolling 90d) mean-reverts
within 48h (trade the rate, not the coin).

Mechanism: Crowded one-sided positioning increases funding; unwind pressure
pushes rate down — behavioral + inventory story common in practitioner
literature (cite exchange research / arXiv 2506.08573 "Designing funding rates…"
https://arxiv.org/html/2506.08573v1 as modern reference on funding design).

Falsifiable form: Event study — after hitting top decile, distribution of
Δfunding over {24h, 48h} vs placebo times.

Required data: Same as H-PERP-001.

Expected sample size in 12 months: ~30–80 events/year (decile crossings).

Expected gross edge per trade: N/A (alpha in **rate space**); translate to $ only
after execution mapping.

Capacity: Research-high; live-low.

Pre-test confidence: Low–medium (regime shifts).
```

---

```
H-PERP-003: Delta-hedged **carry** on BTC — scheduled funding to a **short**
perp leg combined with **spot–perp log-return difference** over each funding
window (discrete hedge), OKX-only, ≥12 months before Phase 4 gates.

Mechanism: Same perpetual funding premium story as H-PERP-001, but PnL
includes explicit **basis / hedge residual** instead of funding-only upper
bound.

Falsifiable form: Pre-registered in `06_backtest_design_H-PERP-003.md` —
same numeric Phase 4 gates as H-PERP-001 after Phase 3 data contract passes.

Required data: OKX BTC-USDT-SWAP funding + aligned perp mark + spot for
≥365d (see dataset provenance).

Pre-test confidence: Low until depth verified.
```

---

```
H-SPOT-001: Simple time-series momentum (e.g., 20d > 120d trend filter) on BTC
spot (Coinbase candles) has positive OOS Sharpe after 0.6% taker round-trip stress.

Mechanism: Cross-asset momentum is a documented anomaly (Jegadeesh & Titman
1993 lineage); crypto exhibits fat tails so risk management dominates — cite
Moskowitz, Ooi, Pedersen (2012) style **time series momentum** as generic
reference https://www.aqr.com/-/media/AQR/Documents/Insights/Research-Journal/
Time-Series-Momentum.pdf (AQR working paper lineage — user should verify link).

Falsifiable form: Pre-registered lookback + hold, walk-forward on Coinbase OHLC.

Required data: Coinbase (or Exchange) BTC-USD daily candles 24+ months.

Expected sample size in 12 months: ~12–24 round turns at daily frequency.

Expected gross edge per trade: 0.5%–3% before fees typical for crude rules.

Capacity: Moderate until size moves market.

Pre-test confidence: Low (crowded; fees hurt at $5).
```

---

```
H-SPOT-002: BTC–ETH ratio mean reversion (band trade) after z-score > 2 on
30d log spread.

Mechanism: Temporary liquidity shocks move ratios; cointegration episodic in
crypto — falsifiable without strong structural claim.

Falsifiable form: Half-life and OOS hit-rate on spread.

Required data: Dual spot series 24+ months.

Expected sample size in 12 months: 20–60 signals.

Expected gross edge per trade: 0.2%–1.5%.

Capacity: Low–moderate.

Pre-test confidence: Low.
```

---

```
H-KALS-001: Kalshi **non-model** cross-bucket consistency — if two mutually
exclusive buckets for the same expiry imply probabilities summing > 1.05 or < 0.95
(mid prices), post passive orders on both sides (pure mechanical sum violation).
Pre-registered observation design: `06_backtest_design_H-KALS-001.md`.

Mechanism: Probability algebra constraint; rare when both sides liquid.

Falsifiable form: Scan historical archived Kalshi books; count violations ×
hypothetical fill model.

Required data: Archived order books or at least daily closes for sibling tickers.

Expected sample size in 12 months: unknown (likely <50).

Expected gross edge per trade: small when it exists.

Capacity: Tiny.

Pre-test confidence: Low (data heavy).
```

---

```
H-KALS-001b: Same mechanical theme as H-KALS-001, but **rule set B** — only
contiguous `between` ladders within one `event_ticker` + `close_time`; toy
ask-sum diagnostics. Pre-registered: `06_backtest_design_H-KALS-001b.md`.
H-KALS-001 (rule set A) is **parked** — see `07_backtest_results_H-KALS-001.md`.

Mechanism: Tighter necessary conditions for a one-dimensional strike partition.

Falsifiable form: Observation scans + optional toy economics fields in JSONL.

Pre-test confidence: Low.
```

---

```
H-KALS-002: CPI / NFP **event window** — 24h before release, fade extreme YES
prices >90 on "surprise impossible" buckets (behavioral).

Mechanism: Attention / lottery ticket bid-up — needs careful falsification vs
liquidity.

Falsifiable form: Event-study on realized surprise vs bucket payoffs.

Required data: Kalshi + FRED + release timestamps.

Expected sample size in 12 months: 12–24 macro events.

Expected gross edge per trade: unknown.

Capacity: Low.

Pre-test confidence: Low.
```

---

```
H-POLY-001: (Paused venue) Cross-venue **replication lag** between Polymarket
and Kalshi on same geopolitical question — buy cheaper YES within 60s of move.

Mechanism: Slow capital across venues + different fee curves.

Falsifiable form: Count lead–lag opportunities in archived mids (if data exists).

Required data: Paired time series — **hard**; pre-test confidence Low.
```

---

```
H-DERI-001: (Research-only at $40) Implied vol > realized vol (VRP) on BTC
options — long ATM straddle loses on average; short vol earns premium with tail risk.

Mechanism: Classical variance risk premium literature.

Falsifiable form: Straddle write P&L vs move — Deribit index data.

Required data: DVOL + underlying — OK for research; **not** live at $40.

Expected sample size in 12 months: 52 weekly rolls.

Expected gross edge per trade: negative for buyers; positive for sellers with
tail risk.

Capacity: N/A at $40.

Pre-test confidence: High theory; **execution NO** at bankroll.
```

---

```
H-CME-001: (Paused) Micro ES trend following at weekly horizon — academic support
but **margin block** at ~$2k+.

Mechanism: Time-series momentum on equity index futures.

Falsifiable form: CME continuous back-adjusted series.

Required data: Vendor OHLC.

Expected sample size in 12 months: 12–52.

Pre-test confidence: Medium theory; **venue paused** (see 02_venue_survey).
```

---

```
H-SPORT-001: (Paused) Closing-line value proxy without official historical feed —
not pursued this cycle.

Mechanism: Sharp vs soft book inefficiency.

Falsifiable form: N/A until data acquired.

Pre-test confidence: N/A.
```

---

```
H-ARB-001: (Mechanical) Coinbase vs Kraken **same-second** mid deviation > 25 bps
on BTC — simultaneous buy/sell (inventory on both venues).

Mechanism: Fragmented liquidity.

Falsifiable form: Count deviations × simulated fill with half-spread crossing.

Required data: L2 or 1s mids both venues ≥1 month.

Expected sample size in 12 months: data-dependent.

Pre-test confidence: Low at retail latency.
```

---

## Phase 2 gate — ranked top 3

| Rank | ID | One-line rationale |
|------|-----|---------------------|
| **1** | **H-PERP-001** | Only hypothesis pairing **citation-backed perpetual pricing theory** with **free high-frequency public funding history** and a crisp pre-registered backtest — best falsification value per engineering hour. |
| **2** | **H-SPOT-001** | Clean US spot data + simple momentum is the best **execution-aligned** second path if perp research dead-ends. |
| **3** | **H-PERP-002** | Tests **behavioral crowding** in rate space; secondary if H-PERP-001 inconclusive. |

**#1 mechanism is not a pattern:** It cites **arXiv:2310.11771v2** and **NBER w32936** on perpetual pricing/funding design, plus exchange API definition of funding sign.

Proceed to Phase 3: `research/datasets/H-PERP-FUND/` and `05_data_quality_H-PERP-001.md`.
