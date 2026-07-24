# 02 — Venue Survey (Edge Research Reset, Phase 1b)

> Date: 2026-04-19. Capital assumption: ~$40 exploratory bankroll, Python stack, US-based operator unless noted. Scores are **1–5** (5 = best fit for *offline research first*, small size, and public data). Citations are authoritative where possible; secondary blogs are marked as such.

---

## Scoring rubric (per dimension)

| # | Dimension | What “5” means |
|---|-------------|----------------|
| A | Minimum viable trade ($0.50–$5 notionally meaningful) | Can size positions in single-digit dollars without being dominated by fixed fees |
| B | Fee drag at sub–$10k volume | All-in round-trip cost as % of a **$5** round-trip |
| C | Public historical data (12+ mo) | Free or cheap OHLC / funding / trades for backtests |
| D | API quality | REST/WS documented, stable, no $10k data vendor gate |
| E | Documented retail edge stories | Published mechanics (academic, exchange research, or verifiable practitioner writeups) |
| F | Latency sensitivity | 5 = minutes-level OK; 1 = sub-100ms arms race |
| G | Edge taxonomy richness | Multiple *distinct* edge families (mechanical / statistical / behavioral) |
| H | Regulatory / ops friction | KYC, geo locks, withdrawal, tax reporting burden (5 = low friction) |

---

## 1. Kalshi (US regulated event contracts)

**Fit note:** Lognormal-vs-strike on crypto hourlies is **falsified** for this project ([01_postmortem.md](01_postmortem.md)). Kalshi remains interesting for **non-model** edges: cross-series consistency, event-window behavior, liquidity provision, settlement microstructure.

| Dim | Score | Notes |
|-----|-------|------|
| A | **4** | Contracts are $1 payoff; 1-contract clips are feasible. |
| B | **4** | Probability-weighted fee: `ceil(0.07 × C × P × (1−P))` taker; lower at extreme prices. Official: [Kalshi fee schedule PDF](https://kalshi.com/docs/kalshi-fee-schedule.pdf), [Fees | Kalshi Help Center](https://help.kalshi.com/trading/fees). |
| C | **3** | API for markets + trades; long history may require self-archiving (we have `kalshi_backfill.py`). |
| D | **4** | [Trade API v2](https://docs.kalshi.com/) — already integrated in-repo. |
| E | **2** | Few rigorous public “retail arb” track records; much is anecdotal. |
| F | **5** | Seconds–minutes latency is enough for many event markets. |
| G | **4** | Mechanical (mis-crossed buckets), behavioral (overweight tail narratives), structural (macro release windows). |
| H | **4** | US-accessible; KYC; ACH generally smooth per [fee schedule PDF](https://kalshi.com/docs/kalshi-fee-schedule.pdf). |

**Round-trip fee example (taker):** At P=$0.50, one contract fee ≈ `0.07*1*0.5*0.5` ≈ **1.75¢** max per side class in simplified reading — see official PDF for exact rounding.

---

## 2. Polymarket (crypto-based prediction CLOB on Polygon)

| Dim | Score | Notes |
|-----|-------|------|
| A | **3** | Share clips can be small but gas + minimum meaningful bankroll interact. |
| B | **2–3** | Category taker fees up to **~7.2%** on some crypto markets (variable by category); see [Polymarket Docs — Fees](https://docs.polymarket.com/trading/fees) and [Help — Trading Fees](https://help.polymarket.com/en/articles/13364478-trading-fees). **Maker 0.** |
| C | **2** | Subgraph / third-party indexers; full clean 12-month OHLC less turnkey than CEX. |
| D | **3** | CLOB APIs exist; operational complexity (wallet, RPC, allowances). |
| E | **2** | Narrative edge claims abound; peer-reviewed performance rare. |
| F | **4** | Not HFT for most retail flows. |
| G | **4** | Information arrival, cross-market replication, tail mispricing narratives. |
| H | **2** | Geo restrictions for US persons have shifted over time — **verify current ToS** before any live use; see official [polymarket.com](https://polymarket.com) legal pages. |

---

## 3. Coinbase Advanced Trade (spot crypto)

| Dim | Score | Notes |
|-----|-------|------|
| A | **5** | Spot crypto can be traded in **fractional** units; $1–$5 clips realistic. |
| B | **3** | Retail tier ≈ **0.40% maker / 0.60% taker** below $10k 30-day volume (sources disagree slightly on tier cutoffs — verify in-product). See [Coinbase Help — Advanced Trade fees](https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/advanced-trade-fees). |
| C | **5** | Public candles via Coinbase Exchange / Advanced APIs — we already use Coinbase in-repo (`spot_feeds.py`). |
| D | **5** | [Advanced Trade API FAQ](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/faq), [Get fees](https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getfees). |
| E | **3** | Academic + practitioner literature on momentum, carry, microstructure — not “get rich” blogs as primary. |
| F | **4** | HFT not required for daily–hourly signals. |
| G | **4** | Trend, mean reversion, cross-exchange lead–lag, on-chain premia (with extra data). |
| H | **4** | US-compliant path; KYC. |

**$5 round-trip at 0.6% taker both sides:** ≈ **1.2%** of notional → score B = 3.

---

## 4. Bybit linear perpetuals (e.g. BTCUSDT)

| Dim | Score | Notes |
|-----|-------|------|
| A | **4** | Small USDT notional possible; minimums contract-specific — verify `minOrderQty` via instruments API. |
| B | **4** | Competitive maker/taker vs retail spot; see [Bybit fees](https://www.bybit.com/en-US/help-center/article/Trading-Fee-Structure). |
| C | **5** | **Historical funding** via public REST — [Get Funding Rate History](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate) (`/v5/market/funding/history`). |
| D | **5** | V5 REST/WS well documented. |
| E | **4** | Funding premium / basis literature (perps vs spot) is exchange-replicated stylized fact. |
| F | **3** | Liquid BTC can be latency-sensitive for *aggressive* arb; fine for **funding accrual** research at 8h cadence. |
| G | **5** | Funding carry, basis trade, cross-exchange perp–spot, liquidation cascades (harder). |
| H | **2** | **US persons:** Bybit has restricted US retail; do not assume access. Treat as **data + research venue** unless compliance confirms eligibility. |

---

## 5. Binance USDT-M perpetuals

Similar to Bybit on **funding + liquidity**; **US retail** generally blocked from `binance.com` derivatives — use **Binance.US** (different product set) or treat Binance global data as **research-only** for signals that could be executed elsewhere (e.g., Coinbase International perps if eligible).

| Dim | Score | Notes |
|-----|-------|------|
| A–D | **4–5** | Excellent public history & APIs for **global** instance. |
| H | **2** | Jurisdiction friction for US-based operator on derivatives. |

---

## 6. Deribit (BTC/ETH options & futures)

| Dim | Score | Notes |
|-----|-------|------|
| A | **3** | Options multipliers can be large **notional per contract** vs $40 bankroll. |
| B | **3** | Fee schedule updates — see [Deribit fees](https://www.deribit.com/kb/fees) and [Fee Schedule PDF (2025)](https://assets.ctfassets.net/k3n74unfin40/4IWUdo374UjltobtmRRx64/8ebfea7afdcefb313453681b56173cc2/Fee_Schedule_-_Update_for_2025.pdf). |
| C | **4** | Historical tick / DVOL available; heavier engineering than funding CSV. |
| D | **4** | Strong API; testnet available. |
| E | **4** | Vol risk premium / skew literature maps cleanly to options. |
| F | **2** | Competitive at the microstructure level for some strategies. |
| G | **5** | Rich options edge taxonomy. |
| H | **2** | Non-US / KYC patterns; equity tiers for VIP discounts ([Deribit Insights — volume discounts](https://insights.deribit.com/education/new-volume-discounts-on-trading-fees/)). |

---

## 7. US sportsbooks (DraftKings, FanDuel, etc.)

| Dim | Score | Notes |
|-----|-------|------|
| A | **2** | Fixed-odds tickets often **$1+** with promos; repeatable $0.50 clips uncommon. |
| B | **2** | Vig embedded in odds; effective fee **large** vs prediction markets’ explicit fee. |
| C | **1** | No unified free historical closing-line archive at retail tier. |
| D | **1–2** | Scraping ToS issues; official APIs not designed for retail systematic use. |
| E | **3** | “Closing line value” (CLV) literature — sharp vs soft books. |
| F | **3** | Seconds usually enough unless live betting. |
| G | **3** | Behavioral + pricing inefficiency — hard to test cleanly offline. |
| H | **3** | State-by-state legality & account limits. |

---

## 8. CME futures & options (e.g., Micro ES)

| Dim | Score | Notes |
|-----|-------|------|
| A | **1** | **Micro ES margin** is **~$2.2k–$2.5k+** maintenance per CME margin page snapshot (search hit — re-verify live): [CME Micro E-mini S&P 500 margins](https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.margins.html). Far above $40 bankroll for prudent trading. |
| B | **4** | Exchange fees modest vs notional **once** you can afford the contract. |
| C | **4** | Vendor data (CQG, Polygon, etc.) — cost/quality tradeoff. |
| D | **3** | IBKR / Tradovate APIs exist; onboarding heavy. |
| E | **4** | Huge academic literature (trend, carry, term structure). |
| F | **2–4** | Strategy-dependent. |
| G | **5** | |
| H | **3** | Futures account, margins, PFOF not applicable but compliance real. |

**Verdict for $40 bankroll:** **Not viable** as execution venue today; keep only as **literature reference** for statistical edges that might be ported to micro-crypto or Kalshi.

---

## 9. US equity options via Interactive Brokers

| Dim | Score | Notes |
|-----|-------|------|
| A | **2** | Contract fees dominate tiny notionals — e.g. **~$0.15–$0.65 per contract** tiered/fixed ([IBKR — Options commissions](https://www.interactivebrokers.com/en/index.php?f=49623)). A 1-lot on a $1 wide spread can be fee-heavy. |
| B | **2** | Great for larger size; poor for $5 tickets. |
| C | **3** | Options chains historical: vendor or IBKR historical data subscriptions. |
| D | **4** | IBKR API mature. |
| E | **5** | Enormous academic + practitioner literature (VRP, skew, dispersion). |
| F | **2–4** | |
| G | **5** | |
| H | **3** | Account minimums low, but **fee physics** hurts micro size. |

---

## Aggregate ranking (sum of scores)

| Venue | Approx total | Notes |
|-------|----------------|-------|
| Coinbase Advanced (spot) | **33–34** | Best **honest** US retail research + execution path at $40. |
| Bybit perps (global) | **32–33** | Best **public funding dataset**; execution may be blocked for US — still top-tier for **Phase 3–4 research**. |
| Kalshi | **30** | Strong infra + fees; edge must be **non-falsified** hypotheses only. |
| Deribit | **28–30** | Great for options research; bankroll mismatch for live. |
| Binance global perps | **27–30** | Similar to Bybit with geo caveats. |
| Polymarket | **24–26** | Interesting; fee + wallet + geo friction. |
| IBKR options | **24–26** | Literature yes; fee physics no at $5. |
| Sportsbooks | **18–22** | Poor offline testability. |
| CME micro futures | **18–22** | Margin floor eliminates $40 live use. |

---

## TOP 2 RECOMMENDATIONS (Phase 1 gate)

### **#1 — Coinbase Advanced Trade (spot crypto)**

**One sentence:** It is the only venue in this survey where a **US retail operator at ~$40** can realistically size trades, pull **12+ months of free OHLC** from a first-party API, and stress fees without derivatives geo-blockers — making it the default **execution-aligned research venue** for statistical edges (trend, mean reversion, cross-asset lead–lag) that we have not yet falsified.

### **#2 — Linear perpetual funding (Bybit as data + optional execution reference)**

**One sentence:** **Perpetual funding** is a **mechanically defined, high-frequency premium** (longs pay shorts when funding > 0) with **documented public history** via Bybit’s `/v5/market/funding/history` ([docs](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate)), giving the cleanest offline testbed for **carry / premium harvesting** hypotheses even if US execution must later map to a permitted perp venue.

---

## PAUSED (explicit non-pursuit for this cycle)

| Venue | Why paused |
|-------|------------|
| **CME micro futures** | **Margin ~$2k+** vs $40 — cannot responsibly execute; only literature value. |
| **IBKR micro options** | **Per-contract fees** dominate $5 notionals; research OK, live micro-scalping not prioritized. |
| **US sportsbooks** | **No clean free historical** + ToS risk for systematic scraping; deprioritized vs CEX APIs. |
| **Polymarket** | **High taker fees** on some categories + **wallet/geo** complexity; revisit only if a **specific mechanical arb** is pre-registered. |
| **Kalshi lognormal model** | **Falsified** — Kalshi remains open only for **new hypotheses** (see postmortem). |

---

## Next document

Proceed to [03_edge_taxonomy.md](03_edge_taxonomy.md) for the **#1 and #2** venues above.
