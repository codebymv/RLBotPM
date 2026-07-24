# 03 — Edge Taxonomy for Top-2 Venues

> Venues: (1) **Coinbase Advanced Trade — spot crypto**, (2) **Linear perpetuals — funding & basis** (Bybit as canonical data source).  
> Categories: **M** mechanical, **S** statistical, **B** behavioral, **T** structural/timing.

---

## Venue 1 — Coinbase Advanced Trade (spot)

### Mechanical (M)

- **M1 — Cross-venue price capture:** Same asset trades at different prices across exchanges; latency and fee budget determine feasibility (often requires inventory on both sides).
- **M2 — Stablecoin / fiat rail arb:** Rare for retail at size; mostly operational.

### Statistical (S)

- **S1 — Time-series momentum / trend:** Positive autocorrelation in **risk-adjusted** returns at horizons from days to months (extensive asset-pricing literature; crypto exhibits time-varying persistence).
- **S2 — Mean reversion on spreads:** Pairs of correlated assets (ETH/BTC, L2 baskets) deviate and revert — cointegration / half-life sensitive.
- **S3 — Realized vs implied volatility:** Without options on the same venue, needs cross-instrument data (Deribit IV vs spot vol).

### Behavioral (B)

- **B1 — Retail flow pressure:** Weekend / US evening patterns hypothesized from flow asymmetry (hard to prove without order flow; often confounded).
- **B2 — Listing / delisting events:** Attention spikes around new listings (exchange-specific; Coinbase listing studies exist).

### Structural / timing (T)

- **T1 — Funding-clock spillovers:** Crypto spot moves around perpetual funding timestamps (perp venue data + Coinbase spot).
- **T2 — Macro release windows:** CPI/FOMC intraday seasonality on BTC/ETH spot (event-study framing).

---

## Venue 2 — Linear perpetuals (funding & basis)

> Execution venue may differ from **data** venue for US operators; taxonomy is **venue-agnostic**.

### Mechanical (M)

- **M1 — Funding accrual (directional):** On Bybit/Binance-style linear perps, **positive** `fundingRate` typically means **longs pay shorts** (short earns the payment); **negative** means **shorts pay longs**. Any strategy must **read the venue’s sign definition** from API docs ([Bybit funding history](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate)) and keep it consistent in simulation.
- **M2 — Delta-neutral basis / funding harvest:** Spot vs perp hedge to isolate funding or basis component (requires two legs, fee model doubles).
- **M3 — Cross-exchange funding arb:** Same asset, different funding schedules / rates — inventory and transfer latency risk.

### Statistical (S)

- **S1 — Funding rate mean reversion:** Extreme positive funding predicts cooling (crowded longs unwind) — time-series on the rate itself.
- **S2 — Momentum in funding:** High positive funding persists short-term (carry in **rate space**) vs reverses next period — empirical only, must be pre-registered.

### Behavioral (B)

- **B1 — Liquidation / crowding cascades:** Retail long crowding proxied by high positive funding + rising OI — **risky** tail shorts.

### Structural (T)

- **T1 — Exchange parameter changes:** Funding interval / caps / formula tweaks (structural breaks in any fitted model).

---

## Cross-venue (Kalshi — allowed only for *new* hypotheses)

> Not in “top 2 execution” list but infra exists; taxonomy for completeness.

- **M:** Identical event priced differently across Polymarket vs Kalshi (geo permitting) — **pure arb** if both legs executable.
- **S:** Weather / CPI **event-study** around public forecast revisions vs market mid.
- **B:** Tail narrative overweight on low-probability YES contracts.

---

## Next

Concrete falsifiable hypotheses: [04_hypothesis_library.md](04_hypothesis_library.md).
