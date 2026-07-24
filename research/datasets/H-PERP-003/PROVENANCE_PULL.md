# H-PERP-003 — pull provenance

| Field | Value |
|-------|--------|
| Retrieved (UTC) | 2026-04-26T00:53:47.313290+00:00 |
| Script | `fetch_hedged_panel.py` |
| Base URL | `https://www.okx.com/api/v5` |
| Swap | `BTC-USDT-SWAP` funding + mark **1H** (OKX rejects `8H` mark candles) |
| Spot | `BTC-USDT` candles **1H** |
| Funding rows | 285 |
| First funding (UTC) | 2026-01-21T08:00:00+00:00 |
| Last funding (UTC) | 2026-04-26T00:00:00+00:00 |
| Approx. span | 94.7 calendar days |
| `align_ok` (both legs in 1h bar + skew-to-boundary ≤ 60s each) | 100.00% |
| Unique 1H candle opens (mark / spot) | 2400 / 2400 |
| Output | `btc_hedged_panel_okx.csv` |

## Funding sign (short)

OKX **positive** `fundingRate` ⇒ **long pays short** ⇒ a **short** receives **+rate × notional**
in the usual linear USDT convention (verify against [OKX funding](https://www.okx.com/help/funding-fee) before live use).
Backtest [H-PERP-003.py](../../backtests/H-PERP-003.py) uses `b_i = fundingRate` as **USDT per $1 notional per interval**
only if that matches your spot check; flip sign in one place in code if docs disagree.

## Reproduce

```bash
python fetch_hedged_panel.py
```
