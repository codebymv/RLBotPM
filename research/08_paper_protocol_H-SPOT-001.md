# 08 — Paper protocol: H-SPOT-001

## Objective

Log **live Coinbase BTC-USD daily closes** and the **pre-registered H-SPOT-001** signal (`pos_raw`) on the same cadence as Kalshi paper scans — **no orders**, no capital at risk — to verify **data plumbing**, timestamp alignment, and long-run **tracking error vs the offline CSV backtest**.

## Implementation

| Piece | Location |
|-------|----------|
| Logger | `append_h_spot_001_research_snapshot()` in [bot/src/strategies/paper_trader.py](../bot/src/strategies/paper_trader.py) |
| Output | `bot/logs/paper_research_H-SPOT-001.jsonl` |
| Activation | Set environment variable **`RESEARCH_LOG_H_SPOT=true`** while running `python main.py kalshi paper-trade ...` |

Each JSON line includes: `timestamp`, `kalshi_scan`, `coinbase_time`, `close`, `sma20`, `sma120`, `pos_raw`.

## Minimum duration & sample

| Requirement | Value |
|-------------|--------|
| Calendar | **≥ 30 days** wall-clock |
| Snapshots | **≥ 50** log lines **or** 30 days, whichever is later |

### Amendment (2026-04-23) — plumbing-only mode

H-SPOT-001 **already failed** Phase 4 before this paper stream began ([07_backtest_results_H-SPOT-001.md](07_backtest_results_H-SPOT-001.md)). In that situation Phase 5 cannot validate economic edge; it only validates **wiring**.

For **plumbing-only** runs, treat Phase 5 as **complete** when **all** are true:

1. **≥ 50** snapshots written without systematic field corruption, **and**
2. No Coinbase outage pattern (see **Drift policy**), **and**
3. Results are written to [09_paper_results_H-SPOT-001.md](09_paper_results_H-SPOT-001.md).

The **30-day** clock remains required for any hypothesis that **passed** Phase 4 and is being tracked for **live fundability**. This amendment does **not** relax the fundability checklist in [10_fundability_review_H-SPOT-001.md](10_fundability_review_H-SPOT-001.md) for capital deployment.

## Tracking vs backtest

| Quantity | Tolerance |
|----------|-----------|
| `pos_raw` vs offline recompute same day | Must match **exactly** when Coinbase candle set identical |
| Sharpe / win-rate vs backtest | **N/A** until we build a dedicated evaluator — offline backtest already **FAIL** |

## Drift policy

If Coinbase API errors exceed **5%** of scans in a rolling week, pause logging and file an incident note in `09_paper_results_H-SPOT-001.md`.

## Command example

```powershell
$env:RESEARCH_LOG_H_SPOT = "true"
cd RLBotPM\bot
python main.py kalshi paper-trade --interval 300 --bankroll 100 --max-scans 10000
```

Stop manually after 30 days, or when plumbing-only completion criteria in **Amendment (2026-04-23)** are met (for already-FAIL hypotheses), or when you intentionally move to the next hypothesis in [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md).
