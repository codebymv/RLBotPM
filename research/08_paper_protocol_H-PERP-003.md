# 08 — Paper protocol: H-PERP-003

> **Status:** Phase 5 ACTIVATED. Phase 4 is **PASS** in
> [07_backtest_results_H-PERP-003.md](07_backtest_results_H-PERP-003.md)
> (`data_contract_ok = true`, D1 ≥ 365d). First paper snapshot is recorded in
> [09_paper_results_H-PERP-003.md](09_paper_results_H-PERP-003.md); the 30-day
> observation window is now in progress.

## Objective

Log **live OKX `BTC-USDT-SWAP` funding accruals + delta-hedge marks** at every
funding boundary on the same cadence as the backtest, **without sending any
orders**. Confirms that the paper interval-PnL stream tracks the offline CSV
rebuild within tolerance.

## Implementation (next)

| Piece | Location |
|-------|----------|
| Logger | `append_h_perp_003_paper_snapshot()` in [bot/src/strategies/paper_trader.py](../bot/src/strategies/paper_trader.py) |
| Dedicated runner | [bot/scripts/run_h_perp_003_paper_logger.py](../bot/scripts/run_h_perp_003_paper_logger.py) |
| Output | `bot/logs/paper_research_H-PERP-003.jsonl` |
| Activation env | Optional `RESEARCH_LOG_H_PERP_003=true` when using the existing Kalshi paper loop |
| Cadence | One snapshot per `fundingTime` boundary (8h on OKX) |
| Source | OKX public REST (no API key needed for paper) — uses `daily_capture.py` schema |

Each JSON line MUST include:

- `timestamp` — wall-clock UTC of snapshot
- `fundingTime` — ms since epoch of the funding boundary
- `fundingRate` — exchange-reported rate at this boundary
- `mark_close`, `spot_close`, `mark_skew_ms`, `spot_skew_ms`, `align_ok`
- `notional_usdt` — paper notional (frozen at `V = 100`, matches `06 §3`)
- `pnl_interval_usdt` — `V·fundingRate + V·(ln S_i / S_{i-1} − ln F_i / F_{i-1})`
- `cum_pnl_usdt` — running sum since logger activation
- `git_sha`, `code_version` — for reproducibility

## Minimum duration & sample (Phase 5)

| Requirement | Value |
|-------------|--------|
| Calendar | **≥ 30 days** wall-clock from first snapshot |
| Snapshots | **≥ 90** funding boundaries (≈ 30 days × 3/day) |
| Tracking error | Live `cum_pnl_usdt` vs offline rebuild from `daily_capture.py` CSV must match within **±5% of |cum_pnl|** at the close of every UTC day |

This protocol is now eligible to activate because Phase 4 returned **PASS**.
Paper days count from the first timestamp in
[09_paper_results_H-PERP-003.md](09_paper_results_H-PERP-003.md).

## Tracking vs backtest

| Quantity | Tolerance | Verifier |
|----------|-----------|----------|
| Per-interval `pnl_interval_usdt` | Within **1e-6 USDT** of offline recompute on identical CSV | [bot/scripts/check_h_perp_003_tracking.py](../bot/scripts/check_h_perp_003_tracking.py) |
| Daily `cum_pnl_usdt` | Within **5%** of offline cumulative for the same window | same verifier (`--daily-tolerance-pct 0.05`) |
| Sharpe / profit factor | Within **30%** of Phase 4 OOS values over the rolling 30d paper window (mirror of `10` item 9). Sharpe must use Phase 4 `sharpe_8h` (`sqrt(365×3)` on 8h returns), not sample `sqrt(n)`. | same verifier (`--phase4-drift-pct 0.30 --drift-min-intervals 90`) |

The verifier exits `0` for `PASS` or `INSUFFICIENT` and `1` for `FAIL`. It
must be re-run after every paper-day rollover (i.e., at least daily) and
its JSON output appended to [09_paper_results_H-PERP-003.md](09_paper_results_H-PERP-003.md).
Self-tests in [tests/test_h_perp_003_tracking.py](../tests/test_h_perp_003_tracking.py)
prove the verifier is correct on a synthesized offline replay before any
real paper days are counted.

## Drift policy

- If OKX returns 5xx or empty for **> 5%** of scheduled funding boundaries in
  any rolling 7-day window, pause logging and file an incident note in
  `09_paper_results_H-PERP-003.md`.
- If `align_ok` drops below **99%** for 3 consecutive days, halt — that is a
  signal the candle endpoint is no longer covering recent bars.

## Activation procedure

1. Verify [07_backtest_results_H-PERP-003.md](07_backtest_results_H-PERP-003.md)
   verdict is **PASS** with `data_contract_ok = true`. **Done 2026-05-05 UTC.**
2. Implement `append_h_perp_003_paper_snapshot()` in `paper_trader.py`
   (mirror the `append_h_spot_001_research_snapshot` shape). **Done 2026-05-05 UTC.**
3. Set `RESEARCH_LOG_H_PERP_003=true` in the bot environment.
4. Run the daily capture cron (already wired) — the paper logger consumes the
   same fetcher.
5. Stop manually after the minimum duration **AND** after writing
   `09_paper_results_H-PERP-003.md`.

## Command example (after activation)

```powershell
cd RLBotPM\bot
python scripts\run_h_perp_003_paper_logger.py --interval 300
```

The runner sends no orders and writes only to
`bot/logs/paper_research_H-PERP-003.jsonl`. The existing Kalshi paper loop can
also call the logger when `RESEARCH_LOG_H_PERP_003=true`, but the dedicated
runner is preferred for clean Phase 5 evidence.
