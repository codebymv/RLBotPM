# 09 — Paper results: H-PERP-003

> **Status (2026-07-23 UTC):** Phase 5 minimum met. Tracking formula gates
> **PASS**. Phase 4 Sharpe/PF drift gate **FAIL** → do not promote to live.
> **Phase 4 source:** [07_backtest_results_H-PERP-003.md](07_backtest_results_H-PERP-003.md)
> **Protocol:** [08_paper_protocol_H-PERP-003.md](08_paper_protocol_H-PERP-003.md)
> **Log:** `bot/logs/paper_research_H-PERP-003.jsonl`

## Activation

The paper logger was activated after the Phase 4 PASS. It uses OKX public REST
only and sends **no orders**.

| Field | Value |
|-------|-------|
| First snapshot UTC | 2026-05-05T00:37:51.795363+00:00 |
| First `fundingTime` | 1777939200000 |
| Last snapshot UTC | 2026-07-23T00:04:28.855326+00:00 |
| Last `fundingTime` | 1784764800000 |
| Calendar span | **~79.0 days** |
| Snapshots | **237** (236 closed intervals with `pnl_interval_usdt`) |
| Cum PnL (paper) | **+0.7036 USDT** on `V = 100` |
| Code version | `h-perp-003.paper.v1` |

## Phase 5 minimum

| Requirement | Threshold | Current | Verifier |
|-------------|-----------|---------|----------|
| Calendar duration | >= 30 days | **~79 days** | wallclock |
| Funding snapshots | >= 90 | **237** | jsonl line count |
| Per-interval rebuild error | <= 1e-6 USDT | **PASS** (`max_abs_diff=0`) | [bot/scripts/check_h_perp_003_tracking.py](../bot/scripts/check_h_perp_003_tracking.py) |
| Daily tracking error | <= 5% of `|cum_pnl|` | **PASS** (0 daily failures / 80 days) | same |
| Paper Sharpe / PF drift | within 30% of Phase 4 OOS values | **FAIL** (Sharpe **4.07** vs 7.97; PF 1.44 vs 1.88 within band) | same (`--drift-min-intervals 90`) |

Verifier run (2026-07-23 UTC) after Sharpe-annualization fix:

- `n_matched_intervals = 236`
- `verdict = FAIL` (formula gates pass; **Sharpe drift gate fails**; PF within 30%)
- Exit code `1` (FAIL)
- Relative Sharpe drift ≈ **48.9%** (`|4.07 − 7.97| / 7.97`) — still outside the 30% band

### Sharpe definition correction (2026-07-23)

Earlier verifier runs reported paper Sharpe ≈ **1.90** using sample
`sqrt(n)` scaling. Phase 4 OOS Sharpe (**7.97**) uses `sharpe_8h` =
`(mean/stdev) * sqrt(365 * 3)` on 8h intervals
([`H-PERP-003.py`](backtests/H-PERP-003.py)). That mismatch inflated
apparent drift. [`check_h_perp_003_tracking.py`](../bot/scripts/check_h_perp_003_tracking.py)
now uses the Phase 4 convention; comparable paper Sharpe is **~4.07**.
**Verdict remains FAIL — DO NOT PROMOTE.** Regression coverage:
[`tests/test_h_perp_003_tracking.py`](../tests/test_h_perp_003_tracking.py)
(`test_stats_sharpe_matches_phase4_annualization`,
`test_phase4_drift_gate_fails_on_low_sharpe_window`).

## Architecture repair note (2026-07-23)

The offline panel had stalled at `2026-05-04` while the paper logger continued
through `2026-07-23`, so the verifier previously reported `INSUFFICIENT`
(`n_matched_intervals=0`). Repair:

1. Ran [daily_capture.py](datasets/H-PERP-003/daily_capture.py) (+238 rows).
2. Synced paper-window candle marks via
   [sync_panel_from_paper_log.py](datasets/H-PERP-003/sync_panel_from_paper_log.py)
   (late REST re-pulls disagree with paper-time closes).
3. Paper logger now dual-writes each new snapshot into the panel CSV.
4. Verifier rebuilds offline PnL along the **paper fundingTime sequence**
   (tolerates a missed paper boundary without false FAIL).
5. `fleet.yaml` now disables parked Kalshi / RL sleeves and enables the
   H-PERP-003 paper logger as the dry-run path.

## Interim verdict

**FAIL — DO NOT PROMOTE.**

Phase 5 sample size and formula tracking are satisfied, but live paper Sharpe
(~4.07 annualized, Phase 4 convention) is still far outside the 30% drift band
versus Phase 4 OOS (~7.97). Treat the paper stream as an honest negative
transfer result for promotion, not as a tuning invitation.

Live capital remains blocked by
[10_fundability_review_H-PERP-003.md](10_fundability_review_H-PERP-003.md).
