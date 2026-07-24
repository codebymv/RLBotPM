# Architecture Audit 03 — Operating Plan

> This file translates the architecture findings into a recommended work plan.  
> Bias: prefer hypotheses with a real mechanism, clean data, and strict gates over model tweaking.

---

## Guiding principle

The next profitable strategy, if one exists in this repo, is more likely to come from:

1. a small documented market mechanism,
2. clean first-party data,
3. conservative execution assumptions,
4. a pre-registered test,

than from adding complexity to either bot.

That points to **H-PERP-003 first**, not RL retraining and not Kalshi model revival.

---

## Track A — H-PERP-003 hedged carry

### Goal

Clear the Phase 3 data contract for H-PERP-003:

- >=365 calendar days
- OKX-only funding + perp mark + spot / index
- aligned at funding timestamps
- no calendar week with >50% missing intervals

### Current state

The public REST puller now produces:

- 285 funding rows
- about 94.7 calendar days
- 100% `align_ok`
- 2400 unique 1H candle opens on mark and spot legs

The blocker is **D1 depth**, not alignment.

### Recommended actions

1. Add an append-only daily capture job for `fetch_hedged_panel.py`.
2. Add provenance per run or a rolling `pull_log.jsonl`.
3. Investigate authenticated OKX / official archive history for >=365d mark and spot.
4. Add a CSV ingest path for externally purchased or exported history.
5. Rerun `backtests/H-PERP-003.py` only when D1 is met.

### Exit criteria

- If D1 cannot be cleared within a reasonable effort window, mark H-PERP-003 as blocked and move to H-PERP-002.
- If D1 clears and Phase 4 fails, close it cleanly and do not tune.
- If Phase 4 passes, write paper protocol and fundability review before any live discussion.

---

## Track B — RL bot architecture repair

### Goal

Make the RL bot capable of producing a credible candidate, not necessarily a profitable one.

### Required repair list

1. Fix `fleet.yaml` after a real model artifact exists.
2. Make evaluator-passing checkpoints the only deployable artifact.
3. Add held-out date split evaluation.
4. Make walk-forward validation a promotion gate.
5. Train under `pnl_only`.
6. Persist fees, equity, order type, and drawdown in the paper trade schema.
7. Require at least 50 closed paper trades or 14 days before interpreting results.

### Exit criteria

- If the next clean run fails held-out / walk-forward gates, archive it as a negative result.
- If it passes offline but paper fails, keep it paper-only and document the train-serve gap.
- Do not put live capital behind RL until paper gates and fundability review pass.

---

## Track C — Kalshi hypothesis reset

### Goal

Keep the Kalshi infrastructure, but stop treating old strategies as candidates.

### Required rules for any future Kalshi hypothesis

Every new Kalshi `06` must include:

- executable bid / ask edge definition,
- spread cap,
- market activity floor,
- holding-period cap,
- bankroll capacity check,
- settlement and recycling cadence,
- explicit mechanism explanation.

### Candidate direction

If Kalshi is reopened, prefer short-cycle mechanical / structural questions over model-fair-value claims:

- settlement microstructure,
- stale quotes with executable spread,
- same-event cross-contract constraints with actual ask / bid economics,
- event-window reactions with fast settlement.

Do not revive the lognormal crypto bucket model without a new mechanism and a new pre-registration.

---

## First 30 days

### Week 1

- Start H-PERP-003 daily append capture.
- Add a run log / provenance extension for repeated pulls.
- Check OKX authenticated / official historical data options.
- Update stale README and fleet model references.

### Week 2

- Add external CSV ingest path for H-PERP-003 if OKX archive data is obtained.
- Add held-out evaluator split for RL.
- Add paper trade schema fields required for real promotion metrics.

### Weeks 3-4

- If H-PERP-003 D1 is solved, run Phase 4 and write the verdict.
- If not solved, keep capture running and decide whether to begin H-PERP-002.
- Train RL only after validation gates are in place.

---

## Stop-doing list

1. Do not run fleet live while `fleet.yaml` references a missing model.
2. Do not run Kalshi paper trading for falsified strategies as if it were research progress.
3. Do not tune H-SPOT-001 windows after its Phase 4 FAIL.
4. Do not reopen H-KALS-001 / 001b without executable economics.
5. Do not treat a training-reward checkpoint as deployable.

---

## Definition of success

A good outcome over the next cycle is not necessarily live profit. A good outcome is one of:

- H-PERP-003 clears data and fails cleanly.
- H-PERP-003 clears data and passes, unlocking paper protocol.
- RL gets a credible validation path and fails honestly.
- Kalshi remains disabled until a real new mechanism appears.

This is how the project becomes a reliable alpha-discovery machine instead of a collection of hopeful bots.

---

## 2026-05-04 update — execution-plan close-out

The comprehensive execution plan derived from this audit (Tracks A, B, C
in [.cursor/plans/rlbotpm_comprehensive_execution_plan_*.plan.md](../.cursor/plans/))
landed in this commit cycle:

- **Track A (H-PERP-003):** all data + backtest hardening complete, and
  the D1 blocker is now cleared via the first-party OKX historical funding
  archive. The daily-capture cron, auth probe, vendor evaluator, ingest path,
  G6 amendment, and Phase 4/5 templates are in place. The new
  [download_okx_funding_archive.py](datasets/H-PERP-003/download_okx_funding_archive.py)
  queries OKX's public historical-data endpoint and normalizes the monthly
  ZIPs for [ingest_external_csv.py](datasets/H-PERP-003/ingest_external_csv.py).
  Phase 4 is **PASS** in
  [07_backtest_results_H-PERP-003.md](07_backtest_results_H-PERP-003.md)
  (`days_span=369.0`, Sharpe 7.97, PF 1.88, 2x-fee stress positive,
  random sign-flip placebo beat-rate 0.0). Next gate is Phase 5 paper
  logging, not live capital.

- **Track B (RL repair):** all six audit-03 §B items complete. Run 174
  is **pre-registered** in [RL_RUN_174_PLAN.md](RL_RUN_174_PLAN.md) and
  gated by [bot/scripts/launch_run_174.py](../bot/scripts/launch_run_174.py),
  which refuses to start training unless every B1–B5 fix is verifiably in
  place. Training kicked off 2026-05-04 and early-stopped at step
  240,000 with one deployable artifact (`best_model_run_176.zip`).
  Stage 1 evaluation is now closed in
  [RL_RUN_174_RESULTS.md](RL_RUN_174_RESULTS.md) — verdict
  **FAIL → DO NOT PROMOTE** (walk-forward 1/3 profitable folds, mean
  Sharpe -1.85). The artifact is archived; `fleet.yaml` remains
  `rl_crypto.enabled=false`. The audit-03 fixes worked exactly as
  designed: they surfaced an honest negative result before paper
  trading instead of after.

  Two follow-up TODOs surfaced by run 174 results §3 (kept here so they
  are not lost):
  - Add an artifact-level walk-forward mode (current harness re-trains
    per fold, so it cannot directly answer "does this checkpoint
    generalize?").
  - The pre-registration template (§C2) should compute and enforce the
    minimum `held_out_days` and `min_data_days` for the configured
    `max_steps` and walk-forward fold count, so future plans cannot
    pre-register a value the env will silently override.

- **Track C (productization sketches):** intentionally deferred per the
  plan. Three sketch documents capture the design without committing
  implementation:
  - [C1_FUNDING_DASHBOARD_SKETCH.md](C1_FUNDING_DASHBOARD_SKETCH.md) +
    working backend adapter at
    [bot/scripts/funding_dashboard_snapshot.py](../bot/scripts/funding_dashboard_snapshot.py)
  - [C2_PREREGISTRATION_TEMPLATE_SKETCH.md](C2_PREREGISTRATION_TEMPLATE_SKETCH.md) +
    a working CI guard at
    [bot/scripts/check_preregistration.py](../bot/scripts/check_preregistration.py)
  - [C3_BACKTEST_AS_A_SERVICE_SKETCH.md](C3_BACKTEST_AS_A_SERVICE_SKETCH.md)
    (with the H-PERP-003 backtest extended to accept `--csv`/`--out` so
    the sketch is operational rather than aspirational).

All three C sketches close with explicit "do not build until A or B
delivers" preconditions. This keeps the sunk-cost-momentum failure mode
out of the next cycle.

---

## 2026-07-23 update — Phase 5 path unblocked + closed

Architecture repair after the panel CSV stalled at 2026-05-04 while the
paper logger continued to 2026-07-23 (`n_matched_intervals=0`):

- Synced panel via `daily_capture.py` + `sync_panel_from_paper_log.py`.
- Paper logger dual-writes into the panel; verifier rebuilds along the
  paper fundingTime sequence.
- `fleet.yaml`: `kalshi.enabled=false`, `rl_crypto.enabled=false`,
  `h_perp_003.enabled=true` (paper-only dry-run path).
- Phase 5 verdict in [09_paper_results_H-PERP-003.md](09_paper_results_H-PERP-003.md):
  formula tracking **PASS**, Sharpe drift **FAIL** → **DO NOT PROMOTE**.
- Follow-up (same day): verifier Sharpe was sample-`sqrt(n)` while Phase 4
  uses `sharpe_8h` (`sqrt(365*3)`). Aligned in
  [`check_h_perp_003_tracking.py`](../bot/scripts/check_h_perp_003_tracking.py);
  comparable paper Sharpe ~4.07 vs 7.97 — still **FAIL**. Regression tests in
  [`tests/test_h_perp_003_tracking.py`](../tests/test_h_perp_003_tracking.py).

