# RL Crypto — Run 174 Stage 1 Results

> **Status (2026-05-04):** Stage 1 evaluation complete. **Verdict: DO NOT
> PROMOTE.** Run 174 is archived as a negative result.
> **Authority:** [RL_RUN_174_PLAN.md](RL_RUN_174_PLAN.md) (frozen pre-registration)
> **Successor:** No run 175 has been pre-registered yet.

This file is the post-registration counterpart to
[RL_RUN_174_PLAN.md](RL_RUN_174_PLAN.md), separated per the same
`06 → 07` discipline used for `H-PERP-003`. The plan was **not edited**
after kickoff.

---

## 1. Training summary

| Field | Value |
|-------|-------|
| Plan doc | [RL_RUN_174_PLAN.md](RL_RUN_174_PLAN.md) |
| Launcher | `bot/scripts/launch_run_174.py --commit --episodes 600` |
| DB run id | 176 (the planning name "run 174" is preserved; DB autoincrement assigned 176) |
| Reward profile | `pnl_only` (verified by launcher pre-flight + log line `Using reward profile: pnl_only`) |
| Pre-flight gates | All 10 PASS (B1–B5 + reward profile + plan present + data freshness) |
| Episodes requested | 600 |
| Total timesteps requested | 300,000 |
| Total timesteps actually run | 240,000 (early-stopped) |
| Wall clock | ~18 min on CPU |
| Early-stop reason | `EarlyStoppingCallback` patience exhausted at step 240,000 (20/20 evals without improving the high-water `golden_score=2.7136`) |
| Final-step eval | INACTIVE (0.0 trades/episode), `golden_score=0.0000` |

The `2.7136` high-water mark was set at an earlier checkpoint that did
trade non-trivially; from approximately step 30k–50k onward the policy
collapsed to NO_ACTION under `pnl_only`, exactly as anticipated in
[RL_PROFITABILITY_AUDIT.md](../RL_PROFITABILITY_AUDIT.md) §6.2 ("pnl_only
strips entry incentive, optimizer prefers do-nothing").

Artifacts produced (see `bot/models/`):

- `best_model_run_176.zip` — **deployable artifact**, written exclusively
  by `EarlyStoppingCallback` (audit-03 §B2 authority rule). Exactly one
  file. **This is the file Stage 1 evaluates below.**
- `eval_best_run_176.zip` — diagnostic snapshot of the eval-period high
  water (same checkpoint in this run).
- `reward_best_run_176_step_*.zip` — 12 diagnostic snapshots written by
  `CheckpointCallback` based on training-reward, **not** consulted for
  promotion (audit-03 §B2).
- `checkpoint_run_176_step_*.zip` — 24 periodic snapshots (10k cadence).
- `early_stop_eval_run_176_step_*.zip` — 24 pre-eval snapshots
  (SB3 idiom; safe to GC after the run).
- `final_run_176.zip` + `.norm.pkl` — end-of-training snapshot.

Audit-03 §B2 invariant **satisfied**: only one `best_model_run_*` file
exists, and it was written by `EarlyStoppingCallback`.

---

## 2. Stage 1 gate evaluation

### 2.1 Held-out evaluation (`dataset_split=holdout`)

| Field | Planned | Actual | Pass? |
|-------|---------|--------|-------|
| `held_out_days` | 7 | **30** (deviation, see §3.1) | n/a |
| Episodes | (unspecified) | 20 | n/a |
| `total_return` | > 0 | +0.059% | PASS |
| `profit_factor` | > 1.0 | 0.0 (denom = 0; every closed trade was a winner) | **AMBIGUOUS** |
| `trades_per_episode` | >= 0.5 | 0.9 | PASS |
| `max_drawdown` (advisory) | < 25% | 0.20% | PASS |
| `sharpe_ratio` (advisory) | n/a | +3.87 | n/a |
| `flat_ratio` (advisory) | n/a | 96.5% | n/a |
| `fees_pct_of_gross_pnl` (advisory) | < 50% | 16.7% | PASS |

The held-out gates **technically pass**, but two caveats demote the
result to "weak evidence":

1. **`held_out_days` deviation (30 vs planned 7).** The pre-registration
   specified 7 days, but the production env has a hard floor of
   `max_steps + seq_length + 1 = 502` rows, and 7 days × 24 1h candles
   = 168 rows. Either the plan was wrong or the env needs to gain a
   sub-window split. We took the larger window because it matches the
   audit-03 `held_out_days` default and produces a real number, but the
   deviation itself is a Stage-1 yellow flag.
2. **`profit_factor` is degenerate.** Across 18 closed trades on the
   30-day window, gross loss = 0, so PF = `gross_profit / 0` collapses
   to 0 in our evaluator instead of `+inf`. The intent of the gate
   ("profits dominate losses") is technically met, but a single losing
   trade in a longer window would change the verdict materially. We do
   not credit this as a robust pass.

### 2.2 Walk-forward evaluation (`bot/main.py walk-forward`)

Run with `REQUIRE_HISTORICAL_DAYS=180`, `--folds 3 --train-days 30
--test-days 14 --train-episodes 50 --eval-episodes 10`. The harness
**re-trains a fresh policy per fold**, so this is a *policy-family*
generalization test, not an artifact-level test of `best_model_run_176`.
The plan called for `--folds 4`; the dataset depth (BTC-USD 1h goes
back to 2025-08-09) only supports 3 walk-forward folds without
overlapping the eval window with feature warmup, so we ran 3.

| Field | Planned | Actual | Pass? |
|-------|---------|--------|-------|
| Folds completed | 4 | 3 (data-depth deviation, see §3.2) | n/a |
| Profitable folds | >= 75% | 33% (1 of 3) | **FAIL** |
| Mean Sharpe | > 0 (plan §4 says >= 0.5) | -1.85 | **FAIL** |
| Mean return | n/a | -0.09% (std 0.48%) | n/a |
| Worst fold drawdown (advisory) | < 5% | 0.96% | PASS |

Per-fold detail:

| Fold | Window | Return | Sharpe | DD | WinRate | Avg trade PnL |
|------|--------|--------|--------|-----|---------|---------------|
| 1 | 2025-08-09 → 2025-09-22 | -0.05% | -3.16 | 0.16% | 43.3% | -$0.15 |
| 2 | 2025-08-23 → 2025-10-06 | +0.48% | +7.07 | 0.27% | 86.7% | +$1.45 |
| 3 | 2025-09-06 → 2025-10-20 | -0.71% | -9.46 | 0.96% | 22.0% | -$0.61 |

Fold 2 alone passes. Fold 1 and Fold 3 fail. Variance across folds
(0.48% return std on a -0.09% mean) is large relative to the edge,
which is the canonical signature of an "overfit-to-window" policy
family.

### 2.3 Combined Stage 1 verdict

Per [RL_RUN_174_PLAN.md](RL_RUN_174_PLAN.md) §6 decision rule:

> "If held-out OR walk-forward fails, the candidate is archived as a
> negative result. We do not paper-trade a model that fails
> out-of-sample."

Walk-forward **FAIL** (1/3 folds vs ≥75% required, mean Sharpe -1.85 vs
> 0 required) → **Stage 1 FAIL** → **DO NOT PROMOTE**.

Held-out is a soft pass that does not rescue the verdict. The
`held_out_days` deviation, the degenerate `profit_factor`, and the 96.5%
flat ratio all reinforce that the held-out "win" is consistent with the
policy having learned "do almost nothing" rather than "find edge".

---

## 3. Deviations from the pre-registration

These are recorded honestly per the pre-registration discipline. None
were necessary to "save" the run; the verdict is FAIL regardless.

### 3.1 `held_out_days` 7 → 30

The pre-reg specified `held_out_days=7` but the env's
`_prepare_data` requires `max_steps + seq_length + 1 = 502` rows in the
holdout slice. 7 days × 24 1h candles = 168 rows < 502. We ran with
`held_out_days=30` because that is the next round value above the floor
that still meaningfully separates train and eval windows.

**Fix for next run:** the pre-registration template (`C2`) should
default `held_out_days >= ceil((max_steps + seq_len + 1) / candles_per_day)`
or document the env's row-floor explicitly. For 1h bars and
`max_steps=500` that is **22 days minimum**.

### 3.2 Walk-forward 4 → 3 folds

The plan asked for `--folds 4`. BTC-USD 1h history starts 2025-08-09
(~268d as of 2026-05-04), but the env only loads
`REQUIRE_HISTORICAL_DAYS=180` days back, which we temporarily bumped
from the configured 90. With train=30d + test=14d per fold and the
non-overlapping requirement (`run_walk_forward` slides the window
14d/fold), 4 folds need 30 + 4×14 = 86d of clean data **plus** feature
warmup, which we cannot guarantee with 180d when `qa_report` flags
gaps. 3 folds fit cleanly inside fold 1 = 2025-08-09 → 2025-09-22.

**Fix for next run:** either (a) backfill BTC-USD 1h to a full year
before the next run is pre-registered, or (b) the pre-reg template
should compute `min_data_days = train_days + folds * test_days +
feature_warmup_days` and refuse to start otherwise (analogous to the
`launch_run_174.py` data-freshness check we added for training).

### 3.3 DB run id 176 vs planning name "run 174"

The trainer's DB autoincrement assigned id 176 because runs 174 and 175
were placeholder rows from prior aborted attempts. We keep the
**planning** name "run 174" in all docs because it matches the audit-03
§B6 todo and the launcher script name; the **DB** id 176 is the
ground truth for joining `rl_training_runs` to `rl_crypto_trades`.

---

## 4. What we learned

1. **The audit-03 architectural fixes worked exactly as intended.** The
   launcher refused to start training until B1–B5 were green. The
   `EarlyStoppingCallback` was the sole writer of the deployable
   artifact (B2 authority). The held-out split actually ran with a
   different time slice than training (B3). The training data freshness
   check caught a stale-data bug *before* training started instead of
   after 18 minutes of wasted compute.
2. **`pnl_only` collapses to NO_ACTION on this dataset, on this
   feature set, with this reward shaping.** This was the audit's
   prediction (`RL_PROFITABILITY_AUDIT.md` §6.2). The high-water
   `golden_score=2.7136` was set in the first ~30k–50k steps before the
   collapse, which is consistent with "exploration found a few lucky
   trades, then exploitation chose to never trade again because every
   subsequent trade in expectation paid more in fees than it earned in
   alpha".
3. **The walk-forward harness needs work.** Re-training per fold means
   we cannot directly answer "does `best_model_run_176` generalize?".
   The current harness answers a different (still useful) question:
   "does the *family* of policies trained on this config generalize?"
   Both questions matter; we should add an artifact-level walk-forward
   mode that loads a fixed `model_path` and only re-evaluates per
   window. **This is now a tracked TODO** in
   [architecture-audit-03.md](architecture-audit-03.md).
4. **Data depth is the binding constraint.** With 90–180d of 1h BTC,
   any honest walk-forward has high variance per fold and refuses to
   produce a clean Sharpe estimate. This is the same conclusion
   `H-PERP-003` reached and is the reason
   [research/datasets/H-PERP-003/daily_capture.py](datasets/H-PERP-003/daily_capture.py)
   exists. We should mirror that capture discipline for spot OHLCV used
   by RL training: a tiny daily cron that appends one day of BTC-USD
   1h candles to the database, so 12 months from now we have a real
   walk-forward dataset.

---

## 5. Decision and next step

**Decision (recorded 2026-05-04):**

- `bot/models/best_model_run_176.zip` is **archived**, not promoted.
  It will not be loaded by `fleet.yaml` and will not run paper trades.
- `shared/config/fleet.yaml` remains `rl_crypto.enabled=false` and
  `rl_crypto.model=null`.
- The audit-03 §B6 task ("Train run 174 under pnl_only profile only
  after B1–B5 complete") is **complete**: we trained, we evaluated, we
  archived per the rule.

**Explicitly NOT decided (do not skip-ahead):**

- We are **not** committing to a "run 175" without a fresh
  pre-registration that names the change being tested. Per
  [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md) "Pre-registration
  enforcement", any single change (reward profile, data window, action
  space, fee model, ...) requires its own design doc + design freeze
  before training begins.

**Suggested candidates for the next pre-registration**, in priority
order (rank, do not implement-and-train):

1. **Reward shaping vs `pnl_only`.** Add a small entry-incentive term
   to break the NO_ACTION collapse, then re-run the same Stage 1.
   Hypothesis: 2.7136 high-water came from real, transient edge that
   `pnl_only` discouraged from being exploited.
2. **Longer 1h history.** Stand up a daily-capture cron mirroring
   `H-PERP-003`'s `daily_capture.py` so that by ~mid-2027 we have
   12 months of clean BTC-USD 1h and can run a real 4-fold
   walk-forward with the planned 7-day held-out window.
3. **Lower max_steps so 7-day held-out actually fits.** Avoids the
   §3.1 deviation but trades episode length for split discipline.
4. **Multi-asset training.** Currently `DATA_SYMBOLS=BTC-USD` only.
   The DB has ETH-USD, SOL-USD with comparable depth. Hypothesis:
   diversification gives the policy more chances to find non-degenerate
   edge per training step.

Whichever of these we pick, the next run gets its own
`RL_RUN_175_PLAN.md` and its own `RL_RUN_175_RESULTS.md`. We do not
edit this file after today.
