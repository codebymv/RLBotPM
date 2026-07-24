# RL Crypto — Run 174 Pre-Registration

> **Status (2026-05-04):** PRE-REGISTERED, training not yet kicked off.
> **Authority:** This document is frozen *before* training starts. Any
> deviation from these gates after training begins is a hindsight bias
> (same failure mode as the original `06` G6 placebo — see
> [research/06_backtest_design_H-PERP-003.md](06_backtest_design_H-PERP-003.md)
> §5 amendment). Pre-register changes here in their own commit per
> [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md) "Pre-registration enforcement".

## 1. Purpose

Train a single fresh RL crypto policy ("run 174") under the `pnl_only`
reward profile, gated by the B1–B5 architecture-repair fixes from
[architecture-audit-03.md](architecture-audit-03.md) §B. Run 173's last
deployable best (`step_130000`) showed a marginal positive expectancy
(+0.005% return, PF 12.0) but only 0.6 trades/episode, and it was scored
under the *old* drawdown guard and *no* held-out split. Run 174 is the
first training that consumes the audit-03 evaluator changes.

## 2. Pre-conditions (must be true at kickoff)

These are gated by `bot/scripts/launch_run_174.py` — it refuses to start
training if any check fails.

| ID | Pre-condition | Source of truth |
|----|---------------|-----------------|
| B1 | `shared/config/fleet.yaml` has `rl_crypto.enabled=false` AND `rl_crypto.model=null` | YAML diff |
| B1 | README + LIVE_TRADING_READINESS + BOT_ARCHITECTURE banners point to audit | grep for "Run 170" stale claims returns 0 hits |
| B2 | `CheckpointCallback` writes `reward_best_run_*` (NOT `best_model_run_*`) | `bot/src/training/callbacks.py` header table |
| B2 | `EarlyStoppingCallback` is the sole writer of `best_model_run_*` | same table |
| B3 | `EarlyStoppingCallback.__init__` accepts `eval_dataset_split`, `held_out_days`, `drawdown_score_floor` | introspection |
| B3 | `evaluator.py` supports `dataset_split in {"all","train","holdout"}` | introspection |
| B3 | `rl_promotion_check.py` requires `--walk-forward` results for `g_wf` | grep for `g_wf` |
| B4 | `rl_crypto_trades` table has columns `fee_usdt`, `cumulative_equity`, `peak_equity`, `order_type`, `fill_was_maker` | `PRAGMA table_info` |
| B5 | `tests/test_train_serve_parity.py` passes (4 hard tests; 1 documented xfail) | pytest exit 0 |

## 3. Training configuration (frozen)

All of the following are read from existing config files. No
hyperparameter overrides are introduced for this run.

| Knob | Value | Source |
|------|-------|--------|
| `reward_profile` | `pnl_only` | `shared/config/model_config.yaml` env block |
| `policy_type` | `MlpPolicy` | `shared/config/model_config.yaml` ppo block |
| `total_timesteps` | 300,000 | `shared/config/model_config.yaml` training block |
| `eval_frequency` | 10,000 | same |
| `checkpoint_frequency` | 10,000 | same |
| `early_stopping.metric` | `golden_score` | same |
| `early_stopping.min_profit_factor` | 1.0 | same |
| `early_stopping.min_total_return` | 0.0 | same |
| `early_stopping.max_drawdown` | 0.25 | same |
| `early_stopping.max_fees_pct_of_gross_pnl` | 0.50 | same |
| `eval_dataset_split` | `train` | audit-03 §B3 default |
| `held_out_days` | 7 | same |
| `drawdown_score_floor` | 0.0 | same |
| Fees | from `risk_config.yaml` (maker 0.05% / taker 0.10%, `default_order_type=maker`) | parity-tested in B5 |
| Data window | `REQUIRE_HISTORICAL_DAYS` from `.env` (no override) | `bot/src/core/config.py` |

## 4. Pass/fail gates for the trained checkpoint (Stage 1)

The `EarlyStoppingCallback` must save **at least one** `best_model_run_174`
file (i.e., a checkpoint that simultaneously sets a new golden_score AND
passes every hard gate in `_passes_hard_gates`).

If zero `best_model_run_174` files are produced, run 174 is recorded as a
**negative result** in [RL_PROFITABILITY_AUDIT.md](../RL_PROFITABILITY_AUDIT.md)
§9, and we DO NOT promote the `eval_best_run_174` or `reward_best_run_174`
fallback artifacts to paper. The decision to retrain or pivot is made
explicitly; we do not deploy a model that failed offline gates.

If at least one `best_model_run_174` file exists:

| Gate | Threshold | Source |
|------|-----------|--------|
| Held-out total_return | > 0% | re-run evaluator with `--dataset-split holdout` |
| Held-out profit_factor | > 1.0 | same |
| Held-out trades_per_episode | >= 0.5 | same (audit-03: tighter than the audit's 0.1 because run 173 plateaued at 0.6) |
| Walk-forward folds positive | >= 3/4 | `python bot/main.py walk-forward --folds 4` |
| Walk-forward mean Sharpe | > 0 | same |

If held-out OR walk-forward fails, the candidate is **archived as a
negative result**. We do not paper-trade a model that fails out-of-sample.

## 5. Paper-trading protocol (Stage 2)

Only triggered if Stage 1 passes. Mirrors `RL_PROFITABILITY_AUDIT.md` §7.2.

| Gate | Threshold |
|------|-----------|
| Closed trades | >= 50 |
| Sessions | >= 5 |
| Calendar days | >= 14 |
| Win rate | >= 45% |
| Lifetime PnL | > $0 |
| Profit factor | >= 1.2 |
| True max drawdown | < 10% |
| Fee drag | < 25% of gross PnL |
| Worst session PnL | > -$30 (on $1000 starting capital) |
| No catastrophic loss | No single trade loss > 3% of capital |

Paper trading uses `python bot/main.py rl-paper-trade --model
bot/models/best_model_run_174 --duration 0` running continuously.
Promotion is checked daily via `python bot/scripts/rl_promotion_check.py`.

## 6. Decision rule

| Stage 1 | Stage 2 (after >=14d / >=50 trades) | Outcome |
|---------|--------------------------------------|---------|
| FAIL | n/a | Archive run 174; revisit reward / data / hyperparameters as a NEW pre-registered run 175 |
| PASS | FAIL | Archive paper run as evidence the offline gates do not transfer; document train-paper drift before run 175 |
| PASS | PASS | Eligible for `10_fundability_review` and live-capital discussion. Not auto-promoted. |

## 7. Cross-cutting hygiene

- This file is **frozen** before kickoff. Any edit after `python
  bot/scripts/launch_run_174.py` succeeds is a pre-registration violation.
- Stage 1 evaluation results land in a NEW file
  `RL_RUN_174_RESULTS.md` (not appended here) — same separation as
  `06_*` vs `07_*`.
- The fleet config remains `rl_crypto.enabled=false` and
  `rl_crypto.model=null` until BOTH Stage 1 and Stage 2 pass.
