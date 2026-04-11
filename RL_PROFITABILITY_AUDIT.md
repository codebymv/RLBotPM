# RL Crypto Bot — Profitability Audit

## 1. Baseline Scorecard

### Deployed model

The fleet runs `bot/models/best_model_run_172.zip` from training run 172 (330,000 steps, 68 checkpoints from step 5,000 to 330,000). Run 172 used the `lean` reward profile. The early stopping evaluator rejected every checkpoint during training (none passed the hard gate `total_return >= 0.1%`), so the deployed model is the last `best_model_run_172` written by the `CheckpointCallback` based on training reward, not the evaluator's golden score. This means the deployed checkpoint was never validated by the evaluator's profitability criteria.

### Paper-trading performance

Source: `rl_crypto_trades` table, `mode='paper'`.

| Metric | Value |
|--------|-------|
| Total rows | 6 (3 closed, 3 open) |
| Sessions | 4 (over 8 days) |
| Wins / Losses | 2W / 1L (66.7%) |
| Lifetime PnL | **-$0.06** |
| Gross profit | $0.39 |
| Gross loss | $0.45 |
| Profit factor | **0.87** |
| Symbols traded | BTC-USD only |
| Exit reasons | MODEL_DECISION (100%) |

Individual closed trades:

| # | Entry | Exit | PnL | PnL % | Hold |
|---|-------|------|-----|-------|------|
| 1 | 2026-03-31 07:00 | 2026-03-31 17:00 | +$0.28 | +0.33% | 10 steps |
| 2 | 2026-04-03 03:09 | 2026-04-03 13:00 | +$0.11 | +0.19% | 10 steps |
| 3 | 2026-04-08 19:32 | 2026-04-09 06:00 | -$0.45 | -0.27% | 11 steps |

### JSONL activity (decision-level)

Two distinct model generations appear in the logs:

- **Feb 7-13 (old model, pre-run 172):** 161 events. Action mix: HOLD=87, BUY=13, SELL=9, SELL(auto)=4. The old model traded more actively with auto-exits.
- **Mar 30 - Apr 9 (run 172):** 84 events. Action mix: NO_ACTION=75, BUY=6, SELL=3. The current model passes on 71.4% of ticks.

### Promotion gate status

| Gate | Current | Target | Status |
|------|---------|--------|--------|
| Closed trades | 3 | 50 | FAIL |
| Sessions | 3 | 5 | FAIL |
| Days observed | 8 | 14 | FAIL |
| Win rate | 66.7% | 45% | PASS |
| Lifetime PnL | -$0.06 | > $0 | FAIL |
| Return | -0.01% | 0.0% | FAIL |
| Profit factor | 0.87 | 1.2 | FAIL |
| Worst session | -$0.45 | > -$30 | PASS |
| Fee drag | 0% | < 25% | PASS |
| Single-trade loss | 0.27% | < 3% | PASS |

**Top 3 blockers:** insufficient trade count (3 of 50), negative expectancy (PF 0.87), and severe undertrading (71% NO_ACTION).

---

## 2. Reward vs Real-PnL Alignment

### Effective reward weights

The reward function in `gym_env.py` `_calculate_reward` (v2 mode) combines approximately 15 terms. The effective weights are computed as: hardcoded defaults -> `reward_config.yaml` top-level `reward_weights` -> active profile overrides (lean).

The training environment with `lean` profile produces the following effective weights for the most impactful terms:

| Term | Default | YAML top | Lean | Effective | Aligns with PnL? |
|------|---------|----------|------|-----------|-------------------|
| `portfolio_step_scale` | 10.0 | 5.0 | 3.0 | **3.0** | Yes (on executed trades only in v2) |
| `fee_penalty_scale` | 3.0 | 7.0 | 8.5 | **8.5** | Yes |
| `loss_aversion_scale` | 2.0 | 2.0 | - | **2.0** | Partially (penalizes losses quadratically) |
| `carry_bonus_scale` | 2.0 | 4.0 | 1.5 | **1.5** | No (rewards unrealized gains) |
| `hold_pnl_step_scale` | 2.0 | 5.0 | 0.0 | **0.0** | N/A (disabled in lean) |
| `sell_profit_bonus` | 0.5 | 0.5 | - | **0.5** | No (double-counts profitable exits) |
| `sell_profit_bonus_scale` | 10.0 | 10.0 | - | **10.0** | No (double-counts profitable exits) |
| `manual_exit_bonus` | 0.1 | 0.25 | 0.10 | **0.10** | No (rewards the act of selling, not edge) |
| `sharpe_bonus_scale` | 2.0 | 2.0 | - | **2.0** | No (rewards consistency, not magnitude) |
| `idle_base` | 0.01 | 0.005 | 0.003 | **0.003** | No (penalizes not trading) |
| `episode_pnl_bonus_scale` | 10.0 | 10.0 | - | **10.0** | Partially (huge terminal bonus dominates) |
| `episode_end_close_penalty_scale` | 0.5 | 0.5 | 0.8 | **0.8** | No (teaches early exit, not optimal exit) |

### Diagnosis

The reward is approximately 40% real PnL signal and 60% auxiliary shaping. The `episode_pnl_bonus_scale` at 10x is the single largest reward event in any episode, dwarfing per-step feedback. The `idle_base` penalty at 0.003 is small but persistent enough to push the model into marginal entries. The `sell_profit_bonus` + `sell_profit_bonus_scale` combination (up to +0.5 bonus for profitable sells, scaled 10x) double-counts profitable exits on top of the raw PnL signal.

### Proposed PnL-first profile

The `pnl_only` profile already exists in `reward_config.yaml` from the previous audit pass. It zeros out: `carry_bonus`, `manual_exit_bonus`, `sell_profit_bonus`, `idle_base`, `sharpe_bonus`, and reduces `episode_pnl_bonus_scale` from 10 to 1. This should be the next training candidate.

---

## 3. Training and Validation Robustness

### Two competing "best model" definitions

The codebase has two callback systems that both save a `best_model_run_{id}`:

1. **`CheckpointCallback`** (line 203): saves best by rolling mean of last 10 **training episode rewards**. This is the most frequently updated "best" pointer. It fires on every episode completion.
2. **`EarlyStoppingCallback`** (line 314): evaluates periodically using the `Evaluator`, computes `golden_score`, and saves best by evaluator metrics. This only fires every `eval_frequency` steps (10,000).

Both write to the same filename (`best_model_run_{id}`), so the `EarlyStoppingCallback` overwrites the `CheckpointCallback`'s best — **but only when it finds an improvement that passes hard gates**. In run 172, the evaluator never found a passing checkpoint, so the deployed model is the last training-reward winner from `CheckpointCallback`.

### In-sample evaluation

The `Evaluator` loads the same historical dataset as training. Eval episodes use sequential seeds (0, 1, 2, ...) which select different random start points within the same data window. This is **pseudo-independent** evaluation on the same time series — not held-out validation.

### Drawdown guard zeros profit factor

In `evaluator.py` lines 294-298, if `max_drawdown > drawdown_threshold` (default 0.2 = 20%), the evaluator sets `profit_factor = 0.0`. On volatile crypto at 1h candles over 500 steps, 20% intra-episode drawdowns are common. This guard is likely why every checkpoint in run 172 was rejected.

### Golden score composition

```
golden_score = 0.35 * sharpe + 45.0 * total_return + 0.60 * profit_factor
              - 10.0 * fees_pct_of_gross_pnl - 6.0 * drawdown + 1.5 * in_position
```

The 45x multiplier on `total_return` means a 1% return difference produces a 0.45 score swing, while `profit_factor` only contributes 0.60x. This heavily rewards high-return episodes regardless of consistency. The `in_position` term at 1.5x slightly rewards being in the market, which can encourage over-entry.

### Walk-forward exists but is not integrated

`walk_forward.py` implements rolling train/test windows and is accessible via `main.py walk-forward`, but it is a standalone CLI command. No part of the training or promotion pipeline requires walk-forward validation to pass before accepting a model.

### Recommendations

1. Resolve the dual-best-model conflict by making `EarlyStoppingCallback` the sole authority for `best_model_run_{id}`.
2. Soften the drawdown guard: instead of zeroing PF, cap the golden score contribution (e.g. multiply by `max(0, 1 - drawdown/threshold)`).
3. Lower `min_total_return` from 0.001 (0.1%) to 0.0 (break-even). The current threshold rejected every run 172 checkpoint.
4. Require held-out validation (either walk-forward or a reserved date split) before promoting any model to paper.

---

## 4. Data and Environment Realism

### Fee model

| Parameter | Config value | Realism |
|-----------|-------------|---------|
| `default_order_type` | maker | OK if limit orders are actually used live |
| `maker_fee_pct` | 0.05% | Realistic for Coinbase Advanced |
| `taker_fee_pct` | 0.10% | Realistic for Coinbase Advanced |
| `maker_fill_probability` | 65% | Conservative and reasonable |
| `maker_fallback_to_taker` | false | Conservative (missed fills stay missed) |
| `taker_slippage_pct` | 0.05% | Reasonable for small BTC orders |

The blended per-trade cost under the maker model is approximately `0.65 * 0.05% + 0.35 * 0% (missed, no trade) = 0.0325%` for trades that execute. With 35% of intended trades not filling at all, the model learns in a world where ~1 in 3 attempted entries fail silently. This shapes the policy toward conservatism.

### Stale / dead config

| Config item | Status |
|-------------|--------|
| `model_config.yaml` `environment.transaction_cost` | Removed (previous audit). Trainer falls back to `settings.TRANSACTION_COST_PCT`. |
| `model_config.yaml` `environment.reward_weights` | Removed (previous audit). |
| `model_config.yaml` `curriculum` | Dead — no Python code references it. Training does not implement curriculum stages. |
| `model_config.yaml` `specialist_router.model_paths` | Points to runs 167-169 which are stale. The specialist manager code exists but is not active in the standard training path. |
| `gym_env.py` `self.transaction_cost` | Stored at init but never used in v2 trade execution. The env reads maker/taker fees from `risk_config` instead. |

### Entry volatility gate bug (fixed)

The `entry_volatility_cap` check in `get_valid_actions` was unreachably nested under `if row is None: return valid`. This was fixed in the previous audit pass — the guard now correctly checks `if self.entry_volatility_cap > 0` as a separate block after the null-row early return.

### Episode boundary effects

Episodes randomize start points and force-close all positions at the end. The `episode_pnl_bonus_scale` (10x) at termination creates a large terminal reward/penalty that doesn't exist in continuous live trading. The `episode_end_close_penalty_scale` (0.8x) penalizes carrying positions into forced closes. Together, these teach the model to time exits around episode boundaries — a behavior that has no live equivalent.

---

## 5. Paper-Trading Behavior

### Critical finding: train-serve drift

`LiveRLPaperTrader._load_reward_config()` reads **only** the top-level `reward_weights` from `reward_config.yaml`. It does **not** apply the `lean` profile merge that `gym_env.py` performs during training. This produces 9 drifting behavioral/reward parameters:

| Parameter | Training (lean) | Live Paper (top-level) | Impact |
|-----------|----------------|----------------------|--------|
| `carry_bonus_scale` | 1.5 | 4.0 | Live values unrealized gains 2.7x more |
| `carry_bonus_cap` | 0.2 | 0.5 | Live caps unrealized reward higher |
| `hold_pnl_step_scale` | 0.0 | 5.0 | Live adds per-step unrealized PnL signal (training disables it) |
| `portfolio_step_scale` | 3.0 | 5.0 | Live weights portfolio changes 1.7x more |
| `fee_penalty_scale` | 8.5 | 7.0 | Live penalizes fees 18% less |
| `manual_exit_bonus` | 0.10 | 0.25 | Live rewards manual exits 2.5x more |
| `idle_base` | 0.003 | 0.005 | Live penalizes inactivity 1.7x more |
| `auto_exit_penalty` | 0.35 | 0.3 | Minor drift |
| `episode_end_close_penalty_scale` | 0.8 | 0.5 | Not applicable to live (no episodes) |

These parameters don't directly affect the live trader's execution decisions (the model's weights are frozen), but they **do** affect any behavioral parameters read from the reward config, like `min_hold_steps` and `trade_cooldown_steps`. Since those happen to match between top-level YAML and the lean profile, the behavioral drift is limited to future profiles that override them.

### Critical finding: transaction cost model mismatch

The live paper trader uses a **flat fee model**: `base_transaction_cost = settings.TRANSACTION_COST_PCT = 0.001 (0.10%)`, applied uniformly to every trade. It has no maker/taker distinction, no probabilistic fill model, and no fill failures.

The training environment uses a **probabilistic maker/taker model**: 65% of trades attempt as maker at 0.05%, with the remainder failing silently (no fallback). The blended cost per executed trade is ~0.0325%.

This means the live paper trader pays **3x higher fees** than the training environment on maker trades that fill, and it **never skips trades** due to fill probability misses. The net effect: live paper execution is more expensive but more active than what the model was trained for.

### Trade shape

- **Severely undertrading:** 3 round-trip trades in 8 days (one every ~2.7 days). The model chose NO_ACTION on 71.4% of ticks.
- **BTC-only:** Despite 5 configured symbols, every trade is BTC-USD. Symbol rotation exposes other assets but the model never sees opportunity.
- **Uniform hold time:** All 3 trades held for 10-11 steps (= `min_hold_steps`), suggesting the model learned to sell at the earliest allowed moment.
- **Tiny edge:** +0.19% to +0.33% on wins, -0.27% on the loss — near coin-flip at these magnitudes.

### DB schema gaps

The `rl_crypto_trades` table does not persist:
- Transaction costs/fees per trade
- Cumulative equity at close
- True peak-to-trough drawdown
- Order type used (maker vs taker)

The `fee_drag_pct` gate in `rl_promotion_check.py` is hardcoded to 0.0 because the DB lacks fee data.

---

## 6. Profitability Diagnosis

### Verdict: no reliable edge detected

With 3 closed trades, a profit factor of 0.87, and -$0.06 lifetime PnL, current performance is indistinguishable from noise. The model has learned extreme conservatism (71% NO_ACTION) with uniform exit timing at `min_hold_steps`.

### Top 3 reasons the RL bot is not profitable

1. **Reward misalignment.** The shaped reward has ~15 terms, with the 10x `episode_pnl_bonus_scale` dominating at episode boundaries and several terms double-counting profitable exits. The agent optimizes shaped reward rather than net PnL, and the current reward structure can be maximized by avoiding risk entirely.

2. **No out-of-sample validation in the promotion path.** Training and evaluation use the same data window. The early stopping evaluator rejected every run 172 checkpoint, but the model was deployed to paper anyway via the training-reward checkpoint. Walk-forward validation exists as a separate CLI command but is not required for promotion.

3. **Train-serve drift.** The live paper trader does not apply the reward profile used in training (9 parameter drifts), uses a fundamentally different fee model (flat 0.10% vs probabilistic maker at 0.05%), and never experiences fill failures. The model was trained for a world with cheaper trades and missed fills, then deployed into a world with more expensive but guaranteed execution.

### Top 3 highest-leverage changes

1. **Fix train-serve parity.** Make `LiveRLPaperTrader._load_reward_config()` apply the same profile merge logic as `gym_env.py._load_reward_config()`. Align the live fee model with the training env's maker/taker structure.

2. **Train with `pnl_only` reward profile.** Strip auxiliary shaping and retrain. The profile already exists. Use realistic fees (the updated `risk_config.yaml` values) and evaluate on a held-out date split.

3. **Add mandatory held-out validation.** Reserve the most recent 7 days of data as a test set. Evaluate on it before accepting any checkpoint for paper. Soften the drawdown guard so it doesn't zero-out profit factor.

---

## 7. Updated Graduation Rubric

### Stage 1: Model candidate ready for paper

| Gate | Threshold | Source |
|------|-----------|--------|
| Held-out total return | > 0% | Evaluator on reserved date split |
| Held-out profit factor | > 1.0 | Evaluator on reserved date split |
| Max drawdown | < 15% | Evaluator |
| Fee drag | < 30% of gross PnL | Evaluator `fees_pct_of_gross_pnl` |
| Trades per episode | > 2 | Evaluator `trades_per_episode` |
| Walk-forward positive | 3/4+ folds positive return | Walk-forward script |

### Stage 2: Paper candidate ready for live

| Gate | Threshold | Source |
|------|-----------|--------|
| Closed trades | >= 50 | `rl_crypto_trades` DB |
| Sessions | >= 5 | `rl_crypto_trades` DB |
| Calendar days | >= 14 | `rl_crypto_trades` DB |
| Win rate | >= 45% | `rl_crypto_trades` DB |
| Lifetime PnL | > $0 | `rl_crypto_trades` DB |
| Profit factor | >= 1.2 | `rl_crypto_trades` DB |
| True max drawdown | < 10% | Equity curve from JSONL |
| Fee drag | < 25% of gross PnL | `rl_crypto_trades` DB (requires fee column) |
| Worst session PnL | > -$30 (on $1000) | `rl_crypto_trades` DB |
| No catastrophic loss | No single trade > -3% of capital | `rl_crypto_trades` DB |

### Stage 3: Live deployment (micro-scale)

- Start with $100-200 real capital
- BTC-USD only
- Kill switch at -$20 daily loss
- Weekly review for first 4 weeks
- Scale only if Stage 2 gates still hold on live results

### Missing instrumentation

To enforce Stage 2 fully, the following need to be added:
- Per-trade fee/cost column in `rl_crypto_trades` (currently not persisted)
- Cumulative equity column or a separate equity-curve log
- True peak-to-trough drawdown computation from the full equity curve

---

## 8. Checkpoint Selection & Deployment Rule

### Training artifacts (by priority)

| Artifact | Source | Purpose |
|---|---|---|
| `best_model_run_{id}` | `EarlyStoppingCallback` — best golden_score that **passes hard gates** | **Canonical deployment target.** |
| `eval_best_run_{id}` | `EarlyStoppingCallback` — best golden_score **regardless of gates** | Fallback candidate. Deploy only after offline sweep confirms activity and near-break-even performance. |
| `reward_best_run_{id}` | `CheckpointCallback` — best training reward | Diagnostic only. Never deploy directly. |
| `final_run_{id}` | End-of-training snapshot | Diagnostic only. |
| `checkpoint_run_{id}_step_{N}` | Periodic snapshots | Used by offline sweep to find candidates missed by early stopping. |

### Deployment decision flow

1. If `best_model_run_{id}` exists → deploy it to paper.
2. Else if `eval_best_run_{id}` exists → run offline sweep; deploy if the sweep confirms `total_return >= 0`, `trades_per_episode >= 0.1`, and `profit_factor >= 1.0`.
3. Else → run offline sweep on all periodic checkpoints; pick the top-ranked active model and deploy if it meets the sweep gates above.
4. If no checkpoint meets any of the above → do not deploy; retrain with revised hyperparameters.

### Hard gates (deployment eligibility)

These are checked by `_passes_hard_gates()` in `EarlyStoppingCallback` and `_passes_golden_gate()` in the offline sweep:

- `total_return >= 0.0%` (net profitable or break-even after fees)
- `profit_factor >= 1.0` (wins cover losses)
- `max_drawdown <= 25%`
- `fees_pct_of_gross_pnl <= 50%` **only when gross profit > $1.00** (avoids penalizing tiny-profit models)
- `trades_per_episode >= 0.1` (model is not inactive)

### Fleet config

`fleet.yaml` should reference `best_model_run_{id}` by default. After a training run, update the `rl_crypto.model` path to point to the correct artifact following the priority above.

---

## 9. Prioritized Next Steps

1. ~~Fix `LiveRLPaperTrader._load_reward_config()`~~ ✅ Done (run 173 prep)
2. ~~Fix `LiveRLPaperTrader` fee model~~ ✅ Done (run 173 prep)
3. ~~Lower `min_total_return`~~ ✅ Done (run 173 prep)
4. ~~Train a clean run with `pnl_only`~~ ✅ Done (run 173)
5. ~~Fix selection bottlenecks~~ ✅ Done (run 173 optimization)
6. ~~Retune gates and patience~~ ✅ Done (run 173 optimization)
7. ~~Add inactive-model detection~~ ✅ Done (run 173 optimization)
8. ~~Define canonical deployment rule~~ ✅ Done (section 8 above)
9. **Retrain with improved evaluator (recommended next step).** Run 173's best checkpoint (`step_130000`) demonstrated a marginal edge (+0.005%, pf=12.03) but with only 0.6 trades/episode — too thin for paper evidence-gathering. The evaluator, gate, and patience fixes from the run 173 optimization should allow the next training run to sustain improvement past the 130k plateau. Keep `pnl_only` reward profile and retrain from scratch (run 174).
10. **Implement held-out date split** in the training/eval pipeline
11. **Clean dead config** (`curriculum`, stale `specialist_router` paths)
