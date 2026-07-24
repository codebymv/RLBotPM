# Architecture Audit 01 — RL Crypto Spot Bot

> Bot: Coinbase spot crypto trader using MaskablePPO.  
> Main files: `bot/src/environment/gym_env.py`, `bot/src/training/trainer.py`, `bot/src/training/evaluator.py`, `bot/src/training/callbacks.py`, `bot/src/execution/live_rl_trader.py`, `shared/config/reward_config.yaml`, `shared/config/fleet.yaml`.

---

## Verdict

The RL crypto bot is **not currently deployable** as a profit-seeking system.

The architecture is reusable, but the current training / promotion chain does not yet produce a model with clean out-of-sample evidence. The configured fleet model path also points to a missing artifact.

---

## Critical findings

### 1. Configured model artifact is missing

`shared/config/fleet.yaml` references:

```yaml
rl_crypto:
  model: "models/best_model_run_174.zip"
```

But `bot/models/` currently contains no `best_model_run_174.zip` artifact. The directory only showed `run173_checkpoint_sweep.json` and `.gitkeep` during this audit.

**Impact:** any fleet startup that tries to load the configured RL model will fail before trading logic starts.

### 2. Reward is still too far from executable PnL

The prior `RL_PROFITABILITY_AUDIT.md` found that the reward function is roughly:

- about 40% real PnL signal
- about 60% auxiliary shaping

High-risk terms include:

- `episode_pnl_bonus_scale`
- `sell_profit_bonus`
- `sell_profit_bonus_scale`
- `manual_exit_bonus`
- `sharpe_bonus_scale`
- `idle_base`

The `pnl_only` profile already exists in `shared/config/reward_config.yaml`, but no current model artifact has been proven under that profile.

### 3. Held-out validation is not mandatory

`walk_forward.py` exists, but it is a standalone command rather than a hard gate in the model promotion path.

Training and evaluator logic have historically reused the same market window with randomized starts. That is not enough to treat a model as out-of-sample profitable.

### 4. Two "best model" definitions compete

The prior audit identified two checkpoint authorities:

- `CheckpointCallback`: best by training reward
- `EarlyStoppingCallback`: best by evaluator metrics / golden score

Any deployment rule must make evaluator-passing checkpoints canonical and treat training-reward winners as diagnostics only.

### 5. Paper evidence is too thin

The prior paper record for the deployed generation showed:

- 3 closed trades
- 2 wins / 1 loss
- lifetime PnL around `-$0.06`
- profit factor `0.87`
- BTC-only trading
- 71% `NO_ACTION`

This is not enough evidence to infer edge. The behavior looks like a highly conservative model with uniform exit timing around `min_hold_steps`.

### 6. Promotion metrics cannot be fully enforced yet

The paper trade schema lacks enough fields to enforce a serious live-readiness gate:

- per-trade fee / cost
- cumulative equity at close
- peak-to-trough drawdown
- order type / fill model

Any `fee_drag` or drawdown gate that cannot read these fields is not yet real.

---

## Recommended fixes before any new RL paper run

1. Update `fleet.yaml` only after a real model artifact exists.
2. Make evaluator-passing checkpoints the only deployable `best_model_run_*`.
3. Add a held-out date split to the evaluator.
4. Require walk-forward or held-out validation before paper deployment.
5. Train a fresh run using the `pnl_only` reward profile.
6. Add fee / equity / drawdown fields to `rl_crypto_trades`.
7. Re-run paper only after the above and require at least 50 closed trades or 14 calendar days before interpreting results.

---

## Recommended status

**Paused for architecture repair.**

The RL bot should not be the primary profit-search track until the model artifact, validation, and promotion chain are fixed. It remains a useful research platform after those repairs.

