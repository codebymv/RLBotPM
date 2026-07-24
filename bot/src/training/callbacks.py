"""
Custom callbacks for training

Callbacks allow injecting custom logic during training:
- Logging to database
- Saving checkpoints
- Circuit breaker checks
- Custom metrics

These integrate with Stable-Baselines3's callback system.

================================================================================
Checkpoint authority (architecture-audit-03 §B2, 2026-05-04)
================================================================================
Multiple writers used to land in the same `best_model_run_*` namespace, which
caused the deployable artifact to flip silently (audit-01 §1). Authority is now
unified per the table below — DO NOT add a new callback that writes
`best_model_run_*` without first reading [research/architecture-audit-01.md](../../../research/architecture-audit-01.md).

| Filename pattern                          | Written by               | Meaning                                             | Use for                  |
|-------------------------------------------|--------------------------|-----------------------------------------------------|--------------------------|
| `best_model_run_<id>`                     | `EarlyStoppingCallback`  | Best eval metric **AND** all hard gates pass        | **Deployment** (sole)    |
| `eval_best_run_<id>`                      | `EarlyStoppingCallback`  | Best eval metric, gates ignored                     | Retrospective analysis   |
| `reward_best_run_<id>`                    | `CheckpointCallback`     | Best running mean *training* reward                 | Diagnostics only         |
| `reward_best_run_<id>_step_<n>`           | `CheckpointCallback`     | Same, snapshot per "new best" event                 | Diagnostics only         |
| `checkpoint_run_<id>_step_<n>`            | `CheckpointCallback`     | Periodic timestep snapshot                          | Resume training          |
| `early_stop_eval_run_<id>_step_<n>`       | `EarlyStoppingCallback`  | Transient pre-eval snapshot (deleted by SB3)        | Internal                 |

Promotion / fleet code MUST consume `best_model_run_*` and never one of the
diagnostic patterns. `bot/scripts/rl_promotion_check.py` and
[shared/config/fleet.yaml](../../../shared/config/fleet.yaml) follow this.
================================================================================
"""

from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import numpy as np

from ..data import Episode, Trade, ModelCheckpoint, get_db_session
from ..risk import CircuitBreaker
from ..core.logger import get_logger
from ..core.config import get_settings
from .evaluator import Evaluator
from .kalshi_evaluator import KalshiEvaluator


logger = get_logger(__name__)


class CircuitBreakerCallback(BaseCallback):
    """
    Monitors trading and triggers circuit breakers if needed
    
    Pauses training if safety limits are violated.
    """
    
    def __init__(self, training_run_id: int, verbose: int = 0):
        super().__init__(verbose)
        self.training_run_id = training_run_id
        self.circuit_breaker = CircuitBreaker()
        self.settings = get_settings()
        self._last_warning_at = None
    
    def _on_step(self) -> bool:
        """
        Called after each step
        
        Returns:
            False to stop training, True to continue
        """
        info = self.locals.get('infos', [{}])[0]
        trade_result = info.get('trade_result')
        if trade_result and trade_result.get('executed'):
            action_name = trade_result.get('action_name', '')
            if action_name.startswith('SELL') or action_name.startswith('CLOSE'):
                pnl = float(trade_result.get('pnl') or 0.0)
                capital = float(info.get('portfolio_value') or 0.0)
                self.circuit_breaker.record_trade(
                    pnl=pnl,
                    capital=capital,
                    is_win=pnl > 0
                )

        # Check if trading is allowed
        if not self.circuit_breaker.can_trade():
            if self.settings.TRAINING_MODE:
                now = datetime.utcnow()
                if not self._last_warning_at or (now - self._last_warning_at).total_seconds() >= 60:
                    logger.warning("Circuit breaker would trigger (training mode)")
                    self._last_warning_at = now
            else:
                logger.critical("Circuit breaker triggered")
                logger.critical("Stopping training due to circuit breaker")
                return False  # Stop training
        
        return True  # Continue training


class PerformanceLogCallback(BaseCallback):
    """
    Logs performance metrics to database
    
    Records episodes and trades for analysis.
    """
    
    def __init__(
        self,
        training_run_id: int,
        log_frequency: int = 100,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.training_run_id = training_run_id
        self.log_frequency = log_frequency
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.pending_trades = []
        self.settings = get_settings()
    
    def _on_step(self) -> bool:
        """Log performance metrics"""
        reward = float(self.locals.get('rewards', [0])[0])
        self.current_episode_reward += reward

        info = self.locals.get('infos', [{}])[0]
        trade_result = info.get('trade_result')
        if trade_result and trade_result.get('executed'):
            self.pending_trades.append(
                {
                    "action": trade_result.get("action"),
                    "action_name": trade_result.get("action_name"),
                    "side": trade_result.get("side"),
                    "size": trade_result.get("size", 0.0),
                    "price": trade_result.get("price", 0.0),
                    "pnl": trade_result.get("pnl"),
                    "cost": trade_result.get("cost"),
                    "symbol": info.get("current_symbol"),
                    "reward": reward,
                }
            )

        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            episode_reward = self.current_episode_reward

            self.episode_rewards.append(float(episode_reward))
            self.episode_lengths.append(info.get('step', 0))

            if self.episode_count % self.log_frequency == 0:
                self._log_to_database(info)

            self.current_episode_reward = 0.0
            self.pending_trades = []

        return True
    
    def _log_to_database(self, info: dict):
        """Write episode data to database"""
        if not self.episode_rewards:
            return
        
        session = get_db_session()
        
        try:
            # Create episode record
            total_return_pct = 0.0
            final_capital = info.get("portfolio_value")
            if final_capital is not None:
                final_capital = float(final_capital)
                total_return_pct = (final_capital - self.settings.INITIAL_CAPITAL) / self.settings.INITIAL_CAPITAL

            episode = Episode(
                training_run_id=self.training_run_id,
                episode_num=self.episode_count,
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow(),
                total_reward=float(self.episode_rewards[-1]),
                total_return_pct=float(total_return_pct),
                num_trades=len(self.pending_trades),
                num_winning_trades=sum(1 for t in self.pending_trades if (t.get("pnl") or 0) > 0),
                final_capital=final_capital,
                markets_traded=[info.get("current_symbol")] if info.get("current_symbol") else None,
                extra_metadata={'logged_at': datetime.utcnow().isoformat()}
            )
            
            session.add(episode)
            session.flush()

            if self.pending_trades:
                trades = []
                for trade in self.pending_trades:
                    raw_pnl = trade.get("pnl")
                    trades.append(
                        Trade(
                            episode_id=episode.id,
                            timestamp=datetime.utcnow(),
                            market_id=trade.get("symbol") or "",
                            action=int(trade.get("action") or 0),
                            action_name=trade.get("action_name"),
                            position_size=float(trade.get("size") or 0.0),
                            price=float(trade.get("price") or 0.0),
                            side=trade.get("side") or "",
                            immediate_reward=float(trade.get("reward") or 0.0),
                            pnl=float(raw_pnl) if raw_pnl is not None else None,
                            features_snapshot=None,
                        )
                    )
                session.add_all(trades)

            session.commit()
            
            logger.debug(f"Logged episode {self.episode_count} to database")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log episode: {str(e)}")
        finally:
            session.close()


class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints during training.

    Diagnostic-only writer (architecture-audit-03 §B2):

    - Periodic snapshots → `checkpoint_run_<id>_step_<n>` (resumable training).
    - "New best mean training reward" snapshots → `reward_best_run_<id>_step_<n>`
      AND a moving-latest pointer at `reward_best_run_<id>`.

    This callback **never** writes `best_model_run_*` — that path is reserved
    for `EarlyStoppingCallback` after a hard-gate-passing eval. Training-mean
    reward is biased by curriculum and exploration, so `reward_best_*` is for
    diagnostics, not deployment.
    """
    
    def __init__(
        self,
        training_run_id: int,
        save_frequency: int = 10000,
        save_path: str = "./models",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.training_run_id = training_run_id
        self.save_frequency = save_frequency
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.best_mean_reward = -np.inf
        self.best_step = 0  # Track when best was achieved
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        """Check if we should save a checkpoint"""
        # Collect rewards
        if self.locals.get('dones', [False])[0]:
            episode_reward = self.locals.get('rewards', [0])[0]
            self.episode_rewards.append(episode_reward)
        
        # Save periodic checkpoint
        if self.n_calls % self.save_frequency == 0:
            self._save_checkpoint(is_best=False)
        
        # Check if this is the best model so far
        if len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards[-10:])
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_step = self.n_calls
                self._save_checkpoint(is_best=True)
                logger.info(f"New best model! Mean reward: {mean_reward:.2f} at step {self.n_calls}")
        
        return True
    
    def _save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint (diagnostics-only — see class docstring).

        Args:
            is_best: True for "new best running training-mean reward" event.
                     Writes `reward_best_run_<id>_step_<n>` plus a moving
                     pointer `reward_best_run_<id>`. Never writes the
                     deployment artifact `best_model_run_<id>` — that is
                     `EarlyStoppingCallback`'s exclusive responsibility per
                     architecture-audit-03 §B2.
        """
        if is_best:
            # Renamed from `best_model_run_*` to remove the namespace
            # collision with EarlyStoppingCallback (audit-03 §B2 fix).
            filename = f"reward_best_run_{self.training_run_id}_step_{self.n_calls}"
            latest_filename = f"reward_best_run_{self.training_run_id}"
            latest_filepath = self.save_path / latest_filename
            self.model.save(str(latest_filepath))
        else:
            filename = f"checkpoint_run_{self.training_run_id}_step_{self.n_calls}"

        filepath = self.save_path / filename
        
        try:
            # Save model
            self.model.save(str(filepath))
            
            # Log to database
            self._log_checkpoint(
                filepath=str(filepath),
                is_best=is_best
            )
            
            logger.info(f"Saved checkpoint: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def _log_checkpoint(self, filepath: str, is_best: bool):
        """Log checkpoint to database"""
        session = get_db_session()
        
        try:
            # Calculate recent performance
            recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            
            checkpoint = ModelCheckpoint(
                training_run_id=self.training_run_id,
                episode_num=len(self.episode_rewards),
                file_path=filepath,
                avg_reward=avg_reward,
                is_best=is_best,
                created_at=datetime.utcnow()
            )
            
            session.add(checkpoint)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log checkpoint: {str(e)}")
        finally:
            session.close()


class EarlyStoppingCallback(BaseCallback):
    """
    Evaluates the model periodically and stops training if performance stalls.

    Sole writer of the deployable artifact `best_model_run_<id>`
    (architecture-audit-03 §B2). The artifact is written **only** when both
    conditions hold:

    1. The eval metric (`metric_name` / `golden_score`) sets a new best, and
    2. All hard gates in `_passes_hard_gates()` pass.

    A separate, ungated `eval_best_run_<id>` snapshot is kept for retrospective
    analysis whenever the metric improves — this is **not** for deployment;
    promotion / fleet code must consume `best_model_run_<id>` instead.
    """

    def __init__(
        self,
        training_run_id: int,
        eval_frequency: int,
        eval_episodes: int,
        policy_type: str,
        sequence_length: int,
        metric_name: str = "sharpe_ratio",
        patience: int = 3,
        min_delta: float = 0.0,
        save_path: str = "./models",
        verbose: int = 0,
        arbitrage_enabled: bool = False,
        strategy: str = "crypto",
        min_profit_factor: float = 0.0,
        min_total_return: float = -1.0,
        max_drawdown: float = 1.0,
        max_fees_pct_of_gross_pnl: float = 1.0,
        eval_dataset_split: str = "train",
        held_out_days: int = 7,
        drawdown_score_floor: float = 0.0,
    ):
        """
        Args:
            eval_dataset_split: Which slice of the dataset is fed to the
                inner Evaluator at each eval. `"train"` (default after
                audit-03 §B3) holds out the trailing `held_out_days` window
                so that the trainer never sees holdout signal during early
                stopping. Use `"all"` only for legacy reproducibility runs.
            held_out_days: Size of the trailing held-out window in days.
            drawdown_score_floor: Audit-03 §B3 amendment to the drawdown
                guard. The previous implementation zero'd out `profit_factor`
                whenever drawdown exceeded the threshold, which made gates
                effectively binary on a noisy quantity. The new default
                applies a multiplicative score penalty in `_apply_drawdown_penalty`
                instead — see that method's docstring. The score floor caps
                how far the penalty can push a model below; default 0 means
                "no extra floor, the penalty just multiplies".
        """
        super().__init__(verbose)
        self.training_run_id = training_run_id
        self.eval_frequency = max(int(eval_frequency), 1)
        self.eval_episodes = max(int(eval_episodes), 1)
        self.policy_type = policy_type
        self.sequence_length = sequence_length
        self.metric_name = metric_name
        self.patience = max(int(patience), 1)
        self.min_delta = float(min_delta)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.arbitrage_enabled = arbitrage_enabled
        self.strategy = strategy
        self.min_profit_factor = float(min_profit_factor)
        self.min_total_return = float(min_total_return)
        self.max_drawdown = float(max_drawdown)
        self.max_fees_pct_of_gross_pnl = float(max_fees_pct_of_gross_pnl)
        if eval_dataset_split not in ("train", "all"):
            raise ValueError(
                f"eval_dataset_split must be 'train' or 'all', got {eval_dataset_split!r}"
            )
        self.eval_dataset_split = eval_dataset_split
        self.held_out_days = max(0, int(held_out_days))
        self.drawdown_score_floor = float(drawdown_score_floor)

        self.best_metric = -np.inf
        self.best_gated_metric = -np.inf
        self.patience_counter = 0

    def _compute_metric(self, metrics: Dict[str, float]) -> float:
        """
        Compute objective metric used by early stopping.
        Supports direct metric lookup and a composite golden score.
        """
        if self.metric_name == "golden_score":
            sharpe = float(metrics.get("sharpe_ratio", 0.0))
            total_return = float(metrics.get("total_return", 0.0))
            profit_factor = float(metrics.get("profit_factor", 0.0))
            fees_pct = float(metrics.get("fees_pct_of_gross_pnl", 1.0))
            drawdown = float(metrics.get("max_drawdown", 1.0))
            in_position = float(metrics.get("in_position_ratio", 0.0))
            return (
                0.35 * sharpe
                + 45.0 * total_return
                + 0.60 * profit_factor
                - 10.0 * fees_pct
                - 6.0 * drawdown
                + 1.5 * in_position
            )
        return float(metrics.get(self.metric_name, -np.inf))

    def _passes_hard_gates(self, metrics: Dict[str, float]) -> tuple:
        """Hard constraints for deployment eligibility.

        Returns (passed, failures) where failures lists the specific
        gates that were not met.
        """
        failures = []
        if float(metrics.get("profit_factor", 0.0)) < self.min_profit_factor:
            failures.append(f"profit_factor={metrics.get('profit_factor', 0):.3f}<{self.min_profit_factor}")
        if float(metrics.get("total_return", -1.0)) < self.min_total_return:
            failures.append(f"total_return={metrics.get('total_return', -1):.4f}<{self.min_total_return}")
        if float(metrics.get("max_drawdown", 1.0)) > self.max_drawdown:
            failures.append(f"max_drawdown={metrics.get('max_drawdown', 1):.4f}>{self.max_drawdown}")
        fees = float(metrics.get("fees_pct_of_gross_pnl", 1.0))
        gross_profit = float(metrics.get("avg_win_size", 0.0)) * max(
            float(metrics.get("trades_per_episode", 0.0)), 0.01
        )
        if fees > self.max_fees_pct_of_gross_pnl and gross_profit > 1.0:
            failures.append(f"fees_pct={fees:.2%}>{self.max_fees_pct_of_gross_pnl:.0%}")
        return len(failures) == 0, failures

    def _apply_drawdown_penalty(self, score: float, metrics: Dict[str, float]) -> float:
        """
        Multiplicative drawdown penalty (architecture-audit-03 §B3 amendment).

        The previous implementation in the gym env zero'd out `profit_factor`
        whenever drawdown crossed `drawdown_threshold`, turning a continuous
        risk metric into a discontinuous gate. That made eval scores discrete
        and punished borderline-good models the same as catastrophic ones.

        New behavior:

            penalty = max(0, 1 - (drawdown / max_drawdown))
            score'  = score * penalty,  floored at `drawdown_score_floor`

        - `drawdown == 0`        → penalty 1.0 → no change
        - `drawdown == max_dd`   → penalty 0.0 → score multiplied by 0
        - `drawdown > max_dd`    → penalty clipped at 0 (still won't go negative)

        This preserves ranking among healthy models while still pushing
        heavily-drawn models toward the bottom of the eval order. Hard
        rejection still happens via `_passes_hard_gates` for the
        `max_drawdown` line — this is just the soft signal for the metric.
        """
        if self.max_drawdown <= 0:
            return score
        dd = float(metrics.get("max_drawdown", 0.0))
        penalty = max(0.0, 1.0 - (dd / self.max_drawdown))
        adjusted = score * penalty
        return max(adjusted, self.drawdown_score_floor)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_frequency != 0:
            return True

        temp_path = self.save_path / f"early_stop_eval_run_{self.training_run_id}_step_{self.n_calls}"
        self.model.save(str(temp_path))

        if self.strategy == "kalshi":
            evaluator = KalshiEvaluator(
                model_path=str(temp_path),
                policy_type=self.policy_type,
            )
            metrics = evaluator.evaluate(num_episodes=self.eval_episodes, deterministic=True)
        else:
            evaluator = Evaluator(
                model_path=str(temp_path),
                policy_type=self.policy_type,
                sequence_length=self.sequence_length,
                arbitrage_enabled=self.arbitrage_enabled,
                dataset_split=self.eval_dataset_split,
                held_out_days=self.held_out_days,
            )
            metrics = evaluator.evaluate(num_episodes=self.eval_episodes, deterministic=True)

        raw_metric = self._compute_metric(metrics)
        current_metric = self._apply_drawdown_penalty(raw_metric, metrics)
        passes_gates, gate_failures = self._passes_hard_gates(metrics)

        trades_per_ep = float(metrics.get("trades_per_episode", 0.0))
        is_inactive = trades_per_ep < 0.1

        if is_inactive:
            logger.info(
                "Eval step %s: INACTIVE (%.1f trades/ep) — score=%.4f",
                self.n_calls, trades_per_ep, current_metric,
            )
        elif not passes_gates:
            logger.info(
                "Eval step %s: gate reject [%s] — score=%.4f ret=%.4f pf=%.2f dd=%.4f fees=%.2f%%",
                self.n_calls, ", ".join(gate_failures), current_metric,
                float(metrics.get("total_return", 0.0)),
                float(metrics.get("profit_factor", 0.0)),
                float(metrics.get("max_drawdown", 0.0)),
                float(metrics.get("fees_pct_of_gross_pnl", 0.0)) * 100,
            )

        improved = current_metric > self.best_metric + self.min_delta
        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            eval_best_path = self.save_path / f"eval_best_run_{self.training_run_id}"
            self.model.save(str(eval_best_path))
            logger.info(
                "Eval step %s: new eval-best %s=%.4f%s",
                self.n_calls, self.metric_name, current_metric,
                " (gates: PASS)" if passes_gates else f" (gates: FAIL [{', '.join(gate_failures)}])",
            )
        else:
            self.patience_counter += 1
            logger.info(
                "Eval step %s: no improvement (%s=%.4f vs best=%.4f), patience %s/%s",
                self.n_calls, self.metric_name, current_metric,
                self.best_metric, self.patience_counter, self.patience,
            )

        if passes_gates and current_metric > self.best_gated_metric + self.min_delta:
            self.best_gated_metric = current_metric
            best_path = self.save_path / f"best_model_run_{self.training_run_id}"
            self.model.save(str(best_path))
            logger.info(
                "Eval step %s: new gated-best %s=%.4f — saved for deployment",
                self.n_calls, self.metric_name, current_metric,
            )

        if self.patience_counter >= self.patience:
            logger.warning("Early stopping triggered at step %s", self.n_calls)
            return False

        return True


class TensorBoardCallback(BaseCallback):
    """
    Logs additional metrics to TensorBoard
    
    Beyond the default SB3 metrics, this logs:
    - Portfolio value
    - Win rate
    - Drawdown
    - Custom trading metrics
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """Log custom metrics"""
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Get info from environment
            info = self.locals.get('infos', [{}])[0]
            
            # Log custom metrics
            if self.episode_count % 10 == 0:  # Every 10 episodes
                self.logger.record('trading/portfolio_value', info.get('portfolio_value', 0))
                self.logger.record('trading/num_positions', info.get('num_positions', 0))
                self.logger.record('trading/win_rate', info.get('win_rate', 0))
                self.logger.record('trading/drawdown', info.get('drawdown', 0))
        
        return True
