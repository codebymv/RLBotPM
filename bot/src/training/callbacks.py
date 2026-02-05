"""
Custom callbacks for training

Callbacks allow injecting custom logic during training:
- Logging to database
- Saving checkpoints
- Circuit breaker checks
- Custom metrics

These integrate with Stable-Baselines3's callback system.
"""

from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

from ..data import Episode, Trade, ModelCheckpoint, get_db_session
from ..risk import CircuitBreaker
from ..core.logger import get_logger
from ..core.config import get_settings


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
            if info.get("portfolio_value") is not None:
                total_return_pct = (
                    info.get("portfolio_value") - self.settings.INITIAL_CAPITAL
                ) / self.settings.INITIAL_CAPITAL

            episode = Episode(
                training_run_id=self.training_run_id,
                episode_num=self.episode_count,
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow(),
                total_reward=float(self.episode_rewards[-1]),
                total_return_pct=float(total_return_pct),
                num_trades=len(self.pending_trades),
                num_winning_trades=sum(1 for t in self.pending_trades if (t.get("pnl") or 0) > 0),
                final_capital=info.get("portfolio_value"),
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
    Saves model checkpoints during training
    
    Saves best models and periodic checkpoints.
    Now preserves the best model with step number to prevent overwriting.
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
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
        """
        # Generate filename - include step number for best models to preserve history
        if is_best:
            # Save both a versioned copy and update "current best" pointer
            filename = f"best_model_run_{self.training_run_id}_step_{self.n_calls}"
            # Also save to a consistent "latest best" location
            latest_filename = f"best_model_run_{self.training_run_id}"
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
