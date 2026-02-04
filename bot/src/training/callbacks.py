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
    
    def _on_step(self) -> bool:
        """
        Called after each step
        
        Returns:
            False to stop training, True to continue
        """
        # Check if trading is allowed
        if not self.circuit_breaker.can_trade():
            logger.critical("Circuit breaker triggered - stopping training")
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
    
    def _on_step(self) -> bool:
        """Log performance metrics"""
        # Collect episode statistics
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Get episode info
            info = self.locals.get('infos', [{}])[0]
            episode_reward = self.locals.get('rewards', [0])[0]
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(info.get('step', 0))
            
            # Log to database every N episodes
            if self.episode_count % self.log_frequency == 0:
                self._log_to_database()
        
        return True
    
    def _log_to_database(self):
        """Write episode data to database"""
        if not self.episode_rewards:
            return
        
        session = get_db_session()
        
        try:
            # Create episode record
            episode = Episode(
                training_run_id=self.training_run_id,
                episode_num=self.episode_count,
                started_at=datetime.utcnow(),
                ended_at=datetime.utcnow(),
                total_reward=float(np.mean(self.episode_rewards[-self.log_frequency:])),
                num_trades=0,  # TODO: Track from environment
                metadata={'logged_at': datetime.utcnow().isoformat()}
            )
            
            session.add(episode)
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
                self._save_checkpoint(is_best=True)
                logger.info(f"New best model! Mean reward: {mean_reward:.2f}")
        
        return True
    
    def _save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
        """
        # Generate filename
        if is_best:
            filename = f"best_model_run_{self.training_run_id}"
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
