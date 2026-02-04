"""
Training orchestration for RL agents

This module handles the complete training pipeline:
1. Environment setup
2. Agent initialization  
3. Training loop with callbacks
4. Checkpoint management
5. Performance logging
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import yaml

from stable_baselines3.common.callbacks import CallbackList

from ..environment import PolymarketTradingEnv
from ..agents import PPOAgent
from ..data import TrainingRun, init_db, get_db_session
from ..core.logger import get_logger
from ..core.config import get_settings
from .callbacks import (
    CircuitBreakerCallback,
    PerformanceLogCallback,
    CheckpointCallback
)


logger = get_logger(__name__)


class Trainer:
    """
    Orchestrates the training process
    
    Usage:
        trainer = Trainer()
        trainer.train(total_episodes=100000)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            config_path: Path to config YAML file (optional)
        """
        self.settings = get_settings()
        
        # Load config if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize database
        init_db()
        
        # Create training run record
        self.training_run = self._create_training_run()
        
        logger.info(f"Trainer initialized (run_id={self.training_run.id})")
    
    def train(
        self,
        total_episodes: int = 100000,
        checkpoint_frequency: int = 10000,
        eval_frequency: int = 10000
    ):
        """
        Train the agent
        
        Args:
            total_episodes: Total number of training episodes
            checkpoint_frequency: Save checkpoint every N episodes
            eval_frequency: Evaluate every N episodes
        """
        logger.info(f"Starting training: {total_episodes:,} episodes")
        
        try:
            # Create environment
            env = PolymarketTradingEnv()
            
            # Create agent
            agent = PPOAgent(
                env=env,
                tensorboard_log=f"./logs/tensorboard/run_{self.training_run.id}"
            )
            
            # Create callbacks
            callbacks = self._create_callbacks(
                checkpoint_frequency=checkpoint_frequency,
                eval_frequency=eval_frequency
            )
            
            # Train
            agent.train(
                total_timesteps=total_episodes,
                callback=callbacks,
                log_interval=10
            )
            
            # Save final model
            final_model_path = f"models/final_run_{self.training_run.id}"
            agent.save(final_model_path)
            
            # Update training run
            self._complete_training_run(success=True)
            
            logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self._complete_training_run(success=False, status='interrupted')
            raise
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self._complete_training_run(success=False, status='failed')
            raise
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        # TODO: Implement checkpoint loading
        raise NotImplementedError("Checkpoint loading not yet implemented")
    
    def _create_callbacks(
        self,
        checkpoint_frequency: int,
        eval_frequency: int
    ) -> CallbackList:
        """
        Create training callbacks
        
        Args:
            checkpoint_frequency: Checkpoint save frequency
            eval_frequency: Evaluation frequency
        
        Returns:
            CallbackList with all callbacks
        """
        callbacks = [
            CircuitBreakerCallback(
                training_run_id=self.training_run.id
            ),
            PerformanceLogCallback(
                training_run_id=self.training_run.id,
                log_frequency=100
            ),
            CheckpointCallback(
                training_run_id=self.training_run.id,
                save_frequency=checkpoint_frequency,
                save_path=self.settings.MODEL_SAVE_PATH
            )
        ]
        
        return CallbackList(callbacks)
    
    def _create_training_run(self) -> TrainingRun:
        """
        Create database record for this training run
        
        Returns:
            TrainingRun instance
        """
        session = get_db_session()
        
        try:
            training_run = TrainingRun(
                started_at=datetime.utcnow(),
                status='running',
                config_snapshot=self._get_config_snapshot()
            )
            
            session.add(training_run)
            session.commit()
            session.refresh(training_run)
            
            logger.info(f"Created training run record: ID={training_run.id}")
            
            return training_run
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create training run: {str(e)}")
            raise
        finally:
            session.close()
    
    def _complete_training_run(
        self,
        success: bool,
        status: str = 'completed'
    ):
        """
        Mark training run as complete
        
        Args:
            success: Whether training succeeded
            status: Final status ('completed', 'failed', 'interrupted')
        """
        session = get_db_session()
        
        try:
            training_run = session.query(TrainingRun).filter_by(
                id=self.training_run.id
            ).first()
            
            if training_run:
                training_run.ended_at = datetime.utcnow()
                training_run.status = status
                session.commit()
                
                logger.info(f"Training run {self.training_run.id} marked as {status}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update training run: {str(e)}")
        finally:
            session.close()
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML config
        
        Returns:
            Config dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}")
            return {}
    
    def _get_config_snapshot(self) -> Dict:
        """
        Get current configuration snapshot
        
        Returns:
            Dict with all relevant config
        """
        return {
            'environment': self.settings.ENVIRONMENT,
            'initial_capital': self.settings.INITIAL_CAPITAL,
            'max_position_size_pct': self.settings.MAX_POSITION_SIZE_PCT,
            'max_open_positions': self.settings.MAX_OPEN_POSITIONS,
            'transaction_cost_pct': self.settings.TRANSACTION_COST_PCT,
            'training_episodes': self.settings.TRAINING_EPISODES,
            'custom_config': self.config
        }
