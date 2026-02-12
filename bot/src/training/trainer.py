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

from datetime import timedelta
from ..environment import (
    CryptoTradingEnv,
    SequenceStackWrapper,
)
from ..environment.strategy_envs import (
    MomentumTradingEnv,
    MeanReversionTradingEnv,
    BreakoutTradingEnv,
)
from ..environment.kalshi_env import KalshiEventEnv, load_kalshi_settled_markets
from ..agents import PPOAgent
from ..data import TrainingRun, init_db, get_db_session, CryptoSymbol
from ..data.collectors import CryptoDataLoader, MultiSourceLoader
from ..data.sources.base import DataUnavailableError
from ..core.logger import get_logger
from ..core.config import get_settings
from .callbacks import (
    CircuitBreakerCallback,
    PerformanceLogCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
)


logger = get_logger(__name__)


class Trainer:
    """
    Orchestrates the training process
    
    Usage:
        trainer = Trainer()
        trainer.train(total_episodes=100000)
    """
    
    def __init__(self, config_path: Optional[str] = None, overrides: Optional[Dict] = None):
        """
        Initialize trainer
        
        Args:
            config_path: Path to config YAML file (optional)
        """
        self.settings = get_settings()
        
        # Load config if provided
        self.config = self._load_config(config_path) if config_path else {}
        if overrides:
            self._apply_overrides(overrides)
        
        # Checkpoint path for resuming training
        self._checkpoint_path = None
        
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
            env_config = self.config.get("environment", {})
            max_steps = env_config.get("max_steps", 500)
            initial_capital = env_config.get("initial_capital", self.settings.INITIAL_CAPITAL)
            transaction_cost = env_config.get("transaction_cost", self.settings.TRANSACTION_COST_PCT)

            recurrent_config = self.config.get("recurrent", {})
            sequence_length = int(recurrent_config.get("sequence_length", 1))

            policy_type = self.config.get("ppo", {}).get("policy_type", "MlpPolicy")
            use_sequence_stack = policy_type == "MlpPolicy" and sequence_length > 1
            env_sequence_length = sequence_length if use_sequence_stack else 1
            min_rows = max_steps + env_sequence_length + 1

            # Check for arbitrage mode
            arbitrage_enabled = self.settings.ARBITRAGE_ENABLED
            if arbitrage_enabled:
                logger.info("Arbitrage mode enabled - using multi-exchange data")

            # --- Kalshi strategy: completely different data & env path ---
            if strategy == "kalshi":
                session = get_db_session()
                try:
                    kalshi_cfg = self.config.get("kalshi", {})
                    series_tickers = kalshi_cfg.get("series_tickers", None)
                    df = load_kalshi_settled_markets(session, series_tickers=series_tickers)
                    if df is None or df.empty:
                        raise DataUnavailableError(
                            "No settled Kalshi markets in DB. Run: python main.py kalshi backfill-settled"
                        )
                    base_env = KalshiEventEnv(
                        settled_markets=df,
                        initial_capital=initial_capital,
                        max_positions_per_event=kalshi_cfg.get("max_positions_per_event", 3),
                        contracts_per_trade=kalshi_cfg.get("contracts_per_trade", 1),
                        transaction_cost_cents=kalshi_cfg.get("transaction_cost_cents", 0.0),
                        min_volume=kalshi_cfg.get("min_volume", 0),
                        min_contracts_per_event=kalshi_cfg.get("min_contracts_per_event", 3),
                    )
                    logger.info(f"Using KalshiEventEnv ({len(base_env.events)} events)")
                finally:
                    session.close()
                env = base_env
            else:
                # --- Standard crypto strategies ---

                # Load real dataset (fail fast if unavailable)
                dataset = self._load_dataset(min_rows=min_rows, arbitrage_enabled=arbitrage_enabled)

                # Create environment (strategy-specific or default crypto)
                env_kwargs = dict(
                    dataset=dataset,
                    interval=self.settings.DATA_INTERVAL,
                    initial_capital=initial_capital,
                    max_steps=max_steps,
                    transaction_cost=transaction_cost,
                    sequence_length=env_sequence_length,
                    arbitrage_enabled=arbitrage_enabled,
                )
                if strategy == "momentum":
                    base_env = MomentumTradingEnv(**env_kwargs)
                    logger.info("Using MomentumTradingEnv")
                elif strategy == "mean_reversion":
                    base_env = MeanReversionTradingEnv(**env_kwargs)
                    logger.info("Using MeanReversionTradingEnv")
                elif strategy == "breakout":
                    base_env = BreakoutTradingEnv(**env_kwargs)
                    logger.info("Using BreakoutTradingEnv")
                else:
                    base_env = CryptoTradingEnv(**env_kwargs)

                env = base_env
                if use_sequence_stack:
                    env = SequenceStackWrapper(env, sequence_length=sequence_length, flatten=True)
            
            # Create agent
            ppo_config = self.config.get("ppo", {})
            training_config = self.config.get("training", {})
            log_root = training_config.get("tensorboard_log", "./logs/tensorboard")
            tensorboard_log = str(Path(log_root) / f"run_{self.training_run.id}")

            policy_kwargs = None
            if policy_type == "MlpLstmPolicy":
                policy_kwargs = {
                    "lstm_hidden_size": int(recurrent_config.get("lstm_hidden_size", 128)),
                    "n_lstm_layers": int(recurrent_config.get("lstm_layers", 1)),
                    "shared_lstm": bool(recurrent_config.get("shared_lstm", False)),
                    "enable_critic_lstm": bool(recurrent_config.get("enable_critic_lstm", True)),
                }

            agent = PPOAgent(
                env=env,
                policy_type=policy_type,
                policy_kwargs=policy_kwargs,
                learning_rate=ppo_config.get("learning_rate", 3e-4),
                n_steps=ppo_config.get("n_steps", 2048),
                batch_size=ppo_config.get("batch_size", 256),
                n_epochs=ppo_config.get("n_epochs", 10),
                gamma=ppo_config.get("gamma", 0.99),
                gae_lambda=ppo_config.get("gae_lambda", 0.95),
                clip_range=ppo_config.get("clip_range", 0.2),
                ent_coef=ppo_config.get("ent_coef", 0.03),
                vf_coef=ppo_config.get("vf_coef", 0.5),
                max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
                use_gpu=training_config.get("use_gpu", True),
                tensorboard_log=tensorboard_log,
                verbose=training_config.get("verbose", 1),
                checkpoint_path=self._checkpoint_path
            )
            
            # Create callbacks
            callbacks = self._create_callbacks(
                checkpoint_frequency=checkpoint_frequency,
                eval_frequency=eval_frequency,
                policy_type=policy_type,
                sequence_length=sequence_length,
                training_config=training_config,
                arbitrage_enabled=arbitrage_enabled,
            )
            
            # Train
            total_timesteps = total_episodes * max_steps
            agent.train(
                total_timesteps=total_timesteps,
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
        logger.info(f"Setting checkpoint for resume: {checkpoint_path}")
        self._checkpoint_path = checkpoint_path
    
    def _create_callbacks(
        self,
        checkpoint_frequency: int,
        eval_frequency: int,
        policy_type: str,
        sequence_length: int,
        training_config: Dict,
        arbitrage_enabled: bool = False,
    ) -> CallbackList:
        """
        Create training callbacks
        
        Args:
            checkpoint_frequency: Checkpoint save frequency
            eval_frequency: Evaluation frequency
            arbitrage_enabled: Whether arbitrage mode is enabled
        
        Returns:
            CallbackList with all callbacks
        """
        early_stopping_config = training_config.get("early_stopping", {})
        eval_episodes = int(early_stopping_config.get("eval_episodes", 50))
        patience = int(early_stopping_config.get("patience", 3))
        min_delta = float(early_stopping_config.get("min_delta", 0.0))
        metric_name = str(early_stopping_config.get("metric", "sharpe_ratio"))
        eval_sequence_length = sequence_length if policy_type == "MlpPolicy" else 1

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
            ),
            EarlyStoppingCallback(
                training_run_id=self.training_run.id,
                eval_frequency=eval_frequency,
                eval_episodes=eval_episodes,
                policy_type=policy_type,
                sequence_length=eval_sequence_length,
                metric_name=metric_name,
                patience=patience,
                min_delta=min_delta,
                save_path=self.settings.MODEL_SAVE_PATH,
                arbitrage_enabled=arbitrage_enabled,
            ),
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
            candidate = Path(config_path)
            if not candidate.is_file():
                repo_root = Path(__file__).resolve().parents[3]
                candidate = repo_root / config_path

            with open(candidate, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {candidate}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}")
            return {}

    def _apply_overrides(self, overrides: Dict) -> None:
        """
        Apply CLI overrides to config.
        """
        for section, values in overrides.items():
            if values is None:
                continue
            if section not in self.config or not isinstance(self.config.get(section), dict):
                self.config[section] = {}
            self.config[section].update(values)
    
    def _get_config_snapshot(self) -> Dict:
        """
        Get current configuration snapshot
        
        Returns:
            Dict with all relevant config
        """
        return {
            'environment': self.settings.ENVIRONMENT,
            'data_source': self.settings.DATA_SOURCE,
            'data_interval': self.settings.DATA_INTERVAL,
            'require_historical_days': self.settings.REQUIRE_HISTORICAL_DAYS,
            'initial_capital': self.settings.INITIAL_CAPITAL,
            'max_position_size_pct': self.settings.MAX_POSITION_SIZE_PCT,
            'max_open_positions': self.settings.MAX_OPEN_POSITIONS,
            'transaction_cost_pct': self.settings.TRANSACTION_COST_PCT,
            'training_episodes': self.settings.TRAINING_EPISODES,
            'custom_config': self.config
        }

    def _load_dataset(self, min_rows: Optional[int] = None, arbitrage_enabled: bool = False):
        """
        Load real OHLCV dataset from database.
        Raises DataUnavailableError if data is missing.
        
        Args:
            min_rows: Minimum rows required per symbol
            arbitrage_enabled: If True, load from multiple exchanges and compute spread features
        """
        source = self.settings.DATA_SOURCE
        interval = self.settings.DATA_INTERVAL
        days = self.settings.REQUIRE_HISTORICAL_DAYS

        # Determine symbols to load
        symbols = []
        if self.settings.DATA_SYMBOLS:
            symbols = [s.strip() for s in self.settings.DATA_SYMBOLS.split(",") if s.strip()]
        else:
            session = get_db_session()
            try:
                symbols = [
                    row.symbol
                    for row in session.query(CryptoSymbol)
                    .filter_by(source=source, status="active")
                    .all()
                ]
            finally:
                session.close()

        if not symbols:
            raise DataUnavailableError(
                "No symbols found in database. Run CryptoDataLoader to sync symbols and candles."
            )

        end = datetime.utcnow()
        start = end - timedelta(days=days)

        # Use MultiSourceLoader if arbitrage is enabled and multiple sources are configured
        if arbitrage_enabled and self.settings.DATA_SOURCES:
            sources = [s.strip() for s in self.settings.DATA_SOURCES.split(",") if s.strip()]
            if len(sources) >= 2:
                logger.info(f"Loading multi-exchange data from: {sources}")
                multi_loader = MultiSourceLoader(sources=sources)
                dataset = multi_loader.load_aligned_dataset(
                    symbols=symbols,
                    interval=interval,
                    start=start,
                    end=end
                )
            else:
                logger.warning("ARBITRAGE_ENABLED but DATA_SOURCES has < 2 exchanges. Using single source.")
                loader = CryptoDataLoader(source=source)
                dataset = loader.load_dataset(
                    symbols=symbols,
                    interval=interval,
                    start=start,
                    end=end
                )
        else:
            loader = CryptoDataLoader(source=source)
            dataset = loader.load_dataset(
                symbols=symbols,
                interval=interval,
                start=start,
                end=end
            )

        if dataset is None or dataset.empty:
            raise DataUnavailableError("Dataset is empty after loading. Check data pipeline.")

        if min_rows:
            counts = dataset.groupby("symbol").size()
            valid_symbols = counts[counts >= min_rows].index.tolist()
            dataset = dataset[dataset["symbol"].isin(valid_symbols)].reset_index(drop=True)
            if dataset.empty:
                raise DataUnavailableError(
                    "No symbols meet minimum row requirement. Increase history or reduce steps."
                )

        return dataset
