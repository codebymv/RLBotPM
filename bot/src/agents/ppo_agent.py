"""
PPO Agent - Proximal Policy Optimization

Wrapper around Stable-Baselines3's PPO implementation with
custom configuration for trading.

PPO (Proximal Policy Optimization) is chosen because:
1. Stable and reliable convergence
2. Good sample efficiency
3. Works well with continuous training
4. Industry standard for this type of problem

Key hyperparameters explained:
- learning_rate: How fast the agent learns (3e-4 is standard)
- n_steps: Steps to collect before each update (2048 is good balance)
- batch_size: Mini-batch size for training (256 works well)
- gamma: Discount factor for future rewards (0.99 = far-sighted)
- clip_range: How much policy can change per update (0.2 is standard)
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Callable
import os

from ..environment import PolymarketTradingEnv
from ..core.logger import get_logger
from ..core.config import get_settings


logger = get_logger(__name__)


class PPOAgent:
    """
    PPO-based trading agent
    
    This wraps Stable-Baselines3's PPO with trading-specific configuration.
    
    Usage:
        agent = PPOAgent()
        agent.train(total_timesteps=100000)
        agent.save("models/my_model")
        
        # Later...
        agent = PPOAgent()
        agent.load("models/my_model")
        action = agent.predict(observation)
    """
    
    def __init__(
        self,
        env: Optional[PolymarketTradingEnv] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_gpu: bool = True,
        tensorboard_log: Optional[str] = "./logs/tensorboard",
        verbose: int = 1
    ):
        """
        Initialize PPO agent
        
        Args:
            env: Trading environment (creates default if None)
            learning_rate: Learning rate for optimizer
            n_steps: Steps per environment before update
            batch_size: Mini-batch size for training
            n_epochs: Training epochs per rollout
            gamma: Discount factor (0-1, higher = more forward-looking)
            gae_lambda: GAE lambda for advantage estimation
            clip_range: PPO clipping parameter
            ent_coef: Entropy bonus coefficient (exploration)
            vf_coef: Value function coefficient
            max_grad_norm: Gradient clipping
            use_gpu: Whether to use GPU if available
            tensorboard_log: Directory for TensorBoard logs
            verbose: Verbosity level (0=none, 1=info, 2=debug)
        """
        self.settings = get_settings()
        
        # Create environment if not provided
        if env is None:
            env = PolymarketTradingEnv()
        
        # Wrap in vectorized environment (required by SB3)
        self.env = DummyVecEnv([lambda: env])
        
        # Determine device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using GPU for training")
        else:
            self.device = "cpu"
            logger.info("Using CPU for training")
        
        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",  # Multi-layer perceptron policy
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            device=self.device,
            tensorboard_log=tensorboard_log
        )
        
        logger.info("PPO agent initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gamma: {gamma}")
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10
    ):
        """
        Train the agent
        
        Args:
            total_timesteps: Total training steps
            callback: Optional callback for custom logic
            log_interval: Log every N episodes
        """
        logger.info(f"Starting training for {total_timesteps:,} timesteps")
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=log_interval,
                progress_bar=True
            )
            logger.info("Training completed successfully")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple[int, Optional[np.ndarray]]:
        """
        Predict action for given observation
        
        Args:
            observation: Current state
            deterministic: If True, use deterministic policy (no exploration)
        
        Returns:
            (action, state) tuple
        """
        action, state = self.model.predict(
            observation,
            deterministic=deterministic
        )
        return int(action), state
    
    def save(self, path: str):
        """
        Save model to disk
        
        Args:
            path: File path to save model
        """
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model from disk
        
        Args:
            path: File path to load model from
        """
        if not os.path.exists(path + ".zip"):
            raise FileNotFoundError(f"Model not found at {path}")
        
        self.model = PPO.load(
            path,
            env=self.env,
            device=self.device
        )
        logger.info(f"Model loaded from {path}")
    
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution
        
        Useful for understanding model's confidence and decision-making.
        
        Args:
            observation: Current state
        
        Returns:
            Array of probabilities for each action
        """
        # Get policy distribution
        obs_tensor = torch.as_tensor(observation).to(self.device)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()
        
        return probs
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render environment
        
        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating agent over {n_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                if render:
                    self.env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        logger.info(f"Evaluation complete: Mean reward = {results['mean_reward']:.2f}")
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dict with model metadata
        """
        return {
            'policy_type': 'MlpPolicy',
            'learning_rate': self.model.learning_rate,
            'n_steps': self.model.n_steps,
            'batch_size': self.model.batch_size,
            'gamma': self.model.gamma,
            'device': str(self.device),
            'total_timesteps': self.model.num_timesteps
        }


def create_parallel_env(n_envs: int = 4) -> SubprocVecEnv:
    """
    Create multiple parallel environments for faster training
    
    Args:
        n_envs: Number of parallel environments
    
    Returns:
        Vectorized environment
    """
    def make_env():
        def _init():
            return PolymarketTradingEnv()
        return _init
    
    return SubprocVecEnv([make_env() for _ in range(n_envs)])
