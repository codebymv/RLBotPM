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

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO, RecurrentPPO
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple
import os
import pickle

from ..environment import CryptoTradingEnv
from ..data.sources.base import DataUnavailableError
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
        env: Optional[CryptoTradingEnv] = None,
        policy_type: str = "MlpPolicy",
        policy_kwargs: Optional[Dict] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_gpu: bool = True,
        tensorboard_log: Optional[str] = "./logs/tensorboard",
        verbose: int = 1,
        checkpoint_path: Optional[str] = None,
        normalize_rewards: bool = False,
        frame_stack: int = 1
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
            normalize_rewards: Whether to normalize rewards (stabilizes training)
            frame_stack: Number of frames to stack (1=no stacking)
        """
        self.settings = get_settings()
        self.policy_type = policy_type
        self.policy_kwargs = policy_kwargs
        self.is_recurrent = policy_type == "MlpLstmPolicy"
        self.normalize_rewards = normalize_rewards
        self.frame_stack = frame_stack
        
        # Require a real-data environment
        if env is None:
            raise DataUnavailableError("PPOAgent requires a real-data environment instance.")
        
        # Wrap in vectorized environment (required by SB3)
        self.env = DummyVecEnv([lambda: env])
        
        # Apply frame stacking if requested
        if self.frame_stack > 1:
            logger.info(f"Enabling frame stacking (n={self.frame_stack})")
            self.env = VecFrameStack(self.env, n_stack=self.frame_stack)
        
        # Apply reward normalization if requested
        if self.normalize_rewards:
            logger.info("Enabling reward normalization (VecNormalize)")
            self.env = VecNormalize(
                self.env, 
                norm_obs=False, 
                norm_reward=True, 
                clip_reward=10.0, 
                gamma=gamma
            )
        
        # Determine device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using GPU for training")
        else:
            self.device = "cpu"
            logger.info("Using CPU for training")
        
        # Load from checkpoint if provided, otherwise create new model
        if checkpoint_path:
            logger.info(f"Fine-tuning from checkpoint: {checkpoint_path}")
            
            # Use appropriate load method for VecNormalize if applicable
            # Note: We need to load stats BEFORE creating the model if we want the env to be synced,
            # but usually we load the model policy. 
            # If normalize_rewards is True, we should try to load stats.
            if self.normalize_rewards:
                stats_path = str(Path(checkpoint_path).with_suffix(".norm.pkl"))
                if os.path.exists(stats_path):
                    logger.info(f"Loading normalization stats from {stats_path}")
                    with open(stats_path, "rb") as f:
                        self.env.training = True # Set to training mode
                        # Manually load generic stats or use VecNormalize.load
                        # Since we already created self.env, we can load onto it or replace it.
                        # VecNormalize.load creates a new instance.
                        # Let's try to just load state_dict if possible? No, load is a classmethod.
                        # We will replace self.env with loaded one, but we need to ensure it wraps the same dummy env.
                        # Actually, better to just load pickle and set attributes? 
                        # Or use VecNormalize.load(stats_path, self.env.venv) -> self.env is the VecNormalize object.
                        # self.env.venv is the DummyVecEnv.
                         # But VecNormalize.load expects the venv as second arg.
                         # Wait, self.env IS VecNormalize now. self.env.venv is the inner env.
                        norm_env = VecNormalize.load(stats_path, self.env.venv)
                        norm_env.training = True
                        norm_env.norm_obs = False
                        norm_env.norm_reward = True
                        self.env = norm_env
                else:
                    logger.warning(f"Normalization stats not found at {stats_path}, starting fresh normalization.")

            # Create fresh model first (with fresh optimizer state)
            if self.is_recurrent:
                self.model = RecurrentPPO(
                    policy=self.policy_type,
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
                    tensorboard_log=tensorboard_log,
                    policy_kwargs=self.policy_kwargs,
                )
                # Load old model and copy policy weights only
                old_model = RecurrentPPO.load(checkpoint_path, device=self.device)
                self.model.policy.load_state_dict(old_model.policy.state_dict())
                del old_model
            else:
                self.model = MaskablePPO(
                    policy=self.policy_type,
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
                    tensorboard_log=tensorboard_log,
                    policy_kwargs=self.policy_kwargs,
                )
                # Load old model and copy policy weights only (avoids corrupted optimizer state)
                old_model = MaskablePPO.load(checkpoint_path, device=self.device)
                self.model.policy.load_state_dict(old_model.policy.state_dict())
                del old_model
            logger.info(f"Policy weights loaded from checkpoint, fresh optimizer initialized")
        elif self.is_recurrent:
            self.model = RecurrentPPO(
                policy=self.policy_type,
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
                tensorboard_log=tensorboard_log,
                policy_kwargs=self.policy_kwargs,
            )
        else:
            # Create MaskablePPO model
            self.model = MaskablePPO(
                policy=self.policy_type,
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
                tensorboard_log=tensorboard_log,
                policy_kwargs=self.policy_kwargs,
            )
        
        logger.info("PPO agent initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gamma: {gamma}")
        logger.info(f"  Reward Normalization: {self.normalize_rewards}")
    
    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None, log_interval: int = 1):
        """
        Train the agent
        
        Args:
            total_timesteps: Number of timesteps to train for
            callback: Callback for monitoring training
            log_interval: Logging interval
            
        Returns:
            Trained model
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            reset_num_timesteps=False
        )
        
        logger.info("Training completed")
        return self.model
    
    def predict(
        self, 
        observation: np.ndarray, 
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ):
        """
        Predict action for a given observation
        
        Args:
            observation: Current state
            state: LSTM state (optional)
            episode_start: Start of episode (optional)
            deterministic: Whether to use deterministic or stochastic policy
            action_masks: Action masks for MaskablePPO (optional)
            
        Returns:
            Tuple of (action, state)
        """
        # We invoke the model's predict method directly
        # If MaskablePPO logic is needed for action masks, we should ensure
        # that the environment we pass is wrapped or provides masks.
        # However, for simple predict with MaskablePPO, it might require action_masks.
        # SB3 Contrib says: model.predict(obs, action_masks=mask)
        # If we use env.action_masks(), `predict` needs access to env?
        # Typically PPOAgent doesn't reference env in predict for masking unless passed.
        # But MaskablePPO *requires* masks if masking is enabled.
        # If we don't pass masks, it might fail or pick invalid action.
        
        # NOTE: In walk_forward, we are passing `obs` from `env.reset()`.
        # `walk_forward` creates `test_env`. `test_env` has `action_masks()`.
        # But `agent` does NOT know about `test_env` (it knows `self.env` which is dummy wrapper around train env).
        
        # If we are using MaskablePPO, we MUST pass action masks.
        # But we can't get masks from `obs` alone.
        # `agent.predict` signature in `walk_forward` is `agent.predict(obs)`.
        # It does NOT pass env or masks.
        
        # Solution: `walk_forward.py` loop likely needs update to pass masks?
        # OR PPOAgent assumes `observation` includes mask? No.
        
        # Wait, if `MaskablePPO` is used, `model.predict` will complain if no mask is passed?
        # Actually, `MaskablePPO.predict` signature:
        # def predict(self, observation, state=None, episode_start=None, deterministic=False, action_masks=None)
        
        # If `action_masks` is None, it calls `self.policy.predict(..., apply_mask=False)`.
        # So it works but might pick invalid action.
        # However, `CryptoTradingEnv` might reject invalid action with penalty (or just no-op).
        # We saw `get_valid_actions` in env.
        
        # For now, let's just forward to model.predict.
        # If we need strict masking during eval, we would need to fetch masks from eval env.
        # But `agent` is decoupled from eval env loop in `walk_forward`.
        
        # Prepare kwargs for predict
        predict_kwargs = {"deterministic": deterministic}
        
        # Add optional args if they are not None
        if state is not None:
            predict_kwargs["state"] = state
        if episode_start is not None:
            predict_kwargs["episode_start"] = episode_start
        if action_masks is not None:
            predict_kwargs["action_masks"] = action_masks
            
        # Check if model accepts action_masks to avoid invalid kwarg error
        import inspect
        sig = inspect.signature(self.model.predict)
        if "action_masks" not in sig.parameters and "action_masks" in predict_kwargs:
            del predict_kwargs["action_masks"]
            
        action, state = self.model.predict(observation, **predict_kwargs)
        
        # Convert numpy array to scalar if it's a single action
        # This prevents "unhashable type: 'numpy.ndarray'" error when using action as dict key
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = action.item()
            elif action.size == 1:
                action = action.item()
                
        return action, state

    
    # ... (train method remains similar) ...

    # ... (predict method remains similar) ...

    # ... (_get_action_masks remains similar) ...
    
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
        
        # Save normalization stats if enabled
        if self.normalize_rewards and isinstance(self.env, VecNormalize):
            stats_path = str(Path(path).with_suffix(".norm.pkl"))
            self.env.save(stats_path)
            logger.info(f"Normalization stats saved to {stats_path}")
    
    def load(self, path: str):
        """
        Load model from disk
        
        Args:
            path: File path to load model from
        """
        # Check if file exists with .zip extension if not provided
        if not path.endswith(".zip"):
            check_path = path + ".zip"
        else:
            check_path = path
            
        if not os.path.exists(check_path):
            raise FileNotFoundError(f"Model not found at {check_path}")
            
        # SB3 load/save handles suffixes, but usually expects path without suffix
        # or handles it if present. We should pass the path that works.
        # Actually SB3 adds .zip if missing.
        # If we pass "model.zip", SB3 might look for "model.zip.zip" if we are not careful?
        # No, SB3 uses pathlib.Path(path).with_suffix(".zip") usually? 
        # Actually checking source: load(path, ...) -> if path is string, verify exists.
        
        # We will use the path without extension for the load call if possible, 
        # or rely on SB3's handling. 
        # But for our check:

        
        if self.is_recurrent:
            self.model = RecurrentPPO.load(
                path,
                env=self.env,
                device=self.device
            )
        else:
            self.model = MaskablePPO.load(
                path,
                env=self.env,
                device=self.device
            )
        
        # Load normalization stats if enabled
        if self.normalize_rewards:
            stats_path = str(Path(path).with_suffix(".norm.pkl"))
            if os.path.exists(stats_path):
                 # Load stats onto existing environment
                 # We need to access the inner environment to pass to VecNormalize.load
                 # If self.env is already VecNormalize, we need its venv.
                 venv = self.env.venv if isinstance(self.env, VecNormalize) else self.env
                 
                 norm_env = VecNormalize.load(stats_path, venv)
                 norm_env.training = False # Default to eval mode upon load? Or training? 
                 # Usually if we load for inference, we want training=False. 
                 # But if we load to continue training (which is init with checkpoint), we want True.
                 # This 'load' method is typically used for inference.
                 norm_env.norm_obs = False
                 norm_env.norm_reward = True
                 self.env = norm_env
                 
                 # Re-attach env to model? The model holds reference to env?
                 # SB3 models usually hold self.env.
                 self.model.set_env(self.env)
                 logger.info(f"Normalization stats loaded from {stats_path}")
            else:
                logger.warning(f"Normalization stats expected but not found at {stats_path}")

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
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
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


def create_parallel_env(
    dataset,
    interval: str = "1h",
    n_envs: int = 4,
    sequence_length: int = 1,
    frame_stack: bool = False,
) -> SubprocVecEnv:
    """
    Create multiple parallel environments for faster training.

    Args:
        dataset: DataFrame with real OHLCV data
        interval: Candle interval
        n_envs: Number of parallel environments
    """
    if dataset is None or len(dataset) == 0:
        raise DataUnavailableError("Parallel env requires real OHLCV dataset.")

    from ..environment import SequenceStackWrapper

    def make_env():
        def _init():
            env = CryptoTradingEnv(dataset=dataset, interval=interval, sequence_length=sequence_length)
            if frame_stack and sequence_length > 1:
                env = SequenceStackWrapper(env, sequence_length=sequence_length, flatten=True)
            return env
        return _init

    return SubprocVecEnv([make_env() for _ in range(n_envs)])
