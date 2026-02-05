"""
Exploration wrapper to encourage action diversity during training

This wrapper injects random actions during early training to prevent
the agent from collapsing to a single action (like always doing nothing).
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any


class ExplorationWrapper(gym.Wrapper):
    """
    Forces exploration by injecting random actions during early training.
    
    The exploration rate starts at `initial_epsilon` and decays linearly
    to `final_epsilon` over `decay_steps`.
    
    Usage:
        env = CryptoTradingEnv(...)
        env = ExplorationWrapper(env, initial_epsilon=0.8, decay_steps=10000)
    """
    
    def __init__(
        self,
        env: gym.Env,
        initial_epsilon: float = 0.8,
        final_epsilon: float = 0.0,
        decay_steps: int = 10000,
        exclude_actions: Optional[list] = None
    ):
        """
        Initialize exploration wrapper
        
        Args:
            env: Base environment
            initial_epsilon: Starting exploration rate (0-1)
            final_epsilon: Final exploration rate after decay
            decay_steps: Steps over which to decay epsilon
            exclude_actions: Actions to exclude from random sampling (e.g., [0] to exclude NO_ACTION)
        """
        super().__init__(env)
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.exclude_actions = exclude_actions or []
        
        self.step_count = 0
        self.current_epsilon = initial_epsilon
        
        # Build list of allowed random actions
        self.allowed_actions = [
            a for a in range(self.action_space.n)
            if a not in self.exclude_actions
        ]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step with exploration: randomly replace action with probability epsilon
        """
        # Decay epsilon
        if self.step_count < self.decay_steps:
            progress = self.step_count / self.decay_steps
            self.current_epsilon = (
                self.initial_epsilon + 
                (self.final_epsilon - self.initial_epsilon) * progress
            )
        else:
            self.current_epsilon = self.final_epsilon
        
        # Replace action with random action with probability epsilon
        if np.random.random() < self.current_epsilon and self.allowed_actions:
            action = np.random.choice(self.allowed_actions)
        
        self.step_count += 1
        
        return self.env.step(action)
    
    def reset(self, **kwargs):
        """Reset environment"""
        return self.env.reset(**kwargs)
