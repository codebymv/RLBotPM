"""Action masking wrapper for training.

Replaces invalid actions with a valid action so the agent learns
from trade outcomes instead of invalid-action penalties.
"""

from __future__ import annotations

from typing import List
import numpy as np
import gymnasium as gym


class ActionMaskingWrapper(gym.Wrapper):
    """Mask invalid actions during training rollouts."""

    def get_valid_actions(self) -> List[int]:
        if hasattr(self.env, "get_wrapper_attr"):
            try:
                return self.env.get_wrapper_attr("get_valid_actions")()
            except AttributeError:
                pass
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_valid_actions"):
            return self.env.unwrapped.get_valid_actions()
        return list(range(self.action_space.n))

    def step(self, action):
        valid_actions = self.get_valid_actions()
        if valid_actions and action not in valid_actions:
            action = int(np.random.choice(valid_actions))
        return self.env.step(action)
