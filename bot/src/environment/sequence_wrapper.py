"""
Sequence stacking wrapper for LSTM and frame-stacked MLP policies.

This wrapper maintains a rolling window of the last N real observations.
It never generates synthetic observations; it only repeats the most
recent real observation to fill the initial buffer on reset.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SequenceStackWrapper(gym.Wrapper):
    """
    Wraps an environment to provide a sequence of recent observations.

    Args:
        env: Base environment with vector observations.
        sequence_length: Number of timesteps in the sequence.
        flatten: If True, returns a flattened vector (seq_len * obs_dim).
    """

    def __init__(self, env: gym.Env, sequence_length: int = 10, flatten: bool = False):
        super().__init__(env)
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")

        self.sequence_length = sequence_length
        self.flatten = flatten
        self._buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)

        obs_space = self.env.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("SequenceStackWrapper requires a Box observation space.")

        obs_dim = int(np.prod(obs_space.shape))
        if flatten:
            shape = (sequence_length * obs_dim,)
        else:
            shape = (sequence_length, obs_dim)

        self.observation_space = spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=shape,
            dtype=obs_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat = np.asarray(obs).reshape(-1)
        self._buffer.clear()
        for _ in range(self.sequence_length):
            self._buffer.append(flat)
        return self._format_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        flat = np.asarray(obs).reshape(-1)
        self._buffer.append(flat)
        return self._format_obs(), reward, terminated, truncated, info

    def _format_obs(self) -> np.ndarray:
        stacked = np.stack(list(self._buffer), axis=0)
        if self.flatten:
            return stacked.reshape(-1)
        return stacked

    def get_valid_actions(self):
        if hasattr(self.env, "get_wrapper_attr"):
            try:
                return self.env.get_wrapper_attr("get_valid_actions")()
            except AttributeError:
                pass
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_valid_actions"):
            return self.env.unwrapped.get_valid_actions()
        return list(range(self.action_space.n))
