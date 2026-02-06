"""Ensemble agent that combines specialized strategies."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .ppo_agent import PPOAgent
from ..environment import CryptoTradingEnv


class EnsembleAgent:
    """
    Combines multiple specialized agents with simple regime detection.

    Each agent is trained on a different strategy config, and the ensemble
    selects which model to use per step based on observation heuristics.
    """

    def __init__(
        self,
        env: CryptoTradingEnv,
        momentum_model_path: str,
        mean_reversion_model_path: str,
        breakout_model_path: str,
        policy_type: str = "MlpPolicy",
    ):
        self.agents = {
            "momentum": PPOAgent(env=env, policy_type=policy_type),
            "mean_reversion": PPOAgent(env=env, policy_type=policy_type),
            "breakout": PPOAgent(env=env, policy_type=policy_type),
        }

        self.agents["momentum"].load(momentum_model_path)
        self.agents["mean_reversion"].load(mean_reversion_model_path)
        self.agents["breakout"].load(breakout_model_path)

        self.agent_performance = {
            "momentum": {"wins": 0, "total": 0},
            "mean_reversion": {"wins": 0, "total": 0},
            "breakout": {"wins": 0, "total": 0},
        }

    def detect_regime(self, observation: np.ndarray) -> str:
        """
        Detect market regime from observation features.

        Uses:
        - return_24h (obs[5])
        - volatility_24h (obs[6])
        - volume_surge (obs[41])
        """
        return_24h = float(observation[5])
        volatility = abs(float(observation[6]))
        volume_surge = float(observation[41])

        if abs(return_24h) > 0.03 and volume_surge > 1.5:
            return "momentum"
        if volatility < 0.01 and volume_surge > 1.8:
            return "breakout"
        if volatility < 0.01:
            return "mean_reversion"
        return "momentum"

    def predict(self, observation: np.ndarray) -> Tuple[int, str, float]:
        regime = self.detect_regime(observation)
        agent = self.agents[regime]
        action, _ = agent.predict(observation, deterministic=True)

        perf = self.agent_performance[regime]
        confidence = perf["wins"] / perf["total"] if perf["total"] > 0 else 0.5
        return action, regime, confidence

    def update_performance(self, agent_name: str, success: bool) -> None:
        stats = self.agent_performance.get(agent_name)
        if not stats:
            return
        stats["total"] += 1
        if success:
            stats["wins"] += 1
