"""Backward-compatible ensemble wrapper around specialist manager."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .specialist_manager import SpecialistManager
from ..environment import CryptoTradingEnv


class EnsembleAgent:
    """Compatibility layer for specialist-regime inference."""

    def __init__(
        self,
        env: CryptoTradingEnv,
        momentum_model_path: str,
        mean_reversion_model_path: str,
        breakout_model_path: str,
        policy_type: str = "MlpPolicy",
    ):
        self.manager = SpecialistManager(
            env=env,
            policy_type=policy_type,
            model_paths={
                "momentum": momentum_model_path,
                "mean_reversion": mean_reversion_model_path,
                "breakout": breakout_model_path,
            },
        )
        self.agent_performance = {
            "momentum": {"wins": 0, "total": 0},
            "mean_reversion": {"wins": 0, "total": 0},
            "breakout": {"wins": 0, "total": 0},
        }

    def detect_regime(self, observation: np.ndarray) -> str:
        return self.manager.router.detect(observation)

    def predict(self, observation: np.ndarray) -> Tuple[int, str, float]:
        action, _, regime = self.manager.predict(observation, deterministic=True)

        perf = self.agent_performance[regime]
        confidence = perf["wins"] / perf["total"] if perf["total"] > 0 else 0.5
        return int(action), regime, confidence

    def update_performance(self, agent_name: str, success: bool) -> None:
        stats = self.agent_performance.get(agent_name)
        if not stats:
            return
        stats["total"] += 1
        if success:
            stats["wins"] += 1
