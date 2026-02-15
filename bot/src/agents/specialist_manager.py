"""Specialist policy manager with regime-based routing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .ppo_agent import PPOAgent
from ..strategies.regime_router import RegimeRouter


class SpecialistManager:
    """Loads specialist PPO models and routes inference by detected regime."""

    SUPPORTED_REGIMES = ("momentum", "mean_reversion", "breakout")

    def __init__(
        self,
        env,
        model_paths: Dict[str, str],
        policy_type: str = "MlpPolicy",
        router: Optional[RegimeRouter] = None,
        use_gpu: bool = False,
    ):
        self.router = router or RegimeRouter()
        self.policy_type = policy_type
        self.agents: Dict[str, PPOAgent] = {}
        self.route_counts: Dict[str, int] = {name: 0 for name in self.SUPPORTED_REGIMES}

        for regime in self.SUPPORTED_REGIMES:
            raw_path = str(model_paths.get(regime, "")).strip()
            if not raw_path:
                continue

            model_path = self._normalize_model_path(raw_path)
            if not model_path:
                continue

            agent = PPOAgent(env=env, policy_type=policy_type, use_gpu=use_gpu)
            agent.load(model_path)
            self.agents[regime] = agent

        if not self.agents:
            raise ValueError("No specialist models could be loaded. Check specialist_router.model_paths.")

        self.default_regime = "momentum" if "momentum" in self.agents else next(iter(self.agents.keys()))

    @staticmethod
    def _normalize_model_path(path_str: str) -> Optional[str]:
        path = Path(path_str)
        if path.exists():
            return str(path)
        if path.suffix != ".zip":
            zip_path = Path(f"{path_str}.zip")
            if zip_path.exists():
                return str(path)
        return None

    @classmethod
    def from_config(cls, env, config: Dict, policy_type: str = "MlpPolicy", use_gpu: bool = False):
        model_paths = config.get("model_paths", {}) or {}
        router_cfg = config.get("router", {}) or {}
        router = RegimeRouter.from_dict(router_cfg)
        return cls(
            env=env,
            model_paths=model_paths,
            policy_type=policy_type,
            router=router,
            use_gpu=use_gpu,
        )

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        state=None,
        episode_start=None,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[int, object, str]:
        regime = self.router.detect(observation)
        if regime not in self.agents:
            regime = self.default_regime

        self.route_counts[regime] = self.route_counts.get(regime, 0) + 1
        action, next_state = self.agents[regime].predict(
            observation,
            deterministic=deterministic,
            state=state,
            episode_start=episode_start,
            action_masks=action_masks,
        )
        action_int = int(action) if isinstance(action, (np.integer, np.ndarray)) else int(action)
        return action_int, next_state, regime
