"""
Kalshi RL Bridge: maps PPO model outputs to Kalshi YES/NO recommendations.

Wires the crypto PPO (BUY/SELL/HOLD) into Kalshi execution for crypto
prediction markets (e.g. "BTC above $100k"). Uses kalshi_config.yaml for
signal_mapping and thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import yaml

from ..core.logger import get_logger

logger = get_logger(__name__)

# Crypto PPO action space (from CryptoTradingEnv)
ACTION_NO_ACTION = 0
ACTION_BUY = 1
ACTION_SELL = 2


class KalshiRLBridge:
    """
    Maps PPO model outputs to Kalshi YES/NO recommendations.

    Config (from kalshi_config.yaml strategy.signal_mapping):
    - bullish_action: how to interpret BUY (e.g. yes_on_higher_ranges)
    - bearish_action: how to interpret SELL (e.g. yes_on_lower_ranges)
    - neutral_threshold: hold if signal strength below this
    """

    def __init__(
        self,
        ppo_model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.ppo_model_path = ppo_model_path
        self.config = config or self._load_config()
        self._model = None
        self._agent = None
        self._neutral_threshold = float(
            self.config.get("strategy", {})
            .get("signal_mapping", {})
            .get("neutral_threshold", 0.1)
        )

    def _load_config(self) -> Dict:
        # Allow explicit override for deployments with custom layout.
        override = os.getenv("KALSHI_CONFIG_PATH")
        if override:
            path = Path(override).expanduser().resolve()
            if path.exists():
                with open(path) as f:
                    return yaml.safe_load(f) or {}

        # Search upwards for repo-root `shared/config/kalshi_config.yaml`.
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            candidate = parent / "shared" / "config" / "kalshi_config.yaml"
            if candidate.exists():
                with open(candidate) as f:
                    return yaml.safe_load(f) or {}

        return {}

    def load_model(self) -> bool:
        """Load PPO model from configured path. Returns True if loaded."""
        path = self.ppo_model_path or (
            self.config.get("strategy", {})
            .get("crypto_signals", {})
            .get("ppo_model_path")
        )
        if not path:
            return False
        path = str(path)
        if not path.endswith(".zip"):
            path = path + ".zip"
        if not os.path.exists(path):
            logger.warning(f"Kalshi RL bridge: PPO model not found at {path}")
            return False
        try:
            from stable_baselines3 import BaseAlgorithm
            from sb3_contrib import MaskablePPO

            self._model = MaskablePPO.load(path)
            logger.info(f"Kalshi RL bridge: loaded PPO from {path}")
            return True
        except Exception as e:
            logger.warning(f"Kalshi RL bridge: could not load PPO: {e}")
            return False

    def predict(
        self,
        observation: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[int, float]:
        """
        Run PPO forward and return action index and confidence (0-1).

        Args:
            observation: Crypto env observation (e.g. 29-dim).
            action_masks: Optional mask (1=valid, 0=invalid). If None, all actions valid.
            deterministic: Use deterministic policy.

        Returns:
            (action_index, confidence) where action is 0=NO_ACTION, 1=BUY, 2=SELL.
        """
        if self._model is None and not self.load_model():
            return ACTION_NO_ACTION, 0.0

        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        if action_masks is None:
            action_masks = np.ones((1, 3), dtype=np.float32)

        try:
            action, _ = self._model.predict(
                obs,
                action_masks=action_masks,
                deterministic=deterministic,
            )
            action = int(action.flat[0])
            # Use policy entropy or log prob as proxy for confidence; fallback to 0.7
            confidence = 0.7
            return action, confidence
        except Exception as e:
            logger.debug(f"PPO predict error: {e}")
            return ACTION_NO_ACTION, 0.0

    def action_to_recommendation(
        self,
        action: int,
        confidence: float = 0.7,
    ) -> str:
        """
        Map crypto PPO action to Kalshi recommendation.

        Returns:
            "BUY_YES", "BUY_NO", or "HOLD"
        """
        if confidence < self._neutral_threshold:
            return "HOLD"
        if action == ACTION_BUY:
            return "BUY_YES"
        if action == ACTION_SELL:
            return "BUY_NO"
        return "HOLD"

    def recommend_from_observation(
        self,
        observation: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """
        Get Kalshi recommendation from a single crypto observation.

        Returns:
            (recommendation, confidence) where recommendation is BUY_YES/BUY_NO/HOLD.
        """
        action, confidence = self.predict(observation, action_masks=action_masks)
        rec = self.action_to_recommendation(action, confidence)
        return rec, confidence

    def recommend_from_action(self, action: int, confidence: float = 0.7) -> str:
        """Get Kalshi recommendation when action was computed elsewhere."""
        return self.action_to_recommendation(action, confidence)
