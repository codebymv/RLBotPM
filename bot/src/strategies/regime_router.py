"""Regime router for specialist policy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class RegimeRouterConfig:
    """Thresholds for mapping observations to a market regime."""

    momentum_abs_return_24h: float = 0.025
    breakout_abs_return_1h: float = 0.008
    low_volatility_threshold: float = 0.012
    momentum_ratio_threshold: float = 1.0
    trend_strength_threshold: float = 0.25


class RegimeRouter:
    """
    Selects one of the specialist regimes from a single observation.

    Observation indices follow `CryptoTradingEnv._get_observation()`:
    - obs[2]: return_1h
    - obs[4]: return_24h
    - obs[5]: volatility_24h
    - obs[8]: trend_direction
    - obs[14]: return_6h / volatility_24h
    """

    def __init__(self, config: RegimeRouterConfig | None = None):
        self.config = config or RegimeRouterConfig()

    def detect(self, observation: np.ndarray) -> str:
        """Return one of: momentum, mean_reversion, breakout."""
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.size < 15:
            return "momentum"

        ret_1h = float(obs[2])
        ret_24h = float(obs[4])
        volatility = abs(float(obs[5]))
        trend = abs(float(obs[8]))
        momentum_ratio = abs(float(obs[14]))

        if (
            abs(ret_24h) >= self.config.momentum_abs_return_24h
            and momentum_ratio >= self.config.momentum_ratio_threshold
        ):
            return "momentum"

        if (
            volatility <= self.config.low_volatility_threshold
            and abs(ret_1h) >= self.config.breakout_abs_return_1h
            and trend >= self.config.trend_strength_threshold
        ):
            return "breakout"

        return "mean_reversion"

    @classmethod
    def from_dict(cls, values: Dict[str, float] | None) -> "RegimeRouter":
        """Build from YAML dict with optional overrides."""
        if not values:
            return cls()

        cfg = RegimeRouterConfig(
            momentum_abs_return_24h=float(
                values.get("momentum_abs_return_24h", RegimeRouterConfig.momentum_abs_return_24h)
            ),
            breakout_abs_return_1h=float(
                values.get("breakout_abs_return_1h", RegimeRouterConfig.breakout_abs_return_1h)
            ),
            low_volatility_threshold=float(
                values.get("low_volatility_threshold", RegimeRouterConfig.low_volatility_threshold)
            ),
            momentum_ratio_threshold=float(
                values.get("momentum_ratio_threshold", RegimeRouterConfig.momentum_ratio_threshold)
            ),
            trend_strength_threshold=float(
                values.get("trend_strength_threshold", RegimeRouterConfig.trend_strength_threshold)
            ),
        )
        return cls(config=cfg)
