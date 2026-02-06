"""
Strategy-specific environment wrappers.

These subclasses adjust rewards based on market regime heuristics
while keeping the same observation space and execution logic.
"""

from __future__ import annotations

from typing import Dict

from .gym_env import CryptoTradingEnv


class MomentumTradingEnv(CryptoTradingEnv):
    """Environment optimized for momentum trading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_threshold = float(self.reward_config.get("momentum_threshold", 0.03))
        self.volume_surge_threshold = float(self.reward_config.get("momentum_volume_surge", 1.5))

    def _calculate_reward(self, trade_result: Dict, current_portfolio_value: float) -> float:
        reward = super()._calculate_reward(trade_result, current_portfolio_value)

        if trade_result.get("executed"):
            row = self._current_row()
            momentum = float(row["return_24h"])
            volume_surge = float(row["volume"]) / max(float(row["volume_ma_24"]), 1.0)

            if abs(momentum) >= self.momentum_threshold and volume_surge >= self.volume_surge_threshold:
                reward += float(self.reward_config.get("momentum_bonus", 2.0))
            elif abs(momentum) < self.momentum_threshold:
                reward -= float(self.reward_config.get("momentum_penalty", 0.5))

        return float(reward)


class MeanReversionTradingEnv(CryptoTradingEnv):
    """Environment optimized for mean reversion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zscore_threshold = float(self.reward_config.get("range_zscore_threshold", 1.0))
        self.trend_slope_cap = float(self.reward_config.get("range_trend_slope_cap", 0.001))

    def _calculate_reward(self, trade_result: Dict, current_portfolio_value: float) -> float:
        reward = super()._calculate_reward(trade_result, current_portfolio_value)

        if trade_result.get("executed"):
            row = self._current_row()
            zscore = float(row["price_zscore_24h"])
            trend_slope = float(row["ma_200_slope"])

            is_ranging = abs(trend_slope) <= self.trend_slope_cap
            if is_ranging and abs(zscore) >= self.zscore_threshold:
                reward += float(self.reward_config.get("range_adherence_bonus", 1.5))
            elif not is_ranging:
                reward -= float(self.reward_config.get("range_violation_penalty", 0.5))

        return float(reward)


class BreakoutTradingEnv(CryptoTradingEnv):
    """Environment optimized for breakout trading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.breakout_return_threshold = float(self.reward_config.get("breakout_return_threshold", 0.03))
        self.breakout_volume_threshold = float(self.reward_config.get("breakout_volume_threshold", 1.8))

    def _calculate_reward(self, trade_result: Dict, current_portfolio_value: float) -> float:
        reward = super()._calculate_reward(trade_result, current_portfolio_value)

        if trade_result.get("executed"):
            row = self._current_row()
            return_1h = float(row["return_1h"])
            volume_surge = float(row["volume"]) / max(float(row["volume_ma_24"]), 1.0)

            is_breakout = abs(return_1h) >= self.breakout_return_threshold
            if is_breakout and volume_surge >= self.breakout_volume_threshold:
                reward += float(self.reward_config.get("consolidation_detection_bonus", 2.0))
            elif is_breakout and volume_surge < self.breakout_volume_threshold:
                reward -= float(self.reward_config.get("breakout_without_volume_penalty", 0.5))

        return float(reward)
