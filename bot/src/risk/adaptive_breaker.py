"""Adaptive circuit breakers based on recent performance."""

from __future__ import annotations

from typing import Dict, List


class AdaptiveCircuitBreaker:
    """
    Dynamic risk limits that adapt to recent performance.

    Increases risk when performing well, tightens when underperforming.
    """

    def __init__(self, max_window_size: int = 100):
        self.performance_window: List[float] = []
        self.max_window_size = max_window_size
        self.current_multiplier = 1.0

    def update(self, trade_pnl: float) -> None:
        self.performance_window.append(trade_pnl)
        if len(self.performance_window) > self.max_window_size:
            self.performance_window.pop(0)

        recent_return = sum(self.performance_window) / max(len(self.performance_window), 1)

        if recent_return > 0.02:
            self.current_multiplier = min(1.5, self.current_multiplier + 0.1)
        elif recent_return < -0.01:
            self.current_multiplier = max(0.5, self.current_multiplier - 0.1)

    def get_adjusted_limits(self, base_limits: Dict[str, float]) -> Dict[str, float]:
        return {
            "max_position_size": base_limits["max_position_size"] * self.current_multiplier,
            "max_daily_loss": base_limits["max_daily_loss"] * self.current_multiplier,
            "stop_loss_pct": base_limits["stop_loss_pct"] * self.current_multiplier,
        }
