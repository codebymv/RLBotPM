"""
Position Sizing - Kelly Criterion based risk management

The Kelly Criterion calculates optimal position size based on:
- Win probability
- Win/loss ratio
- Available capital

We use a FRACTIONAL Kelly (25%) to be conservative.
This prevents over-betting and reduces variance.

Formula:
    Kelly% = (win_prob * win_amount - loss_prob * loss_amount) / win_amount
    Position = Capital * Kelly% * Fraction
"""

import numpy as np
from typing import Optional, Dict

from ..core.logger import get_logger
from ..core.config import get_settings


logger = get_logger(__name__)


class PositionSizer:
    """
    Calculate position sizes using Kelly Criterion
    
    Usage:
        sizer = PositionSizer()
        
        size = sizer.calculate_position_size(
            capital=1000,
            win_probability=0.55,
            avg_win=50,
            avg_loss=30
        )
    """
    
    def __init__(self, kelly_fraction: float = 0.25):
        """
        Initialize position sizer
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = 25% Kelly)
                           Lower values are more conservative
        """
        self.settings = get_settings()
        self.kelly_fraction = kelly_fraction
        
        logger.info(f"Position sizer initialized with {kelly_fraction:.0%} Kelly")

    def set_kelly_fraction(self, kelly_fraction: float) -> None:
        self.kelly_fraction = max(0.0, min(kelly_fraction, 1.0))
        logger.debug("Updated Kelly fraction to %.2f", self.kelly_fraction)
    
    def calculate_position_size(
        self,
        capital: float,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        model_confidence: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal position size
        
        Args:
            capital: Available capital
            win_probability: Probability of winning (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            model_confidence: Optional model confidence (0-1)
        
        Returns:
            Dict with:
                - suggested_size: Recommended position size
                - kelly_percentage: Full Kelly percentage
                - fractional_kelly: Fractional Kelly percentage used
                - max_size: Maximum allowed size (from risk limits)
        """
        # Validate inputs
        if not (0 < win_probability < 1):
            logger.debug("Invalid win probability %.3f, using 0.5", win_probability)
            win_probability = 0.5
        
        if avg_win <= 0 or avg_loss <= 0:
            logger.warning("Invalid win/loss amounts, using defaults")
            avg_win = max(avg_win, 1.0)
            avg_loss = max(avg_loss, 1.0)
        
        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        loss_probability = 1 - win_probability
        
        # Kelly formula
        kelly_pct = (
            (win_probability * win_loss_ratio - loss_probability) /
            win_loss_ratio
        )
        
        # Ensure Kelly is non-negative
        kelly_pct = max(kelly_pct, 0.0)
        
        # Apply fractional Kelly for safety
        fractional_kelly = kelly_pct * self.kelly_fraction
        
        # Calculate suggested size
        suggested_size = capital * fractional_kelly
        
        # Apply confidence adjustment if provided
        if model_confidence is not None:
            confidence_multiplier = self._confidence_to_multiplier(model_confidence)
            suggested_size *= confidence_multiplier
        
        # Enforce maximum position size from settings
        max_size = capital * self.settings.MAX_POSITION_SIZE_PCT
        final_size = min(suggested_size, max_size)
        
        result = {
            'suggested_size': final_size,
            'kelly_percentage': kelly_pct,
            'fractional_kelly': fractional_kelly,
            'max_size': max_size,
            'size_pct_of_capital': final_size / capital if capital > 0 else 0.0
        }
        
        logger.debug(
            f"Position sizing: Kelly={kelly_pct:.2%}, "
            f"Fractional={fractional_kelly:.2%}, "
            f"Size=${final_size:.2f} ({result['size_pct_of_capital']:.1%} of capital)"
        )
        
        return result
    
    def calculate_from_recent_trades(
        self,
        capital: float,
        recent_trades: list,
        model_confidence: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate position size based on recent trading history
        
        Args:
            capital: Available capital
            recent_trades: List of recent trade P&Ls
            model_confidence: Optional model confidence
        
        Returns:
            Position sizing recommendation
        """
        if not recent_trades or len(recent_trades) < 10:
            # Not enough history, use conservative default
            logger.debug("Insufficient trade history, using conservative sizing")
            return {
                'suggested_size': capital * 0.12,  # 12% default (increased for profitability)
                'kelly_percentage': 0.0,
                'fractional_kelly': 0.0,
                'max_size': capital * self.settings.MAX_POSITION_SIZE_PCT,
                'size_pct_of_capital': 0.12
            }
        
        # Calculate win probability and average win/loss
        wins = [t for t in recent_trades if t > 0]
        losses = [abs(t) for t in recent_trades if t < 0]
        
        win_probability = len(wins) / len(recent_trades)
        avg_win = np.mean(wins) if wins else 1.0
        avg_loss = np.mean(losses) if losses else 1.0
        
        return self.calculate_position_size(
            capital=capital,
            win_probability=win_probability,
            avg_win=avg_win,
            avg_loss=avg_loss,
            model_confidence=model_confidence
        )

    def calculate_with_correlation(
        self,
        capital: float,
        symbol: str,
        current_positions: Dict[str, Dict],
        model_confidence: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate position size with a simple correlation check.

        Reduces size when adding positions correlated to existing holdings.
        """
        base_sizing = self.calculate_position_size(
            capital=capital,
            win_probability=0.5,
            avg_win=1.0,
            avg_loss=1.0,
            model_confidence=model_confidence,
        )

        if not current_positions:
            return base_sizing

        if symbol.startswith("BTC") and any(
            pos_symbol.startswith("BTC") for pos_symbol in current_positions.keys()
        ):
            base_sizing["suggested_size"] *= 0.5

        return base_sizing
    
    def _confidence_to_multiplier(self, confidence: float) -> float:
        """
        Convert model confidence to position size multiplier
        
        Args:
            confidence: Model confidence (0-1)
        
        Returns:
            Multiplier (0.5-1.5)
        """
        # Map confidence to multiplier
        # 0.0 confidence -> 0.5x size
        # 0.5 confidence -> 1.0x size
        # 1.0 confidence -> 1.5x size
        multiplier = 0.5 + confidence
        return np.clip(multiplier, 0.5, 1.5)
    
    def validate_position_size(
        self,
        position_size: float,
        capital: float,
        current_positions: int
    ) -> tuple[bool, str]:
        """
        Validate if a position size is acceptable
        
        Args:
            position_size: Proposed position size
            capital: Available capital
            current_positions: Number of current open positions
        
        Returns:
            (is_valid, reason)
        """
        # Check if we have capital
        if position_size > capital:
            return False, f"Insufficient capital (need ${position_size:.2f}, have ${capital:.2f})"
        
        # Check maximum position size
        max_size = capital * self.settings.MAX_POSITION_SIZE_PCT
        if position_size > max_size:
            return False, f"Exceeds max position size ({self.settings.MAX_POSITION_SIZE_PCT:.0%} of capital)"
        
        # Check maximum number of positions
        if current_positions >= self.settings.MAX_OPEN_POSITIONS:
            return False, f"Maximum open positions reached ({self.settings.MAX_OPEN_POSITIONS})"
        
        # Check minimum position size (avoid dust trades)
        min_size = capital * 0.01  # 1% minimum
        if 0 < position_size < min_size:
            return False, f"Position too small (min ${min_size:.2f})"
        
        return True, "Position size validated"
