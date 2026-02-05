"""
Circuit Breaker - Safety system for trading bot

This system enforces hard stops and safety limits that cannot be bypassed.
The bot will automatically pause trading if any circuit breaker condition is met.

Circuit breakers protect against:
- Excessive losses (daily, weekly, total)
- Overexposure (position size, number of positions)
- Poor performance (consecutive losses, low win rate)
- Technical issues (API errors, data feed problems)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

from ..core.logger import get_logger
from ..core.config import get_settings


logger = get_logger(__name__)


class CircuitBreakerStatus(Enum):
    """Circuit breaker states"""
    ACTIVE = "active"  # Trading allowed
    PAUSED = "paused"  # Trading paused, awaiting review
    TRIGGERED = "triggered"  # Circuit breaker just triggered


@dataclass
class CircuitBreakerEvent:
    """Records when a circuit breaker triggers"""
    timestamp: datetime
    rule_violated: str
    description: str
    current_value: float
    threshold: float
    severity: str  # 'warning' or 'critical'


class CircuitBreaker:
    """
    Enforces safety limits and stops trading when violated
    
    Usage:
        breaker = CircuitBreaker()
        
        # Check before each trade
        if not breaker.can_trade():
            print("Trading paused due to circuit breaker")
            return
        
        # Record trade outcome
        breaker.record_trade(pnl=50.0, capital=1050.0)
    """
    
    def __init__(self):
        """Initialize circuit breaker with configuration from settings"""
        self.settings = get_settings()
        
        # Status
        self.status = CircuitBreakerStatus.ACTIVE
        self.events: List[CircuitBreakerEvent] = []
        self._last_triggered_at: Dict[str, datetime] = {}
        
        # Tracking metrics
        self.daily_pnl: Dict[str, float] = {}  # date -> total P&L
        self.weekly_pnl: Dict[str, float] = {}  # week -> total P&L
        self.peak_capital = self.settings.INITIAL_CAPITAL
        self.current_capital = self.settings.INITIAL_CAPITAL
        
        # Trade tracking
        self.recent_trades: List[Dict] = []  # Last 20 trades
        self.consecutive_losses = 0
        
        # Error tracking
        self.api_error_count = 0
        self.last_api_error: Optional[datetime] = None
        
        logger.info("Circuit breaker initialized")
    
    def can_trade(self) -> bool:
        """
        Check if trading is allowed
        
        Returns:
            True if trading is allowed, False if paused
        """
        if self.status == CircuitBreakerStatus.PAUSED:
            logger.warning("Trading blocked: Circuit breaker is PAUSED")
            return False
        
        # Run all checks
        self._check_daily_loss()
        self._check_weekly_loss()
        self._check_drawdown()
        self._check_consecutive_losses()
        self._check_win_rate()
        self._check_api_errors()
        
        return self.status == CircuitBreakerStatus.ACTIVE
    
    def record_trade(
        self,
        pnl: float,
        capital: float,
        is_win: bool,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a trade for tracking
        
        Args:
            pnl: Profit/loss from the trade
            capital: Current capital after trade
            is_win: Whether the trade was profitable
            timestamp: Trade timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.now()
        
        # Update capital tracking
        self.current_capital = capital
        self.peak_capital = max(self.peak_capital, capital)
        
        # Update daily P&L
        date_key = timestamp.strftime("%Y-%m-%d")
        self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0.0) + pnl
        
        # Update weekly P&L
        week_key = timestamp.strftime("%Y-W%W")
        self.weekly_pnl[week_key] = self.weekly_pnl.get(week_key, 0.0) + pnl
        
        # Track trade
        trade = {
            'timestamp': timestamp,
            'pnl': pnl,
            'is_win': is_win,
            'capital': capital
        }
        self.recent_trades.append(trade)
        
        # Keep only last 20 trades
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
        
        # Update consecutive losses
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        logger.debug(f"Trade recorded: P&L=${pnl:.2f}, Capital=${capital:.2f}")
    
    def record_api_error(self):
        """Record an API error"""
        self.api_error_count += 1
        self.last_api_error = datetime.now()
        logger.warning(f"API error recorded (count: {self.api_error_count})")
    
    def reset_api_errors(self):
        """Reset API error counter (after successful connection)"""
        if self.api_error_count > 0:
            logger.info("API errors reset after successful connection")
        self.api_error_count = 0
        self.last_api_error = None
    
    def manual_resume(self):
        """
        Manually resume trading after review
        
        This should only be called after human review of the
        circuit breaker trigger.
        """
        logger.warning("Circuit breaker manually resumed by human")
        self.status = CircuitBreakerStatus.ACTIVE
        
        # Reset some counters
        self.consecutive_losses = 0
        self.api_error_count = 0
    
    def get_status_report(self) -> Dict:
        """
        Get current status and metrics
        
        Returns:
            Dict with status information
        """
        today = datetime.now().strftime("%Y-%m-%d")
        this_week = datetime.now().strftime("%Y-W%W")
        
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        win_count = sum(1 for t in self.recent_trades if t['is_win'])
        win_rate = win_count / len(self.recent_trades) if self.recent_trades else 0.0
        
        return {
            'status': self.status.value,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown': drawdown,
            'daily_pnl': self.daily_pnl.get(today, 0.0),
            'weekly_pnl': self.weekly_pnl.get(this_week, 0.0),
            'consecutive_losses': self.consecutive_losses,
            'win_rate_last_20': win_rate,
            'api_error_count': self.api_error_count,
            'recent_events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'rule': e.rule_violated,
                    'description': e.description
                }
                for e in self.events[-5:]  # Last 5 events
            ]
        }
    
    def _check_daily_loss(self):
        """Check daily loss limit"""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_pnl = self.daily_pnl.get(today, 0.0)
        
        # Check absolute loss
        if daily_pnl < -self.settings.MAX_DAILY_LOSS_USD:
            self._trigger(
                rule="MAX_DAILY_LOSS_USD",
                description=f"Daily loss ${abs(daily_pnl):.2f} exceeds limit ${self.settings.MAX_DAILY_LOSS_USD}",
                current_value=daily_pnl,
                threshold=-self.settings.MAX_DAILY_LOSS_USD
            )
        
        # Check percentage loss
        daily_loss_pct = abs(daily_pnl) / self.settings.INITIAL_CAPITAL
        if daily_loss_pct > self.settings.MAX_DAILY_LOSS_PCT:
            self._trigger(
                rule="MAX_DAILY_LOSS_PCT",
                description=f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.settings.MAX_DAILY_LOSS_PCT:.2%}",
                current_value=daily_loss_pct,
                threshold=self.settings.MAX_DAILY_LOSS_PCT
            )
    
    def _check_weekly_loss(self):
        """Check weekly loss limit"""
        this_week = datetime.now().strftime("%Y-W%W")
        weekly_pnl = self.weekly_pnl.get(this_week, 0.0)
        
        # Check absolute loss
        if weekly_pnl < -self.settings.MAX_WEEKLY_LOSS_USD:
            self._trigger(
                rule="MAX_WEEKLY_LOSS_USD",
                description=f"Weekly loss ${abs(weekly_pnl):.2f} exceeds limit ${self.settings.MAX_WEEKLY_LOSS_USD}",
                current_value=weekly_pnl,
                threshold=-self.settings.MAX_WEEKLY_LOSS_USD
            )
        
        # Check percentage loss
        weekly_loss_pct = abs(weekly_pnl) / self.settings.INITIAL_CAPITAL
        if weekly_loss_pct > self.settings.MAX_WEEKLY_LOSS_PCT:
            self._trigger(
                rule="MAX_WEEKLY_LOSS_PCT",
                description=f"Weekly loss {weekly_loss_pct:.2%} exceeds limit {self.settings.MAX_WEEKLY_LOSS_PCT:.2%}",
                current_value=weekly_loss_pct,
                threshold=self.settings.MAX_WEEKLY_LOSS_PCT
            )
    
    def _check_drawdown(self):
        """Check maximum drawdown"""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if drawdown > self.settings.MAX_TOTAL_DRAWDOWN:
            self._trigger(
                rule="MAX_TOTAL_DRAWDOWN",
                description=f"Drawdown {drawdown:.2%} exceeds limit {self.settings.MAX_TOTAL_DRAWDOWN:.2%}",
                current_value=drawdown,
                threshold=self.settings.MAX_TOTAL_DRAWDOWN
            )
    
    def _check_consecutive_losses(self):
        """Check consecutive losses"""
        if self.consecutive_losses >= self.settings.MAX_CONSECUTIVE_LOSSES:
            self._trigger(
                rule="MAX_CONSECUTIVE_LOSSES",
                description=f"{self.consecutive_losses} consecutive losses (limit: {self.settings.MAX_CONSECUTIVE_LOSSES})",
                current_value=self.consecutive_losses,
                threshold=self.settings.MAX_CONSECUTIVE_LOSSES
            )
    
    def _check_win_rate(self):
        """Check win rate threshold"""
        if len(self.recent_trades) < 20:
            return  # Need at least 20 trades
        
        win_count = sum(1 for t in self.recent_trades if t['is_win'])
        win_rate = win_count / len(self.recent_trades)
        
        if win_rate < self.settings.MIN_WIN_RATE_THRESHOLD:
            self._trigger(
                rule="MIN_WIN_RATE_THRESHOLD",
                description=f"Win rate {win_rate:.2%} below threshold {self.settings.MIN_WIN_RATE_THRESHOLD:.2%}",
                current_value=win_rate,
                threshold=self.settings.MIN_WIN_RATE_THRESHOLD
            )
    
    def _check_api_errors(self):
        """Check API error count"""
        if self.api_error_count >= 3:
            self._trigger(
                rule="API_ERROR_THRESHOLD",
                description=f"{self.api_error_count} API errors detected",
                current_value=self.api_error_count,
                threshold=3
            )
    
    def _trigger(
        self,
        rule: str,
        description: str,
        current_value: float,
        threshold: float,
        severity: str = "critical"
    ):
        """
        Trigger a circuit breaker
        
        Args:
            rule: Rule that was violated
            description: Human-readable description
            current_value: Current metric value
            threshold: Threshold that was exceeded
            severity: 'warning' or 'critical'
        """
        now = datetime.now()

        if self.settings.TRAINING_MODE:
            last_triggered = self._last_triggered_at.get(rule)
            if last_triggered and (now - last_triggered) < timedelta(seconds=60):
                return

            self._last_triggered_at[rule] = now
            event = CircuitBreakerEvent(
                timestamp=now,
                rule_violated=rule,
                description=description,
                current_value=current_value,
                threshold=threshold,
                severity="warning"
            )
            self.events.append(event)
            self.status = CircuitBreakerStatus.TRIGGERED

            logger.warning(f"Circuit breaker would trigger (training mode): {description}")
            return

        event = CircuitBreakerEvent(
            timestamp=now,
            rule_violated=rule,
            description=description,
            current_value=current_value,
            threshold=threshold,
            severity=severity
        )

        self.events.append(event)
        self.status = CircuitBreakerStatus.PAUSED

        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {description}")
        logger.critical("Trading PAUSED - Human review required")
