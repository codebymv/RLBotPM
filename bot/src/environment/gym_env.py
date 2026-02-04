"""
Custom Gymnasium Environment for Polymarket Trading

This is the core environment where the RL agent "lives" and makes trading decisions.
The environment simulates realistic market conditions including:
- Transaction costs
- Slippage
- Liquidity constraints
- Market impact

STATE SPACE (38 dimensions):
- Market features (15): prices, spread, volume, volatility, etc.
- Portfolio features (8): capital, positions, P&L, etc.
- Temporal features (5): time of day, day of week
- Historical context (10): similar market outcomes, category performance

ACTION SPACE (8 discrete actions):
0: NO_ACTION    - Hold current position
1: BUY_SMALL    - Buy 5% of capital
2: BUY_MEDIUM   - Buy 10% of capital
3: BUY_LARGE    - Buy 20% of capital
4: SELL_SMALL   - Close 33% of position
5: SELL_MEDIUM  - Close 66% of position
6: SELL_LARGE   - Close 100% of position
7: CLOSE_ALL    - Emergency exit all positions

REWARD FUNCTION:
Reward = Sharpe ratio component
         - Transaction costs
         - Drawdown penalty
         + Profitable close bonus
         - Risk violation penalty
         + Long-term profitability component
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import random

from ..core.logger import get_logger
from ..core.config import get_settings


logger = get_logger(__name__)


class PolymarketTradingEnv(gym.Env):
    """
    Custom trading environment for Polymarket
    
    This environment simulates trading on Polymarket prediction markets
    with realistic constraints and costs.
    """
    
    metadata = {'render_modes': ['human']}
    
    # Action space constants
    ACTION_NO_ACTION = 0
    ACTION_BUY_SMALL = 1
    ACTION_BUY_MEDIUM = 2
    ACTION_BUY_LARGE = 3
    ACTION_SELL_SMALL = 4
    ACTION_SELL_MEDIUM = 5
    ACTION_SELL_LARGE = 6
    ACTION_CLOSE_ALL = 7
    
    ACTION_NAMES = {
        0: "NO_ACTION",
        1: "BUY_SMALL",
        2: "BUY_MEDIUM",
        3: "BUY_LARGE",
        4: "SELL_SMALL",
        5: "SELL_MEDIUM",
        6: "SELL_LARGE",
        7: "CLOSE_ALL"
    }
    
    def __init__(
        self,
        initial_capital: Optional[float] = None,
        max_steps: int = 500,
        transaction_cost: Optional[float] = None
    ):
        """
        Initialize the trading environment
        
        Args:
            initial_capital: Starting capital (defaults to config)
            max_steps: Maximum steps per episode
            transaction_cost: Transaction cost percentage (defaults to config)
        """
        super().__init__()
        
        # Load settings
        self.settings = get_settings()
        self.initial_capital = initial_capital or self.settings.INITIAL_CAPITAL
        self.transaction_cost = transaction_cost or self.settings.TRANSACTION_COST_PCT
        self.max_steps = max_steps
        
        # Define action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)
        
        # Define observation space: 38-dimensional continuous state
        # All features normalized to roughly [-1, 1] or [0, 1]
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(38,),
            dtype=np.float32
        )
        
        # Portfolio state
        self.capital = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # market_id -> {size, entry_price, timestamp}
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.episode_returns: List[float] = []
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        
        # Episode state
        self.current_step = 0
        self.current_market = None
        self.terminated = False
        
        # Historical performance (for meta-features)
        self.recent_trades: List[float] = []  # Recent P&Ls
        self.win_count = 0
        self.trade_count = 0
        
        logger.info(f"Environment initialized with capital ${self.initial_capital}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset portfolio
        self.capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        
        # Reset performance tracking
        self.episode_returns = []
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        
        # Reset episode state
        self.current_step = 0
        self.terminated = False
        
        # Sample a random market for this episode
        # TODO: Load actual market data
        self.current_market = self._sample_market()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Episode reset: {info}")
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action index to take
        
        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        if self.terminated:
            logger.warning("Step called on terminated environment")
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Execute the action
        trade_result = self._execute_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(trade_result)
        
        # Update state
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        info['trade_result'] = trade_result
        
        if terminated or truncated:
            self.terminated = True
            logger.info(f"Episode ended: {info}")
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> Dict:
        """
        Execute a trading action
        
        Args:
            action: Action index
        
        Returns:
            Dict with trade execution details
        """
        action_name = self.ACTION_NAMES.get(action, "UNKNOWN")
        
        result = {
            'action': action,
            'action_name': action_name,
            'executed': False,
            'reason': None,
            'pnl': 0.0,
            'cost': 0.0
        }
        
        # NO_ACTION - do nothing
        if action == self.ACTION_NO_ACTION:
            result['reason'] = 'No action taken'
            return result
        
        # BUY actions
        if action in [self.ACTION_BUY_SMALL, self.ACTION_BUY_MEDIUM, self.ACTION_BUY_LARGE]:
            size_pct = {
                self.ACTION_BUY_SMALL: 0.05,
                self.ACTION_BUY_MEDIUM: 0.10,
                self.ACTION_BUY_LARGE: 0.20
            }[action]
            
            return self._execute_buy(size_pct)
        
        # SELL actions
        if action in [self.ACTION_SELL_SMALL, self.ACTION_SELL_MEDIUM, self.ACTION_SELL_LARGE, self.ACTION_CLOSE_ALL]:
            close_pct = {
                self.ACTION_SELL_SMALL: 0.33,
                self.ACTION_SELL_MEDIUM: 0.66,
                self.ACTION_SELL_LARGE: 1.0,
                self.ACTION_CLOSE_ALL: 1.0
            }[action]
            
            return self._execute_sell(close_pct)
        
        return result
    
    def _execute_buy(self, size_pct: float) -> Dict:
        """Execute a buy order"""
        # Calculate position size
        position_value = self.capital * size_pct
        
        # Check if we can afford it
        cost = position_value * (1 + self.transaction_cost)
        if cost > self.capital:
            return {
                'action_name': f'BUY_{int(size_pct*100)}%',
                'executed': False,
                'reason': 'Insufficient capital',
                'pnl': 0.0,
                'cost': 0.0
            }
        
        # Check position limits
        if len(self.positions) >= self.settings.MAX_OPEN_POSITIONS:
            return {
                'action_name': f'BUY_{int(size_pct*100)}%',
                'executed': False,
                'reason': 'Max positions reached',
                'pnl': 0.0,
                'cost': 0.0
            }
        
        # Execute buy (simplified - assumes can always buy at current price)
        current_price = self.current_market['current_price']
        market_id = self.current_market['id']
        
        # Deduct from capital
        self.capital -= cost
        transaction_cost_paid = position_value * self.transaction_cost
        
        # Add position
        self.positions[market_id] = {
            'size': position_value,
            'entry_price': current_price,
            'timestamp': self.current_step
        }
        
        self.trade_count += 1
        
        return {
            'action_name': f'BUY_{int(size_pct*100)}%',
            'executed': True,
            'reason': 'Buy executed',
            'size': position_value,
            'price': current_price,
            'cost': transaction_cost_paid,
            'pnl': 0.0
        }
    
    def _execute_sell(self, close_pct: float) -> Dict:
        """Execute a sell order"""
        market_id = self.current_market['id']
        
        # Check if we have a position
        if market_id not in self.positions:
            return {
                'action_name': f'SELL_{int(close_pct*100)}%',
                'executed': False,
                'reason': 'No position to close',
                'pnl': 0.0,
                'cost': 0.0
            }
        
        position = self.positions[market_id]
        current_price = self.current_market['current_price']
        
        # Calculate sell amount
        sell_size = position['size'] * close_pct
        
        # Calculate P&L
        price_change = (current_price - position['entry_price']) / position['entry_price']
        pnl_before_costs = sell_size * price_change
        transaction_cost_paid = sell_size * self.transaction_cost
        pnl = pnl_before_costs - transaction_cost_paid
        
        # Add proceeds to capital
        self.capital += sell_size + pnl
        
        # Update position
        if close_pct >= 0.99:  # Close completely
            del self.positions[market_id]
        else:  # Partial close
            position['size'] *= (1 - close_pct)
        
        # Track performance
        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 10:
            self.recent_trades.pop(0)
        
        if pnl > 0:
            self.win_count += 1
        
        return {
            'action_name': f'SELL_{int(close_pct*100)}%',
            'executed': True,
            'reason': 'Sell executed',
            'size': sell_size,
            'price': current_price,
            'pnl': pnl,
            'cost': transaction_cost_paid
        }
    
    def _calculate_reward(self, trade_result: Dict) -> float:
        """
        Calculate reward for current step
        
        Reward components:
        1. Returns (normalized)
        2. Transaction costs (penalty)
        3. Drawdown penalty
        4. Position close bonus (if profitable)
        5. Risk violation penalty
        """
        reward = 0.0
        
        # Component 1: Portfolio value change
        current_portfolio_value = self._get_portfolio_value()
        returns = (current_portfolio_value - self.initial_capital) / self.initial_capital
        reward += returns * 10.0  # Scale up for learning
        
        # Component 2: Transaction cost penalty
        if trade_result['executed']:
            cost_penalty = trade_result['cost'] / self.initial_capital
            reward -= cost_penalty * 100.0  # Penalize costs
        
        # Component 3: Drawdown penalty (exponential)
        self.peak_capital = max(self.peak_capital, current_portfolio_value)
        self.current_drawdown = (self.peak_capital - current_portfolio_value) / self.peak_capital
        
        if self.current_drawdown > 0.20:  # Drawdown > 20%
            drawdown_penalty = np.exp(self.current_drawdown * 5) - 1
            reward -= drawdown_penalty * 10.0
        
        # Component 4: Profitable position close bonus
        if trade_result.get('pnl', 0) > 0:
            reward += 0.5  # Small bonus for profitable trades
        
        # Component 5: Risk violation penalty
        if len(self.positions) > self.settings.MAX_OPEN_POSITIONS:
            reward -= 1.0
        
        # Component 6: Long-term profitability (moving average of recent trades)
        if len(self.recent_trades) >= 5:
            avg_recent_pnl = np.mean(self.recent_trades)
            reward += (avg_recent_pnl / self.initial_capital) * 3.0
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if capital too low (bankruptcy)
        if self.capital < self.initial_capital * 0.1:
            logger.warning("Episode terminated: capital too low")
            return True
        
        # Terminate if max drawdown exceeded
        if self.current_drawdown > self.settings.MAX_TOTAL_DRAWDOWN:
            logger.warning(f"Episode terminated: drawdown {self.current_drawdown:.2%}")
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation (38 dimensions)
        
        Returns:
            numpy array of state features
        """
        obs = np.zeros(38, dtype=np.float32)
        
        # Market features (15 dimensions) [0-14]
        if self.current_market:
            obs[0] = self.current_market['current_price']  # Current price
            obs[1] = self.current_market['bid_ask_spread']
            obs[2] = np.log1p(self.current_market['volume_24h']) / 10.0  # Log-scaled volume
            obs[3] = np.log1p(self.current_market['liquidity']) / 10.0
            obs[4] = self.current_market['time_until_resolution'] / 86400.0  # Normalized to days
            obs[5] = self.current_market['time_since_creation'] / 86400.0
            obs[6] = self.current_market['price_change_1h']
            obs[7] = self.current_market['price_change_6h']
            obs[8] = self.current_market['price_change_24h']
            obs[9] = self.current_market['volatility_24h']
            obs[10] = self.current_market['trend_direction']  # -1, 0, 1
            obs[11] = self.current_market['volume_trend']
            obs[12:15] = self.current_market['category_embedding']  # 3D embedding
        
        # Portfolio features (8 dimensions) [15-22]
        portfolio_value = self._get_portfolio_value()
        obs[15] = self.capital / self.initial_capital  # Normalized capital
        obs[16] = len(self.positions) / self.settings.MAX_OPEN_POSITIONS  # Position count (normalized)
        obs[17] = portfolio_value / self.initial_capital  # Total portfolio value
        obs[18] = (portfolio_value - self.initial_capital) / self.initial_capital  # Unrealized P&L
        obs[19] = sum([t for t in self.recent_trades]) / self.initial_capital if self.recent_trades else 0.0  # Realized P&L
        
        # Current position in this market
        market_id = self.current_market['id'] if self.current_market else None
        if market_id and market_id in self.positions:
            obs[20] = self.positions[market_id]['size'] / self.initial_capital
        else:
            obs[20] = 0.0
        
        obs[21] = self.current_step / self.max_steps  # Progress through episode
        obs[22] = self.win_count / max(self.trade_count, 1)  # Win rate
        
        # Temporal features (5 dimensions) [23-27]
        now = datetime.now()
        obs[23] = np.sin(2 * np.pi * now.hour / 24)  # Hour of day (sin)
        obs[24] = np.cos(2 * np.pi * now.hour / 24)  # Hour of day (cos)
        obs[25] = np.sin(2 * np.pi * now.weekday() / 7)  # Day of week (sin)
        obs[26] = np.cos(2 * np.pi * now.weekday() / 7)  # Day of week (cos)
        obs[27] = 1.0 if now.weekday() >= 5 else 0.0  # Is weekend
        
        # Historical context (10 dimensions) [28-37]
        # These would be populated with market history analysis
        # For now, placeholder with recent performance
        if len(self.recent_trades) > 0:
            obs[28] = np.mean(self.recent_trades) / (self.initial_capital * 0.01)  # Avg recent performance
            obs[29] = np.std(self.recent_trades) / (self.initial_capital * 0.01) if len(self.recent_trades) > 1 else 0.0
        
        # Recent trade outcomes (last 5 trades, binary: win/loss)
        for i in range(5):
            if i < len(self.recent_trades):
                obs[30 + i] = 1.0 if self.recent_trades[-(i+1)] > 0 else -1.0
            else:
                obs[30 + i] = 0.0
        
        obs[35] = self.current_drawdown  # Current drawdown
        obs[36] = 0.5  # Placeholder: market category performance
        obs[37] = 0.5  # Placeholder: similar market outcomes
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        portfolio_value = self._get_portfolio_value()
        
        return {
            'step': self.current_step,
            'capital': self.capital,
            'portfolio_value': portfolio_value,
            'num_positions': len(self.positions),
            'total_trades': self.trade_count,
            'win_rate': self.win_count / max(self.trade_count, 1),
            'drawdown': self.current_drawdown,
            'current_market': self.current_market['id'] if self.current_market else None
        }
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total = self.capital
        
        # Add value of open positions
        for market_id, position in self.positions.items():
            # In real implementation, this would use current market price
            # For now, assume no change (simplified)
            total += position['size']
        
        return total
    
    def _sample_market(self) -> Dict:
        """
        Sample a market for trading
        
        In real implementation, this would load actual Polymarket data.
        For now, we generate synthetic market data for testing.
        """
        # Synthetic market data for Phase 1 development
        return {
            'id': f"market_{random.randint(1000, 9999)}",
            'question': "Sample market question",
            'category': random.choice(['sports', 'politics', 'crypto', 'entertainment']),
            'current_price': random.uniform(0.3, 0.7),
            'bid_ask_spread': random.uniform(0.01, 0.05),
            'volume_24h': random.uniform(1000, 50000),
            'liquidity': random.uniform(500, 10000),
            'time_until_resolution': random.uniform(86400, 604800),  # 1-7 days
            'time_since_creation': random.uniform(0, 604800),
            'price_change_1h': random.uniform(-0.1, 0.1),
            'price_change_6h': random.uniform(-0.2, 0.2),
            'price_change_24h': random.uniform(-0.3, 0.3),
            'volatility_24h': random.uniform(0.01, 0.15),
            'trend_direction': random.choice([-1, 0, 1]),
            'volume_trend': random.uniform(-0.5, 0.5),
            'category_embedding': np.random.randn(3).astype(np.float32)
        }
    
    def render(self):
        """Render the environment (human-readable output)"""
        if not hasattr(self, '_render_initialized'):
            print("\n" + "="*60)
            print("RL Trading Bot - Environment State")
            print("="*60)
            self._render_initialized = True
        
        portfolio_value = self._get_portfolio_value()
        returns = (portfolio_value - self.initial_capital) / self.initial_capital
        
        print(f"\nStep: {self.current_step}/{self.max_steps}")
        print(f"Capital: ${self.capital:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f} ({returns:+.2%})")
        print(f"Open Positions: {len(self.positions)}/{self.settings.MAX_OPEN_POSITIONS}")
        print(f"Win Rate: {self.win_count}/{self.trade_count} ({self.win_count/max(self.trade_count,1):.1%})")
        print(f"Drawdown: {self.current_drawdown:.2%}")
        
        if self.current_market:
            print(f"\nCurrent Market: {self.current_market['id']}")
            print(f"  Price: {self.current_market['current_price']:.3f}")
            print(f"  Category: {self.current_market['category']}")
    
    def close(self):
        """Clean up resources"""
        pass
