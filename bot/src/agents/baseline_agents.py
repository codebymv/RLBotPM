"""
Baseline Trading Strategies

These simple strategies serve as performance benchmarks.
The RL agent should outperform these to be considered useful.

Strategies implemented:
1. Random Agent - Takes random actions
2. Buy and Hold - Buys at start, holds until end
3. Mean Reversion - Buys when price is low, sells when high
4. Momentum - Follows price trends
"""

import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod

from ..core.logger import get_logger


logger = get_logger(__name__)


class BaselineAgent(ABC):
    """Base class for baseline strategies"""
    
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {name} agent")
    
    @abstractmethod
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict action given observation
        
        Args:
            observation: Current state
        
        Returns:
            Action index
        """
        pass
    
    def reset(self):
        """Reset agent state for new episode"""
        pass


class RandomAgent(BaselineAgent):
    """
    Random agent - takes random actions from the 7-action space.
    """
    
    def __init__(self):
        super().__init__("Random")
        self.action_space_size = 7
    
    def predict(self, observation: np.ndarray) -> int:
        """Return random action"""
        return np.random.randint(0, self.action_space_size)


class BuyAndHoldAgent(BaselineAgent):
    """
    Buy and hold strategy
    
    Buys at the beginning and holds until the end.
    This is a common benchmark for trading strategies.
    """
    
    def __init__(self):
        super().__init__("Buy and Hold")
        self.has_position = False
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Buy slot 0 at start, then hold.
        
        New 67-dim obs layout:
        - obs[55]: slot 0 position size (>0 if position exists)
        - obs[58]: slot 0 has_position flag
        """
        has_position = observation[58] > 0.5
        
        if not has_position:
            return 1  # ACTION_BUY_0
        else:
            return 0  # ACTION_NO_ACTION
    
    def reset(self):
        """Reset for new episode"""
        self.has_position = False


class MeanReversionAgent(BaselineAgent):
    """
    Mean reversion strategy
    
    Assumes prices revert to mean:
    - Buy when price is below historical average
    - Sell when price is above historical average
    """
    
    def __init__(self, threshold: float = 0.05):
        super().__init__("Mean Reversion")
        self.threshold = threshold  # 5% deviation threshold
        self.price_history = []
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Buy low, sell high based on mean reversion (slot 0).
        
        New 67-dim obs layout:
        - obs[0]: price deviation from MA24 (slot 0)
        - obs[4]: return_24h (slot 0)
        - obs[58]: slot 0 has_position flag
        """
        price_deviation = observation[0]
        has_position = observation[58] > 0.5
        
        # Track price history
        self.price_history.append(price_deviation)
        if len(self.price_history) > 50:
            self.price_history.pop(0)
        
        if len(self.price_history) < 10:
            return 0  # ACTION_NO_ACTION
        
        if not has_position:
            if price_deviation < -self.threshold:
                return 1  # ACTION_BUY_0
        else:
            if price_deviation > self.threshold:
                return 4  # ACTION_SELL_0
        
        return 0  # ACTION_NO_ACTION
    
    def reset(self):
        """Reset for new episode"""
        self.price_history = []


class MomentumAgent(BaselineAgent):
    """
    Momentum strategy
    
    Follows price trends:
    - Buy when price is trending up
    - Sell when price is trending down
    """
    
    def __init__(self, threshold: float = 0.02):
        super().__init__("Momentum")
        self.threshold = threshold  # 2% momentum threshold
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Follow momentum on slot 0.
        
        New 67-dim obs layout:
        - obs[3]: return_6h (slot 0)
        - obs[4]: return_24h (slot 0)
        - obs[8]: trend_direction (slot 0)
        - obs[58]: slot 0 has_position flag
        """
        price_change_6h = observation[3]
        price_change_24h = observation[4]
        trend_direction = observation[8]
        has_position = observation[58] > 0.5
        
        momentum = (price_change_6h + price_change_24h) / 2.0
        
        if not has_position:
            if momentum > self.threshold and trend_direction > 0:
                return 1  # ACTION_BUY_0
        else:
            if momentum < -self.threshold or trend_direction < 0:
                return 4  # ACTION_SELL_0
        
        return 0  # ACTION_NO_ACTION


class ConservativeAgent(BaselineAgent):
    """
    Conservative strategy
    
    Only trades when highly confident:
    - Strong signals required
    - Takes profits quickly
    - Cuts losses fast
    """
    
    def __init__(self):
        super().__init__("Conservative")
        self.holding_time = 0
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Conservative trading on slot 0.
        
        New 67-dim obs layout:
        - obs[0]: price deviation from MA24 (slot 0)
        - obs[4]: return_24h (slot 0)
        - obs[5]: volatility_24h (slot 0)
        - obs[58]: slot 0 has_position flag
        """
        price_deviation = observation[0]
        price_change_24h = observation[4]
        volatility = observation[5]
        has_position = observation[58] > 0.5
        
        if not has_position:
            self.holding_time = 0
            if (volatility < 0.05 and 
                price_change_24h > 0.03):
                return 1  # ACTION_BUY_0
        else:
            self.holding_time += 1
            if price_change_24h < -0.05:
                return 4  # ACTION_SELL_0
            elif price_change_24h > 0.08:
                return 4  # ACTION_SELL_0
            elif self.holding_time > 20:
                return 4  # ACTION_SELL_0
        
        return 0  # ACTION_NO_ACTION
    
    def reset(self):
        """Reset for new episode"""
        self.holding_time = 0


def get_baseline_agents() -> Dict[str, BaselineAgent]:
    """
    Get all baseline agents
    
    Returns:
        Dict mapping agent name to agent instance
    """
    return {
        'random': RandomAgent(),
        'buy_and_hold': BuyAndHoldAgent(),
        'mean_reversion': MeanReversionAgent(),
        'momentum': MomentumAgent(),
        'conservative': ConservativeAgent()
    }


def compare_agents(
    agents: Dict[str, BaselineAgent],
    env,
    n_episodes: int = 10
) -> Dict[str, Dict]:
    """
    Compare multiple agents on the same environment
    
    Args:
        agents: Dict of agent name -> agent instance
        env: Trading environment
        n_episodes: Number of episodes to evaluate
    
    Returns:
        Dict of agent name -> performance metrics
    """
    results = {}
    
    for name, agent in agents.items():
        logger.info(f"Evaluating {name} agent...")
        
        episode_rewards = []
        episode_returns = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            agent.reset()
            done = False
            episode_reward = 0.0
            initial_capital = info['capital']
            
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            final_capital = info['capital']
            returns = (final_capital - initial_capital) / initial_capital
            
            episode_rewards.append(episode_reward)
            episode_returns.append(returns)
        
        results[name] = {
            'mean_reward': np.mean(episode_rewards),
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'sharpe_ratio': np.mean(episode_returns) / (np.std(episode_returns) + 1e-8),
            'best_return': np.max(episode_returns),
            'worst_return': np.min(episode_returns)
        }
        
        logger.info(
            f"{name}: "
            f"Return={results[name]['mean_return']:.2%}, "
            f"Sharpe={results[name]['sharpe_ratio']:.3f}"
        )
    
    return results
