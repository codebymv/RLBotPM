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
    Random agent - takes random actions
    
    This is the worst-case baseline. Any learning agent should
    easily outperform random actions.
    """
    
    def __init__(self):
        super().__init__("Random")
        self.action_space_size = 3
    
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
        Buy once at start, then do nothing
        
        State breakdown (from gym_env):
        - obs[0]: current_price
        - obs[15]: capital (normalized)
        - obs[20]: position_size in current market
        """
        # Check if we have a position (obs[17] is position size)
        has_position = observation[17] > 0.01
        
        if not has_position:
            # No position, buy
            return 1  # ACTION_BUY
        else:
            # Have position, hold (do nothing)
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
        Buy low, sell high based on mean reversion
        
        State features:
        - obs[0]: current_price
        - obs[6]: price_change_1h
        - obs[7]: price_change_6h
        - obs[20]: position_size
        """
        current_price = observation[0]
        price_change_24h = observation[5]  # 24h price change
        has_position = observation[17] > 0.01
        
        # Track price history
        self.price_history.append(current_price)
        if len(self.price_history) > 50:
            self.price_history.pop(0)
        
        # Need sufficient history
        if len(self.price_history) < 10:
            return 0  # ACTION_NO_ACTION
        
        # Calculate mean and deviation
        mean_price = np.mean(self.price_history)
        deviation = (current_price - mean_price) / mean_price
        
        # Mean reversion logic
        if not has_position:
            # Price below mean -> Buy
            if deviation < -self.threshold:
                return 1  # ACTION_BUY
        else:
            # Price above mean -> Sell
            if deviation > self.threshold:
                return 2  # ACTION_SELL
        
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
        Follow momentum
        
        State features:
        - obs[6]: price_change_1h
        - obs[7]: price_change_6h
        - obs[8]: price_change_24h
        - obs[10]: trend_direction (-1, 0, 1)
        - obs[20]: position_size
        """
        price_change_6h = observation[4]
        price_change_24h = observation[5]
        trend_direction = observation[11]
        has_position = observation[17] > 0.01
        
        # Calculate momentum signal
        momentum = (price_change_6h + price_change_24h) / 2.0
        
        # Momentum logic
        if not has_position:
            # Strong upward momentum -> Buy
            if momentum > self.threshold and trend_direction > 0:
                return 1  # ACTION_BUY
        else:
            # Strong downward momentum -> Sell
            if momentum < -self.threshold or trend_direction < 0:
                return 2  # ACTION_SELL
        
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
        Conservative trading with strict conditions
        
        State features:
        - obs[0]: current_price
        - obs[8]: price_change_24h
        - obs[9]: volatility_24h
        - obs[20]: position_size
        """
        current_price = observation[0]
        price_change_24h = observation[5]
        volatility = observation[6]
        has_position = observation[17] > 0.01
        
        if not has_position:
            self.holding_time = 0
            
            # Only buy if:
            # 1. Low volatility (stable market)
            # 2. Positive momentum
            # 3. Price not extreme
            if (volatility < 0.05 and 
                price_change_24h > 0.03 and
                0.3 < current_price < 0.7):
                return 1  # ACTION_BUY
        else:
            self.holding_time += 1
            
            # Take profits or cut losses quickly
            if price_change_24h < -0.05:  # 5% loss
                return 2  # ACTION_SELL
            elif price_change_24h > 0.08:  # 8% profit
                return 2  # ACTION_SELL
            elif self.holding_time > 20:  # Held too long
                return 2  # ACTION_SELL
        
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
