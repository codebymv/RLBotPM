"""
Tests for the Gym trading environment

These tests verify that:
- Environment initializes correctly
- State space has correct shape
- Actions are valid
- Rewards are calculated
- Episodes terminate properly
"""

import pytest
import numpy as np
from src.environment import PolymarketTradingEnv


def test_environment_initialization():
    """Test that environment initializes correctly"""
    env = PolymarketTradingEnv()
    
    assert env is not None
    assert env.action_space.n == 8  # 8 discrete actions
    assert env.observation_space.shape == (38,)  # 38-dimensional state


def test_environment_reset():
    """Test environment reset"""
    env = PolymarketTradingEnv()
    obs, info = env.reset()
    
    # Check observation shape
    assert obs.shape == (38,)
    assert isinstance(obs, np.ndarray)
    
    # Check info dict
    assert 'capital' in info
    assert 'portfolio_value' in info
    assert info['capital'] == env.initial_capital


def test_action_space():
    """Test that all actions are valid"""
    env = PolymarketTradingEnv()
    
    # Check action space
    assert env.action_space.n == 8
    
    # Test each action
    for action in range(8):
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


def test_no_action():
    """Test that NO_ACTION doesn't change capital"""
    env = PolymarketTradingEnv()
    obs, info = env.reset()
    initial_capital = info['capital']
    
    # Take NO_ACTION
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Capital should be unchanged (no transaction cost)
    assert info['capital'] == initial_capital


def test_buy_action():
    """Test buy action reduces capital"""
    env = PolymarketTradingEnv()
    obs, info = env.reset()
    initial_capital = info['capital']
    
    # Take BUY_MEDIUM action
    obs, reward, terminated, truncated, info = env.step(2)
    
    # Capital should be reduced
    assert info['capital'] < initial_capital
    
    # Should have an open position
    assert info['num_positions'] >= 0


def test_episode_length():
    """Test that episodes respect max_steps"""
    env = PolymarketTradingEnv(max_steps=10)
    obs, info = env.reset()
    
    steps = 0
    done = False
    
    while not done and steps < 20:  # Safety limit
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        steps += 1
    
    # Should terminate within max_steps
    assert steps <= 10


def test_reward_calculation():
    """Test that rewards are calculated"""
    env = PolymarketTradingEnv()
    obs, info = env.reset()
    
    # Take a few actions
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        
        # Reward should be a number
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)


def test_observation_bounds():
    """Test that observations are within reasonable bounds"""
    env = PolymarketTradingEnv()
    obs, info = env.reset()
    
    # Take some random actions
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        
        if terminated or truncated:
            break
        
        # Check bounds
        assert obs.shape == (38,)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))


def test_info_dict():
    """Test that info dict contains expected keys"""
    env = PolymarketTradingEnv()
    obs, info = env.reset()
    
    required_keys = [
        'step',
        'capital',
        'portfolio_value',
        'num_positions',
        'total_trades',
        'win_rate',
        'drawdown'
    ]
    
    for key in required_keys:
        assert key in info, f"Missing key: {key}"


def test_multiple_episodes():
    """Test multiple episodes run correctly"""
    env = PolymarketTradingEnv()
    
    for episode in range(3):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            done = terminated or truncated
            steps += 1
        
        # Each episode should complete
        assert done or steps >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
