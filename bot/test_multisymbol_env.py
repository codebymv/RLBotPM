#!/usr/bin/env python3
"""Test multi-symbol environment with 57-dim observation space"""

from src.environment.gym_env import CryptoTradingEnv
from src.data.collectors.crypto_loader import CryptoDataLoader
from src.core.config import get_settings
import pandas as pd

# Load data
settings = get_settings()
loader = CryptoDataLoader(source=settings.DATA_SOURCE)

print("Loading dataset...")
df = loader.load_dataset(
    symbols=['BTC-USD', 'ETH-USD', 'SOL-USD', 'ALGO-USD', 'MATIC-USD'],
    interval='1h',
    limit=1000
)


if df.empty:
    print("ERROR: No data available!")
    exit(1)

print(f"Loaded {len(df)} rows across {df['symbol'].nunique()} symbols")
print(f"Symbols: {sorted(df['symbol'].unique())}")

# Create environment
env = CryptoTradingEnv(dataset=df, max_steps=100)

print(f"\nEnvironment created:")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space}")
print(f"  Symbol encoding: {len(env.symbol_to_id)} symbols")

# Test reset
obs, info = env.reset()
print(f"\nAfter reset:")
print(f"  Observation shape: {obs.shape}")
print(f"  Episode symbols: {env.episode_symbols}")
print(f"  Current symbol: {env.current_symbol}")

# Check observation structure
print(f"\nObservation structure (57 dims):")
print(f"  Market features (0-41): {obs[0:42][:10]}... (showing first 10)")
print(f"  Position slot 1 (42-46): {obs[42:47]} [symbol_id, size, entry_dev, pnl%, hold_norm]")
print(f"  Position slot 2 (47-51): {obs[47:52]}")
print(f"  Position slot 3 (52-56): {obs[52:57]}")

# Run a few steps
print(f"\nRunning 10 steps:")
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  Step {step+1}: symbol={env.current_symbol}, "
          f"action={env.ACTION_NAMES[action]}, "
          f"positions={len(env.positions)}, "
          f"reward={reward:.4f}")
    
    if env.positions:
        for sym, pos in env.positions.items():
            print(f"    → {sym}: size=${pos['size']:.2f}, hold_steps={pos.get('hold_steps', 0)}")
    
    if terminated or truncated:
        print(f"  Episode ended!")
        break

print(f"\nTest completed successfully!")
print(f"Final observation shape: {obs.shape}")
print(f"Expected shape: (57,)")
assert obs.shape == (57,), f"Wrong shape! Got {obs.shape}"
print("✅ All tests passed!")
