"""
Deep trace of a single step to find where hold_steps breaks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.environment.gym_env import CryptoTradingEnv
from src.data.collectors import CryptoDataLoader
from src.data.database import CryptoCandle, DatabaseSession


# Load data
source = "coinbase"
loader = CryptoDataLoader(source=source)

with DatabaseSession() as session:
    symbols = [s[0] for s in session.query(CryptoCandle.symbol).filter_by(source=source, interval="1h").distinct().all()]

dataset =loader.load_dataset(symbols=symbols, interval="1h", limit=10000)

# Create environment
env = CryptoTradingEnv(dataset=dataset, max_steps=100)
obs, info = env.reset()

# Open a position
print("Opening position...")
while len(env.positions) == 0:
    obs, reward, terminated, truncated, info = env.step(env.ACTION_BUY)
    if terminated or truncated:
        break

if len(env.positions) == 0:
    print("Failed to open position!")
    sys.exit(1)

symbol = list(env.positions.keys())[0]
print(f"\nPosition opened in {symbol}")
print(f"Initial hold_steps: {env.positions[symbol]['hold_steps']}")

# Take one step and trace
print(f"\nBefore step:")
print(f"  env.positions[{symbol}]['hold_steps'] = {env.positions[symbol]['hold_steps']}")
print(f"  id(env.positions) = {id(env.positions)}")
print(f"  id(env.positions[{symbol}]) = {id(env.positions[symbol])}")

obs, reward, terminated, truncated, info = env.step(env.ACTION_NO_ACTION)

print(f"\nAfter step:")
print(f"  env.positions[{symbol}]['hold_steps'] = {env.positions[symbol]['hold_steps']}")
print(f"  id(env.positions) = {id(env.positions)}")
print(f"  id(env.positions[{symbol}]) = {id(env.positions[symbol])}")

# Check if it's a reference issue
print(f"\n\nDirect modification test:")
print(f"Before: {env.positions[symbol]['hold_steps']}")
env.positions[symbol]['hold_steps'] += 1
print(f"After manual increment: {env.positions[symbol]['hold_steps']}")
