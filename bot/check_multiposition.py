#!/usr/bin/env python3
"""Check if Run 75 uses multiple concurrent positions"""

from src.environment.gym_env import CryptoTradingEnv
from sb3_contrib import MaskablePPO
import numpy as np

# Load model
model = MaskablePPO.load('models/final_run_75')

# Create env
env = CryptoTradingEnv()

# Run 10 episodes and track positions
all_max_concurrent = []
all_avg_positions = []

for ep in range(10):
    obs, info = env.reset()
    max_concurrent = 0
    position_counts = []
    
    for step in range(100):
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        num_positions = len(env.positions)
        position_counts.append(num_positions)
        max_concurrent = max(max_concurrent, num_positions)
        
        if terminated or truncated:
            break
    
    all_max_concurrent.append(max_concurrent)
    all_avg_positions.append(np.mean(position_counts))
    
    print(f'Episode {ep+1}: Max={max_concurrent}, Avg={np.mean(position_counts):.2f}')

print(f'\nOverall:')
print(f'  Max concurrent ever: {max(all_max_concurrent)}')
print(f'  Avg of maximums: {np.mean(all_max_concurrent):.2f}')
print(f'  Avg positions: {np.mean(all_avg_positions):.2f}')
