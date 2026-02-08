#!/usr/bin/env python3
"""Test Run 75 with current multi-symbol environment"""

from src.environment.gym_env import CryptoTradingEnv
from src.data.collectors.crypto_loader import CryptoDataLoader
from src.core.config import get_settings
from sb3_contrib import MaskablePPO
import numpy as np

settings = get_settings()
loader = CryptoDataLoader(source=settings.DATA_SOURCE)

print("Loading dataset with multiple symbols...")
df = loader.load_dataset(symbols=['ALGO-USD', 'BTC-USD'], interval='1h', limit=1000)

print(f"\nDataset: {len(df)} rows, {df['symbol'].nunique()} symbols")
print(f"Symbols: {sorted(df['symbol'].unique())}")

# Try to create environment
try:
    env = CryptoTradingEnv(dataset=df, max_steps=100)
    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Max positions: {settings.MAX_OPEN_POSITIONS}")
    
    # Try to load Run 75
    try:
        model = MaskablePPO.load('models/final_run_75')
        print(f"\nRun 75 model loaded")
        print(f"  Expected observation shape: {model.observation_space.shape}")
        
        # Test compatibility
        obs, info = env.reset()
        print(f"\nEnvironment reset:")
        print(f"  Actual observation shape: {obs.shape}")
        print(f"  Episode symbols: {env.episode_symbols}")
        
        if obs.shape != model.observation_space.shape:
            print(f"\n❌ INCOMPATIBLE: Environment has {obs.shape[0]} dims, model expects {model.observation_space.shape[0]} dims")
            print(f"\nRun 75 was trained on 42-dim single-symbol environment")
            print(f"Current environment has 57-dim multi-symbol architecture")
            print(f"\nConclusion: Run 75 CANNOT trade BTC+ETH+SOL simultaneously")
        else:
            print(f"\n✓ Compatible! Testing multi-symbol trading...")
            
            # Run episode and track which symbols get traded
            symbols_traded = set()
            positions_held = []
            
            for step in range(100):
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if env.positions:
                    symbols_in_portfolio = list(env.positions.keys())
                    symbols_traded.update(symbols_in_portfolio)
                    positions_held.append(len(env.positions))
                    
                    if len(symbols_in_portfolio) > 1:
                        print(f"  Step {step}: MULTI-ASSET PORTFOLIO: {symbols_in_portfolio}")
                
                if terminated or truncated:
                    break
            
            print(f"\nResults:")
            print(f"  Symbols traded: {symbols_traded}")
            print(f"  Max concurrent positions: {max(positions_held) if positions_held else 0}")
            print(f"  Different assets simultaneously: {'YES' if len(symbols_traded) > 1 else 'NO'}")
            
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        
except Exception as e:
    print(f"\n❌ Error creating environment: {e}")
    import traceback
    traceback.print_exc()
