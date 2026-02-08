"""
Diagnose if portfolio features are working correctly in Run 77.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.environment.gym_env import CryptoTradingEnv
from src.data.collectors import CryptoDataLoader
from src.data.database import CryptoCandle, DatabaseSession


def diagnose_portfolio_features():
    """Check if portfolio-level features are being calculated correctly."""
    
    print("=" * 80)
    print("RUN 77 PORTFOLIO FEATURES DIAGNOSIS")
    print("=" * 80)
    
    # Load data
    source = "coinbase"
    loader = CryptoDataLoader(source=source)
    
    with DatabaseSession() as session:
        symbols = (
            session.query(CryptoCandle.symbol)
            .filter_by(source=source, interval="1h")
            .distinct()
            .all()
        )
        symbols = [s[0] for s in symbols]
    
    dataset = loader.load_dataset(symbols=symbols, interval="1h", limit=10000)
    
    # Create environment
    env = CryptoTradingEnv(dataset=dataset, max_steps=100)
    obs, info = env.reset()
    
    print(f"\n📊 OBSERVATION SPACE CHECK:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Expected: (50,) - Got: {obs.shape}")
    print(f"   ✓ Correct shape!" if obs.shape[0] == 50 else f"   ✗ WRONG SHAPE!")
    
    # Force open multiple positions
    print(f"\n🔨 FORCING MULTI-POSITION SCENARIO:")
    
    positions_opened = 0
    step = 0
    while positions_opened < 3 and step < 50:
        if env.current_symbol not in env.positions:
            action = env.ACTION_BUY
        else:
            action = env.ACTION_NO_ACTION
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if len(env.positions) > positions_opened:
            positions_opened = len(env.positions)
            print(f"   Step {step}: Opened position #{positions_opened} in {env.current_symbol}")
        
        step += 1
        
        if terminated or truncated:
            break
    
    print(f"\n   Final positions: {list(env.positions.keys())}")
    print(f"   Total: {len(env.positions)}")
    
    # Analyze portfolio features
    print(f"\n🔍 PORTFOLIO FEATURES ANALYSIS:")
    print(f"   Current symbol: {env.current_symbol}")
    print(f"   Current step: {env.current_step}")
    print(f"   Symbol pool: {env.episode_symbols}")
    
    # Take a few more steps and track portfolio features
    print(f"\n📈 TRACKING PORTFOLIO FEATURES (10 steps):")
    print(f"   {'Step':<6} {'Symbol':<12} {'Pos':<4} {'#Pos':<5} {'AvgPnL':<10} {'AvgHold':<10} {'Exposure':<10} {'Worst':<10} {'Best':<10}")
    print(f"   {'-'*90}")
    
    for i in range(10):
        has_pos = env.current_symbol in env.positions
        
        # Portfolio features from observation
        num_positions = obs[42]  # Feature 42: total positions (normalized)
        avg_pnl = obs[43]        # Feature 43: average unrealized P&L
        avg_hold = obs[44]       # Feature 44: average hold duration
        exposure = obs[45]       # Feature 45: total exposure
        has_current = obs[46]    # Feature 46: current symbol has position
        worst_pnl = obs[47]      # Feature 47: worst position P&L
        best_pnl = obs[48]       # Feature 48: best position P&L
        
        print(f"   {i:<6} {env.current_symbol:<12} {'Y' if has_pos else 'N':<4} "
              f"{num_positions:.3f} {avg_pnl:>9.3f} {avg_hold:>9.3f} "
              f"{exposure:>9.3f} {worst_pnl:>9.3f} {best_pnl:>9.3f}")
        
        obs, reward, terminated, truncated, info = env.step(env.ACTION_NO_ACTION)
        
        if terminated or truncated:
            break
    
    # Check if features are non-zero
    print(f"\n✅ FEATURE VALIDATION:")
    non_zero_features = []
    feature_names = {
        42: "Total positions",
        43: "Avg P&L",
        44: "Avg hold duration", 
        45: "Total exposure",
        46: "Has current position",
        47: "Worst P&L",
        48: "Best P&L",
    }
    
    for feat_idx in [42, 43, 44, 45, 46, 47, 48]:
        if abs(obs[feat_idx]) > 0.001:
            non_zero_features.append(feat_idx)
            print(f"   ✓ Feature {feat_idx} ({feature_names[feat_idx]}): {obs[feat_idx]:.4f}")
        else:
            print(f"   ✗ Feature {feat_idx} ({feature_names[feat_idx]}): {obs[feat_idx]:.4f} (ZERO!)")
    
    if len(non_zero_features) >= 4:
        print(f"\n   ✅ Portfolio features ARE working ({len(non_zero_features)}/7 active)")
    else:
        print(f"\n   ❌ Portfolio features NOT working ({len(non_zero_features)}/7 active)")
    
    # Deep dive into hold_steps tracking
    print(f"\n🔬 HOLD_STEPS TRACKING TEST:")
    for symbol, position in env.positions.items():
        hold_steps = position.get("hold_steps", -1)
        print(f"   {symbol}: hold_steps = {hold_steps}")
    
    # Test if hold_steps increment
    print(f"\n⏱️  HOLD_STEPS INCREMENT TEST (5 steps):")
    print(f"   Current positions: {list(env.positions.keys())}")
    print(f"   Episode terminated: {terminated}, truncated: {truncated}")
    initial_hold_steps = {sym: pos.get("hold_steps", 0) for sym, pos in env.positions.items()}
    
    if terminated or truncated:
        print(f"   ⚠️  Episode already ended, resetting...")
        obs, info = env.reset()
        # Force positions again
        for _ in range(10):
            if len(env.positions) < 3:
                obs, reward, terminated, truncated, info = env.step(env.ACTION_BUY)
            else:
                break
        initial_hold_steps = {sym: pos.get("hold_steps", 0) for sym, pos in env.positions.items()}
    
    steps_completed = 0
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(env.ACTION_NO_ACTION)
        steps_completed += 1
        if terminated or truncated:
            print(f"   ⚠️  Episode ended at step {i+1}!")
            break
    
    print(f"   Completed {steps_completed}/5 steps")
    
    for symbol, position in env.positions.items():
        initial = initial_hold_steps.get(symbol, 0)
        current = position.get("hold_steps", 0)
        increment = current - initial
        expected = steps_completed
        status = "✓" if increment == expected else "✗"
        print(f"   {status} {symbol}: {initial} → {current} (increment: {increment}, expected: {expected})")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    diagnose_portfolio_features()
