"""
Diagnose why Run 76 learned to buy but never sell.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.environment.gym_env import CryptoTradingEnv
from src.data.collectors import CryptoDataLoader
from src.data.database import CryptoCandle, DatabaseSession


def diagnose_run_76():
    """Analyze episode behavior to understand the pathology."""
    
    print("=" * 80)
    print("RUN 76 DIAGNOSIS - Why does agent buy but never sell?")
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
    
    print(f"\n📊 ENVIRONMENT SETUP:")
    print(f"   Symbol pool size: {env.settings.MAX_OPEN_POSITIONS * 2}")
    print(f"   Max positions: {env.settings.MAX_OPEN_POSITIONS}")
    print(f"   Episode length: {env.max_steps} steps")
    
    # Run one episode and track what agent sees
    obs, info = env.reset()
    
    print(f"\n🔄 SYMBOL ROTATION TEST:")
    print(f"   Episode symbols: {env.episode_symbols}")
    
    symbol_position_visibility = {sym: [] for sym in env.episode_symbols}
    
    for step in range(20):
        current_sym = env.current_symbol
        has_position = current_sym in env.positions
        
        # Check position features in observation
        obs_position_size = obs[17]  # Feature 17: current symbol position size
        obs_unrealized_pnl = obs[36]  # Feature 36: unrealized P&L
        obs_hold_duration = obs[37]  # Feature 37: hold duration
        
        symbol_position_visibility[current_sym].append({
            'step': step,
            'has_position': has_position,
            'obs_shows_position': obs_position_size > 0,
            'obs_pnl': obs_unrealized_pnl,
            'obs_hold': obs_hold_duration,
        })
        
        # Randomly buy to create positions
        if not has_position and len(env.positions) < env.settings.MAX_OPEN_POSITIONS:
            action = env.ACTION_BUY
        else:
            action = env.ACTION_NO_ACTION
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Analysis
    print(f"\n🔍 POSITION VISIBILITY ANALYSIS:")
    print(f"   Total positions opened: {len(env.positions)}")
    print(f"   Positions: {list(env.positions.keys())}")
    
    for symbol, history in symbol_position_visibility.items():
        if not history:
            continue
        
        times_viewed = len(history)
        times_with_position = sum(1 for h in history if h['has_position'])
        times_obs_showed_position = sum(1 for h in history if h['obs_shows_position'])
        
        print(f"\n   {symbol}:")
        print(f"      Viewed: {times_viewed} times")
        print(f"      Had position: {times_with_position} times")
        print(f"      Observation showed position: {times_obs_showed_position} times")
        
        if times_with_position > 0:
            visibility_rate = times_obs_showed_position / times_viewed * 100
            print(f"      Visibility rate: {visibility_rate:.1f}%")
    
    # Critical insight
    print(f"\n⚠️  CRITICAL INSIGHT:")
    print(f"   With {len(env.episode_symbols)} symbols rotating:")
    print(f"   - Agent sees each symbol only ~{100/len(env.episode_symbols):.1f}% of the time")
    print(f"   - When viewing Symbol A, agent CANNOT see positions in Symbol B or C")
    print(f"   - Position features (obs[36-38]) only show CURRENT symbol's position")
    print(f"   - Agent has {100 - (100/len(env.episode_symbols)):.1f}% blind spot to existing positions!")
    
    print(f"\n🐛 ROOT CAUSE:")
    print(f"   The symbol rotation creates information asymmetry:")
    print(f"   1. When seeing Symbol A: Agent can decide to BUY Symbol A")
    print(f"   2. Next step sees Symbol B: Agent forgets about Symbol A position!")
    print(f"   3. By the time Symbol A rotates back: Too late to manage it properly")
    print(f"   4. Result: Agent learns to BUY (always sees opportunities)")
    print(f"           But never learns to SELL (can't track positions)")
    
    print(f"\n💡 SOLUTION:")
    print(f"   Option 1: Don't rotate symbols - stay on same symbol per episode")
    print(f"            (Loses multi-position benefit)")
    print(f"   Option 2: Add portfolio-level features showing ALL positions")
    print(f"            (obs[42-50]: position count, total P&L, avg hold time, etc.)")
    print(f"   Option 3: Expand observation to show all N positions explicitly")
    print(f"            (obs[42-50+N*3]: Each position's symbol, size, P&L)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    diagnose_run_76()
