# Multi-Position Portfolio Training Guide

## What Changed?

The environment now supports **true multi-asset portfolio management**:

### ✅ Enhancements Implemented

1. **Multi-Symbol Episodes**: Each episode now uses a pool of symbols (default: MAX_OPEN_POSITIONS × 2)
   - Previous: 1 symbol per episode → 1 position max
   - Now: 4-6 symbols per episode → 3 concurrent positions across different assets

2. **Symbol Rotation**: Agent sees different symbols each step (round-robin)
   - Enables building diversified portfolio within single episode
   - Agent learns cross-symbol opportunities and correlation patterns

3. **Per-Position Tracking**: Each position tracks its own `hold_steps` and P&L
   - Previous: Global `position_hold_steps` (broken for multi-position)
   - Now: Position dict includes `{"size", "entry_price", "timestamp", "hold_steps"}`

4. **Portfolio-Aware Actions**: BUY checks symbol-specific positions
   - Can open position in BTC while holding ETH and SOL
   - Max 3 concurrent positions (configurable via `MAX_OPEN_POSITIONS`)

5. **Backward Compatible**: Observation and action spaces unchanged
   - Existing models will run but won't leverage multi-position capability
   - New models automatically learn diversification strategies

## Training a Multi-Position Model

### Quick Start - Run 76

```bash
cd /workspaces/RLBotPM/bot

# Train new model with multi-position learning (10,000 episodes)
python main.py train \
  --episodes 10000 \
  --policy MlpPolicy \
  --checkpoint-frequency 2000 \
  --eval-frequency 2000
```

The model will be saved to `models/best_model_run_76.zip`

### What to Expect

**Training Behavior:**
- Episodes now span 4-6 symbols instead of 1
- Agent will learn to:
  - Identify best entry opportunities across multiple assets
  - Build diversified portfolio (reduce correlation risk)
  - Manage position sizes across 3 concurrent holdings
  - Exit positions independently based on symbol-specific signals

**Performance Improvements:**
- **Expected win rate**: 52-57% (vs 50.94% baseline)
  - Diversification reduces drawdowns
  - More trading opportunities per episode
  
- **Expected Sharpe ratio**: 3.5-4.5 (vs 3.2 baseline)
  - Portfolio-level risk reduction from uncorrelated assets
  
- **Capital efficiency**: 20-40% (vs 7.38% baseline)
  - Multiple positions = more capital deployed
  - Better utilization of training data

**Potential Issues:**
1. **Slower convergence**: Multi-symbol complexity requires more episodes
2. **Symbol preference bias**: Agent may favor certain symbols initially
3. **Correlation blindness**: Agent doesn't explicitly see cross-symbol correlation (future enhancement)

### Advanced Training Options

#### Longer Training Run (Recommended)

```bash
# 50,000 episodes for optimal convergence
python main.py train \
  --episodes 50000 \
  --policy MlpPolicy \
  --checkpoint-frequency 5000 \
  --eval-frequency 5000
```

#### Resume from Checkpoint

```bash
# Continue training from best checkpoint
python main.py train \
  --episodes 20000 \
  --checkpoint models/best_model_run_76.zip \
  --policy MlpPolicy
```

#### Use LSTM for Sequence Learning

```bash
# LSTM can better capture multi-symbol patterns
python main.py train \
  --episodes 30000 \
  --policy MlpLstmPolicy \
  --sequence-length 5 \
  --checkpoint-frequency 3000 \
  --eval-frequency 3000
```

## Evaluation

### Compare Against Baseline

```bash
# Baseline: final_run_75 (single-position, 50.94% win rate)
python main.py evaluate \
  --model models/final_run_75 \
  --episodes 500 \
  --policy MlpPolicy

# New: Run 76 (multi-position)
python main.py evaluate \
  --model models/best_model_run_76 \
  --episodes 500 \
  --policy MlpPolicy
```

### Key Metrics to Compare

| Metric | Baseline (Run 75) | Target (Run 76) | Improvement |
|--------|-------------------|-----------------|-------------|
| Win Rate | 50.94% | 52-57% | +2-6% |
| Sharpe Ratio | 3.200 | 3.5-4.5 | +9-41% |
| Drawdown | 0.86% | <0.7% | Better risk mgmt |
| In-Position % | 7.38% | 20-40% | 3-5x efficiency |
| Trades/Ep | 4.48 | 8-12 | More opportunities |

## Configuration Tuning

### Adjust Max Positions

Edit `bot/src/core/config.py`:

```python
class TradingSettings(BaseModel):
    MAX_OPEN_POSITIONS: int = Field(default=5, ...)  # Increase to 5 positions
```

### Adjust Symbol Pool Size

Edit `bot/src/environment/gym_env.py` line ~430:

```python
# Increase pool for more diversification
pool_size = min(self.settings.MAX_OPEN_POSITIONS * 3, len(valid_symbols))  # 3x instead of 2x
```

### Enable Correlation Features (Future Enhancement)

Add to observation space:
- Cross-symbol correlation matrix
- Portfolio volatility
- Herfindahl concentration index
- Sector diversification score

## Troubleshooting

### Issue: Agent opens max positions immediately then holds forever

**Solution**: Increase `hold_step_penalty` in reward config to encourage turnover:

```yaml
# shared/config/reward_config.yaml
reward_weights:
  hold_step_penalty: 0.01  # Increase from 0.005
```

### Issue: Agent prefers only 1-2 symbols (not diversifying)

**Solution**: Add diversification bonus to reward function:

```python
# In _calculate_reward()
num_unique_positions = len(set(self.positions.keys()))
diversification_bonus = num_unique_positions * 0.1
reward += diversification_bonus
```

### Issue: Training loss not decreasing

**Solution**: 
1. Check tensorboard: `tensorboard --logdir logs/tensorboard`
2. Reduce learning rate: Edit `shared/config/model_config.yaml`
   ```yaml
   ppo:
     learning_rate: 0.0001  # Down from 0.0003
   ```
3. Increase batch size for stability

## Next Steps

1. **Train Run 76**: Start training with default config
2. **Monitor Progress**: Watch tensorboard and checkpoint evaluations
3. **Compare Results**: Evaluate after 10k episodes vs baseline
4. **Iterate**: Tune reward weights based on behavior
5. **Production Deploy**: If Run 76 > 52% win rate and Sharpe > 3.5

## Expected Timeline

- **10,000 episodes**: ~2-4 hours (depending on hardware)
- **50,000 episodes**: ~10-20 hours
- **Evaluation (500 episodes)**: ~5-10 minutes

## Architecture Notes

The environment maintains backward compatibility:
- ✅ Observation space: 42 features (unchanged)
- ✅ Action space: 3 discrete actions (unchanged)
- ✅ Existing models work but don't use multi-position capability
- ✅ New models automatically learn diversification

**Why this works**: The agent learns from:
1. Portfolio value changes (multi-position P&L)
2. Symbol rotation exposes multiple opportunities
3. Position capacity constraint (MAX_OPEN_POSITIONS)
4. Per-symbol hold tracking in observation features

The reward function already tracks portfolio-level returns, so the agent naturally learns to optimize across all positions simultaneously.
