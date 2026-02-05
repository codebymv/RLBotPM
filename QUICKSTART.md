# RLTrade - Quick Start Guide

Get up and running in 10 minutes!

## Prerequisites

✅ Python 3.10+ installed  
✅ PostgreSQL database on Railway (you have this!)

## Step 1: Setup Bot Environment

```powershell
# Navigate to project
cd C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLTrade

# Go to bot directory
cd bot

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate

# Install dependencies (takes 2-3 minutes)
pip install -r requirements.txt
```

## Step 2: Initialize Database

```powershell
# Still in bot directory with venv activated
python -c "from src.data import init_db; init_db()"
```

You should see: `✓ Database schema initialized`

## Step 3: Load Real Data (Required)

```powershell
# Example: load 30 days of BTC-USD and ETH-USD candles
python main.py collect-data --source coinbase --symbols BTC-USD,ETH-USD --interval 1h --days 30
```

If real data is unavailable, the system will error by design.

## Step 4: Test the Environment

```powershell
python main.py test-env
```

This creates the Gym environment and runs a few test steps on **real data**.
If you skipped Step 3, this will error.

```
Testing Gym Environment...
  State space: Box(-10.0, 10.0, (38,), float32)
  Action space: Discrete(8)
  
  Initial observation shape: (38,)
  Step 1: action=2, reward=0.1234
  Step 2: action=5, reward=-0.5678
  ...
✓ Environment test passed!
```

## Step 5: Run Your First Training

```powershell
python main.py train --episodes 1000
```

This trains the agent for 1000 episodes (~5-10 minutes). You'll see:

```
Starting training: 1,000 episodes

Episode 100: reward=5.23, win_rate=0.48
Episode 200: reward=7.45, win_rate=0.52
Episode 300: reward=9.12, win_rate=0.55
...

✓ Training completed!
```

## Step 5: View Training Metrics

**TensorBoard:**

```powershell
# In a new terminal (bot directory)
tensorboard --logdir=logs/tensorboard
```

Open http://localhost:6006 to see:
- Episode rewards over time
- Policy & value loss
- Exploration (entropy)

## What Just Happened?

1. **Environment Created**: The bot can now trade on real crypto data
2. **Database Connected**: Training metrics are stored in your Railway PostgreSQL
3. **Agent Trained**: PPO algorithm learned a basic trading strategy
4. **Checkpoints Saved**: Model saved in `bot/models/`
5. **Metrics Logged**: Performance data in database + TensorBoard

## Next Steps

### Run Longer Training

```powershell
python main.py train --episodes 10000
```

### Try LSTM Policy

```powershell
python main.py train --episodes 20000 --policy MlpLstmPolicy --sequence-length 10
```

### A/B Compare LSTM vs MLP

```powershell
python main.py compare --model-a models/lstm_model --policy-a MlpLstmPolicy `
  --model-b models/mlp_model --policy-b MlpPolicy --seq-b 10 --episodes 200
```

More episodes = better learning!

### Evaluate Your Model

```powershell
python main.py evaluate --model models/best_model_run_1 --episodes 100
```

### Check Bot Info

```powershell
python main.py info
```

Shows your configuration and risk limits.

### Compare to Baselines

Create a simple Python script:

```python
from src.environment import CryptoTradingEnv
from src.agents.baseline_agents import get_baseline_agents, compare_agents

env = CryptoTradingEnv(dataset=dataset, interval="1h")
agents = get_baseline_agents()
results = compare_agents(agents, env, n_episodes=10)

for name, metrics in results.items():
    print(f"{name}: {metrics['mean_return']:.2%}")
```

## Common Commands

```powershell
# Quick training (testing)
python main.py train --episodes 1000

# Longer training (real learning)
python main.py train --episodes 50000

# Test environment
python main.py test-env

# View configuration
python main.py info

# Collect market data (Phase 1+ feature)
python main.py collect-data --months 6

# Evaluate trained model
python main.py evaluate --model models/best_model_run_1
```

## Troubleshooting

**"No module named 'src'"**
→ Activate virtual environment: `.\venv\Scripts\activate`

**"Database connection failed"**
→ Check `.env` file has correct `DATABASE_URL`

**"Out of memory"**
→ Reduce `n_steps` in `shared/config/model_config.yaml` to 1024

## Project Structure (Where Things Are)

```
bot/
  ├── main.py              # CLI entry point
  ├── src/
  │   ├── environment/     # Gym trading environment
  │   ├── agents/          # PPO agent & baselines
  │   ├── risk/            # Safety systems
  │   ├── training/        # Training loop
  │   └── data/            # Database models
  ├── models/              # Saved checkpoints
  └── logs/                # Training logs
```

## What to Expect

**After 1,000 episodes:**
- Win rate: ~50-52% (barely better than random)
- Sharpe ratio: ~0.2-0.5
- Agent is still exploring

**After 10,000 episodes:**
- Win rate: ~53-56%
- Sharpe ratio: ~0.8-1.2
- Basic strategy emerging

**After 100,000 episodes:**
- Win rate: >55%
- Sharpe ratio: >1.0
- Solid strategy with risk management

## Monitoring Training

**Real-time in terminal:**
Watch episode numbers and rewards increase.

**TensorBoard (visual):**
```powershell
tensorboard --logdir=logs/tensorboard
```

**Database (detailed):**
Connect to your Railway PostgreSQL and query:
```sql
SELECT * FROM training_runs ORDER BY started_at DESC LIMIT 1;
SELECT * FROM episodes WHERE training_run_id = 1 ORDER BY episode_num DESC LIMIT 10;
```

## Success Checklist

After following this guide, you should have:

✅ Virtual environment activated  
✅ Dependencies installed  
✅ Database initialized  
✅ Environment tested successfully  
✅ Completed at least one training run  
✅ TensorBoard showing metrics  
✅ Model checkpoint saved  

## Next: Full System

Want the full monitoring experience?

1. **Setup API** (see `docs/SETUP.md`)
2. **Setup Dashboard** (see `dashboard/README.md`)
3. **Train with monitoring** (view in browser)

But for Phase 1, the CLI is all you need!

## Resources

- **Full Setup Guide**: `docs/SETUP.md`
- **RL Concepts**: `docs/RL_PRIMER.md`
- **Configuration**: `shared/config/`
- **Examples**: `notebooks/` (coming soon)

---

**🎉 Congratulations!** You're now training a reinforcement learning trading bot!
