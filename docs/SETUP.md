# RLTrade - Setup Guide

Complete step-by-step instructions to get the RL trading bot running.

## Prerequisites

- **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18+** (for dashboard) - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/)
- **PostgreSQL database** - Already set up on Railway ✅

## Installation

### 1. Clone the Repository

```bash
cd C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLTrade
```

### 2. Setup Bot (Python)

```bash
cd bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py info
```

### 3. Initialize Database

The database schema needs to be created once:

```bash
# Still in bot directory with venv activated
python -c "from src.data import init_db; init_db()"
```

You should see: `✓ Database schema initialized`

### 4. Setup API (Python)

```bash
cd ../api

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 5. Setup Dashboard (Node.js)

```bash
cd ../dashboard

# Install dependencies
npm install

# Build for development
npm run dev
```

## Configuration

The `.env` file in the root directory contains all configuration. It's already set up with your Railway PostgreSQL connection.

### Key Configuration Options

**Risk Management** (adjust as needed):
```
MAX_DAILY_LOSS_USD=20.0
MAX_POSITION_SIZE_PCT=0.20
MAX_OPEN_POSITIONS=3
```

**Training Settings**:
```
TRAINING_EPISODES=100000
CHECKPOINT_FREQUENCY=10000
```

**Data Source Settings**:
```
DATA_SOURCE=coinbase
DATA_INTERVAL=1h
REQUIRE_HISTORICAL_DAYS=30
DATA_SYMBOLS=BTC-USD,ETH-USD
```

## Real Data Requirement (No Synthetic Data)

This system **does not allow synthetic data**. You must load real OHLCV
data from a supported exchange before training or testing the environment.

### Load Real OHLCV Data

```bash
# Example: load 30 days of BTC-USD and ETH-USD candles
python main.py collect-data --source coinbase --symbols BTC-USD,ETH-USD --interval 1h --days 30
```

If data is unavailable, the system will raise an error and stop.

## Quick Start

### Option 1: CLI Training (Simplest)

```bash
cd bot
.\venv\Scripts\activate
python main.py train --episodes 1000
```

This runs a quick training session with 1000 episodes.

### Option 2: Full System (Bot + API + Dashboard)

**Terminal 1 - Bot Training:**
```bash
cd bot
.\venv\Scripts\activate
python main.py train --episodes 10000
```

**Terminal 2 - API Server:**
```bash
cd api
.\venv\Scripts\activate
python main.py
```

**Terminal 3 - Dashboard:**
```bash
cd dashboard
npm run dev
```

Then open http://localhost:3000 to see the dashboard.

## Testing the Environment

Before training, test that everything works:

```bash
cd bot
.\venv\Scripts\activate
python main.py test-env
```

This creates the Gym environment and runs a few steps on **real** data to verify setup.
If no real data is loaded, it will error by design.

## First Training Run

Start with a short training run to verify everything works:

```bash
python main.py train --episodes 1000
```

This should:
- Initialize the environment ✅
- Create a training run in the database ✅
- Train for 1000 episodes (~5-10 minutes) ✅
- Save checkpoints ✅
- Log to TensorBoard ✅

### Monitoring Training

**View TensorBoard:**
```bash
tensorboard --logdir=logs/tensorboard
```

Then open http://localhost:6006

You'll see:
- Episode rewards over time
- Policy loss
- Value loss
- Entropy (exploration)

## Common Issues

### Issue: "No module named 'src'"

**Solution:** Make sure you activated the virtual environment:
```bash
.\venv\Scripts\activate
```

### Issue: "Database connection failed"

**Solution:** Check your `.env` file has the correct `DATABASE_URL`

### Issue: "CUDA out of memory"

**Solution:** The bot will automatically use CPU if GPU fails. If you want to force CPU:
```python
# In bot/src/agents/ppo_agent.py
use_gpu=False
```

### Issue: "ImportError: DLL load failed"

**Solution:** On Windows, you may need Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

## Next Steps

After successful setup:

1. **Read the RL Primer** - `docs/RL_PRIMER.md` for RL concepts
2. **Explore Data** - Run `notebooks/01_data_exploration.ipynb`
3. **Train Longer** - Try 10k-100k episodes
4. **Compare Baselines** - Evaluate vs simple strategies
5. **Analyze Results** - Use the dashboard and notebooks

## Development Workflow

### Daily Workflow

1. **Start Training:**
```bash
cd bot
python main.py train --episodes 10000
```

2. **Monitor Progress:**
- TensorBoard: http://localhost:6006
- Dashboard: http://localhost:3000 (if API running)

3. **Evaluate:**
```bash
python main.py evaluate --model models/best_model_run_1
```

### Making Changes

After modifying code:
```bash
# Run tests
pytest tests/

# Check environment still works
python main.py test-env

# Resume training
python main.py train --episodes 5000
```

## Deployment (Phase 2+)

Deployment to Railway will be covered in Phase 2. For now, everything runs locally.

## Getting Help

**Check Logs:**
```bash
# Bot logs
tail -f logs/bot_YYYYMMDD.log

# API logs
# (printed to console)
```

**Database Inspection:**
Use any PostgreSQL client to connect to your Railway database and inspect tables:
- `training_runs` - All training sessions
- `episodes` - Episode-level metrics
- `trades` - Individual trades
- `models` - Saved checkpoints

**Common Commands:**
```bash
# Info
python main.py info

# Test environment
python main.py test-env

# Quick training
python main.py train --episodes 1000

# Evaluate model
python main.py evaluate --model models/best_model_run_1 --episodes 100
```

## Success Checklist

✅ Python 3.10+ installed  
✅ Virtual environment created and activated  
✅ Dependencies installed  
✅ Database initialized (tables created)  
✅ `.env` file configured  
✅ Test environment runs without errors  
✅ Short training run completes successfully  
✅ TensorBoard shows metrics  

If all checks pass, you're ready to start serious training!

## Resources

- **Documentation:** `docs/` directory
- **Examples:** `notebooks/` directory
- **Configuration:** `shared/config/` directory
- **Logs:** `bot/logs/` directory

---

**Need help?** Check the troubleshooting section above or review the logs for error messages.
