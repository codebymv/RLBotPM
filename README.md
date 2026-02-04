# RLTrade - Reinforcement Learning Crypto Trading Bot

A comprehensive, safety-focused reinforcement learning trading bot for crypto spot markets. Built with a phased approach prioritizing learning, safety, and realistic performance expectations.

## 🎯 Project Overview

This bot uses **Proximal Policy Optimization (PPO)** to learn profitable trading strategies on real crypto market data while incorporating robust risk management and safety controls. The system is designed for gradual deployment across three phases:

- **Phase 1 (Current)**: Historical backtesting environment
- **Phase 2**: Paper trading with live data
- **Phase 3**: Minimal live deployment with strict controls

## Real Data Only Policy

This system **does not allow synthetic data**. Training and evaluation only run
when real OHLCV data is available from configured crypto exchanges. If data is
missing or the source is unavailable, the system fails fast with a clear error.

## 🏗️ Architecture

```
RLTrade/
├── bot/              # Core RL trading system (Python)
├── api/              # FastAPI monitoring backend
├── dashboard/        # React monitoring dashboard
├── shared/           # Shared configs & schemas
├── infrastructure/   # Docker & Railway configs
├── notebooks/        # Jupyter analysis notebooks
└── tests/           # Integration tests
```

### Monorepo Structure

This project uses a monorepo architecture similar to modern web applications, allowing independent deployment of:
- **Bot**: Runs locally for training (Phase 1) or on Railway (Phase 2+)
- **API**: Always-on FastAPI backend for monitoring
- **Dashboard**: Web UI for remote monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for dashboard)
- PostgreSQL (Railway instance provided)
- Git

### Installation

1. **Clone and setup:**
```bash
cd C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLTrade
```

2. **Setup Bot (Python):**
```bash
cd bot
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
# Copy and edit .env file
cp .env.example .env
# Add your Railway PostgreSQL URL
```

4. **Setup API (Python):**
```bash
cd api
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

5. **Setup Dashboard (Node.js):**
```bash
cd dashboard
npm install
```

### Running the System

**Phase 1 - Local Training:**

```bash
# Terminal 1: Start the bot training
cd bot
python main.py train --episodes 10000

# Terminal 2: Start the API (for monitoring)
cd api
uvicorn main:app --reload

# Terminal 3: Start the dashboard
cd dashboard
npm run dev
```

Access the dashboard at `http://localhost:3000`

## 📊 Key Features

### Reinforcement Learning
- **Algorithm**: PPO (Proximal Policy Optimization) via Stable-Baselines3
- **State Space**: 38 dimensions (market features, portfolio state, temporal data)
- **Action Space**: 8 discrete actions (no action, buy/sell at different sizes)
- **Reward Function**: Sharpe ratio-based with penalties for risk violations

### Safety & Risk Management
- **Hard Stops**: Daily/weekly loss limits, position size limits
- **Circuit Breakers**: Automatic pause on consecutive losses or low win rate
- **Position Sizing**: Kelly Criterion-based (fractional, conservative)
- **Trade Validation**: Pre-trade checks for capital, liquidity, correlation

### Monitoring & Analysis
- **Real-time Dashboard**: Track training progress remotely
- **TensorBoard Integration**: Detailed training metrics
- **Performance Analytics**: Sharpe ratio, drawdown, win rate tracking
- **Trade Logging**: Complete audit trail of all decisions

## 📈 Training Pipeline

### Curriculum Learning
Training progresses through stages of increasing difficulty:

1. **Stage 1 (0-20k steps)**: Simple, liquid markets
2. **Stage 2 (20k-50k)**: Mixed categories, medium liquidity
3. **Stage 3 (50k-100k)**: All market types
4. **Stage 4 (100k+)**: Edge cases and challenging markets

### Evaluation Metrics
- Total return (absolute and percentage)
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown
- Win rate and profit factor
- Comparison vs baseline strategies

## 🛡️ Safety Philosophy

This bot is designed with **conservative safety-first principles**:

1. **Phase 1**: Pure backtesting, no real money
2. **Phase 2**: Paper trading only, proving consistency
3. **Phase 3**: Minimal capital ($50-100), strict limits
4. **Gradual scaling**: Only after sustained profitability

### Risk Controls
- Max position size: 20% of capital
- Max open positions: 3 concurrent
- Max daily loss: $20 or 5% (whichever smaller)
- Max total drawdown: 30% (triggers full pause)

## 📚 Documentation

- `docs/RL_PRIMER.md` - Reinforcement learning concepts explained
- `docs/SETUP.md` - Detailed setup instructions
- `docs/ARCHITECTURE.md` - System design and data flow
- `docs/TRAINING_GUIDE.md` - How to train and evaluate models
- `docs/API_REFERENCE.md` - API endpoint documentation

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_environment.py
pytest tests/test_risk_system.py
```

## 📦 Deployment

### Railway Deployment (Phase 2+)

The monorepo structure allows deploying individual services to Railway:

1. **Database**: Already deployed (PostgreSQL)
2. **API**: Deploy from `/api` subdirectory
3. **Dashboard**: Deploy from `/dashboard` subdirectory
4. **Bot**: Deploy from `/bot` subdirectory (Phase 2+)

See `infrastructure/railway.json` for configuration.

## 🔧 Technology Stack

**Bot:**
- Stable-Baselines3 (RL algorithms)
- PyTorch (neural networks)
- Gymnasium (environment interface)
- SQLAlchemy (database ORM)
- Pandas/NumPy (data processing)

**API:**
- FastAPI (async web framework)
- Pydantic (validation)
- Alembic (database migrations)
- PostgreSQL (data storage)

**Dashboard:**
- Next.js 14 (React framework)
- TailwindCSS (styling)
- Recharts (visualizations)
- TanStack Query (data fetching)

## ⚠️ Important Disclaimers

- **Educational Purpose**: This is a learning project, not financial advice
- **Risk Warning**: Prediction markets involve financial risk
- **No Guarantees**: Past performance does not guarantee future results
- **Regulatory Compliance**: Understand local gambling/trading laws
- **Conservative Approach**: Always start with backtesting and minimal capital

## 🎓 Learning Resources

New to reinforcement learning? Start here:
- Read `docs/RL_PRIMER.md` for RL basics
- Work through `notebooks/01_data_exploration.ipynb`
- Review baseline strategies in `bot/src/agents/baseline_agents.py`
- Watch training metrics in TensorBoard

## 🤝 Contributing

This is currently a personal project for learning. However, suggestions and feedback are welcome!

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Stable-Baselines3 team for excellent RL implementations
- Coinbase and Kraken for public market data APIs
- OpenAI for PPO algorithm research

---

**Status**: Phase 1 - In Development
**Last Updated**: February 2026
