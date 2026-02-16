# RLTrade Bot Architecture - Complete Overview

## 🤖 Two Separate Trading Bots

### Bot #1: RL Crypto Trading Bot (PRIMARY)
**What it does:** Trades actual crypto spot on Coinbase using a trained reinforcement learning model

```
Coinbase Live Data → RL Model (PPO) → BUY/SELL/HOLD → Crypto Portfolio
```

**Key Details:**
- **Assets**: BTC-USD, ETH-USD, SOL-USD, DOGE-USD, XRP-USD
- **Strategy**: Trained PPO agent (MaskablePPO)
- **Model**: Trained on historical OHLCV data
- **Actions**: BUY, SELL, NO_ACTION
- **Environment**: `CryptoTradingEnv` (Gym-style)
- **Training**: `python main.py train --episodes 10000`
- **Paper Trading**: `python main.py rl-paper-trade --model models/best_model_run_170`

**This is the bot you've been training!** (Run 170, 168, 169, etc.)

---

### Bot #2: Kalshi Prediction Market Bot (SECONDARY)
**What it does:** Trades binary prediction market contracts on Kalshi about crypto prices

```
Crypto Prices → Statistical Model → Edge Detection → Kalshi YES/NO Contracts
```

**Key Details:**
- **Assets**: Kalshi markets (KXBTC, KXETH, KXSOL, etc.)
- **Strategy**: Lognormal statistical edge detector
- **Model**: No ML - pure statistical arbitrage
- **Actions**: BUY_YES or BUY_NO on binary contracts
- **Training**: None (rule-based)
- **Paper Trading**: `python main.py kalshi paper-trade --live`

**Example Kalshi Market:**
- "Will BTC close above $100,000 by Feb 28?"
- YES contract: 12¢, NO contract: 88¢
- If model says YES is overpriced → BUY_NO

---

## 📊 Comparison Table

| Feature | RL Crypto Bot | Kalshi Market Bot |
|---------|---------------|-------------------|
| **Trading** | Actual crypto spot | Binary prediction contracts |
| **Exchange** | Coinbase | Kalshi |
| **Strategy** | Reinforcement Learning (PPO) | Statistical edge detection |
| **Training Required?** | YES - 10k+ episodes | NO - rule-based |
| **Model Type** | Neural network (MaskablePPO) | Lognormal probability |
| **Actions** | BUY, SELL, HOLD | BUY_YES, BUY_NO |
| **Complexity** | High | Medium |
| **Backtest Win Rate** | ~50-60% (varies) | 100% (BUY_NO historical) |
| **Primary Use** | Main trading strategy | Alternative/hedge strategy |
| **Paper Trade Command** | `rl-paper-trade` | `kalshi paper-trade` |

---

## 🚀 Which Bot Should You Focus On?

### For **Profitable Live Trading** → **RL Crypto Bot (#1)**

**Why?**
- This is your main project with trained models
- You've already trained Run 170 (+0.80% return, 9.45 Sharpe)
- Sophisticated RL strategy with regime specialists
- Better scalability and capital efficiency
- Direct crypto exposure

**Next Steps:**
1. Train for 10,000 episodes (as planned)
2. Run extended paper trading with best model
3. Validate performance consistency
4. Deploy to live with small capital

---

### For **Quick Statistical Arbitrage** → **Kalshi Bot (#2)**

**Why?**
- No training required (rule-based)
- Historical backtest: 100% win rate on BUY_NO
- Lower complexity, easier to understand
- Good for learning paper trading mechanics
- Can run alongside RL bot

**Next Steps:**
1. Start paper trading immediately
2. Collect 2-4 weeks of data
3. Validate edge still exists
4. Deploy to live with tiny capital ($100)

---

## 🎯 Recommended Approach: Run BOTH

### Parallel Strategy
```
┌─────────────────────────────────┐
│  RL Crypto Bot (PRIMARY)        │
│  Coinbase spot trading          │
│  Trained PPO model              │
│  Capital: $1000-5000            │
└─────────────────────────────────┘
              +
┌─────────────────────────────────┐
│  Kalshi Market Bot (SECONDARY)  │
│  Prediction market edges        │
│  Statistical arbitrage          │
│  Capital: $100-500              │
└─────────────────────────────────┘
```

**Benefits:**
- Diversification across strategies
- Uncorrelated returns (crypto vs prediction markets)
- Kalshi bot validates paper trading infrastructure
- Learn from simpler bot before scaling RL bot

---

## 📝 Complete Command Reference

### RL Crypto Bot (Primary)

#### Training
```bash
# Train new model
python bot/main.py train --episodes 10000

# Resume from checkpoint
python bot/main.py train --episodes 10000 --checkpoint models/checkpoint_run_170

# Train with specialist router
python bot/main.py train --episodes 10000 --config shared/config/model_config.yaml
```

#### Evaluation
```bash
# Evaluate model
python bot/main.py evaluate --model models/best_model_run_170 --episodes 100

# Evaluate with specialist router
python bot/main.py evaluate --model models/best_model_run_170 --specialist-router
```

#### Paper Trading (Live Data)
```bash
# Run indefinitely
python bot/main.py rl-paper-trade --model models/best_model_run_170 --duration 0

# Run for 24 hours
python bot/main.py rl-paper-trade --model models/best_model_run_170 --duration 24 --capital 1000

# Specific symbols only
python bot/main.py rl-paper-trade --model models/best_model_run_170 --symbols BTC-USD,ETH-USD
```

---

### Kalshi Market Bot (Secondary)

#### Paper Trading
```bash
# Run with defaults (BUY_NO only, recommended)
python bot/main.py kalshi paper-trade --live --interval 300 --bankroll 100

# Custom parameters
python bot/main.py kalshi paper-trade \
  --live \
  --interval 300 \
  --bankroll 100 \
  --min-edge 0.02 \
  --max-edge 0.10 \
  --side no
```

#### Check Status
```bash
# View portfolio
python bot/main.py kalshi paper-status

# View logs
Get-Content bot/logs/paper_trades.jsonl -Tail 20
```

---

## 🎓 Which to Start With?

### Option A: Start with Kalshi Bot (Easier)
**Pros:**
- ✅ No training required
- ✅ Can start immediately
- ✅ Simpler to understand
- ✅ Tests dashboard/infrastructure

**Cons:**
- ❌ Limited scalability
- ❌ Smaller edge
- ❌ Depends on Kalshi market availability

**Timeline:** Start today, validate in 2 weeks

---

### Option B: Start with RL Crypto Bot (Better Long-term)
**Pros:**
- ✅ Your main project with trained models
- ✅ Better scalability
- ✅ More sophisticated strategy
- ✅ Direct crypto exposure

**Cons:**
- ❌ Needs training run first
- ❌ More complex to validate
- ❌ Requires ML expertise

**Timeline:** Train now (24-48h), paper trade 2-4 weeks

---

## 🚀 Recommended Action Plan

### Week 1-2: Start Both Bots

#### RL Crypto Bot
```bash
# 1. Start 10k episode training (Terminal 1)
cd bot
python main.py train --episodes 10000

# 2. While training, monitor in dashboard
cd ../api && python main.py  # Terminal 2
cd ../dashboard && npm run dev  # Terminal 3
```

#### Kalshi Bot (Parallel)
```bash
# 3. Start Kalshi paper trading (Terminal 4)
cd bot
python main.py kalshi paper-trade --live --interval 300 --bankroll 100
```

### Week 2-4: Validate Both

#### RL Crypto Bot
```bash
# When training completes, start paper trading
python main.py rl-paper-trade --model models/best_model_run_171 --duration 0
```

#### Kalshi Bot
- Review 2-week performance
- Check win rate, P&L, edge accuracy
- Decide if strategy still works

### Week 4+: Live Trading Decision

**Go live with:**
- ✅ Whichever bot has better paper trading results
- ✅ Both bots with small capital ($100-500 each)
- ✅ RL bot only if training + paper results are strong

---

## 💡 Pro Tips

1. **Run Both**: They're uncorrelated, so you get diversification
2. **Start Small**: $100-500 total across both bots
3. **Monitor Daily**: Use dashboard for both
4. **Compare Results**: See which strategy works better
5. **Scale Winner**: After 2-4 weeks, increase capital on better performer

---

## 📁 Key Files for Each Bot

### RL Crypto Bot
- Training: `bot/src/training/trainer.py`
- Environment: `bot/src/environment/gym_env.py`
- Live Trading: `bot/src/execution/live_rl_trader.py`
- Models: `bot/models/best_model_run_XXX/`
- Config: `shared/config/model_config.yaml`

### Kalshi Bot
- Paper Trading: `bot/src/strategies/paper_trader.py`
- Edge Detection: `bot/src/strategies/kalshi_edges.py`
- Live Trading: `bot/src/strategies/live_trader.py`
- Logs: `bot/logs/paper_trades.jsonl`

---

## ❓ Which Bot for What?

### Use RL Crypto Bot for:
- Main trading strategy
- Larger capital deployment ($1k-10k+)
- Sophisticated ML-based decisions
- Long-term profitability

### Use Kalshi Bot for:
- Quick statistical arbitrage
- Small capital side bets ($100-500)
- Learning paper trading
- Hedging crypto exposure
- Testing infrastructure

---

## 🎯 Final Recommendation

**Start BOTH bots in parallel:**

1. **RL Bot**: Kick off 10k training run NOW
2. **Kalshi Bot**: Start paper trading IMMEDIATELY
3. **Compare**: After 2-4 weeks, see which performs better
4. **Deploy**: Go live with whichever shows consistent profitability

This gives you:
- Diversification
- Faster feedback (Kalshi results in days)
- Insurance (if one fails, other might work)
- Learning opportunity from simpler bot

**Commands to run RIGHT NOW:**

```bash
# Terminal 1: Start RL training
cd bot && python main.py train --episodes 10000

# Terminal 2: Start Kalshi paper trading
cd bot && python main.py kalshi paper-trade --live

# Terminal 3: Monitor via API
cd api && python main.py

# Terminal 4: Monitor via Dashboard
cd dashboard && npm run dev
```

**Then monitor both at http://localhost:3000 and compare results!**
