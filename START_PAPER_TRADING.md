# Start Paper Trading - Complete Guide

## 🎯 What This Does

Paper trades **Kalshi prediction markets** about crypto prices by:
1. Polling Kalshi API every 5 minutes for active crypto markets
2. Using statistical edge detection (lognormal model) to find mispriced contracts
3. Opening hypothetical positions when edge > 2%
4. Tracking all trades in database
5. Monitoring settlements and calculating P&L

**NO REAL MONEY IS USED - All trades are simulated**

---

## 🚀 Quick Start Command

```bash
cd bot
python main.py kalshi paper-trade --interval 300 --bankroll 100 --live
```

**What this does:**
- Scans Kalshi markets every 300 seconds (5 minutes)
- Starts with $100 simulated capital
- Uses LIVE Kalshi API (real market data)
- Only trades BUY_NO (default, best backtest performance)
- Logs everything to `bot/logs/paper_trades.jsonl`
- Saves trades to database for dashboard

---

## 📊 Full Paper Trading Options

### Basic Paper Trading
```bash
# Run with defaults (recommended to start)
python main.py kalshi paper-trade --live

# Run for specific duration (stop after N scans)
python main.py kalshi paper-trade --live --max-scans 100

# Use demo API for testing
python main.py kalshi paper-trade --demo
```

### Advanced Options
```bash
# Customize all parameters
python main.py kalshi paper-trade \
  --interval 300 \          # Scan every 5 minutes
  --bankroll 100.0 \        # Start with $100
  --min-edge 0.02 \         # 2% minimum edge
  --max-edge 0.10 \         # 10% max (larger = suspicious)
  --min-price 1 \           # Min 1¢ contracts
  --max-price 15 \          # Max 15¢ contracts
  --max-contracts 10 \      # Max 10 contracts per trade
  --max-positions 20 \      # Max 20 open positions
  --side no \               # BUY_NO only (best strategy)
  --series KXBTC,KXETH \    # Only BTC & ETH markets
  --live                    # Use live API
```

### Check Status
```bash
# View current portfolio
python main.py kalshi paper-status

# View in dashboard
# Navigate to http://localhost:3000 after starting API
```

---

## ✅ Pre-Flight Checklist

Before starting paper trading, ensure:

### 1. **Environment Variables Set**
```bash
# Check .env file exists
cat .env

# Must have:
KALSHI_API_KEY=your_key_here
KALSHI_API_SECRET=your_secret_here
KALSHI_MODE=live  # or 'demo'
```

### 2. **Database Ready**
```bash
# Check database connection
python -c "from bot.src.data.database import get_db_session; get_db_session().close(); print('✅ DB OK')"
```

### 3. **API Server Running** (for dashboard)
```bash
cd api
python main.py
# Should start on http://localhost:8000
```

### 4. **Dashboard Running** (optional, for monitoring)
```bash
cd dashboard
npm run dev
# Should start on http://localhost:3000
```

---

## 📝 The AI Prompt to Start Everything

**Copy and paste this to an AI assistant:**

```
I need to start paper trading the RLTrade bot. Please:

1. Navigate to C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLTrade

2. Check prerequisites:
   - Verify .env file has KALSHI_API_KEY and KALSHI_API_SECRET
   - Confirm database is accessible
   - Check API server status (port 8000)

3. Start paper trading with these settings:
   - Command: python bot/main.py kalshi paper-trade --live --interval 300 --bankroll 100
   - Let it run continuously (use screen/tmux or background process)
   - Monitor initial output for errors

4. Set up monitoring:
   - Ensure API server is running (cd api && python main.py)
   - Ensure dashboard is running (cd dashboard && npm run dev)
   - Open browser to http://localhost:3000

5. After 30 minutes, check status:
   - Run: python bot/main.py kalshi paper-status
   - Review trades in dashboard at http://localhost:3000/positions
   - Check logs at bot/logs/paper_trades.jsonl

6. Monitoring checklist:
   - Are markets being scanned successfully?
   - Are trades being opened when edges detected?
   - Is data flowing to database?
   - Is dashboard showing live data?
   - Any API errors or rate limiting?

7. Let it run for 2-4 weeks to collect sufficient data for analysis.

Expected behavior:
- Scans every 5 minutes
- Opens 0-5 positions per day (depends on edge availability)
- Most trades should be BUY_NO on crypto markets
- Each trade should have edge >= 2%
- Positions close automatically when markets settle
```

---

## 🔍 What to Monitor

### First Hour
- ✅ Bot starts without errors
- ✅ Connects to Kalshi API successfully
- ✅ Scans markets and detects edges
- ✅ Opens at least 1 position (if edge found)
- ✅ Logs appear in `bot/logs/paper_trades.jsonl`

### First Day
- ✅ 2-10 trades opened (varies by market conditions)
- ✅ Trades appear in database
- ✅ Dashboard shows positions
- ✅ No API rate limiting issues
- ✅ Portfolio tracking works

### First Week
- ✅ 10-50 trades completed
- ✅ Some positions settled (markets expired)
- ✅ Win rate around 50-60%
- ✅ P&L is tracked correctly
- ✅ No crashes or hangs

### After 2-4 Weeks
- ✅ 100+ trades for statistical significance
- ✅ Consistent positive P&L
- ✅ Edge detector performance validated
- ✅ Ready for risk management layer

---

## 🚨 Troubleshooting

### "API Key not found"
```bash
# Check .env file
cat .env | grep KALSHI

# Set keys if missing
export KALSHI_API_KEY="your_key"
export KALSHI_API_SECRET="your_secret"
```

### "No markets found"
- Check if using `--live` (not `--demo`)
- Kalshi might not have active crypto markets at the moment
- Try different `--series` filter

### "Database error"
```bash
# Reinitialize database
cd bot
python -c "from src.data.database import init_db; init_db()"
```

### "Rate limited"
- Increase `--interval` to 600 (10 minutes)
- Check Kalshi API rate limits
- Ensure only one instance running

---

## 📈 Success Metrics

After 2-4 weeks, paper trading should show:

| Metric | Target | Good | Concerning |
|--------|--------|------|------------|
| **Total Trades** | 100+ | 50-200 | <20 |
| **Win Rate** | 50-60% | 45-65% | <40% or >75% |
| **Total Return** | +1-5% | +0.5-10% | Negative |
| **Sharpe Ratio** | >1.5 | >1.0 | <0.5 |
| **Max Drawdown** | <10% | <15% | >20% |
| **Avg Edge** | 3-5% | 2-8% | <1% or >10% |

---

## 🎓 Next Steps After Successful Paper Trading

Once you have 2-4 weeks of solid paper trading data:

1. **Analyze Results**
   ```bash
   python bot/scripts/analyze_paper_trading.py
   ```

2. **Run Statistical Validation**
   - Bootstrap analysis
   - Regime breakdown
   - Out-of-sample testing

3. **Implement Risk Management**
   - Add safety rails from `risk_limits.yaml`
   - Build alerting system
   - Test circuit breakers

4. **Micro-Scale Live Testing**
   - Start with $100-500 real capital
   - Same parameters as paper
   - Monitor for 2-4 weeks

5. **Gradual Scale-Up**
   - Only increase capital after proven success
   - Follow scaling schedule in `LIVE_TRADING_READINESS.md`

---

## 💡 Pro Tips

1. **Run in Background**: Use `screen`, `tmux`, or `nohup` for long-running sessions
2. **Monitor Daily**: Check dashboard and logs every day
3. **Save Logs**: Archive `paper_trades.jsonl` weekly
4. **Test Variations**: Try different `--min-edge` thresholds
5. **Compare Sides**: Test `--side yes` vs `--side no` vs `--side both`

---

## 🔗 Related Files

- Configuration: `shared/config/risk_limits.yaml`
- Paper Trading Code: `bot/src/strategies/paper_trader.py`
- Edge Detection: `bot/src/strategies/kalshi_edges.py`
- Database Schema: `bot/src/data/database.py`
- Dashboard: `dashboard/src/app/page.tsx`
