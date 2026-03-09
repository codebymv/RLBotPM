# Live Trading Readiness Checklist

## 🎯 Current Status Assessment

### What We Know
- ✅ Paper trading infrastructure operational
- ✅ Dashboard with Paper/Live mode separation
- ✅ Specialist router system implemented
- ✅ Recent training run (170) showed strong metrics:
  - +0.80% return
  - 9.45 Sharpe ratio
  - 6.35 Profit Factor
- ⚠️ **Need validation**: Performance consistency over time

### Critical Gaps
- ❌ No sustained multi-week paper trading track record
- ❌ No walk-forward validation results
- ❌ No risk management limits configured
- ❌ No live trading safety rails
- ❌ No alerting/monitoring system
- ❌ No small-scale live testing framework

---

## 📋 Phase 1: Validation (2-4 weeks)

### 1.1 Extended Paper Trading Run ⚠️ **CRITICAL**
**Goal**: Prove consistent profitability over time

```bash
# Run paper trading continuously for 2-4 weeks
python bot/main.py paper-trade --continuous --log-level INFO
```

**Success Criteria**:
- [ ] 100+ completed trades across multiple market conditions
- [ ] Positive total return (target: +1% minimum after fees)
- [ ] Sharpe ratio > 1.5
- [ ] Profit factor > 1.3
- [ ] Max drawdown < 10%
- [ ] Fee drag < 15% of gross PnL
- [ ] Win rate: 45-60% (realistic range)
- [ ] No catastrophic losses (single trade < -5% of capital)

**What to Watch**:
- Performance during volatility spikes
- Behavior during low-liquidity periods
- Edge degradation over time
- Specialist router routing accuracy

### 1.2 Statistical Validation
**Goal**: Confirm edge is real, not luck

- [ ] **Bootstrap Analysis**: Resample trades to test robustness
- [ ] **Regime Analysis**: Performance breakdown by market condition
- [ ] **Monte Carlo Simulation**: Test capital survival under stress
- [ ] **Out-of-Sample Testing**: Evaluate on recent unseen data
- [ ] **Backtest vs Live Comparison**: Check for overfitting

### 1.3 Model Confidence Assessment
- [ ] Review recent training run results (Run 170+)
- [ ] Analyze specialist routing effectiveness
- [ ] Check for any warning signs in edge-health metrics
- [ ] Verify no data leakage or lookahead bias

---

## 📋 Phase 2: Risk Management Setup (1 week)

### 2.1 Capital & Position Limits
**Create `shared/config/risk_limits.yaml`**

```yaml
live_trading:
  # Capital limits
  max_total_capital: 1000.00  # Start SMALL
  max_position_size: 50.00    # Per position
  max_open_positions: 5
  
  # Risk limits
  max_daily_loss: 100.00      # Kill switch at -10%
  max_weekly_loss: 200.00
  max_single_trade_loss: 30.00
  
  # Throttling
  max_trades_per_day: 20
  max_trades_per_hour: 5
  min_seconds_between_trades: 60
  
  # Edge requirements (stricter for live)
  min_edge_threshold: 0.08    # 8% minimum (vs 5% paper)
  min_confidence: 0.7
  
  # Circuit breakers
  consecutive_losses_halt: 5  # Stop after 5 losses in a row
  drawdown_halt_pct: 0.15     # Stop at 15% drawdown
```

### 2.2 Safety Rails Implementation
**Add to `bot/src/strategies/live_safety.py`**

```python
class LiveTradingSafety:
    """Safety checks before every live trade"""
    
    def pre_trade_checks(self):
        # Capital limits
        # Position concentration
        # Daily loss limits
        # Edge threshold
        # Liquidity check
        # Price sanity check
        # API connectivity
        pass
    
    def should_halt_trading(self):
        # Circuit breaker logic
        # Drawdown check
        # Consecutive losses
        # API failures
        pass
```

### 2.3 Order Execution Validation
- [ ] Test order placement with $1 positions
- [ ] Verify fill prices match expectations
- [ ] Confirm fee calculations are correct
- [ ] Test maker/taker fallback logic
- [ ] Validate position tracking accuracy

---

## 📋 Phase 3: Monitoring & Alerts (1 week)

### 3.1 Real-Time Monitoring Dashboard
**Enhance existing dashboard with live trading view**

- [ ] Live P&L tracker (updated every 30s)
- [ ] Open position risk meters
- [ ] Circuit breaker status indicators
- [ ] Recent trades feed with alerts
- [ ] Capital utilization gauges
- [ ] Performance vs risk limits

### 3.2 Alert System
**Implemented: `bot/src/monitoring/alerter.py`**

The `AlertSystem` class supports three transports (tried in order):

| Transport | Trigger | Config |
|-----------|---------|--------|
| **AICallerSaaS voice call** | All severities — phone rings | `AICALLER_*` env vars |
| **Generic webhook POST** | All severities | `ALERT_WEBHOOK_URL` |
| **SMTP email** | All severities | `ALERT_SMTP_*` env vars |

**Alert events wired in `live_trader.py`:**
| Event | Severity | Description |
|-------|----------|-------------|
| Session started | info | Mode, limits, sides |
| Kill switch triggered | critical | Consecutive losses or daily loss |
| Order failure | warning | Ticker, side, price |
| Unhandled crash | critical | Exception details |

**Circuit breaker events** (wired via callback in `circuit_breaker.py`):
| Rule | Severity |
|------|----------|
| MAX_DAILY_LOSS_USD / PCT | critical |
| MAX_WEEKLY_LOSS_USD / PCT | critical |
| MAX_TOTAL_DRAWDOWN | critical |
| MAX_CONSECUTIVE_LOSSES | critical |
| MIN_WIN_RATE_THRESHOLD | critical |
| API_ERROR_THRESHOLD | critical |

### 3.2.1 AICallerSaaS Voice Alert Setup

Your phone will ring when any alert fires. Set up as follows:

#### A. Configure the AICallerSaaS Agent
1. Open your AICallerSaaS dashboard (Gleam)
2. Create a new Agent with these settings:
   - **Name:** `RLBot Alert`
   - **Mode:** `OUTBOUND`
   - **Communication Channel:** `VOICE_ONLY`
   - **System Prompt:**
     ```
     You are an emergency alert system for an automated trading bot
     called RLBot. When a call connects, immediately say: "ALERT.
     Your RLBot trading system has triggered a safety event. Please
     check your trading dashboard immediately. Do you acknowledge?"
     If the user acknowledges, say "Acknowledged. Goodbye." and end
     the call. If no response after 10 seconds, repeat the alert
     once, then end. Keep the tone urgent but calm.
     ```
   - **Call Window:** `00:00` – `23:59` (24/7 for trading alerts)
   - **Max Call Duration:** `60` seconds
   - **Voicemail Message:**
     ```
     URGENT: Your RLBot trading system triggered a safety alert.
     Check your dashboard immediately.
     ```
   - **Voice:** Pick a clear, authoritative ElevenLabs voice
3. Assign a Twilio phone number to the agent
4. Note the **Agent ID** from the URL (UUID)

#### B. Generate an API Key
1. Go to **API Keys** in the AICallerSaaS dashboard
2. Create a key — copy the `sk_live_…` value (shown only once)

#### C. Set Environment Variables
```bash
AICALLER_BASE_URL=https://your-gleam-instance.railway.app
AICALLER_API_KEY=<your-api-key-here>
AICALLER_AGENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AICALLER_PHONE_TO=+14155551234
```

#### D. Test
```bash
python -c "
from bot.src.monitoring.alerter import AlertSystem
a = AlertSystem([])
a.send_alert('Test Alert', 'Integration test call', 'info')
"
```
Your phone should ring within seconds.

**Info Notifications** (Dashboard):
- Specialist routing changes
- Market regime shifts
- Model predictions

### 3.3 Logging & Audit Trail
- [ ] All live trades logged to DB with full context
- [ ] Decision reasoning captured (why trade/no trade)
- [ ] API responses saved
- [ ] Daily performance summaries
- [ ] Weekly reconciliation reports

---

## 📋 Phase 4: Micro-Scale Live Testing (1-2 weeks)

### 4.1 Initial Live Capital: $100-$500
**Goal**: Test infrastructure with minimal risk

- [ ] Deploy with $100 starting capital
- [ ] Max $10 per position
- [ ] Max 3 open positions
- [ ] Same edge thresholds as paper
- [ ] Run for 1-2 weeks or 20+ trades

**Success Criteria**:
- Infrastructure works flawlessly
- No unexpected bugs or API issues
- Performance similar to paper trading
- Risk limits enforce correctly
- Alerts fire appropriately

### 4.2 What Could Go Wrong
**Common failure modes to watch for**:

1. **Slippage**: Live fills worse than paper assumptions
2. **Liquidity**: Can't fill at desired prices
3. **Latency**: Delays cause missed opportunities
4. **Fees**: Higher than expected (wrong tier?)
5. **Behavioral**: Model acts differently in live
6. **API Failures**: Downtime, rate limits, errors
7. **Edge Decay**: Paper edge doesn't transfer to live

### 4.3 Kill Criteria
**Stop immediately if**:
- Any critical bug or infrastructure failure
- Performance diverges significantly from paper (>50% worse)
- Risk limits don't enforce properly
- Unexpected losses exceed $50
- Unable to exit positions cleanly

---

## 📋 Phase 5: Gradual Scale-Up (Months)

### 5.1 Scaling Plan
**Only increase capital after proven success**

| Phase | Capital | Max Position | Min Runtime | Exit Criteria |
|-------|---------|--------------|-------------|---------------|
| Micro | $100-500 | $10 | 2 weeks | Profitable, no issues |
| Small | $1,000 | $50 | 4 weeks | Sharpe > 1.5, < 5% DD |
| Medium | $2,500 | $100 | 8 weeks | Proven consistency |
| Standard | $5,000+ | $200 | Ongoing | Full operations |

### 5.2 Performance Benchmarks
**Before scaling up, require**:
- Positive returns in current phase
- All metrics within target ranges
- No major incidents or issues
- Model confidence remains high

---

## 🚨 Critical Pre-Flight Checks

### Before ANY live trading:

#### Legal & Compliance
- [ ] Confirm you're legally allowed to trade (jurisdiction, age, etc.)
- [ ] Understand tax implications of frequent trading
- [ ] Review Kalshi Terms of Service for automated trading
- [ ] Check if API access permits algorithmic trading
- [ ] Consider liability and loss scenarios

#### Infrastructure
- [ ] Separate live API keys from paper keys
- [ ] Enable 2FA on exchange accounts
- [ ] Set up secure credential storage
- [ ] Test emergency shutdown procedure
- [ ] Document runbook for common scenarios

#### Mental Preparation
- [ ] Accept you WILL lose money on individual trades
- [ ] Commit to following risk limits (no manual overrides)
- [ ] Prepare for emotional volatility
- [ ] Have a plan for different scenarios
- [ ] Only risk capital you can afford to lose

---

## 📊 Recommended Immediate Next Steps

### This Week (CRITICAL)
1. **Start Extended Paper Trading Run**
   - Goal: 2-4 weeks continuous operation
   - Log everything for analysis
   
2. **Set Up Monitoring**
   - Daily review of paper trading results
   - Track metrics dashboard
   - Weekly performance reports

3. **Build Risk Management Config**
   - Define all limits in YAML
   - Implement safety checks
   - Test circuit breakers in paper

### Next 2 Weeks
4. **Statistical Validation**
   - Bootstrap analysis of paper trades
   - Regime breakdown
   - Out-of-sample testing

5. **Alert System**
   - Critical alerts via email/SMS
   - Trade notifications
   - Performance tracking

6. **Live Trading Safety Rails**
   - Pre-trade checks
   - Position limits
   - Loss limits

### Week 3-4
7. **Review Paper Trading Results**
   - Did we hit success criteria?
   - Any concerning patterns?
   - Edge still strong?

8. **Final Infrastructure Testing**
   - Test with $1 orders
   - Verify execution logic
   - Confirm monitoring works

9. **Go/No-Go Decision**
   - Review checklist
   - Assess readiness
   - Start micro-scale live OR continue paper

---

## 🎓 Key Principles

1. **Start Small**: $100-500 initial capital max
2. **Move Slowly**: Weeks/months between scale-ups
3. **Trust the Process**: Follow the checklist, no shortcuts
4. **Risk First**: Safety rails before profit optimization
5. **Be Ready to Stop**: Most strategies fail in live trading
6. **Learn Continuously**: Every trade is data

---

## ❓ Open Questions to Answer

1. **Paper Trading Track Record**: How long have we been running paper successfully?
2. **Walk-Forward Results**: Do we have validated out-of-sample performance?
3. **Capital Allocation**: How much are you willing to risk?
4. **Time Commitment**: Can you monitor live trading daily?
5. **Kalshi API Limits**: Any restrictions on automated trading?
6. **Tax/Legal**: Have you consulted with a professional?

---

## 🚦 Go/No-Go Checklist

Only proceed to live trading when ALL are ✅:

### Validation
- [ ] 2+ weeks profitable paper trading
- [ ] 100+ completed paper trades
- [ ] Positive Sharpe ratio (>1.5)
- [ ] Statistical validation complete
- [ ] No major issues or bugs

### Infrastructure  
- [ ] Risk limits configured and tested
- [ ] Safety rails implemented
- [ ] Monitoring dashboard live
- [ ] Alert system working
- [ ] Emergency procedures documented

### Preparation
- [ ] Legal/tax considerations addressed
- [ ] Capital allocation decided
- [ ] Mental preparation complete
- [ ] Support system in place
- [ ] Realistic expectations set

### Final Check
- [ ] Paper performance meets ALL success criteria
- [ ] No outstanding bugs or concerns
- [ ] Team consensus (if applicable)
- [ ] Ready to start with micro capital ($100-500)

---

**Remember**: The goal isn't to go live quickly. The goal is to go live SAFELY when truly ready. Most algorithmic traders fail because they rush this process.

Take your time. The market will still be there tomorrow.
