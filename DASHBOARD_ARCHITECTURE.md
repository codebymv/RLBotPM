# Dashboard Architecture - Multi-Bot Support

## 🔍 Current State Analysis

### Database Tables
```
├── kalshi_trades (Kalshi Bot)
│   ├── ticker, side, entry_price_cents
│   ├── edge_value, edge_type
│   ├── mode (paper/live)
│   └── status (open/settled)
│
└── trades (RL Bot - TRAINING ONLY)
    ├── market_id, action, position_size
    ├── immediate_reward, pnl
    └── episode_id (linked to training runs)
```

**Problem:** RL bot paper/live trading doesn't persist to database yet!

### Current Dashboard Pages
1. ✅ **Overview** - Shows Kalshi metrics only
2. ✅ **Positions** - Shows Kalshi trades only
3. ✅ **Market (Crypto)** - Shows Coinbase prices (both bots use this)
4. ✅ **Edge Health** - Shows Kalshi edge stats only
5. ✅ **Bot Status** - Shows Kalshi bot config only

**Verdict:** Dashboard is 100% Kalshi-focused, no RL bot visibility!

---

## 🎯 Proposed Architecture

### Option 1: Bot Selector (RECOMMENDED)

Add a **bot/strategy selector** to the nav, similar to Paper/Live toggle:

```
┌─────────────────────────────────────────────────────────┐
│ [Icon] RLTRADE   [RL Bot ▼] [Overview] [Positions]... [PAPER ▼] │
└─────────────────────────────────────────────────────────┘
           ↑
      Bot Selector:
      - RL Crypto Bot
      - Kalshi Market Bot
      - All Strategies (unified view)
```

**Benefits:**
- ✅ Clean UI, familiar pattern
- ✅ Easy to switch context
- ✅ Can show "All" for combined view
- ✅ Maintains Paper/Live mode independence

**Pages Behavior:**
- **Overview**: Shows selected bot's metrics (or combined)
- **Positions**: Filters to selected bot's trades
- **Edge Health**: Shows selected bot's performance
- **Bot Status**: Shows selected bot's config

---

### Option 2: Separate Nav Sections

Split navigation into two sections:

```
┌──────────────────────────────────────────────────────────┐
│ [Icon] RLTRADE                             [PAPER ▼]      │
├──────────────────────────────────────────────────────────┤
│ RL CRYPTO BOT                                            │
│ [Overview] [Positions] [Performance]                     │
├──────────────────────────────────────────────────────────┤
│ KALSHI MARKETS                                           │
│ [Overview] [Positions] [Edge Health]                     │
├──────────────────────────────────────────────────────────┤
│ SHARED                                                   │
│ [Market Data] [System Status]                            │
└──────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ Very clear separation
- ✅ Bot-specific pages possible
- ✅ No confusion about what you're viewing

**Drawbacks:**
- ❌ More nav items (cluttered)
- ❌ Can't easily compare bots side-by-side

---

### Option 3: Unified View with Filters (SIMPLEST)

Keep current structure, add bot filters to each page:

```
Overview Page:
┌─────────────────────────────────────────┐
│ Trading Performance                     │
│ [All Strategies ▼] [PAPER ▼]           │
│                                         │
│ ┌─────────────┬─────────────┐         │
│ │ RL Crypto   │ Kalshi      │         │
│ │ +$45.20     │ +$12.80     │         │
│ │ 12 trades   │ 8 trades    │         │
│ └─────────────┴─────────────┘         │
└─────────────────────────────────────────┘
```

**Benefits:**
- ✅ Minimal UI changes
- ✅ Easy to compare side-by-side
- ✅ Unified metrics calculation

**Drawbacks:**
- ❌ Bot-specific metrics might not fit same format
- ❌ Could get crowded with more bots

---

## ✅ Recommended Solution: Hybrid Approach

**Combine Option 1 + Option 3:**

### 1. Add Bot Selector to Nav (like Mode Toggle)
```tsx
<nav>
  <Logo />
  <BotSelector />  ← NEW: Select RL / Kalshi / All
  <NavLinks />
  <ModeToggle />   ← Existing: Paper / Live
</nav>
```

### 2. Pages Adapt to Selected Bot

#### Overview Page (Unified)
```
When "All Strategies" selected:
├── Combined Metrics
│   ├── Total P&L: $58.00
│   ├── Combined Win Rate: 56%
│   └── Total Trades: 20
│
├── By Strategy Breakdown
│   ├── RL Crypto: +$45.20 (12 trades)
│   └── Kalshi: +$12.80 (8 trades)
│
└── Recent Activity (All Bots)

When "RL Crypto Bot" selected:
├── RL-Specific Metrics
│   ├── Total Return: +4.5%
│   ├── Sharpe Ratio: 1.8
│   ├── Max Drawdown: -2.3%
│   └── Model: best_model_run_171
│
└── Recent RL Trades

When "Kalshi Market Bot" selected:
├── Kalshi-Specific Metrics
│   ├── Edge Accuracy: 58%
│   ├── Avg Edge: 3.2%
│   ├── Win Rate by Side: NO: 62%, YES: 45%
│   └── Active Markets: 5
│
└── Recent Kalshi Trades
```

#### Positions Page (Filtered)
```
Shows positions for selected bot only
- RL Bot: Shows crypto positions (BTC-USD, ETH-USD, etc.)
- Kalshi Bot: Shows prediction market contracts
- All: Shows both, grouped by strategy
```

#### New: Performance Comparison Page
```
Side-by-side metrics:
├── RL Crypto Bot          │ Kalshi Market Bot
├──────────────────────────┼──────────────────────
│ Return: +4.5%            │ Return: +1.3%
│ Sharpe: 1.8              │ Sharpe: 2.1
│ Win Rate: 52%            │ Win Rate: 58%
│ Avg Trade: $3.77         │ Avg Trade: $1.60
│ Max Drawdown: -2.3%      │ Max Drawdown: -0.8%
└──────────────────────────┴──────────────────────
```

---

## 🛠️ Implementation Plan

### Phase 1: Database Schema (CRITICAL)

**Add RL bot paper/live trading persistence:**

```python
# New table: rl_trades
class RLTrade(Base):
    __tablename__ = "rl_trades"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)  # BTC-USD, ETH-USD
    action = Column(String(10), nullable=False)  # buy, sell
    position_size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    
    # RL-specific
    model_path = Column(String(255), nullable=False)  # which model
    confidence = Column(Float, nullable=True)
    regime = Column(String(50), nullable=True)  # momentum, breakout, etc.
    
    # Standard fields
    mode = Column(String(20), default='paper')  # paper or live
    status = Column(String(20), default='open')  # open or closed
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
```

**Or: Unified trades table (alternative)**

```python
class UnifiedTrade(Base):
    __tablename__ = "trades_v2"
    
    id = Column(Integer, primary_key=True)
    strategy = Column(String(50), nullable=False)  # 'rl_crypto', 'kalshi'
    mode = Column(String(20), default='paper')
    status = Column(String(20), default='open')
    
    # Common fields
    entry_value = Column(Float, nullable=False)
    exit_value = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    
    # Strategy-specific data (JSON)
    strategy_data = Column(JSON, nullable=True)  # Flexible schema
```

### Phase 2: API Endpoints

**Add new endpoints:**

```python
@app.get("/api/strategies/list")
async def get_strategies():
    """List available trading strategies"""
    return {
        "strategies": [
            {"id": "rl_crypto", "name": "RL Crypto Bot", "status": "active"},
            {"id": "kalshi", "name": "Kalshi Market Bot", "status": "active"}
        ]
    }

@app.get("/api/metrics/combined")
async def get_combined_metrics(mode: str = "paper"):
    """Get combined metrics across all strategies"""
    # Aggregate from both rl_trades and kalshi_trades
    pass

@app.get("/api/metrics/by-strategy")
async def get_metrics_by_strategy(strategy: str, mode: str = "paper"):
    """Get metrics for specific strategy"""
    # Query rl_trades or kalshi_trades based on strategy
    pass
```

### Phase 3: Dashboard UI Components

**1. Bot Selector Component**

```tsx
// components/BotSelector.tsx
export type TradingBot = "all" | "rl_crypto" | "kalshi";

export function BotSelector() {
  const [bot, setBot] = useBot(); // Similar to useMode()
  
  return (
    <div className="inline-flex rounded-lg border border-gray-700/60 bg-gray-900/40">
      <button onClick={() => setBot("all")} 
              className={bot === "all" ? "active" : ""}>
        ALL
      </button>
      <button onClick={() => setBot("rl_crypto")}
              className={bot === "rl_crypto" ? "active" : ""}>
        RL CRYPTO
      </button>
      <button onClick={() => setBot("kalshi")}
              className={bot === "kalshi" ? "active" : ""}>
        KALSHI
      </button>
    </div>
  );
}
```

**2. Strategy Badge Component**

```tsx
// components/StrategyBadge.tsx
export function StrategyBadge({ strategy }: { strategy: "rl_crypto" | "kalshi" }) {
  const styles = {
    rl_crypto: "bg-purple-500/20 text-purple-300 border-purple-700/60",
    kalshi: "bg-blue-500/20 text-blue-300 border-blue-700/60"
  };
  
  const labels = {
    rl_crypto: "RL",
    kalshi: "KALSHI"
  };
  
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[9px] font-bold ${styles[strategy]}`}>
      {labels[strategy]}
    </span>
  );
}
```

**3. Updated Overview Page**

```tsx
// app/OverviewClient.tsx
const bot = useBot(); // "all" | "rl_crypto" | "kalshi"
const mode = useMode(); // "paper" | "live"

// Fetch data based on bot selection
const { data: metrics } = useSWR(
  `/api/metrics/${bot === "all" ? "combined" : "by-strategy"}?strategy=${bot}&mode=${mode}`
);

// Show bot-specific or combined view
{bot === "all" ? (
  <CombinedMetricsView metrics={metrics} />
) : (
  <StrategyMetricsView strategy={bot} metrics={metrics} />
)}
```

---

## 📊 Wireframe: Updated Nav

```
┌────────────────────────────────────────────────────────────────┐
│ [Logo]  [All Strategies ▼]  ‖  OVERVIEW  POSITIONS  MARKET...  │
│                                                    [PAPER ▼]    │
└────────────────────────────────────────────────────────────────┘
         ↓
    Dropdown:
    ┌──────────────────────┐
    │ ● All Strategies     │
    │ ─────────────────    │
    │   RL Crypto Bot      │
    │   Kalshi Market Bot  │
    └──────────────────────┘
```

---

## 📊 Wireframe: Updated Overview

```
┌────────────────────────────────────────────────────────────┐
│ RLTRADE                                         ● ALL      │
│                                                 [PAPER]    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│ Trading Performance                                        │
│ Combined metrics across all strategies                    │
│                                                            │
│ ┌──────────┬──────────┬──────────┬──────────┐            │
│ │ Total    │ Combined │ Combined │ Total    │            │
│ │ Return   │ Win Rate │ Trades   │ Capital  │            │
│ │ +$58.00  │ 56%      │ 20       │ $2,000   │            │
│ └──────────┴──────────┴──────────┴──────────┘            │
│                                                            │
│ By Strategy                                                │
│ ┌─────────────────────────┬─────────────────────────┐    │
│ │ RL Crypto Bot      [RL] │ Kalshi Market Bot  [K]  │    │
│ │ +$45.20  |  12 trades   │ +$12.80  |  8 trades   │    │
│ │ 52% WR   |  4.5% return │ 58% WR   |  1.3% return│    │
│ └─────────────────────────┴─────────────────────────┘    │
│                                                            │
│ Recent Activity (All Bots)                                │
│ ┌────────────────────────────────────────────────┐       │
│ │ [RL]  BUY BTC-USD @ $98,234  →  +$3.45        │       │
│ │ [K]   BUY_NO KXBTC-25FEB  →  Pending          │       │
│ │ [RL]  SELL ETH-USD @ $3,421  →  -$0.87        │       │
│ └────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────┘
```

---

## 🎨 Visual Language for Bot Types

### Color Coding
- **RL Crypto Bot**: Purple/Violet (`#9333ea`)
- **Kalshi Market Bot**: Blue (`#3b82f6`)
- **Combined/All**: Cyan (`#22d3ee`)

### Icons
- **RL**: `◆` (diamond - sophisticated)
- **Kalshi**: `▣` (prediction market grid)
- **All**: `⬢` (hexagon - combined)

---

## 🚀 Implementation Priority

### Week 1: Foundation (HIGH PRIORITY)
1. ✅ Add `rl_trades` table to database
2. ✅ Update RL bot paper trader to persist trades
3. ✅ Add API endpoints for combined metrics
4. ✅ Create `BotSelector` component
5. ✅ Add bot selection state management

### Week 2: UI Updates (MEDIUM PRIORITY)
6. ✅ Update Overview page for multi-bot view
7. ✅ Update Positions page with strategy filter
8. ✅ Add strategy badges to trade listings
9. ✅ Create comparison view page

### Week 3: Polish (LOW PRIORITY)
10. ✅ Add charts comparing bot performance
11. ✅ Bot-specific configuration pages
12. ✅ Strategy correlation analysis
13. ✅ Performance attribution

---

## 💡 Quick Win: Minimal Changes

**If you want to ship fast, do this:**

1. **Add bot column to existing tables** (1 hour)
   - `ALTER TABLE kalshi_trades ADD COLUMN strategy VARCHAR(50) DEFAULT 'kalshi'`
   - Update RL bot to write to `kalshi_trades` with `strategy='rl_crypto'`

2. **Add bot filter to UI** (2 hours)
   - Add dropdown to nav: All / RL / Kalshi
   - Filter data in frontend based on selection
   - No API changes needed

3. **Add strategy badge** (1 hour)
   - Show `[RL]` or `[K]` badge on each trade
   - Color code by strategy

**Total time: 4 hours, gets you 80% of the value!**

---

## ❓ Decision: Which Approach?

**My recommendation:** **Hybrid Option 1 + Minimal Changes**

**Implementation:**
1. Add `strategy` column to existing trade tables
2. Add BotSelector to nav (like ModeToggle)
3. Filter views based on selected bot
4. Show combined view when "All" selected
5. Add strategy badges to differentiate trades

**Why?**
- ✅ Fastest to implement (1 week)
- ✅ Clean UI, familiar pattern
- ✅ Easy to compare strategies
- ✅ Minimal breaking changes
- ✅ Extensible for future bots

**What do you think? Want me to implement this?**
