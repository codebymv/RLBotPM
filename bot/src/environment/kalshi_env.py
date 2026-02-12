"""
Custom Gymnasium Environment for Kalshi Prediction Markets Trading.

This environment is designed for binary prediction market trading,
where outcomes are YES/NO and prices represent probabilities.

Key differences from crypto trading:
- Binary outcomes (settle at $1 or $0)
- Time decay towards expiration
- Price = probability (0-1)
- Position: long YES, long NO, or flat
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, List, Any, TYPE_CHECKING
from pathlib import Path
import yaml

from ..core.logger import get_logger
from ..core.config import get_settings
from ..data.sources.base import DataUnavailableError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger(__name__)


class KalshiTradingEnv(gym.Env):
    """
    Prediction market trading environment for Kalshi.
    
    Observation Space (10 dimensions):
        [0] yes_price: Current YES price (0-1)
        [1] no_price: Current NO price (0-1), typically 1 - yes_price
        [2] spread: yes_ask - yes_bid (0-1)
        [3] time_to_expiry: Normalized time remaining (0-1)
        [4] momentum_5: Price change over last 5 steps
        [5] momentum_20: Price change over last 20 steps
        [6] volume_ratio: Recent volume / average volume
        [7] position: Current position (-1=NO, 0=flat, 1=YES)
        [8] position_pnl: Unrealized P&L of position (normalized)
        [9] model_signal: External signal (e.g., from crypto PPO model)
    
    Action Space: Discrete(5)
        0: HOLD - Do nothing
        1: BUY_YES - Buy YES contracts
        2: BUY_NO - Buy NO contracts  
        3: SELL_YES - Sell YES position (if holding)
        4: SELL_NO - Sell NO position (if holding)
    
    Reward:
        - Realized P&L from trades
        - Small penalty for spread crossing
        - Bonus for correct directional bets
        - Penalty for holding through expiry with wrong bet
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Actions
    ACTION_HOLD = 0
    ACTION_BUY_YES = 1
    ACTION_BUY_NO = 2
    ACTION_SELL_YES = 3
    ACTION_SELL_NO = 4
    
    ACTION_NAMES = {
        0: "HOLD",
        1: "BUY_YES",
        2: "BUY_NO",
        3: "SELL_YES",
        4: "SELL_NO",
    }
    
    def __init__(
        self,
        dataset: pd.DataFrame,
        initial_capital: float = 1000.0,
        max_steps: int = 200,
        position_size_pct: float = 0.01,
        max_position_pct: float = 0.05,
        spread_cost: float = 0.02,  # 2 cents typical spread
        include_signal: bool = True,
    ):
        """
        Initialize Kalshi trading environment.
        
        Args:
            dataset: DataFrame with columns:
                - ticker: Market identifier
                - timestamp: Time
                - yes_price: YES contract price (0-100 cents)
                - yes_bid: Best bid for YES
                - yes_ask: Best ask for YES
                - volume: Trading volume
                - time_to_expiry: Seconds until expiration
                - outcome: Final outcome (1=YES, 0=NO, None if not settled)
                - signal: Optional external signal (-1 to 1)
            initial_capital: Starting capital in dollars
            max_steps: Maximum steps per episode
            position_size_pct: Fraction of capital per trade
            max_position_pct: Max position in single market
            spread_cost: Estimated spread cost (as fraction)
            include_signal: Whether to include external signal in obs
        """
        super().__init__()
        
        self.settings = get_settings()
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        self.position_size_pct = position_size_pct
        self.max_position_pct = max_position_pct
        self.spread_cost = spread_cost
        self.include_signal = include_signal
        
        # Load config
        self.config = self._load_kalshi_config()
        
        # Validate dataset
        if dataset is None or dataset.empty:
            raise DataUnavailableError("Dataset is empty. Real market data required.")
        
        required_cols = {"ticker", "timestamp", "yes_price"}
        if not required_cols.issubset(set(dataset.columns)):
            missing = required_cols - set(dataset.columns)
            raise DataUnavailableError(f"Dataset missing required columns: {missing}")
        
        self.dataset = dataset.copy()
        self._prepare_data()
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space: 10 dimensions
        self.obs_dim = 10
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        
        # Portfolio state
        self.capital = self.initial_capital
        self.position: Dict[str, Any] = {}  # Current position info
        self.trade_history: List[Dict] = []
        
        # Episode state
        self.current_step = 0
        self.current_index = 0
        self.current_ticker = ""
        self.market_data: pd.DataFrame = pd.DataFrame()
        self.terminated = False
        
        # Performance tracking
        self.episode_pnl = 0.0
        self.win_count = 0
        self.trade_count = 0
        self.correct_predictions = 0
        
        # Price history for momentum calculation
        self.price_history: List[float] = []
        
        logger.info(f"KalshiTradingEnv initialized with {len(self.dataset)} records")

    @classmethod
    def from_db(
        cls,
        session: "Session",
        tickers: Optional[List[str]] = None,
        min_rows_per_market: int = 50,
        **env_kwargs: Any,
    ) -> "KalshiTradingEnv":
        """Build env from Kalshi market history in the database."""
        from ..data.sources.kalshi import load_kalshi_dataset_from_db
        dataset = load_kalshi_dataset_from_db(
            session, tickers=tickers, min_rows_per_market=min_rows_per_market
        )
        if dataset is None or dataset.empty:
            raise DataUnavailableError(
                "No Kalshi history in DB. Run: python main.py kalshi backfill"
            )
        return cls(dataset=dataset, **env_kwargs)
    
    def _load_kalshi_config(self) -> Dict:
        """Load Kalshi configuration."""
        config_path = Path(__file__).parents[3] / "shared" / "config" / "kalshi_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _prepare_data(self) -> None:
        """Prepare and validate dataset."""
        # Convert prices from cents to probability (0-1)
        if self.dataset["yes_price"].max() > 1:
            self.dataset["yes_price"] = self.dataset["yes_price"] / 100.0
        
        # Add no_price if not present
        if "no_price" not in self.dataset.columns:
            self.dataset["no_price"] = 1.0 - self.dataset["yes_price"]
        elif self.dataset["no_price"].max() > 1:
            self.dataset["no_price"] = self.dataset["no_price"] / 100.0
        
        # Add bid/ask if not present
        if "yes_bid" not in self.dataset.columns:
            self.dataset["yes_bid"] = self.dataset["yes_price"] - self.spread_cost / 2
        elif self.dataset["yes_bid"].max() > 1:
            self.dataset["yes_bid"] = self.dataset["yes_bid"] / 100.0
            
        if "yes_ask" not in self.dataset.columns:
            self.dataset["yes_ask"] = self.dataset["yes_price"] + self.spread_cost / 2
        elif self.dataset["yes_ask"].max() > 1:
            self.dataset["yes_ask"] = self.dataset["yes_ask"] / 100.0
        
        # Ensure volume exists
        if "volume" not in self.dataset.columns:
            self.dataset["volume"] = 100
        
        # Ensure time_to_expiry exists (normalized 0-1)
        if "time_to_expiry" not in self.dataset.columns:
            self.dataset["time_to_expiry"] = 1.0
        elif self.dataset["time_to_expiry"].max() > 1:
            # Normalize if in seconds
            max_tte = self.dataset["time_to_expiry"].max()
            self.dataset["time_to_expiry"] = self.dataset["time_to_expiry"] / max_tte
        
        # Add signal column if missing
        if "signal" not in self.dataset.columns:
            self.dataset["signal"] = 0.0
        
        # Get unique markets
        self.tickers = self.dataset["ticker"].unique().tolist()
        logger.info(f"Prepared data with {len(self.tickers)} unique markets")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset portfolio
        self.capital = self.initial_capital
        self.position = {}
        self.trade_history = []
        
        # Reset episode state
        self.current_step = 0
        self.terminated = False
        self.episode_pnl = 0.0
        self.win_count = 0
        self.trade_count = 0
        self.correct_predictions = 0
        self.price_history = []
        
        # Select a random market and starting point
        self._select_episode_market()
        
        # Initialize price history
        for i in range(min(20, self.current_index)):
            idx = self.current_index - 20 + i
            if idx >= 0 and idx < len(self.market_data):
                self.price_history.append(float(self.market_data.iloc[idx]["yes_price"]))
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _select_episode_market(self) -> None:
        """Select a market and time window for the episode."""
        # Pick random ticker
        self.current_ticker = str(self.np_random.choice(self.tickers))
        
        # Get data for this ticker
        self.market_data = self.dataset[
            self.dataset["ticker"] == self.current_ticker
        ].reset_index(drop=True)
        
        # Pick random starting point (allow room for episode)
        max_start = max(0, len(self.market_data) - self.max_steps - 1)
        self.current_index = int(self.np_random.integers(0, max(1, max_start)))
        
        logger.debug(f"Selected market {self.current_ticker} at index {self.current_index}")
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        if self.current_index >= len(self.market_data):
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        row = self.market_data.iloc[self.current_index]
        
        # Current prices
        yes_price = float(row["yes_price"])
        no_price = float(row["no_price"])
        spread = float(row["yes_ask"]) - float(row["yes_bid"])
        time_to_expiry = float(row["time_to_expiry"])
        
        # Update price history
        self.price_history.append(yes_price)
        if len(self.price_history) > 50:
            self.price_history.pop(0)
        
        # Momentum features
        momentum_5 = 0.0
        momentum_20 = 0.0
        if len(self.price_history) >= 5:
            momentum_5 = yes_price - self.price_history[-5]
        if len(self.price_history) >= 20:
            momentum_20 = yes_price - self.price_history[-20]
        
        # Volume ratio
        volume = float(row["volume"])
        avg_volume = max(1, self.market_data["volume"].mean())
        volume_ratio = np.clip(volume / avg_volume, 0, 5) - 1  # Center around 0
        
        # Position info
        position_side = 0.0
        position_pnl = 0.0
        if self.position:
            if self.position["side"] == "YES":
                position_side = 1.0
                position_pnl = (yes_price - self.position["entry_price"]) * self.position["contracts"]
            else:  # NO
                position_side = -1.0
                position_pnl = (no_price - self.position["entry_price"]) * self.position["contracts"]
            # Normalize P&L
            position_pnl = np.clip(position_pnl / self.initial_capital, -1, 1)
        
        # External signal
        signal = float(row.get("signal", 0.0)) if self.include_signal else 0.0
        
        obs = np.array([
            yes_price * 2 - 1,  # Scale to [-1, 1]
            no_price * 2 - 1,
            spread * 10,  # Amplify spread signal
            time_to_expiry * 2 - 1,
            momentum_5 * 10,
            momentum_20 * 10,
            volume_ratio,
            position_side,
            position_pnl,
            signal,
        ], dtype=np.float32)
        
        return np.clip(obs, -5.0, 5.0)
    
    def _get_info(self) -> Dict:
        """Return episode info."""
        return {
            "ticker": self.current_ticker,
            "step": self.current_step,
            "capital": self.capital,
            "position": self.position.copy() if self.position else None,
            "episode_pnl": self.episode_pnl,
            "trade_count": self.trade_count,
            "win_rate": self.win_count / max(1, self.trade_count),
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.terminated or self.current_index >= len(self.market_data):
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        row = self.market_data.iloc[self.current_index]
        yes_price = float(row["yes_price"])
        no_price = float(row["no_price"])
        yes_ask = float(row["yes_ask"])
        yes_bid = float(row["yes_bid"])
        
        reward = 0.0
        trade_executed = False
        
        # Get valid actions
        valid_actions = self._get_valid_actions()
        
        # Execute action
        action_name = self.ACTION_NAMES.get(action, "UNKNOWN")
        
        if action not in valid_actions:
            # Invalid action penalty
            reward -= 0.001
        elif action == self.ACTION_BUY_YES and not self.position:
            # Buy YES contracts
            trade_executed = self._execute_buy("YES", yes_ask)
            reward -= self.spread_cost  # Spread cost
            
        elif action == self.ACTION_BUY_NO and not self.position:
            # Buy NO contracts
            no_ask = 1.0 - yes_bid  # NO ask = 1 - YES bid
            trade_executed = self._execute_buy("NO", no_ask)
            reward -= self.spread_cost
            
        elif action == self.ACTION_SELL_YES and self.position.get("side") == "YES":
            # Sell YES position
            pnl = self._execute_sell(yes_bid)
            reward += pnl
            trade_executed = True
            
        elif action == self.ACTION_SELL_NO and self.position.get("side") == "NO":
            # Sell NO position
            no_bid = 1.0 - yes_ask  # NO bid = 1 - YES ask
            pnl = self._execute_sell(no_bid)
            reward += pnl
            trade_executed = True
        
        # Advance time
        self.current_step += 1
        self.current_index += 1
        
        # Check for expiration
        time_remaining = float(row["time_to_expiry"])
        terminated = False
        truncated = False
        
        if time_remaining <= 0.01:  # Near expiration
            # Settle position if any
            if self.position:
                outcome = row.get("outcome")
                if outcome is not None:
                    settlement_reward = self._settle_position(int(outcome))
                    reward += settlement_reward
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True
        elif self.current_index >= len(self.market_data):
            truncated = True
        
        self.terminated = terminated
        
        observation = self._get_observation()
        info = self._get_info()
        info["action"] = action_name
        info["trade_executed"] = trade_executed
        
        return observation, reward, terminated, truncated, info
    
    def _get_valid_actions(self) -> List[int]:
        """Return list of valid actions given current state."""
        valid = [self.ACTION_HOLD]
        
        if not self.position:
            # Can open new position
            valid.extend([self.ACTION_BUY_YES, self.ACTION_BUY_NO])
        else:
            # Can close existing position
            if self.position["side"] == "YES":
                valid.append(self.ACTION_SELL_YES)
            else:
                valid.append(self.ACTION_SELL_NO)
        
        return valid
    
    def _execute_buy(self, side: str, price: float) -> bool:
        """Execute a buy order."""
        # Calculate position size
        position_value = self.capital * self.position_size_pct
        position_value = min(position_value, self.capital * self.max_position_pct)
        
        # contracts * price = position_value
        contracts = position_value / max(0.01, price)
        cost = contracts * price
        
        if cost > self.capital:
            return False
        
        self.capital -= cost
        self.position = {
            "side": side,
            "contracts": contracts,
            "entry_price": price,
            "entry_step": self.current_step,
        }
        
        self.trade_count += 1
        self.trade_history.append({
            "type": "BUY",
            "side": side,
            "price": price,
            "contracts": contracts,
            "step": self.current_step,
        })
        
        return True
    
    def _execute_sell(self, price: float) -> float:
        """Execute a sell order, return P&L."""
        if not self.position:
            return 0.0
        
        contracts = self.position["contracts"]
        entry_price = self.position["entry_price"]
        
        # Proceeds from sale
        proceeds = contracts * price
        cost = contracts * entry_price
        pnl = proceeds - cost
        
        self.capital += proceeds
        self.episode_pnl += pnl
        
        if pnl > 0:
            self.win_count += 1
        
        self.trade_history.append({
            "type": "SELL",
            "side": self.position["side"],
            "price": price,
            "contracts": contracts,
            "pnl": pnl,
            "step": self.current_step,
        })
        
        self.position = {}
        return pnl
    
    def _settle_position(self, outcome: int) -> float:
        """Settle position at expiration."""
        if not self.position:
            return 0.0
        
        contracts = self.position["contracts"]
        entry_price = self.position["entry_price"]
        side = self.position["side"]
        
        # Settlement price: 1.0 if correct, 0.0 if wrong
        if side == "YES":
            settlement_price = 1.0 if outcome == 1 else 0.0
        else:  # NO
            settlement_price = 1.0 if outcome == 0 else 0.0
        
        proceeds = contracts * settlement_price
        cost = contracts * entry_price
        pnl = proceeds - cost
        
        self.capital += proceeds
        self.episode_pnl += pnl
        
        if pnl > 0:
            self.win_count += 1
            self.correct_predictions += 1
        
        self.trade_history.append({
            "type": "SETTLEMENT",
            "side": side,
            "outcome": outcome,
            "settlement_price": settlement_price,
            "contracts": contracts,
            "pnl": pnl,
            "step": self.current_step,
        })
        
        self.position = {}
        return pnl
    
    def render(self, mode: str = "human") -> None:
        """Render current state."""
        if mode != "human":
            return
        
        print(f"\n=== Step {self.current_step} ===")
        print(f"Market: {self.current_ticker}")
        print(f"Capital: ${self.capital:.2f}")
        
        if self.position:
            print(f"Position: {self.position['contracts']:.2f} {self.position['side']} @ {self.position['entry_price']:.3f}")
        else:
            print("Position: Flat")
        
        if self.current_index < len(self.market_data):
            row = self.market_data.iloc[self.current_index]
            print(f"YES Price: {row['yes_price']:.3f}")
            print(f"Time to Expiry: {row['time_to_expiry']:.3f}")
    
    def get_action_mask(self) -> np.ndarray:
        """Return action mask for masked PPO."""
        mask = np.zeros(5, dtype=np.float32)
        for action in self._get_valid_actions():
            mask[action] = 1.0
        return mask


# ---------------------------------------------------------------------------
# Event-level binary outcome environment (trained on settled market snapshots)
# ---------------------------------------------------------------------------

@dataclass
class _KalshiPos:
    """A position in a single Kalshi contract."""
    ticker: str
    side: str           # 'yes' or 'no'
    entry_price: float  # cents (0-100)
    contracts: int
    outcome: int        # 1=YES won, 0=NO won


@dataclass
class _EventGroup:
    """A group of contracts from the same event (same expiry)."""
    event_ticker: str
    series_ticker: str
    close_time: Any
    open_time: Any
    expiration_value: float
    contracts: List[Dict]


class KalshiEventEnv(gym.Env):
    """
    RL environment for trading settled Kalshi binary contracts, grouped by event.

    Each episode replays one event (e.g., one hourly BTC expiry) with N contracts
    at different strike levels. The agent steps through contracts and decides
    whether to BUY_YES, BUY_NO, or HOLD for each one. At the end of the event,
    all positions are settled using the actual outcome.

    This is the environment you train on after running ``kalshi backfill-settled``.

    Observation space (17 dims):
        [0]  yes_price (0-1)
        [1]  spread (0-1)             - bid/ask spread normalized
        [2]  volume_log               - log(1 + volume) normalized
        [3]  open_interest_log        - log(1 + OI) normalized
        [4]  time_to_expiry           - hours until settlement, normalized
        [5]  strike_distance          - how far strike is from underlying
        [6]  strike_direction         - 1=above, -1=below, 0=range
        [7]  implied_prob             - yes_price / 100
        [8]  previous_price_delta     - change from previous close
        [9]  liquidity_log            - log(1 + liquidity) normalized
        [10] contract_index_norm      - position in event's contract list
        [11] capital_ratio            - remaining capital / initial
        [12] num_positions_norm       - current positions / max allowed
        [13] total_exposure_ratio     - total $ at risk / capital
        [14] unrealized_edge          - avg (1 - entry_price/100) for positions
        [15] win_rate                 - rolling win rate
        [16] episode_return           - cumulative return this episode

    Action space: Discrete(3) — 0=HOLD, 1=BUY_YES, 2=BUY_NO
    """

    metadata = {"render_modes": ["human"]}

    ACTION_HOLD = 0
    ACTION_BUY_YES = 1
    ACTION_BUY_NO = 2
    ACTION_NAMES = {0: "HOLD", 1: "BUY_YES", 2: "BUY_NO"}

    def __init__(
        self,
        settled_markets: pd.DataFrame,
        initial_capital: float = 25.0,
        max_positions_per_event: int = 3,
        contracts_per_trade: int = 1,
        transaction_cost_cents: float = 0.0,
        min_volume: int = 0,
        min_contracts_per_event: int = 3,
    ):
        """
        Args:
            settled_markets: DataFrame from ``load_kalshi_settled_markets()``.
            initial_capital: Starting capital in dollars.
            max_positions_per_event: Max contracts to buy per event episode.
            contracts_per_trade: Contracts per BUY action.
            transaction_cost_cents: Fee per contract (cents).
            min_volume: Filter contracts with volume below this.
            min_contracts_per_event: Minimum strike-level contracts to form a valid event.
        """
        super().__init__()

        self.initial_capital = initial_capital
        self.max_positions_per_event = max_positions_per_event
        self.contracts_per_trade = contracts_per_trade
        self.transaction_cost_cents = transaction_cost_cents

        self.events = self._build_events(settled_markets, min_volume, min_contracts_per_event)
        if not self.events:
            raise DataUnavailableError(
                "No valid events built from settled_markets. Run: python main.py kalshi backfill-settled"
            )

        logger.info(
            f"KalshiEventEnv: {len(self.events)} events, "
            f"{sum(len(e.contracts) for e in self.events)} total contracts"
        )

        self.action_space = spaces.Discrete(3)
        self.obs_dim = 17
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # State — will be set in reset()
        self.capital = initial_capital
        self.positions: List[_KalshiPos] = []
        self.current_event_idx = 0
        self.current_contract_idx = 0
        self.episode_positions = 0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.episode_trades: List[float] = []

    # ------------------------------------------------------------------
    # data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_events(
        df: pd.DataFrame, min_volume: int, min_contracts: int
    ) -> List[_EventGroup]:
        if df is None or df.empty:
            return []
        if min_volume > 0:
            df = df[df["volume"] >= min_volume]

        events: List[_EventGroup] = []
        for event_ticker, grp in df.groupby("event_ticker"):
            contracts = grp.to_dict("records")
            if len(contracts) < min_contracts:
                continue
            contracts.sort(key=lambda c: c.get("floor_strike") or c.get("cap_strike") or 0)
            first = contracts[0]
            events.append(_EventGroup(
                event_ticker=str(event_ticker),
                series_ticker=str(first.get("series_ticker", "")),
                close_time=first.get("close_time"),
                open_time=first.get("open_time"),
                expiration_value=first.get("expiration_value") or 0.0,
                contracts=contracts,
            ))
        events.sort(key=lambda e: str(e.close_time or ""))
        return events

    @classmethod
    def from_db(
        cls,
        session: "Session",
        series_tickers: Optional[List[str]] = None,
        **env_kwargs: Any,
    ) -> "KalshiEventEnv":
        """Convenience constructor: load data from DB and create env."""
        df = load_kalshi_settled_markets(session, series_tickers=series_tickers)
        if df is None or df.empty:
            raise DataUnavailableError(
                "No settled Kalshi markets in DB. Run: python main.py kalshi backfill-settled"
            )
        return cls(settled_markets=df, **env_kwargs)

    # ------------------------------------------------------------------
    # gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_event_idx = int(self.np_random.integers(0, len(self.events)))
        self.current_contract_idx = 0
        self.capital = self.initial_capital
        self.positions = []
        self.episode_positions = 0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.episode_trades = []
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        event = self.events[self.current_event_idx]
        contract = event.contracts[self.current_contract_idx]
        reward = 0.0

        if action in (self.ACTION_BUY_YES, self.ACTION_BUY_NO):
            side = "yes" if action == self.ACTION_BUY_YES else "no"
            reward = self._execute_trade(contract, side)

        self.current_contract_idx += 1
        terminated = self.current_contract_idx >= len(event.contracts)
        truncated = False

        if terminated:
            reward += self._settle_all(event)

        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # trade execution & settlement
    # ------------------------------------------------------------------

    def _execute_trade(self, contract: Dict, side: str) -> float:
        if self.episode_positions >= self.max_positions_per_event:
            return -0.01
        if side == "yes":
            entry = contract.get("yes_ask", contract.get("last_price", 50))
        else:
            entry = 100 - contract.get("yes_bid", contract.get("last_price", 50))
        cost = (entry / 100.0) * self.contracts_per_trade
        fee = (self.transaction_cost_cents / 100.0) * self.contracts_per_trade
        if cost + fee > self.capital or entry <= 0 or entry >= 100:
            return -0.01
        self.capital -= cost + fee
        self.positions.append(_KalshiPos(
            ticker=contract["ticker"], side=side, entry_price=entry,
            contracts=self.contracts_per_trade, outcome=contract.get("outcome", 0),
        ))
        self.episode_positions += 1
        self.trade_count += 1
        return -fee / self.initial_capital

    def _settle_all(self, event: _EventGroup) -> float:
        total_reward = 0.0
        for pos in self.positions:
            won = (pos.side == "yes" and pos.outcome == 1) or (pos.side == "no" and pos.outcome == 0)
            payout = 1.0 * pos.contracts if won else 0.0
            cost = (pos.entry_price / 100.0) * pos.contracts
            pnl = payout - cost
            self.total_pnl += pnl
            self.capital += payout
            if pnl > 0:
                self.win_count += 1
            self.episode_trades.append(pnl)
            total_reward += pnl / self.initial_capital
        # Episode-level shaping
        total_reward += (self.total_pnl / self.initial_capital) * 0.5
        return total_reward

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.current_event_idx >= len(self.events):
            return obs
        event = self.events[self.current_event_idx]
        if self.current_contract_idx >= len(event.contracts):
            return obs
        c = event.contracts[self.current_contract_idx]

        yes_price = c.get("last_price", 50)
        yes_bid = c.get("yes_bid", 0)
        yes_ask = c.get("yes_ask", 100)
        spread = max(0, yes_ask - yes_bid)
        volume = c.get("volume", 0)
        oi = c.get("open_interest", 0)
        liquidity = c.get("liquidity", 0)
        prev_price = c.get("previous_price", yes_price)

        exp_val = event.expiration_value or 0
        floor_s = c.get("floor_strike")
        cap_s = c.get("cap_strike")
        st = c.get("strike_type", "")
        if st == "greater" and floor_s:
            strike_ref, strike_dir = floor_s, 1.0
        elif st == "less" and cap_s:
            strike_ref, strike_dir = cap_s, -1.0
        elif floor_s and cap_s:
            strike_ref, strike_dir = (floor_s + cap_s) / 2, 0.0
        else:
            strike_ref, strike_dir = exp_val, 0.0
        strike_dist = (strike_ref - exp_val) / exp_val if exp_val else 0.0

        ct = event.close_time
        ot = event.open_time
        try:
            if hasattr(ct, "timestamp") and hasattr(ot, "timestamp"):
                dur_h = max(0.1, (ct.timestamp() - ot.timestamp()) / 3600)
            else:
                dur_h = 1.0
        except Exception:
            dur_h = 1.0

        obs[0] = np.clip(yes_price / 100.0, 0, 1)
        obs[1] = np.clip(spread / 100.0, 0, 1)
        obs[2] = np.clip(np.log1p(volume) / 10.0, 0, 1)
        obs[3] = np.clip(np.log1p(oi) / 10.0, 0, 1)
        obs[4] = np.clip(dur_h / 24.0, 0, 1)
        obs[5] = np.clip(strike_dist, -2, 2)
        obs[6] = strike_dir
        obs[7] = np.clip(yes_price / 100.0, 0, 1)
        obs[8] = np.clip((yes_price - prev_price) / 100.0, -1, 1)
        obs[9] = np.clip(np.log1p(liquidity) / 15.0, 0, 1)
        obs[10] = self.current_contract_idx / max(1, len(event.contracts) - 1)
        obs[11] = np.clip(self.capital / self.initial_capital, 0, 2)
        obs[12] = self.episode_positions / self.max_positions_per_event
        total_exp = sum((p.entry_price / 100.0) * p.contracts for p in self.positions)
        obs[13] = np.clip(total_exp / self.initial_capital, 0, 1)
        if self.positions:
            avg_edge = np.mean([1.0 - p.entry_price / 100.0 for p in self.positions])
        else:
            avg_edge = 0.0
        obs[14] = np.clip(avg_edge, -1, 1)
        obs[15] = self.win_count / max(1, self.trade_count)
        obs[16] = np.clip(self.total_pnl / self.initial_capital, -1, 1)
        return obs

    def _get_info(self) -> Dict:
        return {
            "capital": self.capital,
            "total_pnl": self.total_pnl,
            "positions": len(self.positions),
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "win_rate": self.win_count / max(1, self.trade_count),
            "episode_return": self.total_pnl / self.initial_capital,
            "event_idx": self.current_event_idx,
            "contract_idx": self.current_contract_idx,
        }

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for MaskablePPO."""
        mask = np.ones(3, dtype=np.bool_)
        can_buy = (
            self.episode_positions < self.max_positions_per_event
            and self.capital > 0.01
        )
        if not can_buy:
            mask[self.ACTION_BUY_YES] = False
            mask[self.ACTION_BUY_NO] = False
        else:
            event = self.events[self.current_event_idx]
            if self.current_contract_idx < len(event.contracts):
                c = event.contracts[self.current_contract_idx]
                ya = c.get("yes_ask", 0)
                yb = c.get("yes_bid", 0)
                if ya <= 0 or ya >= 100:
                    mask[self.ACTION_BUY_YES] = False
                if yb <= 0 or yb >= 100:
                    mask[self.ACTION_BUY_NO] = False
                if (ya / 100.0) * self.contracts_per_trade > self.capital:
                    mask[self.ACTION_BUY_YES] = False
                if ((100 - yb) / 100.0) * self.contracts_per_trade > self.capital:
                    mask[self.ACTION_BUY_NO] = False
        return mask

    def render(self, mode="human"):
        event = self.events[self.current_event_idx]
        print(f"Event: {event.event_ticker} | Contract {self.current_contract_idx}/{len(event.contracts)}")
        print(f"Capital: ${self.capital:.2f} | PnL: ${self.total_pnl:.2f} | Positions: {len(self.positions)}")
        if self.current_contract_idx < len(event.contracts):
            c = event.contracts[self.current_contract_idx]
            print(f"  {c.get('subtitle', c.get('ticker', ''))}")
            print(f"  YES: bid={c.get('yes_bid',0)} ask={c.get('yes_ask',0)} | Result: {c.get('result','?')}")


# ---------------------------------------------------------------------------
# DB loaders
# ---------------------------------------------------------------------------

def load_kalshi_settled_markets(
    session: "Session",
    series_tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load settled market snapshots from ``kalshi_settled_markets`` table."""
    from ..data.database import KalshiSettledMarket

    q = session.query(KalshiSettledMarket)
    if series_tickers:
        q = q.filter(KalshiSettledMarket.series_ticker.in_(series_tickers))
    rows = q.all()
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        records.append({
            "ticker": r.ticker, "event_ticker": r.event_ticker,
            "series_ticker": r.series_ticker, "title": r.title,
            "subtitle": r.subtitle, "strike_type": r.strike_type,
            "floor_strike": r.floor_strike, "cap_strike": r.cap_strike,
            "result": r.result, "outcome": r.outcome,
            "expiration_value": r.expiration_value,
            "settlement_value": r.settlement_value,
            "last_price": r.last_price, "yes_bid": r.yes_bid,
            "yes_ask": r.yes_ask, "no_bid": r.no_bid, "no_ask": r.no_ask,
            "previous_price": r.previous_price, "volume": r.volume,
            "open_interest": r.open_interest, "liquidity": r.liquidity,
            "open_time": r.open_time, "close_time": r.close_time,
            "settlement_ts": r.settlement_ts, "created_time": r.created_time,
            "category": r.category,
        })
    return pd.DataFrame(records)
