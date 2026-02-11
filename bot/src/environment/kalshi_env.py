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
