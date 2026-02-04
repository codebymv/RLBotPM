"""
Custom Gymnasium Environment for Crypto Spot Trading (Real Data Only)

This environment consumes real OHLCV candles from the database.
No synthetic data is allowed. If data is unavailable, it raises
DataUnavailableError and training must stop.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, List

from ..core.logger import get_logger
from ..core.config import get_settings
from ..data.sources.base import DataUnavailableError


logger = get_logger(__name__)


class CryptoTradingEnv(gym.Env):
    """
    Crypto spot trading environment backed by real OHLCV data.
    """

    metadata = {"render_modes": ["human"]}

    # Action space constants
    ACTION_NO_ACTION = 0
    ACTION_BUY_SMALL = 1
    ACTION_BUY_MEDIUM = 2
    ACTION_BUY_LARGE = 3
    ACTION_SELL_SMALL = 4
    ACTION_SELL_MEDIUM = 5
    ACTION_SELL_LARGE = 6
    ACTION_CLOSE_ALL = 7

    ACTION_NAMES = {
        0: "NO_ACTION",
        1: "BUY_SMALL",
        2: "BUY_MEDIUM",
        3: "BUY_LARGE",
        4: "SELL_SMALL",
        5: "SELL_MEDIUM",
        6: "SELL_LARGE",
        7: "CLOSE_ALL",
    }

    def __init__(
        self,
        dataset: pd.DataFrame,
        interval: str = "1h",
        initial_capital: Optional[float] = None,
        max_steps: int = 500,
        transaction_cost: Optional[float] = None,
    ):
        """
        Initialize environment with a real OHLCV dataset.

        Args:
            dataset: DataFrame with columns:
                symbol, timestamp, open, high, low, close, volume
            interval: Candle interval (e.g., 1m, 5m, 1h, 1d)
            initial_capital: Starting capital
            max_steps: Maximum steps per episode
            transaction_cost: Transaction cost percentage
        """
        super().__init__()

        self.settings = get_settings()
        self.initial_capital = initial_capital or self.settings.INITIAL_CAPITAL
        self.transaction_cost = transaction_cost or self.settings.TRANSACTION_COST_PCT
        self.max_steps = max_steps
        self.interval = interval

        if dataset is None or dataset.empty:
            raise DataUnavailableError("Dataset is empty. Real OHLCV data required.")

        required_cols = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(dataset.columns)):
            missing = required_cols - set(dataset.columns)
            raise DataUnavailableError(f"Dataset missing required columns: {missing}")

        self.dataset = dataset.copy()
        self._prepare_data()

        # Action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)

        # Observation space: 24-dimensional continuous state
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(24,),
            dtype=np.float32,
        )

        # Portfolio state
        self.capital = self.initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []

        # Performance tracking
        self.episode_returns: List[float] = []
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0

        # Episode state
        self.current_step = 0
        self.current_symbol: Optional[str] = None
        self.current_index = 0
        self.episode_data: Optional[pd.DataFrame] = None
        self.terminated = False

        # Historical performance
        self.recent_trades: List[float] = []
        self.win_count = 0
        self.trade_count = 0

        logger.info("CryptoTradingEnv initialized with real data")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Reset portfolio
        self.capital = self.initial_capital
        self.positions = {}
        self.trade_history = []

        # Reset performance tracking
        self.episode_returns = []
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0

        # Reset episode state
        self.current_step = 0
        self.terminated = False

        # Select symbol and window
        self._select_episode_window()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.terminated:
            return self._get_observation(), 0.0, True, False, self._get_info()

        trade_result = self._execute_action(action)
        reward = self._calculate_reward(trade_result)

        self.current_step += 1
        self.current_index += 1

        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()
        info["trade_result"] = trade_result

        if terminated or truncated:
            self.terminated = True

        return observation, reward, terminated, truncated, info

    def _prepare_data(self) -> None:
        """Precompute rolling features from real OHLCV data."""
        df = self.dataset
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        interval_steps = _interval_to_steps(self.interval)

        def compute_features(group: pd.DataFrame) -> pd.DataFrame:
            g = group.copy()
            g["return_1h"] = g["close"].pct_change(periods=interval_steps["1h"])
            g["return_6h"] = g["close"].pct_change(periods=interval_steps["6h"])
            g["return_24h"] = g["close"].pct_change(periods=interval_steps["24h"])
            g["volatility_24h"] = g["close"].pct_change().rolling(
                interval_steps["24h"]
            ).std()
            g["ma_6"] = g["close"].rolling(interval_steps["6h"]).mean()
            g["ma_24"] = g["close"].rolling(interval_steps["24h"]).mean()
            g["ma_48"] = g["close"].rolling(interval_steps["48h"]).mean()
            g["volume_ma_24"] = g["volume"].rolling(interval_steps["24h"]).mean()
            g["hl_spread_pct"] = (g["high"] - g["low"]) / g["close"]
            g["trend_direction"] = np.sign(g["return_24h"])
            return g

        df = df.groupby("symbol", group_keys=False).apply(compute_features)
        df = df.dropna().reset_index(drop=True)

        if df.empty:
            raise DataUnavailableError("Dataset has insufficient history after feature prep.")

        self.dataset = df
        self.data_by_symbol = {
            symbol: group.reset_index(drop=True)
            for symbol, group in df.groupby("symbol")
        }

    def _select_episode_window(self) -> None:
        """Select a real symbol and time window for the episode."""
        valid_symbols = [
            symbol
            for symbol, data in self.data_by_symbol.items()
            if len(data) > self.max_steps + 1
        ]
        if not valid_symbols:
            raise DataUnavailableError("No symbols with sufficient data for episode.")

        self.current_symbol = np.random.choice(valid_symbols)
        data = self.data_by_symbol[self.current_symbol]

        start_index = np.random.randint(0, len(data) - self.max_steps - 1)
        self.episode_data = data.iloc[start_index : start_index + self.max_steps + 1]
        self.current_index = 0

    def _current_row(self) -> pd.Series:
        if self.episode_data is None:
            raise DataUnavailableError("Episode data not initialized.")
        return self.episode_data.iloc[self.current_index]

    def _execute_action(self, action: int) -> Dict:
        action_name = self.ACTION_NAMES.get(action, "UNKNOWN")
        result = {
            "action": action,
            "action_name": action_name,
            "executed": False,
            "reason": None,
            "pnl": 0.0,
            "cost": 0.0,
        }

        if action == self.ACTION_NO_ACTION:
            result["reason"] = "No action taken"
            return result

        if action in [self.ACTION_BUY_SMALL, self.ACTION_BUY_MEDIUM, self.ACTION_BUY_LARGE]:
            size_pct = {
                self.ACTION_BUY_SMALL: 0.05,
                self.ACTION_BUY_MEDIUM: 0.10,
                self.ACTION_BUY_LARGE: 0.20,
            }[action]
            return self._execute_buy(size_pct)

        if action in [self.ACTION_SELL_SMALL, self.ACTION_SELL_MEDIUM, self.ACTION_SELL_LARGE, self.ACTION_CLOSE_ALL]:
            close_pct = {
                self.ACTION_SELL_SMALL: 0.33,
                self.ACTION_SELL_MEDIUM: 0.66,
                self.ACTION_SELL_LARGE: 1.0,
                self.ACTION_CLOSE_ALL: 1.0,
            }[action]
            return self._execute_sell(close_pct)

        return result

    def _execute_buy(self, size_pct: float) -> Dict:
        row = self._current_row()
        current_price = float(row["close"])
        symbol = self.current_symbol

        position_value = self.capital * size_pct
        cost = position_value * (1 + self.transaction_cost)
        if cost > self.capital:
            return {
                "action_name": f"BUY_{int(size_pct*100)}%",
                "executed": False,
                "reason": "Insufficient capital",
                "pnl": 0.0,
                "cost": 0.0,
            }

        if len(self.positions) >= self.settings.MAX_OPEN_POSITIONS:
            return {
                "action_name": f"BUY_{int(size_pct*100)}%",
                "executed": False,
                "reason": "Max positions reached",
                "pnl": 0.0,
                "cost": 0.0,
            }

        self.capital -= cost
        transaction_cost_paid = position_value * self.transaction_cost

        self.positions[symbol] = {
            "size": position_value,
            "entry_price": current_price,
            "timestamp": self.current_step,
        }

        self.trade_count += 1

        return {
            "action_name": f"BUY_{int(size_pct*100)}%",
            "executed": True,
            "reason": "Buy executed",
            "size": position_value,
            "price": current_price,
            "cost": transaction_cost_paid,
            "pnl": 0.0,
        }

    def _execute_sell(self, close_pct: float) -> Dict:
        symbol = self.current_symbol
        if symbol not in self.positions:
            return {
                "action_name": f"SELL_{int(close_pct*100)}%",
                "executed": False,
                "reason": "No position to close",
                "pnl": 0.0,
                "cost": 0.0,
            }

        row = self._current_row()
        current_price = float(row["close"])
        position = self.positions[symbol]

        sell_size = position["size"] * close_pct
        price_change = (current_price - position["entry_price"]) / position["entry_price"]
        pnl_before_costs = sell_size * price_change
        transaction_cost_paid = sell_size * self.transaction_cost
        pnl = pnl_before_costs - transaction_cost_paid

        self.capital += sell_size + pnl

        if close_pct >= 0.99:
            del self.positions[symbol]
        else:
            position["size"] *= (1 - close_pct)

        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 10:
            self.recent_trades.pop(0)
        if pnl > 0:
            self.win_count += 1

        return {
            "action_name": f"SELL_{int(close_pct*100)}%",
            "executed": True,
            "reason": "Sell executed",
            "size": sell_size,
            "price": current_price,
            "pnl": pnl,
            "cost": transaction_cost_paid,
        }

    def _calculate_reward(self, trade_result: Dict) -> float:
        reward = 0.0

        current_portfolio_value = self._get_portfolio_value()
        returns = (current_portfolio_value - self.initial_capital) / self.initial_capital
        reward += returns * 10.0

        if trade_result["executed"]:
            cost_penalty = trade_result["cost"] / self.initial_capital
            reward -= cost_penalty * 100.0

        self.peak_capital = max(self.peak_capital, current_portfolio_value)
        self.current_drawdown = (self.peak_capital - current_portfolio_value) / self.peak_capital
        if self.current_drawdown > 0.20:
            drawdown_penalty = np.exp(self.current_drawdown * 5) - 1
            reward -= drawdown_penalty * 10.0

        if trade_result.get("pnl", 0) > 0:
            reward += 0.5

        if len(self.positions) > self.settings.MAX_OPEN_POSITIONS:
            reward -= 1.0

        if len(self.recent_trades) >= 5:
            avg_recent_pnl = np.mean(self.recent_trades)
            reward += (avg_recent_pnl / self.initial_capital) * 3.0

        return float(reward)

    def _check_termination(self) -> bool:
        if self.capital < self.initial_capital * 0.1:
            return True
        if self.current_drawdown > self.settings.MAX_TOTAL_DRAWDOWN:
            return True
        return False

    def _get_observation(self) -> np.ndarray:
        row = self._current_row()

        obs = np.zeros(24, dtype=np.float32)

        obs[0] = float(row["close"])
        obs[1] = float(row["hl_spread_pct"])
        obs[2] = np.log1p(float(row["volume"])) / 10.0
        obs[3] = float(row["return_1h"])
        obs[4] = float(row["return_6h"])
        obs[5] = float(row["return_24h"])
        obs[6] = float(row["volatility_24h"])
        obs[7] = float(row["ma_6"] / row["close"]) - 1.0
        obs[8] = float(row["ma_24"] / row["close"]) - 1.0
        obs[9] = float(row["ma_48"] / row["close"]) - 1.0
        obs[10] = float(row["volume_ma_24"]) / max(float(row["volume"]), 1.0)
        obs[11] = float(row["trend_direction"])

        portfolio_value = self._get_portfolio_value()
        obs[12] = self.capital / self.initial_capital
        obs[13] = len(self.positions) / self.settings.MAX_OPEN_POSITIONS
        obs[14] = portfolio_value / self.initial_capital
        obs[15] = (portfolio_value - self.initial_capital) / self.initial_capital
        obs[16] = sum(self.recent_trades) / self.initial_capital if self.recent_trades else 0.0
        obs[17] = self.positions.get(self.current_symbol, {}).get("size", 0.0) / self.initial_capital
        obs[18] = self.win_count / max(self.trade_count, 1)
        obs[19] = self.current_drawdown

        ts = pd.to_datetime(row["timestamp"])
        obs[20] = np.sin(2 * np.pi * ts.hour / 24)
        obs[21] = np.cos(2 * np.pi * ts.hour / 24)
        obs[22] = np.sin(2 * np.pi * ts.weekday() / 7)
        obs[23] = np.cos(2 * np.pi * ts.weekday() / 7)

        return obs

    def _get_info(self) -> Dict:
        portfolio_value = self._get_portfolio_value()
        return {
            "step": self.current_step,
            "capital": self.capital,
            "portfolio_value": portfolio_value,
            "num_positions": len(self.positions),
            "total_trades": self.trade_count,
            "win_rate": self.win_count / max(self.trade_count, 1),
            "drawdown": self.current_drawdown,
            "current_symbol": self.current_symbol,
        }

    def _get_portfolio_value(self) -> float:
        total = self.capital
        row = self._current_row()
        current_price = float(row["close"])

        for symbol, position in self.positions.items():
            entry_price = position["entry_price"]
            size = position["size"]
            if entry_price > 0:
                total += size * (current_price / entry_price)
            else:
                total += size

        return total

    def render(self):
        row = self._current_row()
        portfolio_value = self._get_portfolio_value()
        returns = (portfolio_value - self.initial_capital) / self.initial_capital

        print(f"\nStep: {self.current_step}/{self.max_steps}")
        print(f"Symbol: {self.current_symbol}")
        print(f"Price: {row['close']:.4f}")
        print(f"Capital: ${self.capital:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f} ({returns:+.2%})")
        print(f"Open Positions: {len(self.positions)}/{self.settings.MAX_OPEN_POSITIONS}")
        print(f"Drawdown: {self.current_drawdown:.2%}")

    def close(self):
        pass


def _interval_to_steps(interval: str) -> Dict[str, int]:
    mapping_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    if interval not in mapping_minutes:
        raise DataUnavailableError(f"Unsupported interval: {interval}")

    minutes = mapping_minutes[interval]
    steps_per_hour = max(1, int(60 / minutes))

    return {
        "1h": steps_per_hour,
        "6h": steps_per_hour * 6,
        "24h": steps_per_hour * 24,
        "48h": steps_per_hour * 48,
    }
