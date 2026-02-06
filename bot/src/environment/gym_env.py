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
from pathlib import Path
import yaml

from ..risk.position_sizer import PositionSizer
from ..risk.adaptive_breaker import AdaptiveCircuitBreaker

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
    ACTION_BUY = 1
    ACTION_SELL = 2

    MIN_POSITION_VALUE_FRACTION = 0.01

    ACTION_NAMES = {
        0: "NO_ACTION",
        1: "BUY",
        2: "SELL",
    }

    def __init__(
        self,
        dataset: pd.DataFrame,
        interval: str = "1h",
        initial_capital: Optional[float] = None,
        max_steps: int = 500,
        transaction_cost: Optional[float] = None,
        sequence_length: int = 1,
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
        self.sequence_length = max(1, int(sequence_length))
        self.reward_config = self._load_reward_config()
        self.risk_config = self._load_risk_config()
        self.min_hold_steps = int(self.reward_config.get("min_hold_steps", 5))
        self.max_hold_steps = int(self.reward_config.get("max_hold_steps", 100))
        self.trade_cooldown_steps = int(self.reward_config.get("trade_cooldown_steps", 3))
        self.auto_exit_enabled = bool(self.reward_config.get("auto_exit_enabled", True))
        self.entry_signal_threshold = float(self.reward_config.get("entry_signal_threshold", 0.0))
        self.entry_volatility_cap = float(self.reward_config.get("entry_volatility_cap", 0.0))
        self.entry_momentum_ratio_threshold = float(
            self.reward_config.get("entry_momentum_ratio_threshold", 0.0)
        )

        position_limits = self.risk_config.get("position_limits", {})
        min_position_pct = float(position_limits.get("min_position_size_pct", self.MIN_POSITION_VALUE_FRACTION))
        max_position_pct = float(position_limits.get("max_position_size_pct", 0.20))
        max_position_value = self.initial_capital * max_position_pct
        self.min_position_value = min(self.initial_capital * min_position_pct, max_position_value)
        self.max_position_size_pct = max_position_pct

        tx_costs = self.risk_config.get("transaction_costs", {})
        self.base_transaction_cost = float(tx_costs.get("base_cost_pct", self.transaction_cost))
        self.large_order_penalty_threshold = float(tx_costs.get("large_order_penalty_threshold", 0.15))
        self.large_order_penalty_pct = float(tx_costs.get("large_order_penalty_pct", 0.0))
        self.slippage_model = str(tx_costs.get("slippage_model", "fixed"))
        self.max_slippage_pct = float(self.risk_config.get("market_requirements", {}).get("max_slippage_pct", 0.02))

        market_req = self.risk_config.get("market_requirements", {})
        self.min_volume_24h = float(market_req.get("min_volume_24h", 0.0))
        self.min_market_liquidity = float(market_req.get("min_market_liquidity", 0.0))

        sizing_config = self.risk_config.get("position_sizing", {})
        kelly_fraction = float(sizing_config.get("kelly_fraction", 0.25))
        self.default_position_size_pct = float(sizing_config.get("default_position_size_pct", 0.05))
        self.min_trades_for_kelly = int(sizing_config.get("min_trades_for_kelly", 10))
        self.base_kelly_fraction = kelly_fraction
        self.position_sizer = PositionSizer(kelly_fraction=kelly_fraction)
        self.adaptive_breaker = AdaptiveCircuitBreaker()

        if dataset is None or dataset.empty:
            raise DataUnavailableError("Dataset is empty. Real OHLCV data required.")

        required_cols = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(dataset.columns)):
            missing = required_cols - set(dataset.columns)
            raise DataUnavailableError(f"Dataset missing required columns: {missing}")

        self.dataset = dataset.copy()
        self._prepare_data()

        # Action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)

        # Observation space: 42-dimensional continuous state
        # (28 original + 8 technical indicators + 3 position-level features + 3 trend context)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(42,),  # Run 48: Expanded from 39 to include long-term trend context
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
        self.prev_portfolio_value = self.initial_capital

        # Episode state
        self.current_step = 0
        self.current_symbol: Optional[str] = None
        self.current_index = 0
        self.episode_data: Optional[pd.DataFrame] = None
        self.terminated = False
        self.position_hold_steps = 0
        self.flat_steps = 0
        self.steps_since_last_close = self.trade_cooldown_steps  # Start ready to trade

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
        self.recent_trades = []
        self.win_count = 0
        self.trade_count = 0
        self.prev_portfolio_value = self.initial_capital

        # Reset episode state
        self.current_step = 0
        self.terminated = False
        self.position_hold_steps = 0
        self.flat_steps = 0
        self.steps_since_last_close = self.trade_cooldown_steps  # Start ready to trade

        # Select symbol and window
        self._select_episode_window()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.terminated:
            return self._get_observation(), 0.0, True, False, self._get_info()

        forced_exit = None
        if self.positions and self.auto_exit_enabled:
            row = self._current_row()
            current_price = float(row["close"])
            position = self.positions.get(self.current_symbol)
            if position:
                price_change = (current_price - position["entry_price"]) / position["entry_price"]
                if price_change <= -self.reward_config.get("stop_loss_pct", 0.01):
                    forced_exit = "AUTO_STOP_LOSS"
                elif price_change >= self.reward_config.get("take_profit_pct", 0.02):
                    forced_exit = "AUTO_TAKE_PROFIT"
                elif self.position_hold_steps >= self.max_hold_steps:
                    forced_exit = "AUTO_MAX_HOLD"

        if forced_exit:
            action = self.ACTION_SELL

        invalid_action = False
        hold_steps_before_action = self.position_hold_steps
        if self.positions:
            if action == self.ACTION_BUY:
                invalid_action = True
            elif self.position_hold_steps < self.min_hold_steps and action == self.ACTION_SELL:
                invalid_action = True
        else:
            if action == self.ACTION_SELL:
                invalid_action = True

        if invalid_action:
            trade_result = {
                "action": action,
                "action_name": "INVALID_ACTION",
                "executed": False,
                "reason": "Invalid action",
                "pnl": 0.0,
                "cost": 0.0,
            }
        else:
            trade_result = self._execute_action(action)

        trade_result["hold_steps"] = hold_steps_before_action
        if forced_exit:
            trade_result["reason"] = forced_exit

        if not self.positions and trade_result.get("reason") in {"No action taken", "Invalid action"}:
            self.flat_steps += 1
        elif trade_result.get("executed") or self.positions:
            self.flat_steps = 0

        current_portfolio_value = self._get_portfolio_value()
        reward = self._calculate_reward(trade_result, current_portfolio_value)
        self.prev_portfolio_value = current_portfolio_value

        self.current_step += 1
        self.current_index += 1

        # CRITICAL FIX: Increment position hold steps to enable manual exit learning
        if self.positions:
            self.position_hold_steps += 1

        # Increment cooldown counter when flat
        if not self.positions:
            self.steps_since_last_close += 1

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
            
            # Price-based features
            g["return_1h"] = g["close"].pct_change(periods=interval_steps["1h"])
            g["return_6h"] = g["close"].pct_change(periods=interval_steps["6h"])
            g["return_24h"] = g["close"].pct_change(periods=interval_steps["24h"])
            g["volatility_24h"] = g["close"].pct_change().rolling(
                interval_steps["24h"]
            ).std()
            g["ma_6"] = g["close"].rolling(interval_steps["6h"]).mean()
            g["ma_24"] = g["close"].rolling(interval_steps["24h"]).mean()
            g["ma_48"] = g["close"].rolling(interval_steps["48h"]).mean()
            g["ma_200"] = g["close"].rolling(interval_steps["200h"]).mean()  # Run 48: Long-term trend
            g["ma_200_slope"] = g["ma_200"].pct_change(periods=interval_steps["24h"])  # Trend direction
            g["price_std_24h"] = g["close"].rolling(interval_steps["24h"]).std()
            g["price_zscore_24h"] = (g["close"] - g["ma_24"]) / (g["price_std_24h"] + 1e-9)
            g["volume_ma_24"] = g["volume"].rolling(interval_steps["24h"]).mean()
            g["volume_ma_200"] = g["volume"].rolling(interval_steps["200h"]).mean()  # Long-term volume baseline
            g["hl_spread_pct"] = (g["high"] - g["low"]) / g["close"]
            g["trend_direction"] = np.sign(g["return_24h"])
            
            # Technical indicators
            g["rsi_14"] = self._calculate_rsi(g["close"], period=14)
            macd_data = self._calculate_macd(g["close"])
            g["macd_line"] = macd_data["macd"]
            g["macd_signal"] = macd_data["signal"]
            g["macd_hist"] = macd_data["histogram"]
            g["atr_14"] = self._calculate_atr(g["high"], g["low"], g["close"], period=14)
            bb_data = self._calculate_bollinger_bands(g["close"], period=20, std_dev=2.0)
            g["bb_upper"] = bb_data["upper"]
            g["bb_middle"] = bb_data["middle"]
            g["bb_lower"] = bb_data["lower"]
            
            return g

        df = df.groupby("symbol", group_keys=True).apply(compute_features, include_groups=False)
        df = df.reset_index(level=0)  # Move 'symbol' from index to column
        df = df.dropna().reset_index(drop=True)

        if df.empty:
            raise DataUnavailableError("Dataset has insufficient history after feature prep.")

        self.dataset = df
        self.data_by_symbol = {
            symbol: group.reset_index(drop=True)
            for symbol, group in df.groupby("symbol")
        }

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        # Normalize to [0, 1] range
        return rsi / 100.0

    @staticmethod
    def _calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Normalize by price to make scale-invariant
        price_ref = close + 1e-9
        return {
            "macd": macd_line / price_ref,
            "signal": signal_line / price_ref,
            "histogram": histogram / price_ref,
        }

    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        # Normalize by price
        return atr / (close + 1e-9)

    @staticmethod
    def _calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        # Normalize by middle band
        middle_ref = middle + 1e-9
        return {
            "upper": upper / middle_ref,
            "middle": middle / middle_ref,
            "lower": lower / middle_ref,
        }

    def _select_episode_window(self) -> None:
        """Select a real symbol and time window for the episode."""
        max_available = max(len(data) for data in self.data_by_symbol.values())
        if max_available <= 2:
            raise DataUnavailableError("No symbols with sufficient data for episode.")

        required = self.max_steps + self.sequence_length + 1
        if max_available <= required:
            self.max_steps = max_available - self.sequence_length - 1
            logger.warning(
                "Insufficient history for requested max_steps; "
                f"reducing max_steps to {self.max_steps}."
            )
            if self.max_steps <= 0:
                raise DataUnavailableError("Insufficient history for requested sequence length.")

        valid_symbols = [
            symbol
            for symbol, data in self.data_by_symbol.items()
            if len(data) > self.max_steps + self.sequence_length + 1
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

        if action == self.ACTION_BUY:
            return self._execute_buy(1.0, action)

        if action == self.ACTION_SELL:
            return self._execute_sell(1.0, action)

        return result

    def get_valid_actions(self) -> List[int]:
        """Return valid actions given current position state."""
        if self.positions:
            if self.position_hold_steps < self.min_hold_steps:
                return [self.ACTION_NO_ACTION]
            return [
                self.ACTION_NO_ACTION,
                self.ACTION_SELL,
            ]

        # Trade cooldown: prevent immediate re-entry after closing
        if self.steps_since_last_close < self.trade_cooldown_steps:
            return [self.ACTION_NO_ACTION]

        row = self._current_row()
        if self.entry_signal_threshold > 0:
            trend_strength = abs(float(row["return_24h"]))
            if trend_strength < self.entry_signal_threshold:
                return [self.ACTION_NO_ACTION]

        if self.entry_volatility_cap > 0:
            volatility = abs(float(row["volatility_24h"]))
            if volatility > self.entry_volatility_cap:
                return [self.ACTION_NO_ACTION]

        if self.entry_momentum_ratio_threshold > 0:
            volatility = abs(float(row["volatility_24h"]))
            momentum_ratio = float(row["return_6h"]) / (volatility + 1e-6)
            if momentum_ratio < self.entry_momentum_ratio_threshold:
                return [self.ACTION_NO_ACTION]

        if self.min_volume_24h > 0:
            volume_ma_24 = float(row["volume_ma_24"])
            if volume_ma_24 < self.min_volume_24h:
                return [self.ACTION_NO_ACTION]

        if self.min_market_liquidity > 0:
            current_price = float(row["close"])
            volume_value = float(row["volume_ma_24"]) * current_price
            if volume_value < self.min_market_liquidity:
                return [self.ACTION_NO_ACTION]

        return [
            self.ACTION_NO_ACTION,
            self.ACTION_BUY,
        ]

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions for MaskablePPO."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action in self.get_valid_actions():
            mask[action] = True
        return mask

    def _load_reward_config(self) -> Dict[str, float]:
        default_config = {
            "base_penalty": 0.02,
            "portfolio_step_scale": 10.0,
            "step_return_scale": 10.0,
            "transaction_cost_penalty": 100.0,
            "blocked_action_penalty": 0.5,
            "invalid_action_penalty": 0.0,
            "idle_base": 0.01,
            "idle_step": 0.0,
            "idle_cap": 0.01,
            "idle_trend_threshold": 0.0,
            "idle_trend_scale": 0.0,
            "idle_trend_cap": 0.0,
            "idle_extra_after_steps": 0,
            "idle_extra_penalty": 0.0,
            "opportunity_scale": 0.0,
            "opportunity_cap": 0.0,
            "drawdown_threshold": 0.20,
            "drawdown_exp": 5.0,
            "drawdown_penalty_scale": 10.0,
            "sell_pnl_scale": 8.0,
            "sell_profit_bonus": 0.5,
            "sell_profit_bonus_scale": 10.0,
            "sell_loss_bonus": 0.1,
            "sell_trend_threshold": -0.003,
            "sell_trend_scale": 5.0,
            "sell_trend_cap": 0.3,
            "loss_aversion_scale": 2.0,
            "pnl_cost_ratio_scale": 0.0,
            "pnl_cost_ratio_cap": 5.0,
            "buy_bonus": 0.0,
            "buy_trend_threshold": 0.0,
            "buy_trend_scale": 0.0,
            "buy_trend_cap": 0.0,
            "buy_trend_penalty_scale": 0.0,
            "buy_trend_penalty_cap": 0.0,
            "signal_alignment_scale": 0.0,
            "signal_misalignment_scale": 0.0,
            "signal_alignment_cap": 0.5,
            "hold_step_penalty": 0.005,
            "hold_penalty_cap": 0.2,
            "max_positions_penalty": 1.0,
            "recent_trade_scale": 3.0,
            "min_hold_steps": 5,
            "trade_cooldown_steps": 3,
            "fee_penalty_scale": 3.0,
            "early_close_penalty": 0.0,
            "carry_bonus_scale": 5.0,
            "carry_bonus_cap": 0.5,
            "hold_pnl_step_scale": 5.0,
            "auto_exit_penalty": 0.0,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.05,
            "stop_loss_penalty": 1.0,
            "take_profit_bonus": 0.5,
            "unrealized_drawdown_scale": 5.0,
            "unrealized_drawdown_cap": 0.5,
            "auto_exit_enabled": True,
            "max_hold_steps": 100,
            "entry_signal_threshold": 0.0,
            "entry_volatility_cap": 0.0,
            "entry_momentum_ratio_threshold": 0.0,
            "sharpe_bonus_scale": 2.0,
            "sharpe_bonus_cap": 2.0,
            "manual_exit_bonus": 0.1,
        }

        reward_config_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "reward_config.yaml"
        config_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "model_config.yaml"
        try:
            if reward_config_path.exists():
                with reward_config_path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
            else:
                with config_path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            logger.warning("Reward config not found at %s; using defaults.", reward_config_path)
            data = {}

        reward_overrides = data.get("reward_weights") or data.get("environment", {}).get("reward_weights", {}) or {}
        merged = default_config.copy()
        for key, value in reward_overrides.items():
            if value is not None:
                merged[key] = value
        merged["_version"] = str(data.get("version", "legacy"))
        return merged

    def _load_risk_config(self) -> Dict[str, Dict]:
        config_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "risk_config.yaml"
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError:
            logger.warning("Risk config not found at %s; using defaults.", config_path)
            return {}

    def _estimate_slippage_pct(self, order_value: float, row: pd.Series) -> float:
        if self.slippage_model == "volume_based":
            volume_ma_24 = float(row.get("volume_ma_24", 0.0))
            current_price = float(row.get("close", 0.0))
            volume_value = max(volume_ma_24 * current_price, 1e-9)
            raw_slippage = (order_value / volume_value) * self.max_slippage_pct
        else:
            raw_slippage = self.max_slippage_pct * 0.5

        return raw_slippage

    def _get_trade_cost_pct(self, order_value: float, portfolio_value: float, row: pd.Series) -> float:
        cost_pct = self.base_transaction_cost
        if portfolio_value > 0:
            order_pct = order_value / portfolio_value
            if order_pct > self.large_order_penalty_threshold:
                cost_pct += self.large_order_penalty_pct

        slippage_pct = min(self.max_slippage_pct, self._estimate_slippage_pct(order_value, row))
        cost_pct += slippage_pct
        return cost_pct

    def _execute_buy(self, size_pct: float, action: int) -> Dict:
        row = self._current_row()
        current_price = float(row["close"])
        symbol = self.current_symbol
        action_name = self.ACTION_NAMES.get(action, "BUY")

        if self.positions:
            return {
                "action": action,
                "action_name": action_name,
                "executed": False,
                "reason": "Position already open",
                "pnl": 0.0,
                "cost": 0.0,
            }

        if len(self.recent_trades) >= self.min_trades_for_kelly:
            sizing = self.position_sizer.calculate_from_recent_trades(
                capital=self.capital,
                recent_trades=self.recent_trades,
            )
            base_size = sizing.get("suggested_size", self.capital * self.default_position_size_pct)
        else:
            base_size = self.capital * self.default_position_size_pct

        base_size = max(base_size, self.min_position_value)
        position_value = min(base_size * size_pct, self.capital * self.max_position_size_pct)
        position_value = max(position_value, self.min_position_value)

        raw_slippage = self._estimate_slippage_pct(position_value, row)
        if raw_slippage > self.max_slippage_pct:
            return {
                "action": action,
                "action_name": action_name,
                "executed": False,
                "reason": "Slippage too high",
                "pnl": 0.0,
                "cost": 0.0,
            }

        cost_pct = self._get_trade_cost_pct(position_value, self._get_portfolio_value(), row)
        cost = position_value * (1 + cost_pct)
        if cost > self.capital:
            return {
                "action": action,
                "action_name": action_name,
                "executed": False,
                "reason": "Insufficient capital",
                "pnl": 0.0,
                "cost": 0.0,
            }

        if len(self.positions) >= self.settings.MAX_OPEN_POSITIONS:
            return {
                "action": action,
                "action_name": action_name,
                "executed": False,
                "reason": "Max positions reached",
                "pnl": 0.0,
                "cost": 0.0,
            }

        self.capital -= cost
        transaction_cost_paid = position_value * cost_pct

        self.positions[symbol] = {
            "size": position_value,
            "entry_price": current_price,
            "timestamp": self.current_step,
        }
        self.position_hold_steps = 0

        return {
            "action": action,
            "action_name": action_name,
            "executed": True,
            "reason": "Buy executed",
            "size": position_value,
            "price": current_price,
            "side": "buy",
            "cost": transaction_cost_paid,
            "pnl": 0.0,
        }

    def _execute_sell(self, close_pct: float, action: int) -> Dict:
        symbol = self.current_symbol
        action_name = self.ACTION_NAMES.get(action, "SELL")
        if symbol not in self.positions:
            return {
                "action": action,
                "action_name": action_name,
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
        raw_slippage = self._estimate_slippage_pct(sell_size, row)
        if raw_slippage > self.max_slippage_pct:
            return {
                "action": action,
                "action_name": action_name,
                "executed": False,
                "reason": "Slippage too high",
                "pnl": 0.0,
                "cost": 0.0,
            }

        cost_pct = self._get_trade_cost_pct(sell_size, self._get_portfolio_value(), row)
        transaction_cost_paid = sell_size * cost_pct
        pnl = pnl_before_costs - transaction_cost_paid

        self.capital += sell_size + pnl

        if close_pct >= 0.99:
            del self.positions[symbol]
            self.position_hold_steps = 0
            self.steps_since_last_close = 0  # Start cooldown
        else:
            position["size"] *= (1 - close_pct)
            if position["size"] < self.min_position_value:
                del self.positions[symbol]
                self.position_hold_steps = 0
                self.steps_since_last_close = 0  # Start cooldown

        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
        if pnl > 0:
            self.win_count += 1
        self.trade_count += 1
        self._update_adaptive_kelly(pnl)

        return {
            "action": action,
            "action_name": action_name,
            "executed": True,
            "reason": "Sell executed",
            "size": sell_size,
            "price": current_price,
            "side": "sell",
            "pnl": pnl,
            "cost": transaction_cost_paid,
            "entry_price": position["entry_price"],
            "price_change": price_change,
        }

    def _update_adaptive_kelly(self, trade_pnl: float) -> None:
        if self.initial_capital <= 0:
            return
        self.adaptive_breaker.update(trade_pnl / self.initial_capital)
        adjusted_fraction = self.base_kelly_fraction * self.adaptive_breaker.current_multiplier
        adjusted_fraction = min(max(adjusted_fraction, 0.15), 0.35)
        self.position_sizer.set_kelly_fraction(adjusted_fraction)

    def _calculate_reward(self, trade_result: Dict, current_portfolio_value: float) -> float:
        """
        Conditional manual exit rewards: Quality over quantity
        
        Philosophy: "Reward profitable manual exits, not just manual exits"
        
        Key components:
        1. Portfolio feedback (10×) - reduced noise, clearer outcome signals
        2. PnL outcomes - direct profit/loss feedback
        3. CONDITIONAL manual bonus - only for profitable manual exits
        4. Auto-exit penalty - discourage reliance on safety nets
        
        Critical change from Run 44:
        - Manual exit bonus NOW CONDITIONAL on profitability
        - Previous: +0.8 for any manual exit (rewarded bad timing)
        - Current: +1.2 only for profitable manual exits
        """
        weights = self.reward_config
        reward_version = str(weights.get("_version", "v2")).lower()
        reward = 0.0

        # Update drawdown tracking (for termination, not reward)
        self.peak_capital = max(self.peak_capital, current_portfolio_value)
        self.current_drawdown = (self.peak_capital - current_portfolio_value) / self.peak_capital

        # 1. Portfolio feedback - apply only on executed trades to reduce noise
        portfolio_change = (current_portfolio_value - self.prev_portfolio_value) / self.initial_capital
        if reward_version in {"v1", "legacy"}:
            reward += portfolio_change * weights.get("portfolio_step_scale", 10.0)
        elif trade_result.get("executed"):
            reward += portfolio_change * weights.get("portfolio_step_scale", 10.0)

        # 1a. Per-step unrealized P&L signal while holding
        if self.positions and not trade_result.get("executed"):
            hold_pnl_step_scale = float(weights.get("hold_pnl_step_scale", 0.0))
            reward += portfolio_change * hold_pnl_step_scale

        # 1b. Sharpe bonus - reward consistent returns over recent trades
        sharpe_scale = float(weights.get("sharpe_bonus_scale", 2.0))
        sharpe_cap = float(weights.get("sharpe_bonus_cap", 2.0))
        if sharpe_scale > 0 and len(self.recent_trades) >= 5:
            returns = np.array(self.recent_trades, dtype=np.float32) / max(self.initial_capital, 1e-9)
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))
            if std_return > 0:
                if reward_version in {"v1", "legacy"}:
                    sharpe = (mean_return / std_return) * np.sqrt(len(returns))
                else:
                    sharpe = mean_return / std_return
                reward += max(-sharpe_cap, min(sharpe_cap, sharpe)) * sharpe_scale

        if reward_version not in {"v1", "legacy"}:
            row = self._current_row()
            current_price = float(row["close"])

            if trade_result.get("reason") == "Invalid action":
                reward -= float(weights.get("invalid_action_penalty", 0.0))

            if not self.positions and trade_result.get("reason") == "No action taken":
                idle_base = float(weights.get("idle_base", 0.0))
                idle_step = float(weights.get("idle_step", 0.0))
                idle_cap = float(weights.get("idle_cap", 0.0))
                idle_penalty = min(idle_cap, idle_base + idle_step * self.flat_steps)
                reward -= idle_penalty

            if self.positions and self.current_symbol in self.positions:
                hold_steps = int(trade_result.get("hold_steps", self.position_hold_steps))
                hold_step_penalty = float(weights.get("hold_step_penalty", 0.0))
                hold_penalty_cap = float(weights.get("hold_penalty_cap", 0.0))
                reward -= min(hold_penalty_cap, hold_step_penalty * hold_steps)

                position = self.positions[self.current_symbol]
                entry_price = float(position.get("entry_price", 0.0))
                if entry_price > 0:
                    unrealized_pct = (current_price - entry_price) / entry_price
                    if unrealized_pct > 0:
                        carry_bonus_scale = float(weights.get("carry_bonus_scale", 0.0))
                        carry_bonus_cap = float(weights.get("carry_bonus_cap", 0.0))
                        reward += min(carry_bonus_cap, unrealized_pct * carry_bonus_scale)

        # 1c. Explicit fee penalty - make agent feel the cost of each trade
        if trade_result.get("executed"):
            trade_cost = float(trade_result.get("cost", 0.0))
            fee_penalty_scale = float(weights.get("fee_penalty_scale", 3.0))
            reward -= (trade_cost / self.initial_capital) * fee_penalty_scale

        # 2. EXIT-BASED REWARDS: Conditional on outcomes
        if trade_result.get("executed"):
            action_name = trade_result.get("action_name", "")
            
            if action_name.startswith(("SELL", "CLOSE")):
                # OUTCOME LEARNING: Main PnL feedback
                realized_pnl = float(trade_result.get("pnl", 0.0))
                realized_pnl_pct = realized_pnl / self.initial_capital
                reward += realized_pnl_pct
                if reward_version not in {"v1", "legacy"}:
                    loss_aversion_scale = float(weights.get("loss_aversion_scale", 0.0))
                    loss_aversion_cap = float(weights.get("loss_aversion_cap", 0.5))
                    if realized_pnl_pct < 0 and loss_aversion_scale > 0:
                        loss_penalty = loss_aversion_scale * (abs(realized_pnl_pct) ** 2)
                        reward -= min(loss_aversion_cap, loss_penalty)
                
                # Check if this is manual or auto-exit
                exit_reason = trade_result.get("reason", "")
                is_auto_exit = exit_reason in {"AUTO_STOP_LOSS", "AUTO_TAKE_PROFIT", "AUTO_MAX_HOLD"}
                
                if is_auto_exit:
                    # Penalty for relying on auto-exit (failure to learn timing)
                    auto_exit_penalty = float(weights.get("auto_exit_penalty", 0.5))
                    if reward_version in {"v1", "legacy"}:
                        reward -= auto_exit_penalty
                    else:
                        if realized_pnl_pct >= 0:
                            target_pct = float(weights.get("take_profit_pct", 0.02))
                            penalty_scale = min(1.0, abs(realized_pnl_pct) / max(target_pct, 1e-9))
                            reward -= auto_exit_penalty * 0.25 * penalty_scale
                        else:
                            target_pct = float(weights.get("stop_loss_pct", 0.01))
                            penalty_scale = min(1.0, abs(realized_pnl_pct) / max(target_pct, 1e-9))
                            reward -= auto_exit_penalty * 0.5 * penalty_scale
                else:
                    # CONDITIONAL MANUAL EXIT BONUS: Only for profitable exits
                    if realized_pnl > 0:
                        # Combined bonus: profit achievement + manual skill
                        if reward_version in {"v1", "legacy"}:
                            reward += weights.get("sell_profit_bonus", 0.5)
                            reward += weights.get("manual_exit_bonus", 1.2)
                        else:
                            profit_bonus_scale = float(weights.get("sell_profit_bonus", 0.5))
                            profit_bonus_multiplier = float(weights.get("sell_profit_bonus_scale", 10.0))
                            profit_bonus = min(profit_bonus_scale, realized_pnl_pct * profit_bonus_multiplier)
                            reward += max(0.0, profit_bonus)
                            manual_exit_bonus = min(float(weights.get("manual_exit_bonus", 0.1)), 0.1)
                            reward += manual_exit_bonus
                    # If manual but unprofitable: no bonus, no penalty
                    # Model learns: "Wait for profitable exit timing"
        
        return float(reward)

    def _check_termination(self) -> bool:
        if self.capital < self.initial_capital * 0.1:
            return True
        if self.current_drawdown > self.settings.MAX_TOTAL_DRAWDOWN:
            return True
        return False

    def _get_observation(self) -> np.ndarray:
        row = self._current_row()

        obs = np.zeros(42, dtype=np.float32)  # Run 48: Expanded from 39 to 42

        # Get current price (used in multiple places)
        current_price = float(row["close"])

        # Normalized price deviation from 24h MA (instead of raw price which clips)
        obs[0] = (current_price - float(row["ma_24"])) / (float(row["ma_24"]) + 1e-9)
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
        volatility = abs(float(row["volatility_24h"]))
        obs[24] = float(row["return_6h"]) / (volatility + 1e-6)
        obs[25] = float(row["return_1h"]) / (volatility + 1e-6)
        obs[26] = float(row["return_24h"]) / (volatility + 1e-6)
        obs[27] = float(row["price_zscore_24h"])

        # Technical indicators (normalized)
        obs[28] = float(row["rsi_14"])  # Already normalized to [0, 1]
        obs[29] = float(row["macd_line"]) * 100.0  # Scale for visibility
        obs[30] = float(row["macd_signal"]) * 100.0
        obs[31] = float(row["macd_hist"]) * 100.0
        obs[32] = float(row["atr_14"]) * 10.0  # Scale normalized ATR
        obs[33] = float(row["bb_upper"]) - 1.0  # Deviation from middle
        obs[34] = float(row["bb_middle"]) - 1.0
        obs[35] = float(row["bb_lower"]) - 1.0

        # Position-level features (critical for exit timing decisions)
        if self.positions and self.current_symbol in self.positions:
            position = self.positions[self.current_symbol]
            entry_price = position["entry_price"]
            
            # Feature 36: Unrealized P&L percentage (how much are we winning/losing?)
            obs[36] = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            
            # Feature 37: Hold duration normalized (how long have we been in this trade?)
            obs[37] = self.position_hold_steps / self.max_hold_steps
            
            # Feature 38: Distance from entry relative to volatility (entry quality indicator)
            volatility = abs(float(row["volatility_24h"]))
            obs[38] = ((current_price - entry_price) / entry_price) / (volatility + 1e-6)
        else:
            # No position: set position features to neutral values
            obs[36] = 0.0  # No unrealized P&L
            obs[37] = 0.0  # No hold duration
            obs[38] = 0.0  # No entry reference

        # Run 48: Long-term trend context features (obs 39-41)
        # These features help model understand market regime and avoid counter-trend trades
        
        # Feature 39: Distance to 200h MA (trend strength)
        # Positive = price above long-term trend (bullish), Negative = price below (bearish)
        ma_200 = float(row["ma_200"])
        obs[39] = (current_price - ma_200) / (ma_200 + 1e-9)
        
        # Feature 40: 200h MA slope (trend direction)
        # Positive = uptrend strengthening, Negative = downtrend or weakening
        obs[40] = float(row["ma_200_slope"])
        
        # Feature 41: Volume surge detection
        # >1.0 = volume spike (breakout/breakdown), <1.0 = low volume (consolidation)
        volume_ma_200 = float(row["volume_ma_200"])
        obs[41] = float(row["volume"]) / max(volume_ma_200, 1.0)

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
        "200h": steps_per_hour * 200,  # Run 48: Long-term trend indicator
    }
