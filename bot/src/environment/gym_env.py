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

    # Action space: 7 explicit actions for multi-asset portfolio
    # Each slot corresponds to a fixed symbol (e.g., slot 0 = BTC, slot 1 = ETH, slot 2 = SOL)
    ACTION_NO_ACTION = 0
    ACTION_BUY_0 = 1   # Buy symbol in slot 0
    ACTION_BUY_1 = 2   # Buy symbol in slot 1
    ACTION_BUY_2 = 3   # Buy symbol in slot 2
    ACTION_SELL_0 = 4   # Sell position in slot 0
    ACTION_SELL_1 = 5   # Sell position in slot 1
    ACTION_SELL_2 = 6   # Sell position in slot 2

    NUM_SLOTS = 3  # Number of asset slots (matches MAX_OPEN_POSITIONS)

    MIN_POSITION_VALUE_FRACTION = 0.01

    ACTION_NAMES = {
        0: "NO_ACTION",
        1: "BUY_0",
        2: "BUY_1",
        3: "BUY_2",
        4: "SELL_0",
        5: "SELL_1",
        6: "SELL_2",
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

        # Action space: 7 discrete actions (NO_ACTION + 3 BUY + 3 SELL)
        self.action_space = spaces.Discrete(7)

        # Assign symbols to fixed slots for the entire training session
        # This eliminates symbol rotation entirely - agent always sees all symbols
        self.slot_symbols: List[str] = sorted(self.data_by_symbol.keys())[:self.NUM_SLOTS]
        self.symbol_to_slot: Dict[str, int] = {s: i for i, s in enumerate(self.slot_symbols)}
        logger.info(f"Fixed symbol slots: {self.slot_symbols}")

        # Observation space: multi-asset simultaneous view
        # Architecture: [slot_0_market(15) + slot_1_market(15) + slot_2_market(15) +
        #               portfolio_global(10) + slot_0_position(4) + slot_1_position(4) + slot_2_position(4)]
        # = 45 market + 10 portfolio + 12 position = 67 dimensions
        # Agent sees ALL symbols at every step - no rotation needed
        self.obs_dim = 67
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Portfolio state
        self.capital = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # Keyed by symbol
        self.trade_history: List[Dict] = []

        # Performance tracking
        self.episode_returns: List[float] = []
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.prev_portfolio_value = self.initial_capital

        # Episode state
        self.current_step = 0
        self.current_index = 0
        self.slot_data: Dict[str, pd.DataFrame] = {}  # Episode data per slot symbol
        self.terminated = False
        self.flat_steps = 0
        self.steps_since_last_close = self.trade_cooldown_steps  # Start ready to trade

        # Historical performance
        self.recent_trades: List[float] = []
        self.win_count = 0
        self.trade_count = 0

        logger.info("CryptoTradingEnv initialized with real data for multi-asset portfolio")

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
        self.flat_steps = 0
        self.steps_since_last_close = self.trade_cooldown_steps  # Start ready to trade

        # Select symbol and window
        self._select_episode_window()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_slot_symbol(self, slot: int) -> Optional[str]:
        """Get the symbol assigned to a given slot index."""
        if 0 <= slot < len(self.slot_symbols):
            return self.slot_symbols[slot]
        return None

    def _get_slot_price(self, slot: int) -> float:
        """Get the current close price for a slot's symbol."""
        symbol = self._get_slot_symbol(slot)
        if symbol and symbol in self.slot_data:
            data = self.slot_data[symbol]
            if len(data) > self.current_index:
                return float(data.iloc[self.current_index]["close"])
        return 0.0

    def _get_slot_row(self, slot: int) -> Optional[pd.Series]:
        """Get the current data row for a slot's symbol."""
        symbol = self._get_slot_symbol(slot)
        if symbol and symbol in self.slot_data:
            data = self.slot_data[symbol]
            if len(data) > self.current_index:
                return data.iloc[self.current_index]
        return None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.terminated:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Phase 1: Auto-exit check across ALL positions ---
        forced_exits: List[Dict] = []
        if self.positions and self.auto_exit_enabled:
            for symbol, position in list(self.positions.items()):
                slot = self.symbol_to_slot.get(symbol)
                if slot is None:
                    continue
                current_price = self._get_slot_price(slot)
                if current_price <= 0:
                    continue
                hold_steps = position.get("hold_steps", 0)
                price_change = (current_price - position["entry_price"]) / position["entry_price"]

                if price_change <= -self.reward_config.get("stop_loss_pct", 0.01):
                    forced_exits.append({"symbol": symbol, "slot": slot, "reason": "AUTO_STOP_LOSS"})
                elif price_change >= self.reward_config.get("take_profit_pct", 0.02):
                    forced_exits.append({"symbol": symbol, "slot": slot, "reason": "AUTO_TAKE_PROFIT"})
                elif hold_steps >= self.max_hold_steps:
                    forced_exits.append({"symbol": symbol, "slot": slot, "reason": "AUTO_MAX_HOLD"})

        # Execute forced exits first (before agent action)
        forced_results = []
        for fe in forced_exits:
            result = self._execute_sell_slot(fe["slot"])
            result["reason"] = fe["reason"]
            forced_results.append(result)

        # --- Phase 2: Determine action type and target slot ---
        action_name = self.ACTION_NAMES.get(action, "UNKNOWN")
        invalid_action = False
        target_slot = -1

        if action == self.ACTION_NO_ACTION:
            target_slot = -1
        elif action in (self.ACTION_BUY_0, self.ACTION_BUY_1, self.ACTION_BUY_2):
            target_slot = action - 1  # BUY_0=1 -> slot 0, BUY_1=2 -> slot 1, BUY_2=3 -> slot 2
            symbol = self._get_slot_symbol(target_slot)
            if symbol is None:
                invalid_action = True
            elif symbol in self.positions:
                invalid_action = True  # Already have position in this symbol
            elif len(self.positions) >= self.settings.MAX_OPEN_POSITIONS:
                invalid_action = True  # Max positions reached
        elif action in (self.ACTION_SELL_0, self.ACTION_SELL_1, self.ACTION_SELL_2):
            target_slot = action - 4  # SELL_0=4 -> slot 0, SELL_1=5 -> slot 1, SELL_2=6 -> slot 2
            symbol = self._get_slot_symbol(target_slot)
            if symbol is None or symbol not in self.positions:
                invalid_action = True  # No position to sell
        else:
            invalid_action = True

        # --- Phase 3: Execute agent's action ---
        hold_steps_before_action = 0
        if invalid_action:
            trade_result = {
                "action": action,
                "action_name": action_name,
                "executed": False,
                "reason": "Invalid action",
                "pnl": 0.0,
                "cost": 0.0,
            }
        elif action == self.ACTION_NO_ACTION:
            trade_result = {
                "action": action,
                "action_name": "NO_ACTION",
                "executed": False,
                "reason": "No action taken",
                "pnl": 0.0,
                "cost": 0.0,
            }
        elif action in (self.ACTION_BUY_0, self.ACTION_BUY_1, self.ACTION_BUY_2):
            trade_result = self._execute_buy_slot(target_slot, action)
        elif action in (self.ACTION_SELL_0, self.ACTION_SELL_1, self.ACTION_SELL_2):
            symbol = self._get_slot_symbol(target_slot)
            if symbol and symbol in self.positions:
                hold_steps_before_action = self.positions[symbol].get("hold_steps", 0)
            trade_result = self._execute_sell_slot(target_slot)
            trade_result["action"] = action
            trade_result["action_name"] = action_name
        else:
            trade_result = {
                "action": action,
                "action_name": "UNKNOWN",
                "executed": False,
                "reason": "Unknown action",
                "pnl": 0.0,
                "cost": 0.0,
            }

        trade_result["hold_steps"] = hold_steps_before_action

        # Track flat steps
        if not self.positions and not trade_result.get("executed"):
            self.flat_steps += 1
        else:
            self.flat_steps = 0

        # --- Phase 4: Calculate reward ---
        current_portfolio_value = self._get_portfolio_value()
        reward = self._calculate_reward(trade_result, current_portfolio_value)

        # Add penalty for forced exits
        for fr in forced_results:
            if fr.get("executed"):
                pnl = float(fr.get("pnl", 0.0))
                pnl_pct = pnl / self.initial_capital
                auto_exit_penalty = float(self.reward_config.get("auto_exit_penalty", 0.5))
                if pnl >= 0:
                    reward -= auto_exit_penalty * 0.25
                else:
                    reward -= auto_exit_penalty * 0.5
                reward += pnl_pct  # PnL feedback

        self.prev_portfolio_value = current_portfolio_value

        # --- Phase 5: Advance state ---
        self.current_step += 1
        self.current_index += 1

        # Increment hold steps for ALL open positions
        for symbol, position in self.positions.items():
            position["hold_steps"] = position.get("hold_steps", 0) + 1

        # Cooldown tracking
        if not self.positions:
            self.steps_since_last_close += 1
        else:
            self.steps_since_last_close = 0

        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()
        info["trade_result"] = trade_result
        if forced_results:
            info["forced_exits"] = forced_results

        if terminated or truncated:
            self.terminated = True
            episode_return = (current_portfolio_value - self.initial_capital) / self.initial_capital
            episode_pnl_scale = float(self.reward_config.get("episode_pnl_bonus_scale", 10.0))
            reward += episode_return * episode_pnl_scale

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
        """
        Select a synchronized time window for all fixed slot symbols.
        
        All slot symbols share the same time window so the agent sees 
        concurrent market data. No rotation - slots are fixed for the 
        entire training session.
        """
        # Find maximum window that works for ALL slot symbols
        min_length = float('inf')
        for symbol in self.slot_symbols:
            if symbol not in self.data_by_symbol:
                raise DataUnavailableError(f"Slot symbol {symbol} not found in data.")
            min_length = min(min_length, len(self.data_by_symbol[symbol]))

        required = self.max_steps + self.sequence_length + 1
        if min_length <= required:
            self.max_steps = int(min_length) - self.sequence_length - 1
            logger.warning(
                "Insufficient history for requested max_steps; "
                f"reducing max_steps to {self.max_steps}."
            )
            if self.max_steps <= 0:
                raise DataUnavailableError("Insufficient history for requested sequence length.")

        # Pick a random start index within the valid range
        max_start = int(min_length) - self.max_steps - 1
        if max_start <= 0:
            start_index = 0
        else:
            start_index = np.random.randint(0, max_start)

        # Slice synchronized windows for each slot symbol
        self.slot_data = {}
        for symbol in self.slot_symbols:
            data = self.data_by_symbol[symbol]
            self.slot_data[symbol] = data.iloc[start_index : start_index + self.max_steps + 1].reset_index(drop=True)

        self.current_index = 0

    def _current_row(self, slot: int = 0) -> pd.Series:
        """Get current data row for a slot (default slot 0 for general market reference)."""
        symbol = self._get_slot_symbol(slot)
        if symbol and symbol in self.slot_data:
            data = self.slot_data[symbol]
            if len(data) > self.current_index:
                return data.iloc[self.current_index]
        raise DataUnavailableError("Episode data not initialized.")

    def get_valid_actions(self) -> List[int]:
        """Return valid actions for the 7-action slot-based space."""
        valid = [self.ACTION_NO_ACTION]

        # Check each slot for SELL validity
        for slot in range(self.NUM_SLOTS):
            symbol = self._get_slot_symbol(slot)
            if symbol and symbol in self.positions:
                position = self.positions[symbol]
                hold_steps = position.get("hold_steps", 0)
                if hold_steps >= self.min_hold_steps:
                    valid.append(self.ACTION_SELL_0 + slot)  # SELL_0=4, SELL_1=5, SELL_2=6

        # Check each slot for BUY validity (only if under max positions and cooldown passed)
        can_buy = (
            len(self.positions) < self.settings.MAX_OPEN_POSITIONS
            and self.steps_since_last_close >= self.trade_cooldown_steps
        )

        if can_buy:
            for slot in range(self.NUM_SLOTS):
                symbol = self._get_slot_symbol(slot)
                if symbol is None:
                    continue
                if symbol in self.positions:
                    continue  # Already have position in this slot

                # Per-slot entry filters
                row = self._get_slot_row(slot)
                if row is None:
                    continue

                if self.entry_signal_threshold > 0:
                    trend_strength = abs(float(row["return_24h"]))
                    if trend_strength < self.entry_signal_threshold:
                        continue

                if self.entry_volatility_cap > 0:
                    volatility = abs(float(row["volatility_24h"]))
                    if volatility > self.entry_volatility_cap:
                        continue

                if self.entry_momentum_ratio_threshold > 0:
                    volatility = abs(float(row["volatility_24h"]))
                    momentum_ratio = float(row["return_6h"]) / (volatility + 1e-6)
                    if momentum_ratio < self.entry_momentum_ratio_threshold:
                        continue

                if self.min_volume_24h > 0:
                    volume_ma_24 = float(row["volume_ma_24"])
                    if volume_ma_24 < self.min_volume_24h:
                        continue

                if self.min_market_liquidity > 0:
                    current_price = float(row["close"])
                    volume_value = float(row["volume_ma_24"]) * current_price
                    if volume_value < self.min_market_liquidity:
                        continue

                valid.append(self.ACTION_BUY_0 + slot)  # BUY_0=1, BUY_1=2, BUY_2=3

        return valid

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
            "carry_bonus_scale": 2.0,
            "carry_bonus_cap": 0.2,
            "hold_pnl_step_scale": 2.0,
            "episode_pnl_bonus_scale": 10.0,
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

    def _execute_buy_slot(self, slot: int, action: int) -> Dict:
        """Execute a buy order for the symbol in the given slot."""
        symbol = self._get_slot_symbol(slot)
        row = self._get_slot_row(slot)
        action_name = self.ACTION_NAMES.get(action, f"BUY_{slot}")

        if symbol is None or row is None:
            return {
                "action": action, "action_name": action_name,
                "executed": False, "reason": "Invalid slot",
                "pnl": 0.0, "cost": 0.0,
            }

        current_price = float(row["close"])

        if symbol in self.positions:
            return {
                "action": action, "action_name": action_name,
                "executed": False, "reason": "Position already open in this symbol",
                "pnl": 0.0, "cost": 0.0,
            }

        if len(self.positions) >= self.settings.MAX_OPEN_POSITIONS:
            return {
                "action": action, "action_name": action_name,
                "executed": False, "reason": "Max positions reached",
                "pnl": 0.0, "cost": 0.0,
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
        position_value = min(base_size, self.capital * self.max_position_size_pct)
        position_value = max(position_value, self.min_position_value)

        raw_slippage = self._estimate_slippage_pct(position_value, row)
        if raw_slippage > self.max_slippage_pct:
            return {
                "action": action, "action_name": action_name,
                "executed": False, "reason": "Slippage too high",
                "pnl": 0.0, "cost": 0.0,
            }

        cost_pct = self._get_trade_cost_pct(position_value, self._get_portfolio_value(), row)
        cost = position_value * (1 + cost_pct)
        if cost > self.capital:
            return {
                "action": action, "action_name": action_name,
                "executed": False, "reason": "Insufficient capital",
                "pnl": 0.0, "cost": 0.0,
            }

        self.capital -= cost
        transaction_cost_paid = position_value * cost_pct

        self.positions[symbol] = {
            "size": position_value,
            "entry_price": current_price,
            "timestamp": self.current_step,
            "hold_steps": 0,
            "slot": slot,
        }

        return {
            "action": action, "action_name": action_name,
            "executed": True, "reason": "Buy executed",
            "size": position_value, "price": current_price,
            "side": "buy", "cost": transaction_cost_paid, "pnl": 0.0,
            "symbol": symbol,
        }

    def _execute_sell_slot(self, slot: int) -> Dict:
        """Execute a sell order for the position in the given slot."""
        symbol = self._get_slot_symbol(slot)
        action_name = f"SELL_{slot}"

        if symbol is None or symbol not in self.positions:
            return {
                "action": self.ACTION_SELL_0 + slot, "action_name": action_name,
                "executed": False, "reason": "No position to close",
                "pnl": 0.0, "cost": 0.0,
            }

        row = self._get_slot_row(slot)
        if row is None:
            return {
                "action": self.ACTION_SELL_0 + slot, "action_name": action_name,
                "executed": False, "reason": "No data for slot",
                "pnl": 0.0, "cost": 0.0,
            }

        current_price = float(row["close"])
        position = self.positions[symbol]
        sell_size = position["size"]
        price_change = (current_price - position["entry_price"]) / position["entry_price"]
        pnl_before_costs = sell_size * price_change

        raw_slippage = self._estimate_slippage_pct(sell_size, row)
        if raw_slippage > self.max_slippage_pct:
            return {
                "action": self.ACTION_SELL_0 + slot, "action_name": action_name,
                "executed": False, "reason": "Slippage too high",
                "pnl": 0.0, "cost": 0.0,
            }

        cost_pct = self._get_trade_cost_pct(sell_size, self._get_portfolio_value(), row)
        transaction_cost_paid = sell_size * cost_pct
        pnl = pnl_before_costs - transaction_cost_paid

        self.capital += sell_size + pnl
        del self.positions[symbol]
        self.steps_since_last_close = 0

        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
        if pnl > 0:
            self.win_count += 1
        self.trade_count += 1
        self._update_adaptive_kelly(pnl)

        return {
            "action": self.ACTION_SELL_0 + slot, "action_name": action_name,
            "executed": True, "reason": "Sell executed",
            "size": sell_size, "price": current_price,
            "side": "sell", "pnl": pnl, "cost": transaction_cost_paid,
            "entry_price": position["entry_price"], "price_change": price_change,
            "symbol": symbol,
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

            if self.positions:
                # Apply hold penalty and carry bonus across ALL open positions
                hold_step_penalty = float(weights.get("hold_step_penalty", 0.0))
                hold_penalty_cap = float(weights.get("hold_penalty_cap", 0.0))
                carry_bonus_scale = float(weights.get("carry_bonus_scale", 0.0))
                carry_bonus_cap = float(weights.get("carry_bonus_cap", 0.0))

                for symbol, position in self.positions.items():
                    hold_steps = position.get("hold_steps", 0)
                    reward -= min(hold_penalty_cap, hold_step_penalty * hold_steps) / max(len(self.positions), 1)

                    slot = self.symbol_to_slot.get(symbol)
                    if slot is not None:
                        pos_price = self._get_slot_price(slot)
                        entry_price = float(position.get("entry_price", 0.0))
                        if entry_price > 0 and pos_price > 0:
                            unrealized_pct = (pos_price - entry_price) / entry_price
                            if unrealized_pct > 0:
                                reward += min(carry_bonus_cap, unrealized_pct * carry_bonus_scale) / max(len(self.positions), 1)

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
        """
        Build 67-dimensional observation: all 3 symbols visible simultaneously.
        
        Layout:
          [0-14]  Slot 0 market features (15)
          [15-29] Slot 1 market features (15)
          [30-44] Slot 2 market features (15)
          [45-54] Portfolio global features (10)
          [55-58] Slot 0 position features (4)
          [59-62] Slot 1 position features (4)
          [63-66] Slot 2 position features (4)
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # --- Per-slot market features (15 each × 3 slots = 45) ---
        for slot in range(self.NUM_SLOTS):
            base = slot * 15
            row = self._get_slot_row(slot)
            if row is None:
                continue  # Leave zeros for missing slot data

            close_price = float(row["close"])
            ma_24 = float(row["ma_24"])

            obs[base + 0] = (close_price - ma_24) / (ma_24 + 1e-9)   # Price deviation from 24h MA
            obs[base + 1] = float(row["hl_spread_pct"])                # High-low spread
            obs[base + 2] = float(row["return_1h"])                    # 1h return
            obs[base + 3] = float(row["return_6h"])                    # 6h return
            obs[base + 4] = float(row["return_24h"])                   # 24h return
            obs[base + 5] = float(row["volatility_24h"])               # 24h volatility
            obs[base + 6] = float(row["ma_6"] / close_price) - 1.0    # MA6 deviation
            obs[base + 7] = float(row["ma_24"] / close_price) - 1.0   # MA24 deviation
            obs[base + 8] = float(row["trend_direction"])              # Trend direction
            obs[base + 9] = float(row["rsi_14"])                       # RSI
            obs[base + 10] = float(row["macd_hist"]) * 100.0           # MACD histogram
            obs[base + 11] = float(row["atr_14"]) * 10.0              # ATR
            obs[base + 12] = float(row["bb_upper"]) - 1.0             # BB upper deviation
            obs[base + 13] = float(row["bb_lower"]) - 1.0             # BB lower deviation
            # Momentum ratio
            vol = abs(float(row["volatility_24h"]))
            obs[base + 14] = float(row["return_6h"]) / (vol + 1e-6)

        # --- Portfolio global features (10) at indices 45-54 ---
        portfolio_value = self._get_portfolio_value()
        obs[45] = self.capital / self.initial_capital                   # Cash ratio
        obs[46] = len(self.positions) / self.settings.MAX_OPEN_POSITIONS  # Position fill ratio
        obs[47] = portfolio_value / self.initial_capital                # Portfolio value ratio
        obs[48] = (portfolio_value - self.initial_capital) / self.initial_capital  # Total return
        obs[49] = sum(self.recent_trades) / self.initial_capital if self.recent_trades else 0.0  # Recent PnL
        obs[50] = self.win_count / max(self.trade_count, 1)            # Win rate
        obs[51] = self.current_drawdown                                 # Current drawdown

        # Time features from slot 0
        row0 = self._get_slot_row(0)
        if row0 is not None:
            ts = pd.to_datetime(row0["timestamp"])
            obs[52] = np.sin(2 * np.pi * ts.hour / 24)                # Hour sine
            obs[53] = np.cos(2 * np.pi * ts.hour / 24)                # Hour cosine
            obs[54] = np.sin(2 * np.pi * ts.weekday() / 7)            # Day sine

        # --- Per-slot position features (4 each × 3 slots = 12) at indices 55-66 ---
        for slot in range(self.NUM_SLOTS):
            base = 55 + slot * 4
            symbol = self._get_slot_symbol(slot)
            if symbol and symbol in self.positions:
                position = self.positions[symbol]
                entry_price = position["entry_price"]
                current_price = self._get_slot_price(slot)
                hold_steps = position.get("hold_steps", 0)

                obs[base + 0] = position["size"] / self.initial_capital                      # Position size
                obs[base + 1] = (current_price - entry_price) / (entry_price + 1e-9)        # Unrealized PnL %
                obs[base + 2] = hold_steps / self.max_hold_steps                              # Hold duration
                obs[base + 3] = 1.0                                                           # Has position flag

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
            "slot_symbols": self.slot_symbols,
            "open_positions": list(self.positions.keys()),
        }

    def _get_portfolio_value(self) -> float:
        """Calculate portfolio value using each position's own symbol price."""
        total = self.capital

        for symbol, position in self.positions.items():
            entry_price = position["entry_price"]
            size = position["size"]
            slot = self.symbol_to_slot.get(symbol)
            if slot is not None:
                current_price = self._get_slot_price(slot)
                if current_price > 0 and entry_price > 0:
                    total += size * (current_price / entry_price)
                else:
                    total += size
            else:
                total += size

        return total

    def render(self):
        row = self._current_row(0)
        portfolio_value = self._get_portfolio_value()
        returns = (portfolio_value - self.initial_capital) / self.initial_capital

        print(f"\nStep: {self.current_step}/{self.max_steps}")
        print(f"Slots: {self.slot_symbols}")
        print(f"Price (slot 0): {row['close']:.4f}")
        print(f"Capital: ${self.capital:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f} ({returns:+.2%})")
        print(f"Open Positions: {len(self.positions)}/{self.settings.MAX_OPEN_POSITIONS} - {list(self.positions.keys())}")
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
