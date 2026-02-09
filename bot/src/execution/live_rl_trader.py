"""
Live RL Paper Trader - Runs the trained RL model on real-time Coinbase data.

Rotates through configured symbols, evaluating one per tick.
Auto-exits are checked across all open positions each tick.

Usage (via CLI):
    python main.py rl-paper-trade --model models/best_model_run_75_step_674000 --duration 0

Architecture:
    For each symbol:
      Coinbase API -> Feature Computation -> Observation -> MaskablePPO -> Paper Portfolio
    (identical pipeline to CryptoTradingEnv, just with live data)
"""

from __future__ import annotations

import json
import os
import time
import numpy as np
import pandas as pd
import torch
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ..data.sources.coinbase import CoinbaseAdapter
from ..environment.gym_env import CryptoTradingEnv, _interval_to_steps
from ..core.logger import get_logger
from ..core.config import get_settings
from ..risk.position_sizer import PositionSizer
from ..risk.adaptive_breaker import AdaptiveCircuitBreaker


logger = get_logger(__name__)


class LiveRLPaperTrader:
    """
    Live paper trading engine powered by the trained RL model.

    Evaluates one symbol per tick (rotating).
    Supports multiple concurrent positions (matching training behavior).
    """

    ACTION_NO_ACTION = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    NUM_ACTIONS = 3
    ACTION_NAMES = {
        0: "NO_ACTION",
        1: "BUY",
        2: "SELL",
    }

    def __init__(
        self,
        model_path: str,
        symbols: Optional[List[str]] = None,
        initial_capital: float = 1000.0,
        interval: str = "1h",
        log_dir: str = "./logs/paper_trading",
    ):
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.interval = interval
        self.log_dir = log_dir

        self.settings = get_settings()
        self.coinbase = CoinbaseAdapter()

        # Resolve symbol list: explicit arg > DATA_SYMBOLS env var > default
        if symbols:
            self.symbols = symbols
        elif self.settings.DATA_SYMBOLS:
            self.symbols = [s.strip() for s in self.settings.DATA_SYMBOLS.split(",") if s.strip()]
        else:
            self.symbols = ["BTC-USD", "ETH-USD"]

        # Load configs (same files the training env reads)
        self.reward_config = self._load_reward_config()
        self.risk_config = self._load_risk_config()

        # Trading parameters from config
        self.min_hold_steps = int(self.reward_config.get("min_hold_steps", 5))
        self.max_hold_steps = int(self.reward_config.get("max_hold_steps", 100))
        self.trade_cooldown_steps = int(self.reward_config.get("trade_cooldown_steps", 3))
        self.stop_loss_pct = float(self.reward_config.get("stop_loss_pct", 0.03))
        self.take_profit_pct = float(self.reward_config.get("take_profit_pct", 0.05))
        self.entry_signal_threshold = float(self.reward_config.get("entry_signal_threshold", 0.0))
        self.entry_volatility_cap = float(self.reward_config.get("entry_volatility_cap", 0.0))
        self.entry_momentum_ratio_threshold = float(
            self.reward_config.get("entry_momentum_ratio_threshold", 0.0)
        )
        position_sizing = self.risk_config.get("position_sizing", {})
        self.default_position_size_pct = float(position_sizing.get("default_position_size_pct", 0.12))
        self.min_trades_for_kelly = int(position_sizing.get("min_trades_for_kelly", 10))
        self.base_kelly_fraction = float(position_sizing.get("kelly_fraction", 0.25))
        self.position_sizer = PositionSizer(kelly_fraction=self.base_kelly_fraction)
        self.adaptive_breaker = AdaptiveCircuitBreaker()

        # Transaction costs
        tx_costs = self.risk_config.get("transaction_costs", {})
        self.base_transaction_cost = float(
            tx_costs.get("base_cost_pct", self.settings.TRANSACTION_COST_PCT)
        )
        self.max_slippage_pct = float(
            self.risk_config.get("market_requirements", {}).get("max_slippage_pct", 0.02)
        )
        self.slippage_model = str(tx_costs.get("slippage_model", "volume_based"))
        self.large_order_penalty_threshold = float(tx_costs.get("large_order_penalty_threshold", 0.15))
        self.large_order_penalty_pct = float(tx_costs.get("large_order_penalty_pct", 0.0005))
        position_limits = self.risk_config.get("position_limits", {})
        self.max_position_size_pct = float(position_limits.get("max_position_size_pct", 0.20))
        self.min_position_size_pct = float(position_limits.get("min_position_size_pct", 0.01))
        market_req = self.risk_config.get("market_requirements", {})
        self.min_volume_24h = float(market_req.get("min_volume_24h", 0.0))
        self.min_market_liquidity = float(market_req.get("min_market_liquidity", 0.0))

        # Portfolio state (multiple positions)
        self.positions: Dict[str, Dict] = {}
        self.symbol_cooldowns: Dict[str, int] = {symbol: self.trade_cooldown_steps for symbol in self.symbols}
        self.current_symbol_index = 0
        self.flat_steps = 0

        # Performance tracking
        self.trade_history: List[Dict] = []
        self.recent_trades: List[float] = []
        self.win_count = 0
        self.trade_count = 0
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.prev_portfolio_value = initial_capital
        self.tick_count = 0

        # Decision log
        self.decision_log: List[Dict] = []

        # Interval helpers
        self.interval_seconds = self._interval_to_seconds(interval)
        self.interval_steps = _interval_to_steps(interval)

        # Track symbols that failed to fetch (skip temporarily)
        self._symbol_errors: Dict[str, int] = {}

        # Load model
        self._load_model()

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        logger.info("LiveRLPaperTrader initialized")
        logger.info(f"  Symbols: {len(self.symbols)} configured")
        logger.info(f"  Capital: ${initial_capital:.2f}")
        logger.info(f"  Interval: {interval}")
        logger.info(f"  Model: {model_path}")

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the trained MaskablePPO model."""
        candles_df = self._fetch_candle_dataframe(self.symbols[0])
        temp_env = CryptoTradingEnv(
            dataset=candles_df,
            interval=self.interval,
            initial_capital=self.initial_capital,
        )
        vec_env = DummyVecEnv([lambda: temp_env])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MaskablePPO.load(self.model_path, env=vec_env, device=device)
        self.device = device
        logger.info(f"Model loaded on {device}")

    # ------------------------------------------------------------------
    # Data Fetching
    # ------------------------------------------------------------------

    def _fetch_candle_dataframe(self, symbol: str, num_candles: int = 300) -> pd.DataFrame:
        """Fetch recent candles from Coinbase for a specific symbol."""
        end = datetime.utcnow()
        start = end - timedelta(seconds=self.interval_seconds * num_candles)

        candles = self.coinbase.get_ohlcv(
            symbol=symbol,
            interval=self.interval,
            start=start,
            end=end,
        )

        if not candles:
            raise RuntimeError(f"No candles returned for {symbol}")

        rows = [
            {
                "symbol": symbol,
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Feature Computation  (mirrors CryptoTradingEnv._prepare_data)
    # ------------------------------------------------------------------

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling features EXACTLY matching CryptoTradingEnv._prepare_data()."""
        g = df.copy()
        steps = self.interval_steps

        g["return_1h"] = g["close"].pct_change(periods=steps["1h"])
        g["return_6h"] = g["close"].pct_change(periods=steps["6h"])
        g["return_24h"] = g["close"].pct_change(periods=steps["24h"])
        g["volatility_24h"] = g["close"].pct_change().rolling(steps["24h"]).std()
        g["ma_6"] = g["close"].rolling(steps["6h"]).mean()
        g["ma_24"] = g["close"].rolling(steps["24h"]).mean()
        g["ma_48"] = g["close"].rolling(steps["48h"]).mean()
        g["ma_200"] = g["close"].rolling(steps["200h"]).mean()
        g["ma_200_slope"] = g["ma_200"].pct_change(periods=steps["24h"])
        g["price_std_24h"] = g["close"].rolling(steps["24h"]).std()
        g["price_zscore_24h"] = (g["close"] - g["ma_24"]) / (g["price_std_24h"] + 1e-9)
        g["volume_ma_24"] = g["volume"].rolling(steps["24h"]).mean()
        g["volume_ma_200"] = g["volume"].rolling(steps["200h"]).mean()
        g["hl_spread_pct"] = (g["high"] - g["low"]) / g["close"]
        g["trend_direction"] = np.sign(g["return_24h"])

        g["rsi_14"] = CryptoTradingEnv._calculate_rsi(g["close"], period=14)
        macd_data = CryptoTradingEnv._calculate_macd(g["close"])
        g["macd_line"] = macd_data["macd"]
        g["macd_signal"] = macd_data["signal"]
        g["macd_hist"] = macd_data["histogram"]
        g["atr_14"] = CryptoTradingEnv._calculate_atr(g["high"], g["low"], g["close"], period=14)
        bb_data = CryptoTradingEnv._calculate_bollinger_bands(g["close"], period=20, std_dev=2.0)
        g["bb_upper"] = bb_data["upper"]
        g["bb_middle"] = bb_data["middle"]
        g["bb_lower"] = bb_data["lower"]

        g = g.dropna().reset_index(drop=True)
        return g

    # ------------------------------------------------------------------
    # Observation Builder  (mirrors CryptoTradingEnv._get_observation)
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_feature(value: float, limit: float = 5.0) -> float:
        return float(np.clip(value, -limit, limit))

    def _build_observation(self, symbol: str, row: pd.Series, current_price: float) -> np.ndarray:
        """Build 29-dim observation vector matching CryptoTradingEnv._get_observation()."""
        obs = np.zeros(29, dtype=np.float32)

        ma_24 = float(row["ma_24"])
        volatility = abs(float(row["volatility_24h"]))

        # --- Market features (0-14) ---
        obs[0] = self._clip_feature((current_price - ma_24) / (ma_24 + 1e-9))
        obs[1] = self._clip_feature(float(row["hl_spread_pct"]))
        obs[2] = self._clip_feature(float(row["return_1h"]))
        obs[3] = self._clip_feature(float(row["return_6h"]))
        obs[4] = self._clip_feature(float(row["return_24h"]))
        obs[5] = self._clip_feature(float(row["volatility_24h"]))
        obs[6] = self._clip_feature(float(row["ma_6"] / row["close"]) - 1.0)
        obs[7] = self._clip_feature(float(row["ma_24"] / row["close"]) - 1.0)
        obs[8] = self._clip_feature(float(row["trend_direction"]))
        obs[9] = self._clip_feature(float(row["rsi_14"]) * 2.0 - 1.0)
        obs[10] = self._clip_feature(float(row["macd_hist"]))
        obs[11] = self._clip_feature(float(row["atr_14"]))
        obs[12] = self._clip_feature(float(row["bb_upper"]) - 1.0)
        obs[13] = self._clip_feature(float(row["bb_lower"]) - 1.0)
        obs[14] = self._clip_feature(float(row["return_6h"]) / (volatility + 1e-6))

        # --- Portfolio features (15-24) ---
        portfolio_value = self._get_portfolio_value()
        obs[15] = self._clip_feature(self.capital / self.initial_capital)
        obs[16] = self._clip_feature(len(self.positions) / self.settings.MAX_OPEN_POSITIONS)
        obs[17] = self._clip_feature(portfolio_value / self.initial_capital)
        obs[18] = self._clip_feature((portfolio_value - self.initial_capital) / self.initial_capital)
        obs[19] = self._clip_feature(sum(self.recent_trades) / self.initial_capital if self.recent_trades else 0.0)
        obs[20] = self._clip_feature(self.win_count / max(self.trade_count, 1))
        obs[21] = self._clip_feature(self.current_drawdown)

        ts = pd.to_datetime(row["timestamp"])
        obs[22] = self._clip_feature(np.sin(2 * np.pi * ts.hour / 24))
        obs[23] = self._clip_feature(np.cos(2 * np.pi * ts.hour / 24))
        obs[24] = self._clip_feature(np.sin(2 * np.pi * ts.weekday() / 7))

        # --- Position features for current symbol (25-28) ---
        position = self.positions.get(symbol)
        if position:
            entry_price = position["entry_price"]
            hold_steps = position.get("hold_steps", 0)
            obs[25] = self._clip_feature(position["size"] / self.initial_capital)
            obs[26] = self._clip_feature((current_price - entry_price) / (entry_price + 1e-9))
            obs[27] = self._clip_feature(hold_steps / self.max_hold_steps)
            obs[28] = 1.0

        return obs

    # ------------------------------------------------------------------
    # Action Masking  (mirrors CryptoTradingEnv.action_masks)
    # ------------------------------------------------------------------

    def _get_action_mask(self, symbol: str, row: pd.Series) -> np.ndarray:
        """Get valid action mask for 3-action space."""
        mask = np.zeros(self.NUM_ACTIONS, dtype=bool)
        mask[self.ACTION_NO_ACTION] = True

        position = self.positions.get(symbol)
        if position:
            if position.get("hold_steps", 0) >= self.min_hold_steps:
                mask[self.ACTION_SELL] = True
        else:
            cooldown_steps = self.symbol_cooldowns.get(symbol, self.trade_cooldown_steps)
            if len(self.positions) < self.settings.MAX_OPEN_POSITIONS and cooldown_steps >= self.trade_cooldown_steps:
                if self.entry_signal_threshold > 0:
                    trend_strength = abs(float(row["return_24h"]))
                    if trend_strength < self.entry_signal_threshold:
                        return mask

                if self.entry_volatility_cap > 0:
                    volatility = abs(float(row["volatility_24h"]))
                    if volatility > self.entry_volatility_cap:
                        return mask

                if self.entry_momentum_ratio_threshold > 0:
                    volatility = abs(float(row["volatility_24h"]))
                    momentum_ratio = float(row["return_6h"]) / (volatility + 1e-6)
                    if momentum_ratio < self.entry_momentum_ratio_threshold:
                        return mask

                if self.min_volume_24h > 0:
                    volume_ma_24 = float(row["volume_ma_24"])
                    if volume_ma_24 < self.min_volume_24h:
                        return mask

                if self.min_market_liquidity > 0:
                    current_price = float(row["close"])
                    volume_value = float(row["volume_ma_24"]) * current_price
                    if volume_value < self.min_market_liquidity:
                        return mask

                mask[self.ACTION_BUY] = True

        return mask

    # ------------------------------------------------------------------
    # Auto-Exit Logic
    # ------------------------------------------------------------------

    def _check_auto_exit(self, position: Dict, current_price: float) -> Optional[str]:
        """Check stop-loss / take-profit / max-hold conditions."""
        entry_price = position.get("entry_price", 0.0)
        if entry_price <= 0:
            return None

        price_change = (current_price - entry_price) / entry_price

        if price_change <= -self.stop_loss_pct:
            return "AUTO_STOP_LOSS"
        elif price_change >= self.take_profit_pct:
            return "AUTO_TAKE_PROFIT"
        elif position.get("hold_steps", 0) >= self.max_hold_steps:
            return "AUTO_MAX_HOLD"

        return None

    # ------------------------------------------------------------------
    # Portfolio Tracking
    # ------------------------------------------------------------------

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value including open positions."""
        total = self.capital
        for position in self.positions.values():
            entry_price = position.get("entry_price", 0.0)
            last_price = position.get("last_price", entry_price)
            size = position.get("size", 0.0)
            if entry_price > 0 and last_price > 0:
                total += size * (last_price / entry_price)
            else:
                total += size
        return total

    # ------------------------------------------------------------------
    # Trade Execution
    # ------------------------------------------------------------------

    def _estimate_slippage_pct(self, order_value: float, row: Optional[pd.Series]) -> float:
        if row is None:
            return self.max_slippage_pct * 0.5
        if self.slippage_model == "volume_based":
            volume_ma_24 = float(row.get("volume_ma_24", 0.0))
            current_price = float(row.get("close", 0.0))
            volume_value = max(volume_ma_24 * current_price, 1e-9)
            return (order_value / volume_value) * self.max_slippage_pct
        return self.max_slippage_pct * 0.5

    def _get_trade_cost_pct(self, order_value: float, portfolio_value: float, row: Optional[pd.Series]) -> float:
        cost_pct = self.base_transaction_cost
        if portfolio_value > 0:
            order_pct = order_value / portfolio_value
            if order_pct > self.large_order_penalty_threshold:
                cost_pct += self.large_order_penalty_pct
        slippage_pct = min(self.max_slippage_pct, self._estimate_slippage_pct(order_value, row))
        cost_pct += slippage_pct
        return cost_pct

    def _update_adaptive_kelly(self, trade_pnl: float) -> None:
        if self.initial_capital <= 0:
            return
        self.adaptive_breaker.update(trade_pnl / self.initial_capital)
        adjusted_fraction = self.base_kelly_fraction * self.adaptive_breaker.current_multiplier
        adjusted_fraction = min(max(adjusted_fraction, 0.15), 0.35)
        self.position_sizer.set_kelly_fraction(adjusted_fraction)

    def _execute_buy(self, symbol: str, current_price: float, row: Optional[pd.Series]) -> Dict:
        """Execute a paper buy order for a specific symbol."""
        if symbol in self.positions:
            return {"executed": False, "reason": "Position already open"}
        if len(self.positions) >= self.settings.MAX_OPEN_POSITIONS:
            return {"executed": False, "reason": "Max positions reached"}

        available_capital = self.capital
        if available_capital <= 0:
            return {"executed": False, "reason": "Insufficient capital"}

        if len(self.recent_trades) >= self.min_trades_for_kelly:
            sizing = self.position_sizer.calculate_from_recent_trades(
                capital=available_capital,
                recent_trades=self.recent_trades,
            )
            base_size = sizing.get("suggested_size", available_capital * self.default_position_size_pct)
        else:
            base_size = available_capital * self.default_position_size_pct

        min_position_value = available_capital * self.min_position_size_pct
        position_value = max(base_size, min_position_value)
        position_value = min(position_value, available_capital * self.max_position_size_pct)

        raw_slippage = self._estimate_slippage_pct(position_value, row)
        if raw_slippage > self.max_slippage_pct:
            return {"executed": False, "reason": "Slippage too high"}

        cost_pct = self._get_trade_cost_pct(position_value, self._get_portfolio_value(), row)
        total_cost = position_value * (1 + cost_pct)

        if total_cost > available_capital:
            return {"executed": False, "reason": "Insufficient capital"}

        transaction_cost = position_value * cost_pct
        self.capital -= total_cost

        self.positions[symbol] = {
            "size": position_value,
            "entry_price": current_price,
            "entry_time": datetime.utcnow().isoformat(),
            "entry_tick": self.tick_count,
            "hold_steps": 0,
            "last_price": current_price,
            "last_row": row,
        }

        return {
            "executed": True,
            "action": "BUY",
            "symbol": symbol,
            "price": current_price,
            "size": position_value,
            "cost": transaction_cost,
            "capital_after": self.capital,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _execute_sell(
        self,
        symbol: str,
        current_price: float,
        reason: str = "MODEL_DECISION",
        row: Optional[pd.Series] = None,
    ) -> Dict:
        """Execute a paper sell order for a specific symbol."""
        if symbol not in self.positions:
            return {"executed": False, "reason": "No position to close"}

        position = self.positions[symbol]
        sell_size = position["size"]
        price_change = (current_price - position["entry_price"]) / position["entry_price"]
        pnl_before_costs = sell_size * price_change

        cost_pct = self._get_trade_cost_pct(sell_size, self._get_portfolio_value(), row)
        transaction_cost = sell_size * cost_pct
        pnl = pnl_before_costs - transaction_cost

        self.capital += sell_size + pnl

        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
        if pnl > 0:
            self.win_count += 1
        self.trade_count += 1
        self._update_adaptive_kelly(pnl)

        trade_record = {
            "executed": True,
            "action": "SELL",
            "reason": reason,
            "symbol": symbol,
            "entry_price": position["entry_price"],
            "exit_price": current_price,
            "size": sell_size,
            "pnl": round(pnl, 4),
            "pnl_pct": round(price_change * 100, 4),
            "cost": round(transaction_cost, 4),
            "hold_steps": position.get("hold_steps", 0),
            "capital_after": round(self.capital, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.trade_history.append(trade_record)

        del self.positions[symbol]
        self.symbol_cooldowns[symbol] = 0

        return trade_record

    # ------------------------------------------------------------------
    # Per-Symbol Evaluation
    # ------------------------------------------------------------------

    def _evaluate_symbol(self, symbol: str) -> Optional[Dict]:
        """Fetch data, compute features, and get model prediction for one symbol.

        Returns a dict with the evaluation result, or None if data fetch failed.
        """
        try:
            raw_df = self._fetch_candle_dataframe(symbol)
            featured_df = self._compute_features(raw_df)

            if featured_df.empty:
                return None

            latest_row = featured_df.iloc[-1]
            current_price = float(latest_row["close"])

            if symbol in self.positions:
                self.positions[symbol]["last_price"] = current_price
                self.positions[symbol]["last_row"] = latest_row

            obs = self._build_observation(symbol, latest_row, current_price)
            action_mask = self._get_action_mask(symbol, latest_row)

            action_raw, _ = self.model.predict(
                obs.reshape(1, -1),
                deterministic=True,
                action_masks=action_mask.reshape(1, -1),
            )
            action = int(action_raw.item()) if hasattr(action_raw, "item") else int(action_raw)

            # Clear error count on success
            self._symbol_errors.pop(symbol, None)

            return {
                "symbol": symbol,
                "price": current_price,
                "action": action,
                "timestamp": str(latest_row["timestamp"]),
                "row": latest_row,
            }

        except Exception as e:
            count = self._symbol_errors.get(symbol, 0) + 1
            self._symbol_errors[symbol] = count
            if count <= 2:
                logger.warning(f"Failed to evaluate {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Main Tick Logic
    # ------------------------------------------------------------------

    def tick(self) -> Dict:
        """
        Process one trading tick on the current symbol.

        - Auto-exit checks run across all open positions.
        - The model acts only on the current symbol (rotating each tick).
        """
        self.tick_count += 1
        forced_results: List[Dict] = []

        # Auto-exit across all open positions
        for symbol, position in list(self.positions.items()):
            try:
                current_price = self.coinbase.get_latest_price(symbol)
            except Exception:
                continue
            if current_price <= 0:
                continue
            position["last_price"] = current_price
            auto_exit = self._check_auto_exit(position, current_price)
            if auto_exit:
                trade_result = self._execute_sell(
                    symbol,
                    current_price,
                    reason=auto_exit,
                    row=position.get("last_row"),
                )
                if trade_result.get("executed"):
                    forced_results.append(trade_result)

        # Evaluate current symbol only
        if not self.symbols:
            self._advance_time()
            return {"tick": self.tick_count, "error": "No symbols configured"}

        current_symbol = self.symbols[self.current_symbol_index % len(self.symbols)]
        eval_result = self._evaluate_symbol(current_symbol)
        if eval_result is None:
            self._advance_time()
            self._update_tracking()
            return {
                "tick": self.tick_count,
                "error": f"Failed to fetch data for symbol {current_symbol}",
            }

        action = eval_result["action"]
        trade_result = None
        if action == self.ACTION_BUY:
            trade_result = self._execute_buy(
                current_symbol,
                eval_result["price"],
                eval_result.get("row"),
            )
        elif action == self.ACTION_SELL:
            position = self.positions.get(current_symbol)
            if position and position.get("hold_steps", 0) >= self.min_hold_steps:
                trade_result = self._execute_sell(
                    current_symbol,
                    eval_result["price"],
                    reason="MODEL_DECISION",
                    row=eval_result.get("row"),
                )

        self._advance_time()
        self._update_tracking()

        action_label = self.ACTION_NAMES.get(action, "?")
        if trade_result and trade_result.get("executed"):
            action_label = trade_result.get("action", action_label)

        result = self._build_tick_result(
            eval_result["timestamp"],
            current_symbol,
            eval_result["price"],
            action_label,
            trade_result,
            forced_results,
        )
        self._log_decision(result)
        return result

    # ------------------------------------------------------------------
    # Tick Result Builder
    # ------------------------------------------------------------------

    def _build_tick_result(
        self,
        timestamp: str,
        symbol: str,
        current_price: float,
        action_label: str,
        trade_result: Optional[Dict],
        forced_exits: Optional[List[Dict]] = None,
    ) -> Dict:
        portfolio_value = self._get_portfolio_value()
        position = self.positions.get(symbol)
        return {
            "tick": self.tick_count,
            "timestamp": timestamp,
            "symbol": symbol,
            "price": current_price,
            "action": action_label,
            "auto_exit": None,
            "forced_exits": forced_exits or [],
            "trade": trade_result if trade_result and trade_result.get("executed") else None,
            "portfolio_value": round(portfolio_value, 2),
            "capital": round(self.capital, 2),
            "position": {
                "active": position is not None,
                "symbol": symbol if position else None,
                "hold_steps": position.get("hold_steps", 0) if position else 0,
                "unrealized_pnl_pct": round(
                    (current_price - position["entry_price"])
                    / position["entry_price"]
                    * 100,
                    3,
                )
                if position and position.get("entry_price")
                else None,
            },
            "open_positions": list(self.positions.keys()),
            "num_positions": len(self.positions),
            "total_return_pct": round(
                (portfolio_value - self.initial_capital) / self.initial_capital * 100, 4
            ),
            "drawdown_pct": round(self.current_drawdown * 100, 4),
            "trades_count": self.trade_count,
            "win_rate": round(self.win_count / max(self.trade_count, 1) * 100, 1),
        }

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _advance_time(self):
        for position in self.positions.values():
            position["hold_steps"] = position.get("hold_steps", 0) + 1

        for symbol in self.symbols:
            self.symbol_cooldowns[symbol] = self.symbol_cooldowns.get(symbol, self.trade_cooldown_steps) + 1

        if self.positions:
            self.flat_steps = 0
        else:
            self.flat_steps += 1

        if self.symbols:
            self.current_symbol_index = (self.current_symbol_index + 1) % len(self.symbols)

    def _update_tracking(self):
        portfolio_value = self._get_portfolio_value()
        self.peak_capital = max(self.peak_capital, portfolio_value)
        self.current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
        self.prev_portfolio_value = portfolio_value

    def _update_tracking_no_price(self):
        """Update tracking when we couldn't get price data."""
        self._advance_time()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_decision(self, result: Dict):
        self.decision_log.append(result)

        log_path = os.path.join(
            self.log_dir,
            f"paper_trade_{datetime.utcnow().strftime('%Y%m%d')}.jsonl",
        )
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                clean = json.loads(json.dumps(result, default=str))
                f.write(json.dumps(clean) + "\n")
        except OSError as e:
            logger.warning(f"Failed to write log: {e}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        portfolio_value = self._get_portfolio_value()
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital

        gross_profit = sum(t["pnl"] for t in self.trade_history if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trade_history if t.get("pnl", 0) < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        trade_pnls = [t.get("pnl", 0) for t in self.trade_history]
        avg_trade_pnl = float(np.mean(trade_pnls)) if trade_pnls else 0.0
        avg_win = float(np.mean([p for p in trade_pnls if p > 0])) if any(p > 0 for p in trade_pnls) else 0.0
        avg_loss = float(np.mean([p for p in trade_pnls if p < 0])) if any(p < 0 for p in trade_pnls) else 0.0

        return {
            "symbols": len(self.symbols),
            "interval": self.interval,
            "ticks": self.tick_count,
            "capital": round(self.capital, 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_return_pct": round(total_return * 100, 4),
            "total_trades": self.trade_count,
            "win_rate": round(self.win_count / max(self.trade_count, 1) * 100, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "avg_trade_pnl": round(avg_trade_pnl, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "max_drawdown_pct": round(self.current_drawdown * 100, 4),
            "open_positions": list(self.positions.keys()),
            "open_position": len(self.positions) > 0,
            "total_fees": round(sum(t.get("cost", 0) for t in self.trade_history), 4),
        }

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def run(self, duration_hours: int = 24, verbose: bool = True) -> Dict:
        if duration_hours > 0:
            end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        else:
            end_time = None

        if verbose:
            self._print_header(duration_hours)

        try:
            while True:
                if end_time and datetime.utcnow() >= end_time:
                    break

                try:
                    result = self.tick()

                    if verbose:
                        self._print_tick(result)

                except Exception as e:
                    logger.error(f"Tick error: {e}")
                    if verbose:
                        print(f"  [ERROR] Tick failed: {e}")

                next_tick = self._next_candle_time()
                sleep_seconds = (next_tick - datetime.utcfromtimestamp(time.time())).total_seconds()

                # Safety cap: never sleep longer than one interval + buffer
                max_sleep = self.interval_seconds + 30
                sleep_seconds = max(10, min(sleep_seconds, max_sleep))

                if verbose:
                    print(
                        f"  Next tick in {sleep_seconds:.0f}s "
                        f"({next_tick.strftime('%H:%M:%S')} UTC)"
                    )
                    print()

                # Chunked sleep: wake every 30s to check if OS suspended us.
                # If laptop slept and woke past the target, we resume immediately.
                target_utc = time.time() + sleep_seconds
                while time.time() < target_utc:
                    remaining = target_utc - time.time()
                    if remaining <= 0:
                        break
                    time.sleep(min(30, remaining))

        except KeyboardInterrupt:
            if verbose:
                print("\n  [INTERRUPTED] Paper trading stopped by user\n")

        metrics = self.get_metrics()

        if verbose:
            self._print_summary(metrics)

        metrics_path = os.path.join(
            self.log_dir,
            f"final_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        )
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)
        except OSError as e:
            logger.warning(f"Failed to write final metrics: {e}")

        return metrics

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _next_candle_time(self) -> datetime:
        """Calculate when the next candle will be available.

        Uses time.time() (always UTC) to avoid the Python timezone trap
        where datetime.utcnow().timestamp() silently applies local TZ offset.
        """
        now_utc = time.time()  # Always UTC, no timezone ambiguity
        seconds = self.interval_seconds
        next_boundary = ((int(now_utc) // seconds) + 1) * seconds
        # Add 10s buffer for Coinbase API lag
        return datetime.utcfromtimestamp(next_boundary + 10)

    # ------------------------------------------------------------------
    # Console Output
    # ------------------------------------------------------------------

    def _print_header(self, duration_hours: int):
        duration_str = f"{duration_hours}h" if duration_hours > 0 else "indefinite"
        print(f"\n{'=' * 65}")
        print(f"  LIVE PAPER TRADING  (RL Model - Multi-Symbol)")
        print(f"{'=' * 65}")
        print(f"  Symbols:     {len(self.symbols)} tracked")
        print(f"               {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}")
        print(f"  Capital:     ${self.initial_capital:,.2f}")
        print(f"  Interval:    {self.interval}")
        print(f"  Duration:    {duration_str}")
        print(f"  Model:       {self.model_path}")
        print(f"  Stop Loss:   {self.stop_loss_pct:.1%}")
        print(f"  Take Profit: {self.take_profit_pct:.1%}")
        print(f"  Position %:  {self.default_position_size_pct:.1%}")
        print(f"  Log Dir:     {self.log_dir}")
        print(f"{'=' * 65}")
        print(f"  Press Ctrl+C to stop at any time")
        print(f"{'=' * 65}\n")

    def _print_tick(self, result: Dict):
        if result.get("error"):
            print(f"  [Tick #{result.get('tick', 0)}] ERROR: {result['error']}")
            return

        action = result.get("action", "?")
        symbol = result.get("symbol", "?")
        price = result.get("price", 0)
        pv = result.get("portfolio_value", 0)
        ret = result.get("total_return_pct", 0)
        dd = result.get("drawdown_pct", 0)
        pos = result.get("position", {})
        num_positions = result.get("num_positions", 0)

        status = "FLAT"
        if pos.get("active"):
            unr = pos.get("unrealized_pnl_pct")
            held = pos.get("symbol", "?")
            unr_str = f"{unr:+.2f}%" if unr is not None else "?"
            status = f"HOLDING {held} (step {pos.get('hold_steps', 0)}, {unr_str})"

        print(f"  [{result.get('timestamp', '')}]  Tick #{result.get('tick', 0)}")
        print(f"    {symbol}: ${price:,.2f}  |  Action: {action}")
        print(
            f"    Portfolio: ${pv:,.2f}  |  Return: {ret:+.2f}%  |  Drawdown: {dd:.2f}%  |  {status}  |  Positions: {num_positions}"
        )

        trade = result.get("trade")
        if trade:
            pnl = trade.get("pnl", 0)
            reason = trade.get("reason", "")
            tsym = trade.get("symbol", "?")
            # BUY uses "price", SELL uses "exit_price"
            trade_price = trade.get("price", 0) or trade.get("exit_price", 0)
            print(f"    >>> TRADE: {trade['action']} {tsym} @ ${trade_price:,.2f}  |  PnL: ${pnl:+.4f}  |  {reason}")

        print(f"    Trades: {result.get('trades_count', 0)}  |  Win Rate: {result.get('win_rate', 0):.0f}%")

    def _print_summary(self, metrics: Dict):
        print(f"\n{'=' * 65}")
        print(f"  PAPER TRADING SUMMARY")
        print(f"{'=' * 65}")
        print(f"  Symbols:        {metrics['symbols']} tracked")
        print(f"  Ticks:          {metrics['ticks']}")
        print(f"  Capital:        ${metrics['capital']:,.2f}")
        print(f"  Portfolio:      ${metrics['portfolio_value']:,.2f}")
        print(f"  Total Return:   {metrics['total_return_pct']:+.2f}%")
        print(f"  Total Trades:   {metrics['total_trades']}")
        print(f"  Win Rate:       {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor:  {metrics['profit_factor']}")
        print(f"  Avg Trade PnL:  ${metrics['avg_trade_pnl']:.4f}")
        print(f"  Max Drawdown:   {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Total Fees:     ${metrics['total_fees']:.4f}")
        open_positions = metrics.get("open_positions", [])
        print(f"  Open Positions: {', '.join(open_positions) if open_positions else 'None'}")
        print(f"{'=' * 65}\n")

    # ------------------------------------------------------------------
    # Config Loaders
    # ------------------------------------------------------------------

    def _load_reward_config(self) -> Dict:
        config_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "reward_config.yaml"
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("reward_weights", {})
        except FileNotFoundError:
            return {}

    def _load_risk_config(self) -> Dict:
        config_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "risk_config.yaml"
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        mapping = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "4h": 14400, "6h": 21600, "1d": 86400,
        }
        return mapping.get(interval, 3600)
