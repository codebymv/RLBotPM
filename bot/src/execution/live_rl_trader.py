"""
Live RL Paper Trader - Runs the trained RL model on real-time Coinbase data.

This module provides a live paper trading loop that:
1. Fetches real-time OHLCV candles from the Coinbase public API
2. Computes the EXACT same features used during training
3. Builds 42-dim observation vectors matching the training environment
4. Gets action predictions from the trained MaskablePPO model
5. Simulates trade execution with paper portfolio tracking
6. Logs every decision and trade to JSONL files for analysis

Usage (via CLI):
    python main.py rl-paper-trade --model models/best_model_run_73_step_955500 --symbol BTC-USD

Architecture:
    Coinbase API -> Feature Computation -> Observation Builder -> MaskablePPO -> Paper Portfolio
    (identical pipeline to CryptoTradingEnv, just with live data instead of historical replay)
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
from typing import Dict, List, Optional, Tuple

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ..data.sources.coinbase import CoinbaseAdapter
from ..environment.gym_env import CryptoTradingEnv, _interval_to_steps
from ..core.logger import get_logger
from ..core.config import get_settings


logger = get_logger(__name__)


class LiveRLPaperTrader:
    """
    Live paper trading engine powered by the trained RL model.

    Replicates CryptoTradingEnv's observation pipeline exactly,
    but feeds live Coinbase candles instead of historical replay.
    Portfolio state is tracked independently (not tied to env episodes).
    """

    ACTION_NO_ACTION = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(
        self,
        model_path: str,
        symbol: str = "BTC-USD",
        initial_capital: float = 1000.0,
        interval: str = "1h",
        log_dir: str = "./logs/paper_trading",
    ):
        self.model_path = model_path
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.interval = interval
        self.log_dir = log_dir

        self.settings = get_settings()
        self.coinbase = CoinbaseAdapter()

        # Load configs (same files the training env reads)
        self.reward_config = self._load_reward_config()
        self.risk_config = self._load_risk_config()

        # Trading parameters from config
        self.min_hold_steps = int(self.reward_config.get("min_hold_steps", 5))
        self.max_hold_steps = int(self.reward_config.get("max_hold_steps", 100))
        self.trade_cooldown_steps = int(self.reward_config.get("trade_cooldown_steps", 3))
        self.stop_loss_pct = float(self.reward_config.get("stop_loss_pct", 0.03))
        self.take_profit_pct = float(self.reward_config.get("take_profit_pct", 0.05))
        self.default_position_size_pct = float(
            self.risk_config.get("position_sizing", {}).get("default_position_size_pct", 0.05)
        )

        # Transaction costs
        tx_costs = self.risk_config.get("transaction_costs", {})
        self.base_transaction_cost = float(
            tx_costs.get("base_cost_pct", self.settings.TRANSACTION_COST_PCT)
        )
        self.max_slippage_pct = float(
            self.risk_config.get("market_requirements", {}).get("max_slippage_pct", 0.02)
        )

        # Portfolio state
        self.position: Optional[Dict] = None
        self.position_hold_steps = 0
        self.steps_since_last_close = self.trade_cooldown_steps  # Ready to trade
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

        # Load model
        self._load_model()

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        logger.info("LiveRLPaperTrader initialized")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Capital: ${initial_capital:.2f}")
        logger.info(f"  Interval: {interval}")
        logger.info(f"  Model: {model_path}")

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the trained MaskablePPO model.

        Creates a temporary CryptoTradingEnv with live data from Coinbase
        so that SB3 can validate observation/action spaces during load.
        """
        candles_df = self._fetch_candle_dataframe()
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

    def _fetch_candle_dataframe(self, num_candles: int = 300) -> pd.DataFrame:
        """Fetch recent candles from Coinbase and convert to DataFrame.

        Coinbase returns at most 300 candles per request.
        For 1h interval this gives ~12.5 days, enough for 200h MA warmup.
        """
        end = datetime.utcnow()
        start = end - timedelta(seconds=self.interval_seconds * num_candles)

        candles = self.coinbase.get_ohlcv(
            symbol=self.symbol,
            interval=self.interval,
            start=start,
            end=end,
        )

        if not candles:
            raise RuntimeError(f"No candles returned for {self.symbol}")

        rows = [
            {
                "symbol": self.symbol,
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

        # Price-based features
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

        # Technical indicators (reuse CryptoTradingEnv static methods)
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

    def _build_observation(self, row: pd.Series) -> np.ndarray:
        """Build 42-dim observation vector EXACTLY matching CryptoTradingEnv._get_observation()."""
        obs = np.zeros(42, dtype=np.float32)

        current_price = float(row["close"])

        # --- Market features (0-11) ---
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

        # --- Portfolio features (12-19) ---
        portfolio_value = self._get_portfolio_value(current_price)
        obs[12] = self.capital / self.initial_capital
        obs[13] = (1.0 if self.position else 0.0) / self.settings.MAX_OPEN_POSITIONS
        obs[14] = portfolio_value / self.initial_capital
        obs[15] = (portfolio_value - self.initial_capital) / self.initial_capital
        obs[16] = sum(self.recent_trades) / self.initial_capital if self.recent_trades else 0.0
        obs[17] = (self.position["size"] if self.position else 0.0) / self.initial_capital
        obs[18] = self.win_count / max(self.trade_count, 1)
        obs[19] = self.current_drawdown

        # --- Time features (20-23) ---
        ts = pd.to_datetime(row["timestamp"])
        obs[20] = np.sin(2 * np.pi * ts.hour / 24)
        obs[21] = np.cos(2 * np.pi * ts.hour / 24)
        obs[22] = np.sin(2 * np.pi * ts.weekday() / 7)
        obs[23] = np.cos(2 * np.pi * ts.weekday() / 7)

        # --- Momentum ratios (24-27) ---
        volatility = abs(float(row["volatility_24h"]))
        obs[24] = float(row["return_6h"]) / (volatility + 1e-6)
        obs[25] = float(row["return_1h"]) / (volatility + 1e-6)
        obs[26] = float(row["return_24h"]) / (volatility + 1e-6)
        obs[27] = float(row["price_zscore_24h"])

        # --- Technical indicators (28-35) ---
        obs[28] = float(row["rsi_14"])
        obs[29] = float(row["macd_line"]) * 100.0
        obs[30] = float(row["macd_signal"]) * 100.0
        obs[31] = float(row["macd_hist"]) * 100.0
        obs[32] = float(row["atr_14"]) * 10.0
        obs[33] = float(row["bb_upper"]) - 1.0
        obs[34] = float(row["bb_middle"]) - 1.0
        obs[35] = float(row["bb_lower"]) - 1.0

        # --- Position-level features (36-38) ---
        if self.position:
            entry_price = self.position["entry_price"]
            obs[36] = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            obs[37] = self.position_hold_steps / self.max_hold_steps
            obs[38] = ((current_price - entry_price) / entry_price) / (volatility + 1e-6)
        else:
            obs[36] = 0.0
            obs[37] = 0.0
            obs[38] = 0.0

        # --- Long-term trend context (39-41) ---
        ma_200 = float(row["ma_200"])
        obs[39] = (current_price - ma_200) / (ma_200 + 1e-9)
        obs[40] = float(row["ma_200_slope"])
        volume_ma_200 = float(row["volume_ma_200"])
        obs[41] = float(row["volume"]) / max(volume_ma_200, 1.0)

        return obs

    # ------------------------------------------------------------------
    # Action Masking  (mirrors CryptoTradingEnv.action_masks)
    # ------------------------------------------------------------------

    def _get_action_mask(self) -> np.ndarray:
        """Get valid action mask matching CryptoTradingEnv.action_masks()."""
        mask = np.zeros(3, dtype=bool)
        mask[self.ACTION_NO_ACTION] = True  # Can always hold

        if self.position:
            if self.position_hold_steps >= self.min_hold_steps:
                mask[self.ACTION_SELL] = True
        else:
            if self.steps_since_last_close >= self.trade_cooldown_steps:
                mask[self.ACTION_BUY] = True

        return mask

    # ------------------------------------------------------------------
    # Auto-Exit Logic  (mirrors CryptoTradingEnv.step auto-exit)
    # ------------------------------------------------------------------

    def _check_auto_exit(self, current_price: float) -> Optional[str]:
        """Check stop-loss / take-profit / max-hold conditions."""
        if not self.position:
            return None

        price_change = (current_price - self.position["entry_price"]) / self.position["entry_price"]

        if price_change <= -self.stop_loss_pct:
            return "AUTO_STOP_LOSS"
        elif price_change >= self.take_profit_pct:
            return "AUTO_TAKE_PROFIT"
        elif self.position_hold_steps >= self.max_hold_steps:
            return "AUTO_MAX_HOLD"

        return None

    # ------------------------------------------------------------------
    # Portfolio Tracking
    # ------------------------------------------------------------------

    def _get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value including open position."""
        total = self.capital
        if self.position:
            entry_price = self.position["entry_price"]
            if entry_price > 0:
                total += self.position["size"] * (current_price / entry_price)
            else:
                total += self.position["size"]
        return total

    # ------------------------------------------------------------------
    # Trade Execution
    # ------------------------------------------------------------------

    def _execute_buy(self, current_price: float) -> Dict:
        """Execute a paper buy order."""
        if self.position:
            return {"executed": False, "reason": "Position already open"}

        position_value = self.capital * self.default_position_size_pct
        cost_pct = self.base_transaction_cost + self.max_slippage_pct * 0.5
        total_cost = position_value * (1 + cost_pct)

        if total_cost > self.capital:
            return {"executed": False, "reason": "Insufficient capital"}

        transaction_cost = position_value * cost_pct
        self.capital -= total_cost

        self.position = {
            "size": position_value,
            "entry_price": current_price,
            "entry_time": datetime.utcnow().isoformat(),
            "entry_tick": self.tick_count,
        }
        self.position_hold_steps = 0

        return {
            "executed": True,
            "action": "BUY",
            "symbol": self.symbol,
            "price": current_price,
            "size": position_value,
            "cost": transaction_cost,
            "capital_after": self.capital,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _execute_sell(self, current_price: float, reason: str = "MODEL_DECISION") -> Dict:
        """Execute a paper sell order."""
        if not self.position:
            return {"executed": False, "reason": "No position to close"}

        position = self.position
        sell_size = position["size"]
        price_change = (current_price - position["entry_price"]) / position["entry_price"]
        pnl_before_costs = sell_size * price_change

        cost_pct = self.base_transaction_cost + self.max_slippage_pct * 0.5
        transaction_cost = sell_size * cost_pct
        pnl = pnl_before_costs - transaction_cost

        self.capital += sell_size + pnl

        # Update stats
        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
        if pnl > 0:
            self.win_count += 1
        self.trade_count += 1

        trade_record = {
            "executed": True,
            "action": "SELL",
            "reason": reason,
            "symbol": self.symbol,
            "entry_price": position["entry_price"],
            "exit_price": current_price,
            "size": sell_size,
            "pnl": round(pnl, 4),
            "pnl_pct": round(price_change * 100, 4),
            "cost": round(transaction_cost, 4),
            "hold_steps": self.position_hold_steps,
            "capital_after": round(self.capital, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.trade_history.append(trade_record)

        # Clear position
        self.position = None
        self.position_hold_steps = 0
        self.steps_since_last_close = 0  # Start cooldown

        return trade_record

    # ------------------------------------------------------------------
    # Main Tick Logic
    # ------------------------------------------------------------------

    def tick(self) -> Dict:
        """
        Process one trading tick:
        1. Fetch latest candles from Coinbase
        2. Compute features
        3. Check auto-exits
        4. Build observation
        5. Get model prediction
        6. Execute action
        7. Update tracking
        8. Log decision

        Returns dict with full tick details.
        """
        self.tick_count += 1

        # 1. Fetch and compute features
        raw_df = self._fetch_candle_dataframe()
        featured_df = self._compute_features(raw_df)

        if featured_df.empty:
            return {"tick": self.tick_count, "error": "No data after feature computation"}

        # 2. Get latest data point
        latest_row = featured_df.iloc[-1]
        current_price = float(latest_row["close"])
        timestamp = str(latest_row["timestamp"])

        # 3. Check auto-exits first
        auto_exit = self._check_auto_exit(current_price)
        if auto_exit:
            trade_result = self._execute_sell(current_price, reason=auto_exit)
            self._update_tracking(current_price)
            result = self._build_tick_result(
                timestamp, current_price, "SELL (auto)", auto_exit, trade_result
            )
            self._log_decision(result)
            return result

        # 4. Build observation
        obs = self._build_observation(latest_row)

        # 5. Get action mask
        action_mask = self._get_action_mask()

        # 6. Get model prediction
        action_raw, _ = self.model.predict(
            obs.reshape(1, -1),
            deterministic=True,
            action_masks=action_mask.reshape(1, -1),
        )
        # model.predict returns ndarray for batched input; extract scalar
        action = int(action_raw.item()) if hasattr(action_raw, "item") else int(action_raw)

        # 7. Execute action
        trade_result = None
        if action == self.ACTION_BUY:
            trade_result = self._execute_buy(current_price)
        elif action == self.ACTION_SELL:
            trade_result = self._execute_sell(current_price, reason="MODEL_DECISION")

        # 8. Update tracking
        self._update_tracking(current_price)

        # 9. Build and log result
        action_label = self.ACTION_NAMES[action]
        if trade_result and trade_result.get("executed"):
            action_label = trade_result["action"]
        result = self._build_tick_result(
            timestamp, current_price, action_label, None, trade_result
        )
        self._log_decision(result)
        return result

    def _build_tick_result(
        self,
        timestamp: str,
        current_price: float,
        action_label: str,
        auto_exit_reason: Optional[str],
        trade_result: Optional[Dict],
    ) -> Dict:
        """Build standardized tick result dict."""
        portfolio_value = self._get_portfolio_value(current_price)
        return {
            "tick": self.tick_count,
            "timestamp": timestamp,
            "price": current_price,
            "action": action_label,
            "auto_exit": auto_exit_reason,
            "trade": trade_result if trade_result and trade_result.get("executed") else None,
            "portfolio_value": round(portfolio_value, 2),
            "capital": round(self.capital, 2),
            "position": {
                "active": self.position is not None,
                "hold_steps": self.position_hold_steps,
                "unrealized_pnl_pct": round(
                    (current_price - self.position["entry_price"])
                    / self.position["entry_price"]
                    * 100,
                    3,
                )
                if self.position
                else None,
            },
            "total_return_pct": round(
                (portfolio_value - self.initial_capital) / self.initial_capital * 100, 4
            ),
            "drawdown_pct": round(self.current_drawdown * 100, 4),
            "trades_count": self.trade_count,
            "win_rate": round(self.win_count / max(self.trade_count, 1) * 100, 1),
        }

    def _update_tracking(self, current_price: float):
        """Update portfolio tracking after each tick."""
        portfolio_value = self._get_portfolio_value(current_price)
        self.peak_capital = max(self.peak_capital, portfolio_value)
        self.current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
        self.prev_portfolio_value = portfolio_value

        if self.position:
            self.position_hold_steps += 1
            self.flat_steps = 0
        else:
            self.steps_since_last_close += 1
            self.flat_steps += 1

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_decision(self, result: Dict):
        """Append decision to in-memory log and write to JSONL file."""
        self.decision_log.append(result)

        log_path = os.path.join(
            self.log_dir,
            f"paper_trade_{self.symbol}_{datetime.utcnow().strftime('%Y%m%d')}.jsonl",
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
        """Get comprehensive performance metrics."""
        # Try to get current price for portfolio valuation
        current_price = 0.0
        try:
            current_price = self.coinbase.get_latest_price(self.symbol)
        except Exception:
            if self.decision_log:
                current_price = self.decision_log[-1].get("price", 0)

        portfolio_value = self._get_portfolio_value(current_price)
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital

        # Profit factor
        gross_profit = sum(t["pnl"] for t in self.trade_history if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trade_history if t.get("pnl", 0) < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        # Average trade metrics
        trade_pnls = [t.get("pnl", 0) for t in self.trade_history]
        avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0.0
        avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0.0
        avg_loss = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0.0

        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "ticks": self.tick_count,
            "capital": round(self.capital, 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_return_pct": round(total_return * 100, 4),
            "total_trades": self.trade_count,
            "win_rate": round(self.win_count / max(self.trade_count, 1) * 100, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "avg_trade_pnl": round(float(avg_trade_pnl), 4),
            "avg_win": round(float(avg_win), 4),
            "avg_loss": round(float(avg_loss), 4),
            "max_drawdown_pct": round(self.current_drawdown * 100, 4),
            "open_position": self.position is not None,
            "total_fees": round(sum(t.get("cost", 0) for t in self.trade_history), 4),
        }

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def run(self, duration_hours: int = 24, verbose: bool = True) -> Dict:
        """
        Main trading loop - runs for specified duration.

        Each tick:
          1. Waits for the next candle interval boundary
          2. Fetches latest candles from Coinbase
          3. Computes features & builds observation
          4. Gets model action prediction
          5. Executes paper trade if applicable
          6. Logs and displays results

        Args:
            duration_hours: How long to run (hours). 0 = run indefinitely.
            verbose: Print real-time console updates.

        Returns:
            Final performance metrics dict.
        """
        if duration_hours > 0:
            end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        else:
            end_time = None  # Run indefinitely

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

                # Wait for next candle
                next_tick = self._next_candle_time()
                sleep_seconds = max(10, (next_tick - datetime.utcnow()).total_seconds())

                if verbose:
                    print(
                        f"  Next tick in {sleep_seconds:.0f}s "
                        f"({next_tick.strftime('%H:%M:%S')} UTC)"
                    )
                    print()

                time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            if verbose:
                print("\n  [INTERRUPTED] Paper trading stopped by user\n")

        metrics = self.get_metrics()

        if verbose:
            self._print_summary(metrics)

        # Save final metrics
        metrics_path = os.path.join(
            self.log_dir,
            f"final_metrics_{self.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
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
        """Calculate when the next candle will be available."""
        now = datetime.utcnow()
        seconds = self.interval_seconds
        # Round up to next interval boundary + buffer for API lag
        next_boundary = datetime.utcfromtimestamp(
            (int(now.timestamp()) // seconds + 1) * seconds
        )
        return next_boundary + timedelta(seconds=10)

    # ------------------------------------------------------------------
    # Console Output
    # ------------------------------------------------------------------

    def _print_header(self, duration_hours: int):
        duration_str = f"{duration_hours}h" if duration_hours > 0 else "indefinite"
        print(f"\n{'=' * 65}")
        print(f"  LIVE PAPER TRADING  (RL Model)")
        print(f"{'=' * 65}")
        print(f"  Symbol:      {self.symbol}")
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
        action = result.get("action", "?")
        price = result.get("price", 0)
        pv = result.get("portfolio_value", 0)
        ret = result.get("total_return_pct", 0)
        dd = result.get("drawdown_pct", 0)
        pos = result.get("position", {})

        status = "FLAT"
        if pos.get("active"):
            unr = pos.get("unrealized_pnl_pct")
            unr_str = f"{unr:+.2f}%" if unr is not None else "?"
            status = f"IN POSITION (hold: {pos.get('hold_steps', 0)}, unrealized: {unr_str})"

        print(f"  [{result.get('timestamp', '')}]  Tick #{result.get('tick', 0)}")
        print(f"    Price: ${price:,.2f}  |  Action: {action}  |  {status}")
        print(f"    Portfolio: ${pv:,.2f}  |  Return: {ret:+.2f}%  |  Drawdown: {dd:.2f}%")

        trade = result.get("trade")
        if trade:
            pnl = trade.get("pnl", 0)
            reason = trade.get("reason", "")
            print(f"    >>> TRADE: {trade.get('action', '')}  |  PnL: ${pnl:+.4f}  |  Reason: {reason}")

        trades_count = result.get("trades_count", 0)
        win_rate = result.get("win_rate", 0)
        print(f"    Trades: {trades_count}  |  Win Rate: {win_rate:.0f}%")

    def _print_summary(self, metrics: Dict):
        print(f"\n{'=' * 65}")
        print(f"  PAPER TRADING SUMMARY")
        print(f"{'=' * 65}")
        print(f"  Symbol:         {metrics['symbol']}")
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
        print(f"  Open Position:  {'Yes' if metrics['open_position'] else 'No'}")
        print(f"{'=' * 65}\n")

    # ------------------------------------------------------------------
    # Config Loaders
    # ------------------------------------------------------------------

    def _load_reward_config(self) -> Dict:
        """Load reward config (same file as CryptoTradingEnv reads)."""
        config_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "reward_config.yaml"
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("reward_weights", {})
        except FileNotFoundError:
            logger.warning("reward_config.yaml not found, using defaults")
            return {}

    def _load_risk_config(self) -> Dict:
        """Load risk config (same file as CryptoTradingEnv reads)."""
        config_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "risk_config.yaml"
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("risk_config.yaml not found, using defaults")
            return {}

    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "6h": 21600,
            "1d": 86400,
        }
        return mapping.get(interval, 3600)
