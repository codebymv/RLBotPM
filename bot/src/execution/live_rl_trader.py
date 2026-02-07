"""
Live RL Paper Trader - Runs the trained RL model on real-time Coinbase data.

Scans ALL configured symbols each tick for trading opportunities.
When flat, evaluates every symbol looking for BUY signals.
When in a position, monitors only the held symbol for SELL/auto-exit.

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


logger = get_logger(__name__)


class LiveRLPaperTrader:
    """
    Live paper trading engine powered by the trained RL model.

    Scans multiple symbols per tick for maximum opportunity coverage.
    Holds one position at a time (matching training behavior).
    """

    ACTION_NO_ACTION = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

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
        self.default_position_size_pct = float(
            self.risk_config.get("position_sizing", {}).get("default_position_size_pct", 0.05)
        )
        # Override for paper trading: use 20% position sizing for meaningful results
        self.default_position_size_pct = 0.20

        # Transaction costs
        tx_costs = self.risk_config.get("transaction_costs", {})
        self.base_transaction_cost = float(
            tx_costs.get("base_cost_pct", self.settings.TRANSACTION_COST_PCT)
        )
        self.max_slippage_pct = float(
            self.risk_config.get("market_requirements", {}).get("max_slippage_pct", 0.02)
        )

        # Portfolio state  (one position at a time, matching training)
        self.position: Optional[Dict] = None
        self.held_symbol: Optional[str] = None
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

    def _build_observation(self, row: pd.Series, current_price: float) -> np.ndarray:
        """Build 42-dim observation vector matching CryptoTradingEnv._get_observation()."""
        obs = np.zeros(42, dtype=np.float32)

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

    def _get_action_mask(self, evaluating_held_symbol: bool = False) -> np.ndarray:
        """Get valid action mask.

        Args:
            evaluating_held_symbol: True when we're evaluating the symbol we hold.
        """
        mask = np.zeros(3, dtype=bool)
        mask[self.ACTION_NO_ACTION] = True

        if self.position:
            # We only allow SELL when checking the held symbol
            if evaluating_held_symbol and self.position_hold_steps >= self.min_hold_steps:
                mask[self.ACTION_SELL] = True
        else:
            if self.steps_since_last_close >= self.trade_cooldown_steps:
                mask[self.ACTION_BUY] = True

        return mask

    # ------------------------------------------------------------------
    # Auto-Exit Logic
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

    def _execute_buy(self, symbol: str, current_price: float) -> Dict:
        """Execute a paper buy order for a specific symbol."""
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
        self.held_symbol = symbol
        self.position_hold_steps = 0

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

    def _execute_sell(self, current_price: float, reason: str = "MODEL_DECISION") -> Dict:
        """Execute a paper sell order for the held position."""
        if not self.position:
            return {"executed": False, "reason": "No position to close"}

        position = self.position
        symbol = self.held_symbol
        sell_size = position["size"]
        price_change = (current_price - position["entry_price"]) / position["entry_price"]
        pnl_before_costs = sell_size * price_change

        cost_pct = self.base_transaction_cost + self.max_slippage_pct * 0.5
        transaction_cost = sell_size * cost_pct
        pnl = pnl_before_costs - transaction_cost

        self.capital += sell_size + pnl

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
            "symbol": symbol,
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

        self.position = None
        self.held_symbol = None
        self.position_hold_steps = 0
        self.steps_since_last_close = 0

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

            obs = self._build_observation(latest_row, current_price)
            is_held = (self.held_symbol == symbol)
            action_mask = self._get_action_mask(evaluating_held_symbol=is_held)

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
        Process one trading tick across all symbols.

        When IN POSITION:
          - Check held symbol for auto-exit
          - Evaluate held symbol for SELL signal

        When FLAT:
          - Shuffle symbols for fairness
          - Evaluate each until one triggers BUY
          - If none trigger, report HOLD

        Returns dict with full tick details.
        """
        self.tick_count += 1
        scan_results: List[Dict] = []

        # ------- IN POSITION: only check the held symbol -------
        if self.position and self.held_symbol:
            eval_result = self._evaluate_symbol(self.held_symbol)
            if eval_result is None:
                # Data fetch failed, skip this tick
                self._update_tracking_no_price()
                return {
                    "tick": self.tick_count,
                    "error": f"Failed to fetch data for held symbol {self.held_symbol}",
                }

            current_price = eval_result["price"]
            symbol = eval_result["symbol"]
            timestamp = eval_result["timestamp"]

            # Check auto-exit
            auto_exit = self._check_auto_exit(current_price)
            if auto_exit:
                trade_result = self._execute_sell(current_price, reason=auto_exit)
                self._update_tracking(current_price)
                result = self._build_tick_result(
                    timestamp, symbol, current_price, "SELL (auto)", trade_result
                )
                self._log_decision(result)
                return result

            # Model decision
            action = eval_result["action"]
            trade_result = None
            if action == self.ACTION_SELL:
                trade_result = self._execute_sell(current_price, reason="MODEL_DECISION")

            self._update_tracking(current_price)
            action_label = self.ACTION_NAMES.get(action, "?")
            if trade_result and trade_result.get("executed"):
                action_label = "SELL"
            result = self._build_tick_result(
                timestamp, symbol, current_price, action_label, trade_result
            )
            self._log_decision(result)
            return result

        # ------- FLAT: scan all symbols for BUY opportunity -------
        order = list(range(len(self.symbols)))
        np.random.shuffle(order)

        buy_triggered = False
        last_timestamp = ""
        symbols_checked = 0

        for idx in order:
            symbol = self.symbols[idx]

            # Skip symbols with repeated errors (retry every 10 ticks)
            if self._symbol_errors.get(symbol, 0) >= 3:
                if self.tick_count % 10 != 0:
                    continue

            eval_result = self._evaluate_symbol(symbol)
            if eval_result is None:
                continue

            symbols_checked += 1
            last_timestamp = eval_result["timestamp"]
            scan_results.append(eval_result)

            if eval_result["action"] == self.ACTION_BUY:
                # Execute buy on this symbol
                trade_result = self._execute_buy(symbol, eval_result["price"])
                if trade_result.get("executed"):
                    self._update_tracking(eval_result["price"])
                    result = self._build_tick_result(
                        eval_result["timestamp"],
                        symbol,
                        eval_result["price"],
                        "BUY",
                        trade_result,
                    )
                    result["symbols_scanned"] = symbols_checked
                    self._log_decision(result)
                    return result
                buy_triggered = True  # Tried but failed (e.g. insufficient capital)

            # Small delay to respect Coinbase rate limits (~10 req/sec)
            time.sleep(0.12)

        # No BUY triggered across all symbols
        self._update_tracking_flat()
        result = {
            "tick": self.tick_count,
            "timestamp": last_timestamp,
            "symbol": "ALL",
            "price": 0.0,
            "action": "HOLD (scanned all)",
            "auto_exit": None,
            "trade": None,
            "portfolio_value": round(self.capital, 2),
            "capital": round(self.capital, 2),
            "position": {"active": False, "hold_steps": 0, "unrealized_pnl_pct": None},
            "total_return_pct": round(
                (self.capital - self.initial_capital) / self.initial_capital * 100, 4
            ),
            "drawdown_pct": round(self.current_drawdown * 100, 4),
            "trades_count": self.trade_count,
            "win_rate": round(self.win_count / max(self.trade_count, 1) * 100, 1),
            "symbols_scanned": symbols_checked,
        }
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
    ) -> Dict:
        portfolio_value = self._get_portfolio_value(current_price)
        return {
            "tick": self.tick_count,
            "timestamp": timestamp,
            "symbol": symbol,
            "price": current_price,
            "action": action_label,
            "auto_exit": None,
            "trade": trade_result if trade_result and trade_result.get("executed") else None,
            "portfolio_value": round(portfolio_value, 2),
            "capital": round(self.capital, 2),
            "position": {
                "active": self.position is not None,
                "symbol": self.held_symbol,
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

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _update_tracking(self, current_price: float):
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

    def _update_tracking_flat(self):
        """Update tracking when flat (no price reference needed)."""
        self.steps_since_last_close += 1
        self.flat_steps += 1

    def _update_tracking_no_price(self):
        """Update tracking when we couldn't get price data."""
        if self.position:
            self.position_hold_steps += 1

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
        current_price = 0.0
        if self.held_symbol:
            try:
                current_price = self.coinbase.get_latest_price(self.held_symbol)
            except Exception:
                pass
        if current_price == 0.0 and self.decision_log:
            current_price = self.decision_log[-1].get("price", 0)

        portfolio_value = self._get_portfolio_value(current_price)
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
            "held_symbol": self.held_symbol,
            "open_position": self.position is not None,
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
        scanned = result.get("symbols_scanned", "")

        status = "FLAT"
        if pos.get("active"):
            unr = pos.get("unrealized_pnl_pct")
            held = pos.get("symbol", "?")
            unr_str = f"{unr:+.2f}%" if unr is not None else "?"
            status = f"HOLDING {held} (step {pos.get('hold_steps', 0)}, {unr_str})"

        print(f"  [{result.get('timestamp', '')}]  Tick #{result.get('tick', 0)}")
        if symbol == "ALL":
            print(f"    Scanned {scanned}/{len(self.symbols)} symbols  |  Action: {action}")
        else:
            print(f"    {symbol}: ${price:,.2f}  |  Action: {action}")
        print(f"    Portfolio: ${pv:,.2f}  |  Return: {ret:+.2f}%  |  Drawdown: {dd:.2f}%  |  {status}")

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
        print(f"  Open Position:  {metrics['held_symbol'] or 'None'}")
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
