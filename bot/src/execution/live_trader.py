"""
Live Trading Engine - Execute real trades using the RL model.

This module connects the trained model to real Coinbase order execution.
Includes safety controls and monitoring.

Usage:
    python -m src.execution.live_trader --model models/best_model_run_118 --mode paper
    python -m src.execution.live_trader --model models/best_model_run_118 --mode live
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .coinbase_client import CoinbaseExecutionClient
from ..data.sources.coinbase import CoinbaseAdapter
from ..environment.gym_env import CryptoTradingEnv, _interval_to_steps
from ..core.logger import get_logger
from ..core.config import get_settings

logger = get_logger(__name__)


def load_env_file(env_path: str = "/workspaces/RLBotPM/.env"):
    """Load environment variables from .env file."""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


class LiveTrader:
    """
    Live trading engine that executes real orders based on RL model predictions.
    
    Features:
    - Real-time market data from Coinbase
    - Limit order execution (maker orders for low fees)
    - Safety controls (daily loss limit, max position size)
    - Trade logging and monitoring
    """
    
    # Action constants (must match training environment)
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    
    def __init__(
        self,
        model_path: str,
        symbol: str = "BTC-USD",
        interval: str = "1h",
        initial_capital: float = 1000.0,
        mode: str = "paper",  # "paper" or "live"
        max_position_pct: float = 0.20,
        daily_loss_limit_pct: float = 0.05,
        maker_offset_pct: float = 0.0005,  # Place limit orders 0.05% inside spread
        log_dir: str = "./logs/live_trading",
    ):
        self.model_path = model_path
        self.symbol = symbol
        self.interval = interval
        self.initial_capital = initial_capital
        self.mode = mode
        self.max_position_pct = max_position_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.maker_offset_pct = maker_offset_pct
        self.log_dir = Path(log_dir)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configs
        self._load_configs()
        
        # Initialize components
        self.settings = get_settings()
        self.coinbase_data = CoinbaseAdapter()  # For market data
        
        if mode == "live":
            self.coinbase_exec = CoinbaseExecutionClient()  # For order execution
        else:
            self.coinbase_exec = None  # Paper mode - no real execution
        
        # State tracking
        self.position: Optional[Dict] = None  # Current position
        self.daily_pnl = 0.0
        self.trade_history: List[Dict] = []
        self.pending_orders: Dict[str, Dict] = {}  # order_id -> order_info
        self.last_action_time: Optional[datetime] = None
        self.tick_count = 0
        
        # Safety flags
        self.trading_halted = False
        self.halt_reason = ""
        
        # Load model
        self._load_model()
        
        # Get initial account state
        self._update_account_state()
        
        # In live mode, use actual portfolio value as initial capital
        if mode == "live":
            current_price = self._get_current_price()
            self.initial_capital = self._get_portfolio_value(current_price)
            logger.info(f"  Initial Capital (from account): ${self.initial_capital:.2f}")
        
        logger.info(f"LiveTrader initialized in {mode.upper()} mode")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Interval: {interval}")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Max Position: {max_position_pct*100:.0f}%")
        logger.info(f"  Daily Loss Limit: {daily_loss_limit_pct*100:.0f}%")
    
    def _load_configs(self):
        """Load reward and risk configs."""
        reward_path = Path("/workspaces/RLBotPM/shared/config/reward_config.yaml")
        risk_path = Path("/workspaces/RLBotPM/shared/config/risk_config.yaml")
        
        with open(reward_path) as f:
            self.reward_config = yaml.safe_load(f)
        
        with open(risk_path) as f:
            self.risk_config = yaml.safe_load(f)
        
        # Extract key parameters
        self.stop_loss_pct = self.reward_config.get("stop_loss_pct", 0.03)
        self.take_profit_pct = self.reward_config.get("take_profit_pct", 0.05)
        self.min_hold_steps = self.reward_config.get("min_hold_steps", 5)
        self.max_hold_steps = self.reward_config.get("max_hold_steps", 100)
    
    def _load_model(self):
        """Load the trained RL model."""
        # Need a temporary environment to load the model with correct obs space
        candles_df = self._fetch_historical_data(num_candles=300)
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
    
    def _fetch_historical_data(self, num_candles: int = 300) -> pd.DataFrame:
        """Fetch historical candle data for feature computation."""
        interval_seconds = self._interval_to_seconds(self.interval)
        end = datetime.utcnow()
        start = end - timedelta(seconds=interval_seconds * num_candles)
        
        candles = self.coinbase_data.get_ohlcv(
            symbol=self.symbol,
            interval=self.interval,
            start=start,
            end=end,
        )
        
        if not candles:
            raise RuntimeError(f"No candles returned for {self.symbol}")
        
        df = pd.DataFrame(candles)
        df["symbol"] = self.symbol
        
        return df
    
    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        mapping = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400
        }
        return mapping.get(interval, 3600)
    
    def _get_current_price(self) -> float:
        """Get current BTC price."""
        if self.mode == "live" and self.coinbase_exec:
            bid, ask = self.coinbase_exec.get_best_bid_ask(self.symbol)
            return (bid + ask) / 2  # Mid price
        else:
            # Get from data adapter
            df = self._fetch_historical_data(num_candles=10)
            return float(df.iloc[-1]["close"])
    
    def _update_account_state(self):
        """Update account balances from Coinbase."""
        if self.mode == "paper":
            # Paper mode - use simulated balances
            self.usd_balance = self.initial_capital
            self.btc_balance = 0.0
            return
        
        usd_avail, _ = self.coinbase_exec.get_usd_balance()
        btc_avail, _ = self.coinbase_exec.get_btc_balance()
        
        self.usd_balance = usd_avail
        self.btc_balance = btc_avail
    
    def _get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value in USD."""
        return self.usd_balance + (self.btc_balance * current_price)
    
    def _build_observation(self, df: pd.DataFrame) -> np.ndarray:
        """Build observation vector from market data (same as training env)."""
        # Create a temporary environment to compute features consistently
        temp_env = CryptoTradingEnv(
            dataset=df,
            interval=self.interval,
            initial_capital=self.initial_capital,
        )
        
        # Reset the environment first to initialize episode data
        temp_env.reset()
        
        # Move to last row of data
        temp_env.current_step = len(df) - 1
        
        # Mirror our current position state
        if self.position:
            temp_env.positions[self.symbol] = {
                "entry_price": self.position["entry_price"],
                "size": self.position["size"],
                "hold_steps": self.position.get("hold_steps", 0),
            }
        
        # Get observation using environment's method
        obs = temp_env._get_observation()
        
        return obs
    
    def _get_action_mask(self) -> np.ndarray:
        """Get valid action mask based on current state."""
        mask = np.ones(3, dtype=np.float32)
        
        if self.position is None:
            # No position - can't sell
            mask[self.ACTION_SELL] = 0
        else:
            # Have position - can't buy more (single position mode)
            mask[self.ACTION_BUY] = 0
            
            # Check min hold time
            hold_steps = self.position.get("hold_steps", 0)
            if hold_steps < self.min_hold_steps:
                mask[self.ACTION_SELL] = 0
        
        return mask
    
    def _check_safety_conditions(self, current_price: float) -> bool:
        """Check if trading should be halted."""
        portfolio_value = self._get_portfolio_value(current_price)
        
        # Daily loss limit
        daily_loss_pct = (self.initial_capital - portfolio_value) / self.initial_capital
        if daily_loss_pct > self.daily_loss_limit_pct:
            self.trading_halted = True
            self.halt_reason = f"Daily loss limit exceeded: {daily_loss_pct*100:.2f}%"
            return False
        
        return True
    
    def _check_auto_exit(self, current_price: float) -> Optional[str]:
        """Check if position should be auto-exited (stop-loss/take-profit)."""
        if self.position is None:
            return None
        
        entry_price = self.position["entry_price"]
        unrealized_pnl_pct = (current_price - entry_price) / entry_price
        hold_steps = self.position.get("hold_steps", 0)
        
        # Stop loss
        if unrealized_pnl_pct < -self.stop_loss_pct:
            return "STOP_LOSS"
        
        # Take profit
        if unrealized_pnl_pct > self.take_profit_pct:
            return "TAKE_PROFIT"
        
        # Max hold time
        if hold_steps >= self.max_hold_steps:
            return "MAX_HOLD_TIME"
        
        return None
    
    def _execute_buy(self, current_price: float, bid_price: float) -> bool:
        """Execute a buy order."""
        if self.trading_halted:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return False
        
        # Calculate position size
        portfolio_value = self._get_portfolio_value(current_price)
        position_value = portfolio_value * self.max_position_pct
        btc_size = position_value / current_price
        
        # Minimum order size check (Coinbase minimum is ~0.0001 BTC)
        if btc_size < 0.0001:
            logger.warning(f"Position size too small: {btc_size:.8f} BTC")
            return False
        
        # Calculate limit price (slightly above bid for maker order)
        limit_price = bid_price * (1 + self.maker_offset_pct)
        
        if self.mode == "live":
            # Place real order
            result = self.coinbase_exec.place_limit_order(
                symbol=self.symbol,
                side="BUY",
                price=round(limit_price, 2),
                size=round(btc_size, 8),
                post_only=True,
            )
            
            if "error" in result:
                logger.error(f"Buy order failed: {result}")
                return False
            
            order_id = result.get("order_id") or result.get("success_response", {}).get("order_id")
            self.pending_orders[order_id] = {
                "side": "BUY",
                "price": limit_price,
                "size": btc_size,
                "timestamp": datetime.utcnow(),
            }
            
            logger.info(f"BUY order placed: {btc_size:.6f} BTC @ ${limit_price:.2f}")
            
        else:
            # Paper mode - simulate instant fill
            self.position = {
                "entry_price": current_price,
                "size": btc_size,
                "hold_steps": 0,
                "entry_time": datetime.utcnow(),
            }
            self.usd_balance -= position_value
            self.btc_balance += btc_size
            
            logger.info(f"[PAPER] BUY executed: {btc_size:.6f} BTC @ ${current_price:.2f}")
        
        # Log trade
        self._log_trade("BUY", current_price, btc_size)
        
        return True
    
    def _execute_sell(self, current_price: float, ask_price: float, reason: str = "MODEL") -> bool:
        """Execute a sell order."""
        if self.position is None:
            return False
        
        btc_size = self.position["size"]
        
        # Calculate limit price (slightly below ask for maker order)
        limit_price = ask_price * (1 - self.maker_offset_pct)
        
        if self.mode == "live":
            # Place real order
            result = self.coinbase_exec.place_limit_order(
                symbol=self.symbol,
                side="SELL",
                price=round(limit_price, 2),
                size=round(btc_size, 8),
                post_only=True,
            )
            
            if "error" in result:
                logger.error(f"Sell order failed: {result}")
                return False
            
            order_id = result.get("order_id") or result.get("success_response", {}).get("order_id")
            self.pending_orders[order_id] = {
                "side": "SELL",
                "price": limit_price,
                "size": btc_size,
                "timestamp": datetime.utcnow(),
                "reason": reason,
            }
            
            logger.info(f"SELL order placed ({reason}): {btc_size:.6f} BTC @ ${limit_price:.2f}")
            
        else:
            # Paper mode - simulate instant fill
            entry_price = self.position["entry_price"]
            pnl = (current_price - entry_price) * btc_size
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            self.usd_balance += btc_size * current_price
            self.btc_balance -= btc_size
            self.daily_pnl += pnl
            
            logger.info(f"[PAPER] SELL executed ({reason}): {btc_size:.6f} BTC @ ${current_price:.2f}")
            logger.info(f"[PAPER] Trade PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            self.position = None
        
        # Log trade
        self._log_trade(f"SELL_{reason}", current_price, btc_size)
        
        return True
    
    def _log_trade(self, action: str, price: float, size: float):
        """Log trade to file."""
        trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "symbol": self.symbol,
            "price": price,
            "size": size,
            "portfolio_value": self._get_portfolio_value(price),
            "daily_pnl": self.daily_pnl,
            "mode": self.mode,
        }
        
        self.trade_history.append(trade)
        
        # Append to trade log file
        log_file = self.log_dir / f"trades_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(trade) + "\n")
    
    def tick(self) -> Dict:
        """
        Process one tick: get data, run model, execute if needed.
        
        Returns:
            Dict with tick results
        """
        self.tick_count += 1
        
        # Get current market data
        df = self._fetch_historical_data(num_candles=300)
        current_row = df.iloc[-1]
        current_price = float(current_row["close"])
        
        # Get bid/ask
        if self.mode == "live":
            bid_price, ask_price = self.coinbase_exec.get_best_bid_ask(self.symbol)
        else:
            bid_price = current_price * 0.9999
            ask_price = current_price * 1.0001
        
        # Check safety conditions
        if not self._check_safety_conditions(current_price):
            return {"action": "HALTED", "reason": self.halt_reason}
        
        # Check auto-exit conditions
        exit_reason = self._check_auto_exit(current_price)
        if exit_reason:
            self._execute_sell(current_price, ask_price, reason=exit_reason)
            return {"action": f"AUTO_EXIT_{exit_reason}", "price": current_price}
        
        # Increment hold steps if in position
        if self.position:
            self.position["hold_steps"] = self.position.get("hold_steps", 0) + 1
        
        # Build observation and get action mask
        obs = self._build_observation(df)
        action_mask = self._get_action_mask()
        
        # Get model prediction
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        action = int(action)
        
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_name = action_names.get(action, "UNKNOWN")
        
        # Execute action
        executed = False
        if action == self.ACTION_BUY and action_mask[self.ACTION_BUY] > 0:
            executed = self._execute_buy(current_price, bid_price)
        elif action == self.ACTION_SELL and action_mask[self.ACTION_SELL] > 0:
            executed = self._execute_sell(current_price, ask_price, reason="MODEL")
        
        # Get portfolio status
        portfolio_value = self._get_portfolio_value(current_price)
        
        result = {
            "tick": self.tick_count,
            "timestamp": datetime.utcnow().isoformat(),
            "price": current_price,
            "bid": bid_price,
            "ask": ask_price,
            "action": action_name,
            "executed": executed,
            "position": self.position.copy() if self.position else None,
            "portfolio_value": portfolio_value,
            "daily_pnl": self.daily_pnl,
            "mode": self.mode,
        }
        
        # Print status
        pos_str = f"IN POSITION (entry: ${self.position['entry_price']:.2f})" if self.position else "FLAT"
        logger.info(
            f"Tick {self.tick_count}: BTC=${current_price:,.2f} | "
            f"Action={action_name} | Executed={executed} | {pos_str} | "
            f"Portfolio=${portfolio_value:,.2f}"
        )
        
        return result
    
    def run(self, duration_hours: float = 0, tick_interval_seconds: int = 60):
        """
        Run the live trading loop.
        
        Args:
            duration_hours: How long to run (0 = indefinitely)
            tick_interval_seconds: Seconds between ticks (for monitoring between candles)
        """
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours) if duration_hours > 0 else None
        
        interval_seconds = self._interval_to_seconds(self.interval)
        
        print("\n" + "=" * 60)
        print(f"LIVE TRADER STARTED - {self.mode.upper()} MODE")
        print("=" * 60)
        print(f"Symbol: {self.symbol}")
        print(f"Interval: {self.interval}")
        print(f"Model: {self.model_path}")
        print(f"Duration: {'Indefinite' if duration_hours == 0 else f'{duration_hours}h'}")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Check if we should stop
                if end_time and datetime.utcnow() >= end_time:
                    logger.info("Duration reached, stopping...")
                    break
                
                # Process tick
                result = self.tick()
                
                if result.get("action") == "HALTED":
                    logger.error(f"Trading halted: {result.get('reason')}")
                    break
                
                # Wait for next candle close
                # For hourly candles, we only need to check once per hour
                # But we check more frequently for order fill status
                time.sleep(tick_interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        
        # Print final summary
        self._print_summary()
    
    def _print_summary(self):
        """Print trading session summary."""
        print("\n" + "=" * 60)
        print("TRADING SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Ticks: {self.tick_count}")
        print(f"Trades Executed: {len(self.trade_history)}")
        print(f"Daily PnL: ${self.daily_pnl:.2f}")
        print(f"Mode: {self.mode.upper()}")
        
        if self.mode == "live":
            self._update_account_state()
            print(f"\nFinal Balances:")
            print(f"  USD: ${self.usd_balance:.2f}")
            print(f"  BTC: {self.btc_balance:.8f}")
        
        print("=" * 60 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Live Trading with RL Model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading pair")
    parser.add_argument("--interval", default="1h", help="Candle interval")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Trading mode")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital (paper mode)")
    parser.add_argument("--duration", type=float, default=0, help="Duration in hours (0=indefinite)")
    parser.add_argument("--tick-interval", type=int, default=60, help="Seconds between ticks")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env_file()
    
    # Create and run trader
    trader = LiveTrader(
        model_path=args.model,
        symbol=args.symbol,
        interval=args.interval,
        initial_capital=args.capital,
        mode=args.mode,
    )
    
    trader.run(duration_hours=args.duration, tick_interval_seconds=args.tick_interval)


if __name__ == "__main__":
    main()
