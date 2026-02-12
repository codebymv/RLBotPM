"""
Backtest the hybrid Kalshi strategy on held-out events.

Compares:
- Pure statistical edge detection
- Pure RL (if model provided)
- Hybrid (statistical + RL timing + Kelly sizing)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

from ..core.logger import get_logger
from ..data.database import get_db_session
from ..environment.kalshi_env import load_kalshi_settled_markets
from .kalshi_hybrid import HybridKalshiEngine, TradeSignal


logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from backtesting a single strategy."""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    roi: float  # return on investment
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_pnl: float
    trades: List[Dict]  # Trade history


class KalshiBacktester:
    """Backtest Kalshi strategies on held-out events."""

    def __init__(
        self,
        initial_capital: float = 25.0,
        test_split: float = 0.2,
        min_liquidity: float = 100,
    ):
        """
        Args:
            initial_capital: Starting bankroll
            test_split: Fraction of events held out for testing
            min_liquidity: Min liquidity to trade a market
        """
        self.initial_capital = initial_capital
        self.test_split = test_split
        self.min_liquidity = min_liquidity

        # Load test data
        session = get_db_session()
        all_markets = load_kalshi_settled_markets(session)
        session.close()

        logger.info(f"Loaded {len(all_markets)} settled markets")

        # Split by events (same as training)
        events = all_markets[["event_ticker", "outcome"]].drop_duplicates()
        events = events.sample(frac=1, random_state=42).reset_index(drop=True)
        n_test = int(len(events) * test_split)
        test_events = set(events.iloc[:n_test]["event_ticker"].values)

        self.test_markets = all_markets[all_markets["event_ticker"].isin(test_events)]
        logger.info(f"Test set: {len(test_events)} events, {len(self.test_markets)} markets")

    def _simulate_trade(
        self,
        market: pd.Series,
        action: str,
        contracts: int,
    ) -> Dict:
        """
        Simulate a trade and compute realized PnL.
        
        Args:
            market: Market row from DataFrame
            action: 'BUY_YES' or 'BUY_NO'
            contracts: Number of contracts
        
        Returns:
            Trade result dict
        """
        yes_price = market["last_price"]
        outcome = market["outcome"]  # 1 for YES, 0 for NO
        
        if action == "BUY_YES":
            cost = contracts * (yes_price / 100.0)
            payout = contracts * 1.0 if outcome == 1 else 0.0
        elif action == "BUY_NO":
            cost = contracts * ((100 - yes_price) / 100.0)
            payout = contracts * 1.0 if outcome == 0 else 0.0
        else:
            return {"contracts": 0, "cost": 0, "payout": 0, "pnl": 0, "won": False}
        
        pnl = payout - cost
        won = pnl > 0
        
        return {
            "ticker": market["ticker"],
            "event_ticker": market["event_ticker"],
            "action": action,
            "contracts": contracts,
            "entry_price": yes_price,
            "cost": cost,
            "payout": payout,
            "pnl": pnl,
            "won": won,
            "outcome": outcome,
        }

    def backtest_hybrid(
        self,
        rl_model_path: Optional[str] = None,
        min_edge: float = 0.05,
        rl_threshold: float = 0.6,
    ) -> BacktestResult:
        """
        Backtest hybrid strategy (statistical + RL + Kelly).
        
        Process:
        1. For each event in test set, get all markets
        2. Run hybrid scanner on markets before settlement
        3. Execute top signal (if any)
        4. Compute realized PnL after settlement
        """
        logger.info(f"Backtesting hybrid strategy (edge≥{min_edge:.1%}, RL≥{rl_threshold:.0%})")
        
        engine = HybridKalshiEngine(
            rl_model_path=rl_model_path,
            rl_threshold=rl_threshold,
            min_edge=min_edge,
            bankroll=self.initial_capital,
        )
        
        # Group markets by event
        events_grouped = self.test_markets.groupby("event_ticker")
        
        trades = []
        capital = self.initial_capital
        equity_curve = [capital]
        
        for event_ticker, event_markets in events_grouped:
            # Convert to dict format for scanner
            market_dicts = []
            for _, row in event_markets.iterrows():
                market_dicts.append({
                    "ticker": row["ticker"],
                    "event_ticker": row["event_ticker"],
                    "series_ticker": row["series_ticker"],
                    "last_price": row["last_price"],
                    "yes_bid": row.get("yes_bid", 0),
                    "yes_ask": row.get("yes_ask", 100),
                    "volume": row.get("volume", 0),
                    "open_interest": row.get("open_interest", 0),
                    "liquidity": row.get("liquidity", 0),
                    "previous_price": row.get("last_price", 50),
                })
            
            # Scan for signals
            signals = engine.scan_and_rank(market_dicts, top_n=1)
            
            if not signals:
                continue  # No signal = skip event
            
            # Take top signal
            signal = signals[0]
            
            # Find corresponding market row for outcome
            market_row = event_markets[event_markets["ticker"] == signal.ticker].iloc[0]
            
            # Simulate trade
            trade = self._simulate_trade(
                market=market_row,
                action=signal.action,
                contracts=signal.contracts,
            )
            
            # Update capital
            capital += trade["pnl"]
            equity_curve.append(capital)
            trades.append(trade)
        
        # Compute metrics
        return self._compute_metrics("Hybrid", trades, equity_curve)

    def backtest_statistical_only(
        self,
        min_edge: float = 0.05,
    ) -> BacktestResult:
        """
        Backtest pure statistical strategy (no RL filter, no Kelly sizing).
        
        Just takes all edges above threshold with fixed position size.
        """
        logger.info(f"Backtesting statistical-only strategy (edge≥{min_edge:.1%})")
        
        engine = HybridKalshiEngine(
            rl_model_path=None,  # No RL
            rl_threshold=0.0,    # Always trade edges
            min_edge=min_edge,
            bankroll=self.initial_capital,
        )
        
        events_grouped = self.test_markets.groupby("event_ticker")
        trades = []
        capital = self.initial_capital
        equity_curve = [capital]
        
        for event_ticker, event_markets in events_grouped:
            market_dicts = []
            for _, row in event_markets.iterrows():
                market_dicts.append({
                    "ticker": row["ticker"],
                    "event_ticker": row["event_ticker"],
                    "series_ticker": row["series_ticker"],
                    "last_price": row["last_price"],
                    "yes_bid": row.get("yes_bid", 0),
                    "yes_ask": row.get("yes_ask", 100),
                    "volume": row.get("volume", 0),
                    "open_interest": row.get("open_interest", 0),
                    "liquidity": row.get("liquidity", 0),
                    "previous_price": row.get("last_price", 50),
                })
            
            signals = engine.scan_and_rank(market_dicts, top_n=1)
            if not signals:
                continue
            
            signal = signals[0]
            market_row = event_markets[event_markets["ticker"] == signal.ticker].iloc[0]
            trade = self._simulate_trade(market_row, signal.action, signal.contracts)
            
            capital += trade["pnl"]
            equity_curve.append(capital)
            trades.append(trade)
        
        return self._compute_metrics("Statistical Only", trades, equity_curve)

    def _compute_metrics(
        self,
        name: str,
        trades: List[Dict],
        equity_curve: List[float],
    ) -> BacktestResult:
        """Compute backtest performance metrics."""
        if not trades:
            return BacktestResult(
                strategy_name=name,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                roi=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_trade_pnl=0.0,
                trades=[],
            )
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t["won"])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(t["pnl"] for t in trades)
        roi = total_pnl / self.initial_capital
        avg_trade_pnl = total_pnl / total_trades
        
        # Sharpe ratio (annualized, assuming daily trades)
        pnls = np.array([t["pnl"] for t in trades])
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if np.std(pnls) > 0 else 0.0
        
        # Max drawdown
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return BacktestResult(
            strategy_name=name,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            roi=roi,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            avg_trade_pnl=avg_trade_pnl,
            trades=trades,
        )

    def compare_strategies(
        self,
        rl_model_path: Optional[str] = None,
        min_edge: float = 0.05,
        rl_threshold: float = 0.6,
    ) -> pd.DataFrame:
        """
        Run backtest on all strategies and compare.
        
        Returns DataFrame with performance comparison.
        """
        logger.info("Running strategy comparison...")
        
        # 1. Pure statistical
        stat_result = self.backtest_statistical_only(min_edge=min_edge)
        
        # 2. Hybrid (statistical + RL + Kelly)
        hybrid_result = self.backtest_hybrid(
            rl_model_path=rl_model_path,
            min_edge=min_edge,
            rl_threshold=rl_threshold,
        )
        
        # Format comparison
        results = [stat_result, hybrid_result]
        
        comparison = pd.DataFrame([{
            "Strategy": r.strategy_name,
            "Trades": r.total_trades,
            "Win Rate": f"{r.win_rate:.1%}",
            "Total P&L": f"${r.total_pnl:.2f}",
            "ROI": f"{r.roi:.1%}",
            "Avg Trade": f"${r.avg_trade_pnl:.2f}",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "Max DD": f"{r.max_drawdown:.1%}",
        } for r in results])
        
        return comparison
