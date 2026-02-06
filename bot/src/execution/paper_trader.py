"""Paper trading engine for live simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import asyncio
import contextlib
from typing import Callable, Dict, List, Optional

from .live_data_feed import LiveDataFeed


@dataclass
class PaperPosition:
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class PaperOrder:
    symbol: str
    side: str
    size: float
    price: float
    timestamp: datetime
    filled: bool = False


class PaperTradingEngine:
    """
    Simulate live trading without real money.

    Tracks capital, fees, and trade outcomes with basic slippage.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct

        self.positions: Dict[str, PaperPosition] = {}
        self.orders: List[PaperOrder] = []
        self.trades: List[Dict] = []
        self.total_pnl = 0.0
        self.total_fees = 0.0

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        current_price: float,
    ) -> Dict:
        """Place a simulated order and update state."""
        fill_price = current_price * (1 + self.slippage_pct if side == "buy" else 1 - self.slippage_pct)
        fees = size * self.transaction_cost_pct

        if side == "buy" and size + fees > self.capital:
            return {"success": False, "reason": "Insufficient capital"}

        order = PaperOrder(
            symbol=symbol,
            side=side,
            size=size,
            price=fill_price,
            timestamp=datetime.utcnow(),
            filled=True,
        )
        self.orders.append(order)

        if side == "buy":
            self._open_position(symbol, size, fill_price)
            self.capital -= (size + fees)
        else:
            pnl = self._close_position(symbol, fill_price)
            self.capital += (size - fees)
            self.total_pnl += pnl

        self.total_fees += fees

        return {"success": True, "order_id": len(self.orders), "fill_price": fill_price, "fees": fees}

    def record_trade(self, trade_result: Dict) -> None:
        """Record trade result from environment execution."""
        if not trade_result.get("executed"):
            return

        pnl = float(trade_result.get("pnl", 0.0))
        cost = float(trade_result.get("cost", 0.0))
        if trade_result.get("side") == "sell":
            self.trades.append(
                {
                    "symbol": trade_result.get("symbol"),
                    "entry_price": trade_result.get("entry_price"),
                    "exit_price": trade_result.get("price"),
                    "pnl": pnl,
                    "timestamp": datetime.utcnow(),
                }
            )
            self.total_pnl += pnl
            self.total_fees += cost

    def _open_position(self, symbol: str, size: float, price: float) -> None:
        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            size=size,
            entry_price=price,
            entry_time=datetime.utcnow(),
        )

    def _close_position(self, symbol: str, exit_price: float) -> float:
        if symbol not in self.positions:
            return 0.0

        position = self.positions[symbol]
        pnl = (exit_price - position.entry_price) / position.entry_price * position.size
        self.trades.append(
            {
                "symbol": symbol,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "timestamp": datetime.utcnow(),
            }
        )
        del self.positions[symbol]
        return pnl

    def get_performance_metrics(self) -> Dict:
        total_return_pct = (self.capital - self.initial_capital) / self.initial_capital
        wins = [t for t in self.trades if t["pnl"] > 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0

        return {
            "capital": self.capital,
            "total_return_pct": total_return_pct,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "num_trades": len(self.trades),
            "win_rate": win_rate,
            "open_positions": len(self.positions),
        }


class LivePaperTradingSession:
    """
    Executes a simple live paper trading loop driven by LiveDataFeed updates.

    This is a minimal integration layer to validate data flow and execution.
    """

    def __init__(
        self,
        symbols: List[str],
        engine: PaperTradingEngine,
        trade_size_pct: float = 0.05,
        entry_threshold_pct: float = 0.002,
        exit_threshold_pct: float = 0.002,
        on_trade: Optional[Callable[[Dict], None]] = None,
    ):
        self.symbols = symbols
        self.engine = engine
        self.trade_size_pct = trade_size_pct
        self.entry_threshold_pct = entry_threshold_pct
        self.exit_threshold_pct = exit_threshold_pct
        self.on_trade = on_trade
        self.feed = LiveDataFeed(symbols=symbols)
        self.last_prices: Dict[str, float] = {}

    def _handle_price_update(self, symbol: str, price: float, _timestamp: datetime) -> None:
        last_price = self.last_prices.get(symbol)
        self.last_prices[symbol] = price
        if last_price is None:
            return

        price_change = (price - last_price) / last_price
        position = self.engine.positions.get(symbol)

        if position is None and price_change >= self.entry_threshold_pct:
            size = self.engine.capital * self.trade_size_pct
            result = self.engine.place_order(symbol=symbol, side="buy", size=size, current_price=price)
            if self.on_trade and result.get("success"):
                self.on_trade({"symbol": symbol, "side": "buy", "price": price, "size": size})
        elif position is not None and price_change <= -self.exit_threshold_pct:
            result = self.engine.place_order(symbol=symbol, side="sell", size=position.size, current_price=price)
            if self.on_trade and result.get("success"):
                self.on_trade({"symbol": symbol, "side": "sell", "price": price, "size": position.size})

    async def run(self, duration_seconds: int) -> None:
        self.feed.subscribe(self._handle_price_update)
        feed_task = asyncio.create_task(self.feed.connect())
        try:
            await asyncio.sleep(duration_seconds)
        finally:
            feed_task.cancel()
            with contextlib.suppress(Exception):
                await feed_task
