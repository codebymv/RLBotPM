"""Real-time data feed from exchanges using WebSocket."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Callable, Dict, List, Optional

import websockets


class LiveDataFeed:
    """
    Stream real-time price data from Coinbase WebSocket.

    This feed emits best-effort tick updates for price monitoring
    and paper trading simulations.
    """

    def __init__(self, symbols: List[str], interval: str = "1m"):
        self.symbols = symbols
        self.interval = interval
        self.callbacks: List[Callable[[str, float, datetime], None]] = []
        self.latest_prices: Dict[str, float] = {}
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    def subscribe(self, callback: Callable[[str, float, datetime], None]) -> None:
        """Register a callback for price updates."""
        self.callbacks.append(callback)

    async def connect(self) -> None:
        """Connect to Coinbase WebSocket and begin streaming."""
        url = "wss://ws-feed.exchange.coinbase.com"
        subscribe_message = {
            "type": "subscribe",
            "product_ids": self.symbols,
            "channels": ["ticker"],
        }

        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            self._ws = ws
            await ws.send(json.dumps(subscribe_message))

            async for message in ws:
                data = json.loads(message)
                await self._handle_message(data)

    async def _handle_message(self, data: Dict) -> None:
        if data.get("type") != "ticker":
            return

        symbol = data.get("product_id")
        price = data.get("price")
        if symbol is None or price is None:
            return

        ts = datetime.utcnow()
        self.latest_prices[symbol] = float(price)
        for callback in self.callbacks:
            await asyncio.to_thread(callback, symbol, float(price), ts)

    def get_latest_price(self, symbol: str) -> float:
        """Get most recent price for symbol, if available."""
        return self.latest_prices.get(symbol, 0.0)
