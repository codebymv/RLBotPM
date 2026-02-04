"""
Base classes and errors for exchange adapters.

All adapters MUST return real data from the configured exchange.
Synthetic data is explicitly запрещено (not allowed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional


class DataUnavailableError(RuntimeError):
    """Raised when real exchange data is missing or unavailable."""


@dataclass(frozen=True)
class OHLCV:
    """Represents one OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class ExchangeAdapter(ABC):
    """
    Abstract base class for exchange adapters.

    All implementations must call the real exchange API and should
    raise DataUnavailableError if data is missing or incomplete.
    """

    name: str

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Return list of tradeable symbols for this exchange."""

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Return latest trade price for the symbol."""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Return historical OHLCV candles for the symbol."""

    def get_orderbook(self, symbol: str, depth: int = 25) -> dict:
        """Return orderbook data if supported by the exchange."""
        raise NotImplementedError("Orderbook not supported by this adapter.")

    def place_order(self, *args, **kwargs) -> dict:
        """Place a live order (Phase 3)."""
        raise NotImplementedError("Trading not implemented for this adapter.")

    def healthcheck(self) -> dict:
        """
        Basic healthcheck. Should raise DataUnavailableError if the
        exchange cannot be reached or returns invalid data.
        """
        symbols = self.get_symbols()
        if not symbols:
            raise DataUnavailableError(f"{self.name} returned no symbols.")
        return {"source": self.name, "ok": True, "symbols": len(symbols)}
