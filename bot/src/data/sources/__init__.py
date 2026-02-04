"""Exchange adapters for real-market data sources."""

from .base import ExchangeAdapter, OHLCV, DataUnavailableError
from .coinbase import CoinbaseAdapter
from .kraken import KrakenAdapter
from .registry import get_adapter, list_adapters

__all__ = [
    "ExchangeAdapter",
    "OHLCV",
    "DataUnavailableError",
    "CoinbaseAdapter",
    "KrakenAdapter",
    "get_adapter",
    "list_adapters",
]
