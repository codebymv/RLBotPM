"""
Exchange adapter registry for multi-source support.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import ExchangeAdapter, DataUnavailableError
from .coinbase import CoinbaseAdapter
from .kraken import KrakenAdapter
from .kalshi import KalshiAdapter


_REGISTRY: Dict[str, Type[ExchangeAdapter]] = {
    "coinbase": CoinbaseAdapter,
    "kraken": KrakenAdapter,
    "kalshi": KalshiAdapter,
}


def get_adapter(name: str) -> ExchangeAdapter:
    """Create an adapter by name."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise DataUnavailableError(f"Unknown data source: {name}")
    return _REGISTRY[key]()


def list_adapters() -> Dict[str, Type[ExchangeAdapter]]:
    """Return registry map."""
    return dict(_REGISTRY)
