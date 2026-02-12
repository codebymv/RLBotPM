"""Data collectors for crypto sources (real data only)."""

from .crypto_loader import CryptoDataLoader, MultiSourceLoader
from .historical_loader import HistoricalDataLoader

try:
    from .polymarket_api import PolymarketAPIClient, get_polymarket_client
except ImportError:
    PolymarketAPIClient = None  # type: ignore
    get_polymarket_client = None  # type: ignore

__all__ = [
    "CryptoDataLoader",
    "MultiSourceLoader",
    "PolymarketAPIClient",
    "get_polymarket_client",
    "HistoricalDataLoader",
]
