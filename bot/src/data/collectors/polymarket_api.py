"""
Polymarket API Client (DEPRECATED)

This project enforces real data only and Polymarket is not supported
in this deployment. Any usage should fail fast with a clear error.
"""

from typing import Dict, List, Optional

from ...core.logger import get_logger
from ...data.sources.base import DataUnavailableError


logger = get_logger(__name__)


class PolymarketAPIClient:
    """
    Deprecated client. Raises DataUnavailableError for all calls.
    """

    def __init__(self, api_key: Optional[str] = None):
        logger.warning("PolymarketAPIClient is deprecated and disabled.")

    def get_active_markets(self, *args, **kwargs) -> List[Dict]:
        raise DataUnavailableError("Polymarket data source is disabled.")

    def get_market_details(self, *args, **kwargs) -> Dict:
        raise DataUnavailableError("Polymarket data source is disabled.")

    def get_market_history(self, *args, **kwargs) -> List[Dict]:
        raise DataUnavailableError("Polymarket data source is disabled.")


# Convenience function
def get_polymarket_client() -> PolymarketAPIClient:
    """Get a configured Polymarket API client"""
    return PolymarketAPIClient()
