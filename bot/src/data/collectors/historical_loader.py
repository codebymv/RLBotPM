"""
Historical Data Loader (DEPRECATED)

Polymarket-specific loader is disabled. Use crypto_loader instead.
All synthetic data paths are removed and this loader now fails fast.
"""

from typing import List, Optional
import pandas as pd

from ...core.logger import get_logger
from ...data.sources.base import DataUnavailableError


logger = get_logger(__name__)


class HistoricalDataLoader:
    """
    Loads and stores historical market data
    
    Usage:
        loader = HistoricalDataLoader()
        loader.collect_historical_data(months=6, min_volume=10000)
    """
    
    def __init__(self):
        """Initialize data loader (deprecated)"""
        logger.warning("HistoricalDataLoader is deprecated and disabled.")
    
    def collect_historical_data(
        self,
        months: int = 6,
        min_volume: float = 10000,
        categories: Optional[List[str]] = None
    ):
        raise DataUnavailableError(
            "HistoricalDataLoader is disabled. Use crypto_loader with real exchange data."
        )
    
    def _store_market(self, *args, **kwargs):
        raise DataUnavailableError("HistoricalDataLoader is disabled.")
    
    def get_training_dataset(
        self,
        min_volume: float = 10000,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        raise DataUnavailableError(
            "HistoricalDataLoader is disabled. Use crypto_loader with real exchange data."
        )
