"""
Historical Data Loader

Collects and stores historical Polymarket data for training.

For Phase 1: Uses synthetic data
For Phase 2+: Connects to real Polymarket data sources
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

from .polymarket_api import PolymarketAPIClient
from ..database import Market, get_db_session, DatabaseSession
from ...core.logger import get_logger


logger = get_logger(__name__)


class HistoricalDataLoader:
    """
    Loads and stores historical market data
    
    Usage:
        loader = HistoricalDataLoader()
        loader.collect_historical_data(months=6, min_volume=10000)
    """
    
    def __init__(self):
        """Initialize data loader"""
        self.api_client = PolymarketAPIClient()
        logger.info("Historical data loader initialized")
    
    def collect_historical_data(
        self,
        months: int = 6,
        min_volume: float = 10000,
        categories: Optional[List[str]] = None
    ):
        """
        Collect historical market data
        
        Args:
            months: Number of months of history to collect
            min_volume: Minimum 24h volume to include
            categories: List of categories to include (None = all)
        """
        logger.info(f"Collecting {months} months of historical data")
        logger.info(f"Filters: min_volume=${min_volume}, categories={categories}")
        
        # Get active markets
        markets = self.api_client.get_active_markets(limit=200)
        
        # Filter markets
        filtered_markets = [
            m for m in markets
            if m['volume_24h'] >= min_volume
            and (categories is None or m['category'] in categories)
        ]
        
        logger.info(f"Found {len(filtered_markets)} markets matching criteria")
        
        # Collect data for each market
        collected_count = 0
        
        for market in filtered_markets:
            try:
                self._store_market(market)
                collected_count += 1
                
                if collected_count % 10 == 0:
                    logger.info(f"Processed {collected_count}/{len(filtered_markets)} markets")
                    
            except Exception as e:
                logger.error(f"Failed to collect market {market['id']}: {str(e)}")
                continue
        
        logger.info(f"✓ Collected data for {collected_count} markets")
    
    def _store_market(self, market_data: Dict):
        """
        Store market data in database
        
        Args:
            market_data: Market dictionary from API
        """
        with DatabaseSession() as session:
            # Check if market already exists
            existing = session.query(Market).filter_by(
                polymarket_id=market_data['id']
            ).first()
            
            if existing:
                # Update existing market
                existing.question = market_data['question']
                existing.category = market_data.get('category')
                existing.volume = market_data.get('volume_24h')
                existing.liquidity = market_data.get('liquidity')
                existing.last_updated = datetime.utcnow()
            else:
                # Create new market
                market = Market(
                    polymarket_id=market_data['id'],
                    question=market_data['question'],
                    category=market_data.get('category'),
                    created_at=datetime.fromisoformat(market_data['created_at'].replace('Z', '+00:00')),
                    volume=market_data.get('volume_24h'),
                    liquidity=market_data.get('liquidity'),
                    metadata={'source': 'api', 'raw_data': market_data}
                )
                session.add(market)
            
            logger.debug(f"Stored market: {market_data['id']}")
    
    def get_training_dataset(
        self,
        min_volume: float = 10000,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get markets suitable for training
        
        Args:
            min_volume: Minimum volume filter
            categories: Category filter
            limit: Maximum number of markets
        
        Returns:
            DataFrame with market data
        """
        with DatabaseSession() as session:
            query = session.query(Market)
            
            if min_volume:
                query = query.filter(Market.volume >= min_volume)
            
            if categories:
                query = query.filter(Market.category.in_(categories))
            
            if limit:
                query = query.limit(limit)
            
            markets = query.all()
            
            # Convert to DataFrame
            data = []
            for market in markets:
                data.append({
                    'id': market.polymarket_id,
                    'question': market.question,
                    'category': market.category,
                    'volume': market.volume,
                    'liquidity': market.liquidity,
                    'created_at': market.created_at,
                    'resolution_date': market.resolution_date
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} markets for training")
            
            return df
