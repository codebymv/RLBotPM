"""
Polymarket API Client

Wrapper for Polymarket API with:
- Rate limiting
- Error handling
- Retry logic
- Data validation

For Phase 1, this provides sample data.
For Phase 2+, connect to real Polymarket API.
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random

from ...core.logger import get_logger
from ...core.config import get_settings


logger = get_logger(__name__)


class PolymarketAPIClient:
    """
    Client for Polymarket API
    
    Usage:
        client = PolymarketAPIClient()
        markets = client.get_active_markets()
        market_data = client.get_market_details(market_id)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API client
        
        Args:
            api_key: Polymarket API key (optional for Phase 1)
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.POLYMARKET_API_KEY
        
        # API endpoints (placeholder - use actual Polymarket endpoints in Phase 2)
        self.base_url = "https://api.polymarket.com"  # Example
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("Polymarket API client initialized")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        retries: int = 3
    ) -> Dict:
        """
        Make API request with error handling
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            retries: Number of retries on failure
        
        Returns:
            Response data
        """
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        for attempt in range(retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=10
                )
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{retries}): {str(e)}")
                
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"API request failed after {retries} attempts")
                    raise
    
    def get_active_markets(
        self,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get list of active markets
        
        Args:
            category: Filter by category (sports, politics, etc.)
            limit: Maximum number of markets
        
        Returns:
            List of market dictionaries
        """
        # Phase 1: Return synthetic data
        # Phase 2+: Replace with actual API call
        
        logger.info(f"Fetching active markets (category={category}, limit={limit})")
        
        # Synthetic data for Phase 1 development
        markets = self._generate_synthetic_markets(limit, category)
        
        logger.info(f"Fetched {len(markets)} markets")
        return markets
    
    def get_market_details(self, market_id: str) -> Dict:
        """
        Get detailed information for a specific market
        
        Args:
            market_id: Market ID
        
        Returns:
            Market details
        """
        logger.debug(f"Fetching market details: {market_id}")
        
        # Phase 1: Synthetic data
        # Phase 2+: Actual API call
        
        return self._generate_synthetic_market_details(market_id)
    
    def get_market_history(
        self,
        market_id: str,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get historical price data for a market
        
        Args:
            market_id: Market ID
            interval: Time interval (1m, 5m, 1h, 1d)
            start_time: Start of time range
            end_time: End of time range
        
        Returns:
            List of price snapshots
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
        if not end_time:
            end_time = datetime.now()
        
        logger.debug(f"Fetching market history: {market_id} from {start_time} to {end_time}")
        
        # Phase 1: Synthetic data
        return self._generate_synthetic_history(market_id, start_time, end_time)
    
    def _generate_synthetic_markets(
        self,
        count: int,
        category: Optional[str] = None
    ) -> List[Dict]:
        """Generate synthetic market data for Phase 1 development"""
        categories = ['sports', 'politics', 'crypto', 'entertainment']
        if category:
            categories = [category]
        
        markets = []
        
        for i in range(count):
            market_category = random.choice(categories)
            
            market = {
                'id': f"market_{i:04d}",
                'question': f"Sample {market_category} market {i}",
                'category': market_category,
                'current_price': random.uniform(0.3, 0.7),
                'volume_24h': random.uniform(1000, 100000),
                'liquidity': random.uniform(500, 50000),
                'created_at': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                'resolution_date': (datetime.now() + timedelta(days=random.randint(1, 90))).isoformat(),
                'status': 'active'
            }
            
            markets.append(market)
        
        return markets
    
    def _generate_synthetic_market_details(self, market_id: str) -> Dict:
        """Generate detailed synthetic market data"""
        return {
            'id': market_id,
            'question': f"Will this market resolve yes? ({market_id})",
            'description': "Detailed market description here",
            'category': random.choice(['sports', 'politics', 'crypto', 'entertainment']),
            'current_price': random.uniform(0.3, 0.7),
            'bid_price': random.uniform(0.3, 0.6),
            'ask_price': random.uniform(0.4, 0.7),
            'volume_24h': random.uniform(1000, 100000),
            'liquidity': random.uniform(500, 50000),
            'num_traders': random.randint(50, 5000),
            'created_at': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            'resolution_date': (datetime.now() + timedelta(days=random.randint(1, 90))).isoformat(),
            'status': 'active',
            'outcomes': ['Yes', 'No'],
            'metadata': {
                'tags': [],
                'source': 'synthetic'
            }
        }
    
    def _generate_synthetic_history(
        self,
        market_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Generate synthetic price history"""
        history = []
        
        current_time = start_time
        current_price = random.uniform(0.4, 0.6)
        
        # Generate hourly snapshots
        while current_time <= end_time:
            # Random walk price
            price_change = random.gauss(0, 0.02)
            current_price = max(0.01, min(0.99, current_price + price_change))
            
            snapshot = {
                'timestamp': current_time.isoformat(),
                'price': current_price,
                'volume': random.uniform(100, 10000),
                'bid': current_price - random.uniform(0.005, 0.02),
                'ask': current_price + random.uniform(0.005, 0.02)
            }
            
            history.append(snapshot)
            current_time += timedelta(hours=1)
        
        return history


# Convenience function
def get_polymarket_client() -> PolymarketAPIClient:
    """Get a configured Polymarket API client"""
    return PolymarketAPIClient()
