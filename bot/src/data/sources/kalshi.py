"""
Kalshi Prediction Markets API Adapter.

API Docs: https://trading-api.kalshi.com/docs
Supports:
- Market discovery and filtering
- Price/probability history
- Orderbook data
- Account balance and positions
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import requests
import pandas as pd

from .base import ExchangeAdapter, DataUnavailableError, OHLCV
from ...core.logger import get_logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger(__name__)

# Try to import cryptography for RSA signing
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("cryptography library not installed - Kalshi auth will be disabled")


@dataclass(frozen=True)
class KalshiMarket:
    """Represents a Kalshi prediction market."""
    ticker: str
    title: str
    subtitle: str
    category: str
    yes_price: float  # 0-100 cents
    no_price: float   # 0-100 cents
    yes_bid: float
    yes_ask: float
    volume: int
    open_interest: int
    expiration_time: datetime
    close_time: datetime
    status: str  # 'open', 'closed', 'settled'
    result: Optional[str]  # 'yes', 'no', None if not settled


@dataclass(frozen=True)
class KalshiCandle:
    """Price history candle for a Kalshi market."""
    timestamp: datetime
    yes_price: float  # 0-1 probability
    volume: int
    open_interest: int


class KalshiAdapter(ExchangeAdapter):
    """
    Adapter for Kalshi prediction markets API.
    
    Focuses on:
    - Crypto range markets (e.g., "BTC above $100k")
    - Economic data markets (CPI, jobs, Fed)
    - Financial ranges (S&P levels)
    """
    
    name = "kalshi"
    
    # Production and demo endpoints
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.elections.kalshi.co/trade-api/v2"
    
    # Market categories we focus on
    SUPPORTED_CATEGORIES = [
        "Crypto",
        "Economics",
        "Financial",
        "Fed",
    ]
    
    def __init__(self, demo: bool = False):
        """
        Initialize Kalshi adapter.
        
        Args:
            demo: If True, use demo API (paper trading)
        """
        self.base_url = self.DEMO_URL if demo else self.PROD_URL
        self.demo = demo
        
        # Load API credentials
        self.api_key = os.getenv("KALSHI_API_KEY", "")
        api_secret_raw = os.getenv("KALSHI_API_SECRET", "")
        
        # Handle escaped newlines in .env file
        self.api_secret = api_secret_raw.replace("\\n", "\n") if api_secret_raw else ""
        
        self._private_key = None
        if self.api_secret and HAS_CRYPTO:
            try:
                self._private_key = serialization.load_pem_private_key(
                    self.api_secret.encode(),
                    password=None,
                )
                logger.info("Kalshi RSA private key loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Kalshi RSA key: {e}")
        
        if not self.api_key or not self._private_key:
            logger.warning(
                "KALSHI credentials not set or invalid - read-only public API mode"
            )
        
        self._session = requests.Session()
        
        logger.info(f"KalshiAdapter initialized (demo={demo})")
    
    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """
        Sign a request using RSA-PSS-SHA256.
        
        Kalshi signature format: timestamp_ms + method + path
        Path must be the full URL path (e.g. /trade-api/v2/markets).
        """
        if not self._private_key:
            return ""
        
        message = f"{timestamp_ms}{method}{path}"
        
        try:
            signature = self._private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            logger.error(f"Failed to sign request: {e}")
            return ""
    
    def _get_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        if not self.api_key or not self._private_key:
            return {}
        
        timestamp_ms = int(time.time() * 1000)
        signature = self._sign_request(method, path, timestamp_ms)
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
    ) -> Dict:
        """Make authenticated API request."""
        url = f"{self.base_url}{endpoint}"
        # Kalshi requires the FULL URL path in the signature (including /trade-api/v2)
        from urllib.parse import urlparse
        full_path = urlparse(url).path
        headers = self._get_auth_headers(method.upper(), full_path)
        
        try:
            resp = self._session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Kalshi API error: {e.response.status_code} - {e.response.text}")
            raise DataUnavailableError(f"Kalshi API error: {e}")
        except Exception as e:
            raise DataUnavailableError(f"Kalshi request failed: {e}")
    
    # === ExchangeAdapter Interface ===
    
    def get_symbols(self) -> List[str]:
        """Return list of active market tickers."""
        markets = self.get_markets(status="open", limit=500)
        return [m.ticker for m in markets]
    
    def get_latest_price(self, symbol: str) -> float:
        """Return latest YES price (0-1) for a market."""
        market = self.get_market(symbol)
        return market.yes_price / 100.0  # Convert cents to probability
    
    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """
        Return price history as OHLCV-like candles.
        
        Note: Kalshi doesn't have traditional OHLCV. We map:
        - open/high/low/close = yes_price (all same for simplicity)
        - volume = trade volume
        """
        history = self.get_market_history(symbol, limit=limit or 100)
        
        return [
            OHLCV(
                timestamp=candle.timestamp,
                open=candle.yes_price,
                high=candle.yes_price,
                low=candle.yes_price,
                close=candle.yes_price,
                volume=float(candle.volume),
            )
            for candle in history
        ]
    
    def get_orderbook(self, symbol: str, depth: int = 25) -> dict:
        """Return orderbook for a market."""
        return self._get_orderbook(symbol, depth)
    
    # === Kalshi-Specific Methods ===
    
    def get_markets(
        self,
        status: str = "open",
        category: Optional[str] = None,
        series_ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> List[KalshiMarket]:
        """
        Get list of markets with optional filters.
        
        Args:
            status: 'open', 'closed', 'settled'
            category: Filter by category (e.g., 'Crypto', 'Economics')
            series_ticker: Filter by series (e.g., 'BTCUSD' for all BTC markets)
            limit: Max results per page
            cursor: Pagination cursor
        """
        params = {
            "status": status,
            "limit": limit,
        }
        if category:
            params["category"] = category
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        
        data = self._request("GET", "/markets", params=params)
        markets_data = data.get("markets", [])
        
        return [self._parse_market(m) for m in markets_data]
    
    def get_market(self, ticker: str) -> KalshiMarket:
        """Get single market by ticker."""
        data = self._request("GET", f"/markets/{ticker}")
        market_data = data.get("market", data)
        return self._parse_market(market_data)
    
    def get_crypto_markets(self, asset: str = "BTC") -> List[KalshiMarket]:
        """
        Get crypto price range markets.
        
        Args:
            asset: Crypto asset (BTC, ETH, etc.)
        """
        # Kalshi crypto markets typically use series like "KXBTC", "KXETH", etc.
        # (Older configs may reference "BTCUSD" / "ETHUSD"; keep as fallback.)
        asset_u = asset.upper()
        series_candidates = [f"KX{asset_u}", f"{asset_u}USD"]
        
        try:
            for series in series_candidates:
                markets = self.get_markets(status="open", series_ticker=series)
                if markets:
                    return markets
            return []
        except DataUnavailableError:
            # Try category filter as fallback
            all_markets = self.get_markets(status="open", limit=500)
            return [
                m for m in all_markets 
                if asset.upper() in m.ticker.upper() or asset.upper() in m.title.upper()
            ]
    
    def get_fed_markets(self) -> List[KalshiMarket]:
        """Get Federal Reserve related markets."""
        try:
            return self.get_markets(status="open", category="Fed")
        except DataUnavailableError:
            all_markets = self.get_markets(status="open", limit=500)
            return [
                m for m in all_markets
                if "fed" in m.title.lower() or "fomc" in m.title.lower()
            ]
    
    def get_economic_markets(self) -> List[KalshiMarket]:
        """Get economic data markets (CPI, jobs, GDP)."""
        keywords = ["cpi", "inflation", "jobs", "unemployment", "gdp", "economic"]
        all_markets = self.get_markets(status="open", limit=500)
        return [
            m for m in all_markets
            if any(kw in m.title.lower() for kw in keywords)
        ]
    
    def get_market_history(
        self,
        ticker: str,
        limit: int = 100,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> List[KalshiCandle]:
        """Get price history for a market."""
        params = {"limit": limit}
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        
        data = self._request("GET", f"/markets/{ticker}/history", params=params)
        history = data.get("history", [])
        
        return [
            KalshiCandle(
                timestamp=datetime.fromtimestamp(h["ts"], tz=timezone.utc),
                yes_price=h.get("yes_price", h.get("price", 50)) / 100.0,
                volume=h.get("volume", 0),
                open_interest=h.get("open_interest", 0),
            )
            for h in history
        ]
    
    def _get_orderbook(self, ticker: str, depth: int = 25) -> Dict:
        """Get orderbook for a market."""
        params = {"depth": depth}
        data = self._request("GET", f"/markets/{ticker}/orderbook", params=params)
        
        return {
            "ticker": ticker,
            "yes_bids": data.get("yes", {}).get("bids", []),
            "yes_asks": data.get("yes", {}).get("asks", []),
            "no_bids": data.get("no", {}).get("bids", []),
            "no_asks": data.get("no", {}).get("asks", []),
        }
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        data = self._request("GET", "/portfolio/balance")
        return {
            "available": data.get("balance", 0) / 100.0,  # Convert cents to dollars
            "total": data.get("portfolio_value", 0) / 100.0,
        }
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions."""
        data = self._request("GET", "/portfolio/positions")
        positions = data.get("market_positions", [])
        
        return [
            {
                "ticker": p["ticker"],
                "position": p.get("position", 0),  # Positive = YES, Negative = NO
                "market_exposure": p.get("market_exposure", 0) / 100.0,
                "realized_pnl": p.get("realized_pnl", 0) / 100.0,
                "total_cost": p.get("total_cost", 0) / 100.0,
            }
            for p in positions
        ]
    
    def _parse_market(self, data: Dict) -> KalshiMarket:
        """Parse API response into KalshiMarket object."""
        return KalshiMarket(
            ticker=data.get("ticker", ""),
            title=data.get("title", ""),
            subtitle=data.get("subtitle", ""),
            category=data.get("category", ""),
            yes_price=data.get("yes_price", data.get("last_price", 50)),
            no_price=100 - data.get("yes_price", data.get("last_price", 50)),
            yes_bid=data.get("yes_bid", 0),
            yes_ask=data.get("yes_ask", 100),
            volume=data.get("volume", 0),
            open_interest=data.get("open_interest", 0),
            expiration_time=self._parse_timestamp(data.get("expiration_time")),
            close_time=self._parse_timestamp(data.get("close_time")),
            status=data.get("status", "unknown"),
            result=data.get("result"),
        )
    
    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp from various formats."""
        if ts is None:
            return datetime.now(timezone.utc)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return datetime.now(timezone.utc)
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        return datetime.now(timezone.utc)
    
    def healthcheck(self) -> dict:
        """Check API connectivity."""
        try:
            markets = self.get_markets(status="open", limit=5)
            return {
                "source": self.name,
                "ok": True,
                "demo": self.demo,
                "markets": len(markets),
                "authenticated": self._private_key is not None,
            }
        except Exception as e:
            return {
                "source": self.name,
                "ok": False,
                "error": str(e),
            }

    # --- Historical data pipeline for RL training ---

    def fetch_and_normalize_history(
        self,
        ticker: str,
        limit: int = 500,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch price history for a market and normalize to a consistent schema.

        Returns list of dicts with: ticker, timestamp, yes_price (0-1), volume, open_interest.
        Does not include outcome or close_time; those are set when backfilling from market metadata.
        """
        candles = self.get_market_history(
            ticker, limit=limit, min_ts=min_ts, max_ts=max_ts
        )
        return [
            {
                "ticker": ticker,
                "timestamp": c.timestamp,
                "yes_price": c.yes_price,
                "volume": c.volume,
                "open_interest": c.open_interest,
            }
            for c in candles
        ]


def backfill_kalshi_market_to_db(
    adapter: KalshiAdapter,
    session: "Session",
    ticker: str,
    limit: int = 500,
    close_time: Optional[datetime] = None,
    outcome: Optional[int] = None,
) -> int:
    """
    Fetch history for a Kalshi market and persist to the database.

    Args:
        adapter: KalshiAdapter instance.
        session: SQLAlchemy session.
        ticker: Market ticker.
        limit: Max history points to fetch.
        close_time: Market expiration (fetched from get_market if None).
        outcome: 1=YES, 0=NO (fetched from get_market.result if None for settled markets).

    Returns:
        Number of rows inserted.
    """
    from ...data.database import KalshiMarketHistory

    try:
        market = adapter.get_market(ticker)
    except DataUnavailableError:
        logger.warning(f"Could not fetch market {ticker}, skipping metadata")
        market = None

    if close_time is None and market is not None:
        close_time = market.close_time
    if outcome is None and market is not None and market.result:
        outcome = 1 if market.result.lower() == "yes" else 0

    rows = adapter.fetch_and_normalize_history(ticker, limit=limit)
    if not rows:
        return 0

    # Dedupe by (ticker, timestamp)
    existing = {
        (r.ticker, r.timestamp)
        for r in session.query(KalshiMarketHistory).filter(
            KalshiMarketHistory.ticker == ticker
        ).all()
    }
    to_insert = []
    for r in rows:
        ts = r["timestamp"]
        if (ticker, ts) in existing:
            continue
        existing.add((ticker, ts))
        rec = KalshiMarketHistory(
            ticker=ticker,
            timestamp=ts,
            yes_price=r["yes_price"],
            volume=r.get("volume", 0),
            open_interest=r.get("open_interest", 0),
            outcome=outcome,
            close_time=close_time,
        )
        to_insert.append(rec)

    session.add_all(to_insert)
    return len(to_insert)


def load_kalshi_dataset_from_db(
    session: "Session",
    tickers: Optional[List[str]] = None,
    min_rows_per_market: int = 50,
    spread_estimate: float = 0.02,
) -> pd.DataFrame:
    """
    Load Kalshi market history from the database into a DataFrame for KalshiTradingEnv.

    Columns produced: ticker, timestamp, yes_price, no_price, yes_bid, yes_ask,
    volume, time_to_expiry, outcome, signal.
    """
    from ...data.database import KalshiMarketHistory

    q = session.query(KalshiMarketHistory).order_by(
        KalshiMarketHistory.ticker, KalshiMarketHistory.timestamp
    )
    if tickers is not None:
        q = q.filter(KalshiMarketHistory.ticker.in_(tickers))
    rows = q.all()

    if not rows:
        return pd.DataFrame()

    # Build per-ticker DataFrames and filter by min_rows_per_market
    by_ticker: Dict[str, List[Dict]] = {}
    for r in rows:
        by_ticker.setdefault(r.ticker, []).append(
            {
                "ticker": r.ticker,
                "timestamp": r.timestamp,
                "yes_price": r.yes_price,
                "volume": r.volume or 0,
                "open_interest": r.open_interest or 0,
                "outcome": r.outcome,
                "close_time": r.close_time,
            }
        )

    def _to_ts(dt: Optional[datetime]) -> float:
        if dt is None:
            return 0.0
        if hasattr(dt, "timestamp"):
            return dt.timestamp()
        delta = dt - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return delta.total_seconds()

    records = []
    for ticker, list_ in by_ticker.items():
        if len(list_) < min_rows_per_market:
            continue
        for rec in list_:
            ts = rec["timestamp"]
            close = rec["close_time"]
            ts_sec = _to_ts(ts) if ts else 0.0
            close_ts = _to_ts(close) if close else ts_sec + 86400 * 7
            time_to_expiry = max(0.0, (close_ts - ts_sec) / (86400 * 30))  # normalize to ~0-1 (30d max)
            yes_price = rec["yes_price"]
            half = spread_estimate / 2
            records.append({
                "ticker": ticker,
                "timestamp": ts,
                "yes_price": yes_price,
                "no_price": 1.0 - yes_price,
                "yes_bid": yes_price - half,
                "yes_ask": yes_price + half,
                "volume": rec["volume"],
                "time_to_expiry": min(1.0, time_to_expiry),
                "outcome": rec["outcome"],
                "signal": 0.0,
            })

    return pd.DataFrame(records)
