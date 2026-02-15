"""
Kalshi Prediction Markets Execution Client.

Handles live order execution, position management, and trade tracking
for Kalshi binary prediction markets.

API Docs: https://trading-api.kalshi.com/docs
"""

from __future__ import annotations

import base64
import os
import time
import uuid
import yaml
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests

from ..core.logger import get_logger
from ..risk.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)

# Try to import cryptography for RSA signing
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("cryptography library not installed - Kalshi auth will be disabled")


class OrderSide(Enum):
    """Order side (YES or NO)."""
    YES = "yes"
    NO = "no"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass
class KalshiOrder:
    """Represents a Kalshi order."""
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    price: int  # Price in cents (1-99)
    contracts: int
    filled_contracts: int
    status: OrderStatus
    created_at: datetime
    client_order_id: Optional[str] = None


@dataclass
class KalshiPosition:
    """Represents a position in a Kalshi market."""
    ticker: str
    position: int  # Positive = YES contracts, Negative = NO contracts
    market_exposure: float  # In dollars
    realized_pnl: float
    total_cost: float


class KalshiExecutionClient:
    """
    Execution client for Kalshi prediction markets.
    
    Handles:
    - Authentication (RSA signature)
    - Order placement (limit and market)
    - Order management (cancel, modify)
    - Position tracking
    - P&L calculation
    """
    
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
    
    def __init__(self, demo: bool = True):
        """
        Initialize execution client.
        
        Args:
            demo: If True, use demo API (paper trading)
        """
        prod_url = os.getenv("KALSHI_API_BASE_URL", self.PROD_URL).rstrip("/")
        demo_url = os.getenv("KALSHI_DEMO_API_BASE_URL", self.DEMO_URL).rstrip("/")
        self.base_url = demo_url if demo else prod_url
        self.demo = demo
        
        # Load credentials
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
            logger.warning("KALSHI credentials not set or invalid - client will not function")
        
        self._session = requests.Session()
        
        # Order tracking
        self._open_orders: Dict[str, KalshiOrder] = {}
        self._filled_orders: List[KalshiOrder] = []
        
        # Risk: circuit breaker and Kalshi-specific rules
        self._circuit_breaker = CircuitBreaker()
        self._risk_config = self._load_risk_config()
        self._kalshi_config = self._load_kalshi_config()
        
        logger.info(f"KalshiExecutionClient initialized (demo={demo}, base_url={self.base_url})")
    
    def _load_risk_config(self) -> Dict:
        cfg_path = Path(__file__).parents[2] / "shared" / "config" / "risk_config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_kalshi_config(self) -> Dict:
        cfg_path = Path(__file__).parents[2] / "shared" / "config" / "kalshi_config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _can_place_kalshi_order(
        self,
        ticker: str,
        close_time_utc: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Enforce circuit breaker and Kalshi-specific risk rules before placing an order.
        Returns (True, None) if allowed, (False, reason) if blocked.
        """
        if not self._circuit_breaker.can_trade():
            return False, "Circuit breaker: trading paused"
        
        no_trade_seconds = (
            self._risk_config.get("market_requirements", {})
            .get("no_trade_window_before_resolution", 7200)
        )
        if no_trade_seconds and close_time_utc:
            now = datetime.now(timezone.utc)
            if close_time_utc.tzinfo is None:
                close_time_utc = close_time_utc.replace(tzinfo=timezone.utc)
            sec_to_close = (close_time_utc - now).total_seconds()
            if 0 < sec_to_close <= no_trade_seconds:
                return False, f"Within no-trade window ({sec_to_close:.0f}s before resolution)"
        
        trading = self._kalshi_config.get("trading", {})
        max_positions = trading.get("max_concurrent_positions", 10)
        positions = self.get_positions()
        if len(positions) >= max_positions:
            return False, f"Max concurrent positions ({max_positions}) reached"
        
        max_exposure_pct = (
            self._kalshi_config.get("trading", {})
            .get("position_sizing", {})
            .get("max_total_exposure", 0.25)
        )
        if max_exposure_pct > 0:
            avail, total = self.get_balance()
            total_exposure = sum(p.market_exposure for p in positions)
            if total > 0 and (total_exposure / total) >= max_exposure_pct:
                return False, f"Max total exposure ({max_exposure_pct:.0%}) reached"
        
        return True, None
    
    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    
    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """
        Sign a request using RSA-PSS-SHA256.
        
        Kalshi signature format: timestamp_ms + method + path
        Path must be the full URL path (e.g. /trade-api/v2/portfolio/balance).
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
        if not self._private_key:
            return {"error": "Authentication not configured"}
        
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
            
            logger.debug(f"API {method} {endpoint} -> {resp.status_code}")
            
            if resp.status_code not in [200, 201]:
                error_msg = resp.text
                logger.error(f"API error: {resp.status_code} - {error_msg}")
                return {"error": error_msg, "status_code": resp.status_code}
            
            return resp.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {endpoint}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}
    
    # ------------------------------------------------------------------
    # Account Methods
    # ------------------------------------------------------------------
    
    def get_balance(self) -> Tuple[float, float]:
        """
        Get account balance.
        
        Returns:
            (available_balance, total_portfolio_value) in dollars
        """
        resp = self._request("GET", "/portfolio/balance")
        
        if "error" in resp:
            return 0.0, 0.0
        
        available = resp.get("balance", 0) / 100.0  # Convert cents to dollars
        total = resp.get("portfolio_value", available * 100) / 100.0
        
        return available, total
    
    def get_positions(self) -> List[KalshiPosition]:
        """Get all open positions."""
        resp = self._request("GET", "/portfolio/positions")
        
        if "error" in resp:
            return []
        
        positions = []
        for pos in resp.get("market_positions", []):
            positions.append(KalshiPosition(
                ticker=pos["ticker"],
                position=pos.get("position", 0),
                market_exposure=pos.get("market_exposure", 0) / 100.0,
                realized_pnl=pos.get("realized_pnl", 0) / 100.0,
                total_cost=pos.get("total_cost", 0) / 100.0,
            ))
        
        return positions
    
    def get_position(self, ticker: str) -> Optional[KalshiPosition]:
        """Get position for a specific market."""
        positions = self.get_positions()
        for pos in positions:
            if pos.ticker == ticker:
                return pos
        return None
    
    # ------------------------------------------------------------------
    # Order Methods
    # ------------------------------------------------------------------
    
    def place_limit_order(
        self,
        ticker: str,
        side: OrderSide,
        price: int,
        contracts: int,
        expiration_seconds: int = 300,
    ) -> Optional[KalshiOrder]:
        """
        Place a limit order.
        
        Args:
            ticker: Market ticker
            side: YES or NO
            price: Price in cents (1-99)
            contracts: Number of contracts
            expiration_seconds: Cancel order after this many seconds (0 = GTC)
        
        Returns:
            KalshiOrder if successful, None otherwise
        """
        allowed, reason = self._can_place_kalshi_order(ticker, close_time_utc=None)
        if not allowed:
            logger.warning(f"Kalshi order blocked: {reason}")
            return None
        
        # Validate price
        price = max(1, min(99, price))
        
        client_order_id = str(uuid.uuid4())
        
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "type": "limit",
            "action": "buy",
            "side": side.value,
            "count": contracts,
            "yes_price" if side == OrderSide.YES else "no_price": price,
        }
        
        if expiration_seconds > 0:
            body["expiration_ts"] = int(time.time()) + expiration_seconds
        
        logger.info(f"Placing limit order: {contracts} {side.value} {ticker} @ {price}c")
        
        resp = self._request("POST", "/portfolio/orders", json_data=body)
        
        if "error" in resp:
            logger.error(f"Order placement failed: {resp['error']}")
            return None
        
        order = resp.get("order", {})
        kalshi_order = KalshiOrder(
            order_id=order.get("order_id", client_order_id),
            ticker=ticker,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            contracts=contracts,
            filled_contracts=order.get("filled_count", 0),
            status=OrderStatus(order.get("status", "pending")),
            created_at=datetime.now(timezone.utc),
            client_order_id=client_order_id,
        )
        
        self._open_orders[kalshi_order.order_id] = kalshi_order
        logger.info(f"Order placed: {kalshi_order.order_id}")
        
        return kalshi_order
    
    def place_market_order(
        self,
        ticker: str,
        side: OrderSide,
        contracts: int,
    ) -> Optional[KalshiOrder]:
        """
        Place a market order (takes liquidity).
        
        Args:
            ticker: Market ticker
            side: YES or NO
            contracts: Number of contracts
        
        Returns:
            KalshiOrder if successful, None otherwise
        """
        allowed, reason = self._can_place_kalshi_order(ticker, close_time_utc=None)
        if not allowed:
            logger.warning(f"Kalshi order blocked: {reason}")
            return None
        
        client_order_id = str(uuid.uuid4())
        
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "type": "market",
            "action": "buy",
            "side": side.value,
            "count": contracts,
        }
        
        logger.info(f"Placing market order: {contracts} {side.value} {ticker}")
        
        resp = self._request("POST", "/portfolio/orders", json_data=body)
        
        if "error" in resp:
            logger.error(f"Market order failed: {resp['error']}")
            return None
        
        order = resp.get("order", {})
        avg_price = order.get("average_fill_price", 50)
        
        kalshi_order = KalshiOrder(
            order_id=order.get("order_id", client_order_id),
            ticker=ticker,
            side=side,
            order_type=OrderType.MARKET,
            price=avg_price,
            contracts=contracts,
            filled_contracts=order.get("filled_count", contracts),
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            client_order_id=client_order_id,
        )
        
        self._filled_orders.append(kalshi_order)
        logger.info(f"Market order filled: {kalshi_order.order_id} @ {avg_price}c")
        
        return kalshi_order
    
    def sell_position(
        self,
        ticker: str,
        contracts: Optional[int] = None,
        use_market: bool = False,
    ) -> Optional[KalshiOrder]:
        """
        Sell (close) a position.
        
        Args:
            ticker: Market ticker
            contracts: Number to sell (None = entire position)
            use_market: If True, use market order
        
        Returns:
            KalshiOrder if successful
        """
        position = self.get_position(ticker)
        if not position or position.position == 0:
            logger.warning(f"No position to sell for {ticker}")
            return None
        
        # Determine side and contracts
        if position.position > 0:
            side = OrderSide.YES
            num_contracts = contracts or position.position
        else:
            side = OrderSide.NO
            num_contracts = contracts or abs(position.position)
        
        client_order_id = str(uuid.uuid4())
        
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "type": "market" if use_market else "limit",
            "action": "sell",
            "side": side.value,
            "count": num_contracts,
        }
        
        logger.info(f"Selling position: {num_contracts} {side.value} {ticker}")
        
        resp = self._request("POST", "/portfolio/orders", json_data=body)
        
        if "error" in resp:
            logger.error(f"Sell order failed: {resp['error']}")
            return None
        
        order = resp.get("order", {})
        kalshi_order = KalshiOrder(
            order_id=order.get("order_id", client_order_id),
            ticker=ticker,
            side=side,
            order_type=OrderType.MARKET if use_market else OrderType.LIMIT,
            price=order.get("average_fill_price", 50),
            contracts=num_contracts,
            filled_contracts=order.get("filled_count", 0),
            status=OrderStatus(order.get("status", "pending")),
            created_at=datetime.now(timezone.utc),
            client_order_id=client_order_id,
        )
        
        return kalshi_order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        resp = self._request("DELETE", f"/portfolio/orders/{order_id}")
        
        if "error" in resp:
            logger.error(f"Cancel failed for {order_id}: {resp['error']}")
            return False
        
        if order_id in self._open_orders:
            del self._open_orders[order_id]
        
        logger.info(f"Order canceled: {order_id}")
        return True
    
    def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """
        Cancel all open orders, optionally for a specific market.
        
        Returns:
            Number of orders canceled
        """
        params = {}
        if ticker:
            params["ticker"] = ticker
        
        resp = self._request("DELETE", "/portfolio/orders", params=params)
        
        if "error" in resp:
            logger.error(f"Cancel all failed: {resp['error']}")
            return 0
        
        canceled = resp.get("canceled_count", 0)
        self._open_orders.clear()
        
        logger.info(f"Canceled {canceled} orders")
        return canceled
    
    def get_order(self, order_id: str) -> Optional[KalshiOrder]:
        """Get order details by ID."""
        resp = self._request("GET", f"/portfolio/orders/{order_id}")
        
        if "error" in resp:
            return None
        
        order = resp.get("order", {})
        return KalshiOrder(
            order_id=order.get("order_id", order_id),
            ticker=order.get("ticker", ""),
            side=OrderSide(order.get("side", "yes")),
            order_type=OrderType(order.get("type", "limit")),
            price=order.get("price", 50),
            contracts=order.get("count", 0),
            filled_contracts=order.get("filled_count", 0),
            status=OrderStatus(order.get("status", "pending")),
            created_at=datetime.now(timezone.utc),
        )
    
    def get_open_orders(self, ticker: Optional[str] = None) -> List[KalshiOrder]:
        """Get all open orders."""
        params = {"status": "open"}
        if ticker:
            params["ticker"] = ticker
        
        resp = self._request("GET", "/portfolio/orders", params=params)
        
        if "error" in resp:
            return []
        
        orders = []
        for order in resp.get("orders", []):
            orders.append(KalshiOrder(
                order_id=order.get("order_id", ""),
                ticker=order.get("ticker", ""),
                side=OrderSide(order.get("side", "yes")),
                order_type=OrderType(order.get("type", "limit")),
                price=order.get("price", 50),
                contracts=order.get("count", 0),
                filled_contracts=order.get("filled_count", 0),
                status=OrderStatus(order.get("status", "open")),
                created_at=datetime.now(timezone.utc),
            ))
        
        return orders
    
    # ------------------------------------------------------------------
    # Trade History
    # ------------------------------------------------------------------
    
    def get_fills(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get trade fill history."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        
        resp = self._request("GET", "/portfolio/fills", params=params)
        
        if "error" in resp:
            return []
        
        return resp.get("fills", [])
    
    def get_settlements(self, limit: int = 100) -> List[Dict]:
        """Get settlement history."""
        resp = self._request("GET", "/portfolio/settlements", params={"limit": limit})
        
        if "error" in resp:
            return []
        
        return resp.get("settlements", [])
    
    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------
    
    def calculate_required_margin(
        self,
        side: OrderSide,
        price: int,
        contracts: int,
    ) -> float:
        """
        Calculate required margin for a trade.
        
        For Kalshi:
        - Buying YES: margin = contracts * (price / 100)
        - Buying NO: margin = contracts * ((100 - price) / 100)
        - Max loss is capped at the margin
        
        Returns:
            Required margin in dollars
        """
        if side == OrderSide.YES:
            return contracts * (price / 100.0)
        else:
            return contracts * ((100 - price) / 100.0)
    
    def calculate_max_profit(
        self,
        side: OrderSide,
        price: int,
        contracts: int,
    ) -> float:
        """
        Calculate max profit for a trade.
        
        Returns:
            Max profit in dollars
        """
        if side == OrderSide.YES:
            # Win = contracts * $1, cost = contracts * price
            return contracts * (1 - price / 100.0)
        else:
            # Win = contracts * $1, cost = contracts * (100-price)
            return contracts * (price / 100.0)
    
    def healthcheck(self) -> Dict:
        """Check API connectivity and auth."""
        try:
            resp = self._request("GET", "/portfolio/balance")
            if "error" in resp:
                return {
                    "source": "kalshi",
                    "ok": False,
                    "demo": self.demo,
                    "authenticated": self._private_key is not None,
                    "error": resp.get("error", "unknown error"),
                }

            available = resp.get("balance", 0) / 100.0
            total = resp.get("portfolio_value", resp.get("balance", 0)) / 100.0
            return {
                "source": "kalshi",
                "ok": True,
                "demo": self.demo,
                "balance_available": available,
                "balance_total": total,
                "authenticated": self._private_key is not None,
            }
        except Exception as e:
            return {
                "source": "kalshi",
                "ok": False,
                "error": str(e),
            }
