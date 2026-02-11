"""
Coinbase Advanced Trade API Client for live order execution.

Uses the new Coinbase Advanced Trade REST API (not the legacy Pro API).
API Docs: https://docs.cdp.coinbase.com/advanced-trade/docs/welcome
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import jwt
import requests
from cryptography.hazmat.primitives import serialization

from ..core.logger import get_logger
from ..core.config import get_settings

logger = get_logger(__name__)


class CoinbaseExecutionClient:
    """
    Client for executing real trades on Coinbase Advanced Trade.
    
    Supports:
    - Limit orders (maker)
    - Market orders (taker)
    - Order status checking
    - Order cancellation
    - Account balance queries
    """
    
    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"
    
    def __init__(self):
        self.settings = get_settings()
        
        # Load API credentials from environment
        self.api_key = os.getenv("COINBASE_API_KEY", "")
        self.api_secret = os.getenv("COINBASE_API_SECRET", "")
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "COINBASE_API_KEY and COINBASE_API_SECRET must be set in environment"
            )
        
        # Parse the EC private key for JWT signing
        self._private_key = self._load_private_key()
        
        logger.info("CoinbaseExecutionClient initialized")
    
    def _load_private_key(self):
        """Load and parse the EC private key from the secret."""
        # Handle escaped newlines in env var (literal \n strings)
        key_str = self.api_secret
        
        # Replace literal \n with actual newlines
        if "\\n" in key_str:
            key_str = key_str.replace("\\n", "\n")
        
        # Also handle double-escaped
        if "\\\\n" in key_str:
            key_str = key_str.replace("\\\\n", "\n")
        
        try:
            private_key = serialization.load_pem_private_key(
                key_str.encode(),
                password=None
            )
            return private_key
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            logger.debug(f"Key starts with: {key_str[:50]}...")
            raise ValueError(f"Invalid COINBASE_API_SECRET format: {e}")
    
    def _generate_jwt(self, method: str, path: str) -> str:
        """Generate a JWT token for Coinbase API authentication."""
        uri = f"{method} api.coinbase.com{path}"
        
        payload = {
            "sub": self.api_key,
            "iss": "coinbase-cloud",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,  # 2 minute expiry
            "uri": uri,
        }
        
        token = jwt.encode(
            payload,
            self._private_key,
            algorithm="ES256",
            headers={"kid": self.api_key, "nonce": str(uuid.uuid4())}
        )
        
        return token
    
    def _request(
        self,
        method: str,
        endpoint: str,
        body: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make an authenticated request to Coinbase API."""
        url = f"{self.BASE_URL}{endpoint}"
        path = f"/api/v3/brokerage{endpoint}"
        
        # Generate JWT
        jwt_token = self._generate_jwt(method, path)
        
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=body, timeout=30)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Log request for debugging
            logger.debug(f"API {method} {endpoint} -> {response.status_code}")
            
            if response.status_code not in [200, 201]:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
            
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {endpoint}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}
    
    # ------------------------------------------------------------------
    # Account Methods
    # ------------------------------------------------------------------
    
    def get_accounts(self) -> List[Dict]:
        """Get all accounts (balances)."""
        response = self._request("GET", "/accounts")
        
        if "error" in response:
            return []
        
        return response.get("accounts", [])
    
    def get_account_balance(self, currency: str = "USD") -> Tuple[float, float]:
        """
        Get available and total balance for a currency.
        
        Returns:
            (available_balance, total_balance)
        """
        accounts = self.get_accounts()
        
        for account in accounts:
            if account.get("currency") == currency:
                available = float(account.get("available_balance", {}).get("value", 0))
                total = float(account.get("hold", {}).get("value", 0)) + available
                return available, total
        
        return 0.0, 0.0
    
    def get_btc_balance(self) -> Tuple[float, float]:
        """Get BTC balance (available, total)."""
        return self.get_account_balance("BTC")
    
    def get_usd_balance(self) -> Tuple[float, float]:
        """Get USD balance (available, total)."""
        return self.get_account_balance("USD")
    
    # ------------------------------------------------------------------
    # Order Methods
    # ------------------------------------------------------------------
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float,
        post_only: bool = True,
        time_in_force: str = "GTC"  # GTC, IOC, FOK
    ) -> Dict:
        """
        Place a limit order (maker order).
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: "BUY" or "SELL"
            price: Limit price
            size: Order size in base currency (e.g., BTC amount)
            post_only: If True, order will only be maker (cancel if would be taker)
            time_in_force: GTC (Good Till Cancel), IOC, FOK
        
        Returns:
            Order response dict with order_id
        """
        client_order_id = str(uuid.uuid4())
        
        order_config = {
            "limit_limit_gtc": {
                "base_size": str(size),
                "limit_price": str(price),
                "post_only": post_only,
            }
        }
        
        if time_in_force == "IOC":
            order_config = {
                "limit_limit_ioc": {
                    "base_size": str(size),
                    "limit_price": str(price),
                }
            }
        elif time_in_force == "FOK":
            order_config = {
                "limit_limit_fok": {
                    "base_size": str(size),
                    "limit_price": str(price),
                }
            }
        
        body = {
            "client_order_id": client_order_id,
            "product_id": symbol,
            "side": side.upper(),
            "order_configuration": order_config,
        }
        
        logger.info(f"Placing limit {side} order: {size} {symbol} @ ${price}")
        
        response = self._request("POST", "/orders", body=body)
        
        if "error" not in response:
            order_id = response.get("order_id") or response.get("success_response", {}).get("order_id")
            logger.info(f"Order placed: {order_id}")
        else:
            logger.error(f"Order failed: {response}")
        
        return response
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        size: Optional[float] = None,
        quote_size: Optional[float] = None,
    ) -> Dict:
        """
        Place a market order (taker order).
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: "BUY" or "SELL"
            size: Order size in base currency (e.g., BTC amount) - for SELL
            quote_size: Order size in quote currency (e.g., USD amount) - for BUY
        
        Returns:
            Order response dict
        """
        client_order_id = str(uuid.uuid4())
        
        if side.upper() == "BUY":
            if quote_size is None:
                raise ValueError("quote_size required for market BUY")
            order_config = {
                "market_market_ioc": {
                    "quote_size": str(quote_size),
                }
            }
        else:
            if size is None:
                raise ValueError("size required for market SELL")
            order_config = {
                "market_market_ioc": {
                    "base_size": str(size),
                }
            }
        
        body = {
            "client_order_id": client_order_id,
            "product_id": symbol,
            "side": side.upper(),
            "order_configuration": order_config,
        }
        
        logger.info(f"Placing market {side} order: {symbol}")
        
        response = self._request("POST", "/orders", body=body)
        
        if "error" not in response:
            order_id = response.get("order_id") or response.get("success_response", {}).get("order_id")
            logger.info(f"Market order placed: {order_id}")
        else:
            logger.error(f"Market order failed: {response}")
        
        return response
    
    def get_order(self, order_id: str) -> Dict:
        """Get order status by ID."""
        return self._request("GET", f"/orders/historical/{order_id}")
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an open order."""
        body = {"order_ids": [order_id]}
        return self._request("POST", "/orders/batch_cancel", body=body)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders, optionally filtered by symbol."""
        params = {"order_status": "OPEN"}
        if symbol:
            params["product_id"] = symbol
        
        response = self._request("GET", "/orders/historical", params=params)
        
        if "error" in response:
            return []
        
        return response.get("orders", [])
    
    # ------------------------------------------------------------------
    # Product/Market Methods
    # ------------------------------------------------------------------
    
    def get_product(self, symbol: str) -> Dict:
        """Get product details (price, min order size, etc.)."""
        return self._request("GET", f"/products/{symbol}")
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker (best bid/ask, last price)."""
        response = self._request("GET", f"/products/{symbol}/ticker")
        return response
    
    def get_best_bid_ask(self, symbol: str) -> Tuple[float, float]:
        """
        Get best bid and ask prices.
        
        Returns:
            (best_bid, best_ask)
        """
        ticker = self.get_ticker(symbol)
        
        if "error" in ticker:
            return 0.0, 0.0
        
        best_bid = float(ticker.get("best_bid", 0))
        best_ask = float(ticker.get("best_ask", 0))
        
        return best_bid, best_ask


# ------------------------------------------------------------------
# Test Connection Function
# ------------------------------------------------------------------

def test_coinbase_connection() -> bool:
    """Test API connection and print account info."""
    try:
        client = CoinbaseExecutionClient()
        
        # Test account access
        print("\n" + "=" * 50)
        print("COINBASE API CONNECTION TEST")
        print("=" * 50)
        
        usd_avail, usd_total = client.get_usd_balance()
        btc_avail, btc_total = client.get_btc_balance()
        
        print(f"\nUSD Balance: ${usd_avail:.2f} available / ${usd_total:.2f} total")
        print(f"BTC Balance: {btc_avail:.8f} available / {btc_total:.8f} total")
        
        # Test market data
        bid, ask = client.get_best_bid_ask("BTC-USD")
        spread = (ask - bid) / bid * 100 if bid > 0 else 0
        
        print(f"\nBTC-USD Market:")
        print(f"  Best Bid: ${bid:,.2f}")
        print(f"  Best Ask: ${ask:,.2f}")
        print(f"  Spread:   {spread:.4f}%")
        
        print("\n✅ API connection successful!")
        print("=" * 50 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API connection failed: {e}")
        return False


if __name__ == "__main__":
    test_coinbase_connection()
