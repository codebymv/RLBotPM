"""
Kraken adapter (public market data).

Uses Kraken public endpoints:
https://api.kraken.com/0/public
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
import time
import requests

from .base import ExchangeAdapter, OHLCV, DataUnavailableError


# Standard symbol -> Kraken symbol mapping
# Kraken uses XBT for Bitcoin and different pair formats
SYMBOL_MAP = {
    "BTC-USD": "XBTUSD",
    "BTC-EUR": "XBTEUR",
    "BTC-GBP": "XBTGBP",
    "ETH-USD": "ETHUSD",
    "ETH-EUR": "ETHEUR",
    "ETH-BTC": "ETHXBT",
    "SOL-USD": "SOLUSD",
    "SOL-EUR": "SOLEUR",
    "DOGE-USD": "XDGUSD",
    "XRP-USD": "XRPUSD",
    "LTC-USD": "LTCUSD",
    "ADA-USD": "ADAUSD",
    "DOT-USD": "DOTUSD",
    "LINK-USD": "LINKUSD",
    "AVAX-USD": "AVAXUSD",
    "MATIC-USD": "MATICUSD",
}


class KrakenAdapter(ExchangeAdapter):
    name = "kraken"

    def __init__(self, base_url: str = "https://api.kraken.com/0/public"):
        self.base_url = base_url.rstrip("/")

    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert standard symbol format (BTC-USD) to Kraken format (XBTUSD)."""
        # Check explicit mapping first
        if symbol in SYMBOL_MAP:
            return SYMBOL_MAP[symbol]
        # Try removing hyphen as fallback
        return symbol.replace("-", "")

    def _get(self, path: str, params: Optional[dict] = None, max_retries: int = 5) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    raise DataUnavailableError(f"Kraken API error: {resp.status_code} {resp.text}")
                data = resp.json()
                if data.get("error"):
                    errors = data['error']
                    # Check for rate limit error
                    if any("Too many requests" in str(e) for e in errors):
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 2  # 2, 4, 8, 16, 32 seconds
                            time.sleep(wait_time)
                            continue
                    raise DataUnavailableError(f"Kraken API error: {errors}")
                return data.get("result", {})
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
                raise DataUnavailableError(f"Kraken request failed: {e}")
        
        raise DataUnavailableError("Kraken API max retries exceeded")

    def get_symbols(self) -> List[str]:
        data = self._get("/AssetPairs")
        if not data:
            raise DataUnavailableError("Kraken returned no asset pairs.")
        return list(data.keys())

    def get_latest_price(self, symbol: str) -> float:
        kraken_symbol = self._to_kraken_symbol(symbol)
        data = self._get("/Ticker", params={"pair": kraken_symbol})
        if not data:
            raise DataUnavailableError(f"Kraken ticker missing for {symbol}.")
        pair_data = next(iter(data.values()))
        price = pair_data.get("c", [None])[0]
        if price is None:
            raise DataUnavailableError(f"Kraken ticker missing price for {symbol}.")
        return float(price)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        kraken_symbol = self._to_kraken_symbol(symbol)
        interval_minutes = _interval_to_minutes(interval)
        params: dict = {"pair": kraken_symbol, "interval": interval_minutes}
        if start:
            params["since"] = int(start.timestamp())
        data = self._get("/OHLC", params=params)
        if not data:
            raise DataUnavailableError(f"Kraken OHLC missing for {symbol}.")

        pair_key = next(iter(data.keys()))
        rows = data[pair_key]
        if not rows:
            raise DataUnavailableError(f"Kraken OHLC empty for {symbol}.")

        candles: List[OHLCV] = []
        for row in rows[: limit or len(rows)]:
            # Kraken format: [time, open, high, low, close, vwap, volume, count]
            try:
                ts = datetime.utcfromtimestamp(row[0])
                candles.append(
                    OHLCV(
                        timestamp=ts,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[6]),
                    )
                )
            except (ValueError, IndexError) as exc:
                raise DataUnavailableError(f"Invalid candle data for {symbol}: {row}") from exc

        return candles


def _interval_to_minutes(interval: str) -> int:
    mapping = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    if interval not in mapping:
        raise DataUnavailableError(f"Unsupported interval for Kraken: {interval}")
    return mapping[interval]
