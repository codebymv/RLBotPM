"""
Coinbase Exchange adapter (public market data).

Uses Coinbase Exchange public endpoints for market data:
https://api.exchange.coinbase.com
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
import requests

from .base import ExchangeAdapter, OHLCV, DataUnavailableError


class CoinbaseAdapter(ExchangeAdapter):
    name = "coinbase"

    def __init__(self, base_url: str = "https://api.exchange.coinbase.com"):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, params: Optional[dict] = None) -> dict | list:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            raise DataUnavailableError(f"Coinbase API error: {resp.status_code} {resp.text}")
        return resp.json()

    def get_symbols(self) -> List[str]:
        data = self._get("/products")
        if not isinstance(data, list) or not data:
            raise DataUnavailableError("Coinbase returned no products.")
        return [item["id"] for item in data if "id" in item]

    def get_latest_price(self, symbol: str) -> float:
        data = self._get(f"/products/{symbol}/ticker")
        price = data.get("price")
        if price is None:
            raise DataUnavailableError(f"Coinbase ticker missing price for {symbol}.")
        return float(price)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        granularity = _interval_to_granularity(interval)
        params: dict = {"granularity": granularity}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()
        data = self._get(f"/products/{symbol}/candles", params=params)

        if not isinstance(data, list) or not data:
            raise DataUnavailableError(f"Coinbase candles empty for {symbol}.")

        candles: List[OHLCV] = []
        for item in data[: limit or len(data)]:
            # Coinbase candle format: [time, low, high, open, close, volume]
            try:
                ts = datetime.utcfromtimestamp(item[0])
                candles.append(
                    OHLCV(
                        timestamp=ts,
                        low=float(item[1]),
                        high=float(item[2]),
                        open=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                    )
                )
            except (ValueError, IndexError) as exc:
                raise DataUnavailableError(f"Invalid candle data for {symbol}: {item}") from exc

        return candles


def _interval_to_granularity(interval: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "6h": 21600,
        "1d": 86400,
    }
    if interval not in mapping:
        raise DataUnavailableError(f"Unsupported interval for Coinbase: {interval}")
    return mapping[interval]
