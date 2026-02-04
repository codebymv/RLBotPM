"""
Kraken adapter (public market data).

Uses Kraken public endpoints:
https://api.kraken.com/0/public
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
import requests

from .base import ExchangeAdapter, OHLCV, DataUnavailableError


class KrakenAdapter(ExchangeAdapter):
    name = "kraken"

    def __init__(self, base_url: str = "https://api.kraken.com/0/public"):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            raise DataUnavailableError(f"Kraken API error: {resp.status_code} {resp.text}")
        data = resp.json()
        if data.get("error"):
            raise DataUnavailableError(f"Kraken API error: {data['error']}")
        return data.get("result", {})

    def get_symbols(self) -> List[str]:
        data = self._get("/AssetPairs")
        if not data:
            raise DataUnavailableError("Kraken returned no asset pairs.")
        return list(data.keys())

    def get_latest_price(self, symbol: str) -> float:
        data = self._get("/Ticker", params={"pair": symbol})
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
        interval_minutes = _interval_to_minutes(interval)
        params: dict = {"pair": symbol, "interval": interval_minutes}
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
