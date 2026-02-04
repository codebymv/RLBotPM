"""
Crypto Data Loader (Real Data Only)

Fetches and stores OHLCV data from real exchange sources.
Raises DataUnavailableError if data cannot be fetched.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd

from ..database import CryptoSymbol, CryptoCandle, DatabaseSession
from ..sources import get_adapter, DataUnavailableError
from ...core.logger import get_logger


logger = get_logger(__name__)


class CryptoDataLoader:
    """
    Loads real OHLCV data from configured exchange adapters.
    """

    def __init__(self, source: str):
        self.source_name = source
        self.adapter = get_adapter(source)

    def sync_symbols(self) -> List[str]:
        """
        Fetch symbols from source and store in DB.
        Returns list of symbols.
        """
        symbols = self.adapter.get_symbols()
        if not symbols:
            raise DataUnavailableError(f"{self.source_name} returned no symbols.")

        with DatabaseSession() as session:
            for symbol in symbols:
                existing = (
                    session.query(CryptoSymbol)
                    .filter_by(source=self.source_name, symbol=symbol)
                    .first()
                )
                if existing:
                    existing.status = "active"
                else:
                    session.add(
                        CryptoSymbol(
                            source=self.source_name,
                            symbol=symbol,
                            status="active",
                            metadata={"source": self.source_name},
                        )
                    )

        logger.info(f"Synced {len(symbols)} symbols from {self.source_name}")
        return symbols

    def collect_ohlcv(
        self,
        symbols: Iterable[str],
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> int:
        """
        Fetch OHLCV candles for symbols and store in DB.
        Returns total candles stored.
        """
        total = 0

        with DatabaseSession() as session:
            for symbol in symbols:
                candles = self.adapter.get_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start=start,
                    end=end,
                    limit=limit,
                )
                if not candles:
                    raise DataUnavailableError(
                        f"No candles returned for {symbol} on {self.source_name}"
                    )

                for candle in candles:
                    session.add(
                        CryptoCandle(
                            source=self.source_name,
                            symbol=symbol,
                            interval=interval,
                            timestamp=candle.timestamp,
                            open=candle.open,
                            high=candle.high,
                            low=candle.low,
                            close=candle.close,
                            volume=candle.volume,
                        )
                    )
                total += len(candles)

        logger.info(
            f"Stored {total} candles from {self.source_name} ({interval})"
        )
        return total

    def load_dataset(
        self,
        symbols: Iterable[str],
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load candles from DB into a DataFrame.
        """
        with DatabaseSession() as session:
            query = session.query(CryptoCandle).filter(
                CryptoCandle.source == self.source_name,
                CryptoCandle.interval == interval,
                CryptoCandle.symbol.in_(list(symbols)),
            )

            if start:
                query = query.filter(CryptoCandle.timestamp >= start)
            if end:
                query = query.filter(CryptoCandle.timestamp <= end)
            if limit:
                query = query.limit(limit)

            rows = query.all()
            if not rows:
                raise DataUnavailableError("No candles in database for requested query.")

            data = [
                {
                    "source": r.source,
                    "symbol": r.symbol,
                    "interval": r.interval,
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]

        return pd.DataFrame(data)
