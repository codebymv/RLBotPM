"""
Crypto Data Loader (Real Data Only)

Fetches and stores OHLCV data from real exchange sources.
Raises DataUnavailableError if data cannot be fetched.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import time
from typing import Iterable, List, Optional

import pandas as pd

from ..database import CryptoSymbol, CryptoCandle, DatabaseSession
from ..sources import get_adapter, DataUnavailableError
from ...core.logger import get_logger
from sqlalchemy.exc import OperationalError


logger = get_logger(__name__)


class CryptoDataLoader:
    """
    Loads real OHLCV data from configured exchange adapters.
    """

    def __init__(self, source: str):
        self.source_name = source
        self.adapter = get_adapter(source)

    def sync_symbols(self, symbols: Optional[Iterable[str]] = None) -> List[str]:
        """
        Fetch symbols from source and store in DB.
        Returns list of symbols.
        """
        symbols = list(symbols) if symbols is not None else self.adapter.get_symbols()
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
                            extra_metadata={"source": self.source_name},
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

        for symbol in symbols:
            ranges = _build_request_ranges(start, end, interval, limit)
            if not ranges:
                raise DataUnavailableError(
                    f"No request ranges computed for {symbol}"
                )

            for range_start, range_end, range_limit in ranges:
                for attempt in range(3):
                    try:
                        with DatabaseSession() as session:
                            existing = set(
                                ts
                                for (ts,) in session.query(CryptoCandle.timestamp)
                                .filter_by(
                                    source=self.source_name,
                                    symbol=symbol,
                                    interval=interval,
                                )
                                .filter(CryptoCandle.timestamp >= range_start)
                                .filter(CryptoCandle.timestamp <= range_end)
                                .all()
                            )
                            candles = self.adapter.get_ohlcv(
                                symbol=symbol,
                                interval=interval,
                                start=range_start,
                                end=range_end,
                                limit=range_limit,
                            )
                            if not candles:
                                raise DataUnavailableError(
                                    f"No candles returned for {symbol} on {self.source_name}"
                                )

                            for candle in candles:
                                if candle.timestamp in existing:
                                    continue
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
                                existing.add(candle.timestamp)
                            total += len(candles)
                        break
                    except OperationalError as exc:
                        if attempt >= 2:
                            raise
                        logger.warning(
                            "Database connection lost for %s (%s to %s), retrying (%s/3)",
                            symbol,
                            range_start,
                            range_end,
                            attempt + 2,
                        )
                        time.sleep(2 * (attempt + 1))

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

            query = query.order_by(CryptoCandle.symbol, CryptoCandle.timestamp)

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


def _build_request_ranges(
    start: Optional[datetime],
    end: Optional[datetime],
    interval: str,
    limit: Optional[int],
    max_candles: int = 300,
):
    if limit:
        return [(start, end, limit)]
    if not start or not end:
        return [(start, end, None)]

    interval_seconds = _interval_to_seconds(interval)
    max_window_seconds = max_candles * interval_seconds

    ranges = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(seconds=max_window_seconds), end)
        ranges.append((current_start, current_end, None))
        current_start = current_end

    return ranges


def _interval_to_seconds(interval: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "6h": 21600,
        "1d": 86400,
    }
    if interval not in mapping:
        raise DataUnavailableError(f"Unsupported interval: {interval}")
    return mapping[interval]
