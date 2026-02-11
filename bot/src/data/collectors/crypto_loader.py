"""
Crypto Data Loader (Real Data Only)

Fetches and stores OHLCV data from real exchange sources.
Raises DataUnavailableError if data cannot be fetched.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
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


class MultiSourceLoader:
    """
    Loads and aligns OHLCV data from multiple exchanges for arbitrage analysis.
    
    Fetches same symbols from multiple sources and aligns by timestamp,
    computing cross-exchange spread features.
    """

    def __init__(self, sources: List[str]):
        """
        Initialize multi-source loader.
        
        Args:
            sources: List of exchange names (e.g., ["coinbase", "kraken"])
        """
        if len(sources) < 2:
            raise DataUnavailableError("MultiSourceLoader requires at least 2 sources for arbitrage")
        
        self.sources = sources
        self.loaders = {source: CryptoDataLoader(source) for source in sources}
        self.primary_source = sources[0]  # Primary exchange for execution
        logger.info(f"MultiSourceLoader initialized with sources: {sources}")

    def sync_all_symbols(self, symbols: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
        """
        Sync symbols across all exchanges.
        
        Returns dict of source -> symbols available.
        """
        result = {}
        for source, loader in self.loaders.items():
            try:
                result[source] = loader.sync_symbols(symbols)
            except DataUnavailableError as e:
                logger.warning(f"Failed to sync symbols from {source}: {e}")
                result[source] = []
        return result

    def collect_all_ohlcv(
        self,
        symbols: Iterable[str],
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Collect OHLCV data from all sources.
        
        Returns dict of source -> candles_stored.
        """
        result = {}
        for source, loader in self.loaders.items():
            try:
                result[source] = loader.collect_ohlcv(symbols, interval, start, end, limit)
            except DataUnavailableError as e:
                logger.warning(f"Failed to collect from {source}: {e}")
                result[source] = 0
        return result

    def load_aligned_dataset(
        self,
        symbols: Iterable[str],
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load and align data from all sources, computing spread features.
        
        Returns DataFrame with columns:
            symbol, timestamp, open, high, low, close, volume (from primary)
            + spread features: price_diff_pct, spread_zscore, spread_ma_ratio, etc.
        """
        symbols_list = list(symbols)
        dfs = {}
        
        # Load from each source
        for source, loader in self.loaders.items():
            try:
                df = loader.load_dataset(symbols_list, interval, start, end)
                df = df.rename(columns={
                    "close": f"close_{source}",
                    "open": f"open_{source}",
                    "high": f"high_{source}",
                    "low": f"low_{source}",
                    "volume": f"volume_{source}",
                })
                dfs[source] = df
            except DataUnavailableError as e:
                logger.warning(f"No data from {source}: {e}")
        
        if not dfs:
            raise DataUnavailableError("No data available from any source")
        
        if len(dfs) < 2:
            logger.warning("Only one source has data - spread features will be zero")
            # Return primary with zero spreads
            primary_df = list(dfs.values())[0]
            primary_source = list(dfs.keys())[0]
            return self._add_zero_spread_features(primary_df, primary_source)
        
        # Merge on symbol + timestamp
        merged = self._merge_sources(dfs, symbols_list)
        
        # Compute spread features
        merged = self._compute_spread_features(merged)
        
        return merged

    def _merge_sources(self, dfs: Dict[str, pd.DataFrame], symbols: List[str]) -> pd.DataFrame:
        """Merge dataframes from multiple sources by symbol and timestamp."""
        
        # Start with primary source
        primary_df = dfs[self.primary_source].copy()
        primary_df["timestamp"] = pd.to_datetime(primary_df["timestamp"])
        
        # Rename primary columns back to standard names
        primary_df = primary_df.rename(columns={
            f"close_{self.primary_source}": "close",
            f"open_{self.primary_source}": "open",
            f"high_{self.primary_source}": "high",
            f"low_{self.primary_source}": "low",
            f"volume_{self.primary_source}": "volume",
        })
        
        # Keep track of other source prices
        for source, df in dfs.items():
            if source == self.primary_source:
                continue
                
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Select only price columns from secondary sources
            price_cols = [f"close_{source}", f"high_{source}", f"low_{source}"]
            df_subset = df[["symbol", "timestamp"] + [c for c in price_cols if c in df.columns]]
            
            # Merge on symbol + timestamp (inner join to keep only aligned rows)
            primary_df = primary_df.merge(
                df_subset,
                on=["symbol", "timestamp"],
                how="inner"
            )
        
        logger.info(f"Merged {len(primary_df)} aligned rows across {len(dfs)} sources")
        return primary_df

    def _compute_spread_features(self, df: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
        """
        Compute cross-exchange spread features.
        
        Features added:
            - price_diff_pct: (primary - secondary) / avg_price
            - spread_zscore: standardized spread deviation
            - spread_ma_ratio: current_spread / rolling_mean_spread
            - spread_direction: sign of spread change
        """
        df = df.copy()
        
        # Find secondary source columns
        secondary_close_cols = [c for c in df.columns if c.startswith("close_") and c != f"close_{self.primary_source}"]
        
        if not secondary_close_cols:
            return self._add_zero_spread_features(df, self.primary_source)
        
        # Use first secondary source for spread calculation
        secondary_col = secondary_close_cols[0]
        secondary_source = secondary_col.replace("close_", "")
        
        # Store symbol mapping before groupby (groupby drops the key column from groups)
        symbol_col = df["symbol"].copy()
        
        def compute_symbol_spreads(group: pd.DataFrame) -> pd.DataFrame:
            g = group.copy()
            
            primary_price = g["close"]
            secondary_price = g[secondary_col]
            avg_price = (primary_price + secondary_price) / 2
            
            # Raw spread as percentage
            g["price_diff_pct"] = (primary_price - secondary_price) / (avg_price + 1e-9)
            
            # Rolling stats for spread
            spread_ma = g["price_diff_pct"].rolling(lookback, min_periods=1).mean()
            spread_std = g["price_diff_pct"].rolling(lookback, min_periods=1).std().fillna(1e-6)
            
            # Z-score of spread (how unusual is current spread)
            g["spread_zscore"] = (g["price_diff_pct"] - spread_ma) / (spread_std + 1e-9)
            
            # Ratio of current spread to rolling mean
            g["spread_ma_ratio"] = g["price_diff_pct"] / (spread_ma.abs() + 1e-6)
            
            # Direction of spread change
            g["spread_direction"] = np.sign(g["price_diff_pct"].diff())
            
            return g
        
        df = df.groupby("symbol", group_keys=False).apply(compute_symbol_spreads)
        
        # Restore symbol column (groupby drops it from groups)
        df["symbol"] = symbol_col.values
        
        # Fill NaN from rolling calcs
        spread_cols = ["price_diff_pct", "spread_zscore", "spread_ma_ratio", "spread_direction"]
        for col in spread_cols:
            df[col] = df[col].fillna(0.0)
        
        # Store secondary source name for reference
        df["secondary_source"] = secondary_source
        
        logger.info(f"Computed spread features: primary={self.primary_source}, secondary={secondary_source}")
        return df

    def _add_zero_spread_features(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Add zero-valued spread features when only one source available."""
        df = df.copy()
        
        # Rename columns back to standard
        rename_map = {
            f"close_{source}": "close",
            f"open_{source}": "open", 
            f"high_{source}": "high",
            f"low_{source}": "low",
            f"volume_{source}": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Add zero spread features
        df["price_diff_pct"] = 0.0
        df["spread_zscore"] = 0.0
        df["spread_ma_ratio"] = 0.0
        df["spread_direction"] = 0.0
        df["secondary_source"] = None
        
        return df
