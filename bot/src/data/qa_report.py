"""
Data quality checks for OHLCV candles.

Generates a simple report for gaps, duplicates, and missing values.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

from .collectors import CryptoDataLoader
from .sources.base import DataUnavailableError
from .database import CryptoSymbol, get_db_session
from ..core.config import get_settings
from ..core.logger import get_logger


logger = get_logger(__name__)


def _interval_to_pandas_freq(interval: str) -> str:
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    if interval not in mapping:
        raise DataUnavailableError(f"Unsupported interval: {interval}")
    return mapping[interval]


def _get_symbols(source: str) -> List[str]:
    settings = get_settings()
    if settings.DATA_SYMBOLS:
        return [s.strip() for s in settings.DATA_SYMBOLS.split(",") if s.strip()]

    session = get_db_session()
    try:
        return [
            row.symbol
            for row in session.query(CryptoSymbol)
            .filter_by(source=source, status="active")
            .all()
        ]
    finally:
        session.close()


def build_data_quality_report(days: int | None = None) -> Dict:
    settings = get_settings()
    source = settings.DATA_SOURCE
    interval = settings.DATA_INTERVAL
    window_days = days or settings.REQUIRE_HISTORICAL_DAYS

    symbols = _get_symbols(source)
    if not symbols:
        raise DataUnavailableError("No symbols found in database.")

    loader = CryptoDataLoader(source=source)
    end = datetime.utcnow()
    start = end - timedelta(days=window_days)

    dataset = loader.load_dataset(
        symbols=symbols,
        interval=interval,
        start=start,
        end=end,
    )

    if dataset is None or dataset.empty:
        raise DataUnavailableError("Dataset is empty after loading.")

    dataset = dataset.copy()
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])

    freq = _interval_to_pandas_freq(interval)
    expected_delta = pd.Timedelta(freq)

    summary = {
        "source": source,
        "interval": interval,
        "days": window_days,
        "symbols": len(symbols),
        "total_rows": int(len(dataset)),
        "total_missing": 0,
        "total_duplicates": 0,
        "symbols_report": {},
    }

    for symbol, group in dataset.groupby("symbol"):
        df = group.sort_values("timestamp")
        unique_timestamps = df["timestamp"].drop_duplicates()
        duplicates = int(df["timestamp"].duplicated().sum())

        if unique_timestamps.empty:
            continue

        expected = pd.date_range(
            start=unique_timestamps.min(),
            end=unique_timestamps.max(),
            freq=freq,
        )
        missing = int(len(expected) - len(unique_timestamps))
        missing_pct = float(missing / len(expected)) if len(expected) else 0.0

        diffs = unique_timestamps.diff().dropna()
        gap_count = int((diffs > expected_delta).sum())
        max_gap = float(diffs.max().total_seconds() / 3600.0) if not diffs.empty else 0.0

        nan_counts = df[["open", "high", "low", "close", "volume"]].isna().sum().to_dict()

        summary["total_missing"] += missing
        summary["total_duplicates"] += duplicates
        summary["symbols_report"][symbol] = {
            "rows": int(len(df)),
            "missing": missing,
            "missing_pct": missing_pct,
            "duplicates": duplicates,
            "gap_count": gap_count,
            "max_gap_hours": max_gap,
            "nan_counts": {k: int(v) for k, v in nan_counts.items()},
        }

    return summary
