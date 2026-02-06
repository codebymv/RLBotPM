"""Collect OHLCV data per symbol with retries."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_settings
from src.data.collectors.crypto_loader import CryptoDataLoader
from src.data.sources.base import DataUnavailableError


def collect_interval(interval: str, days: int, max_retries: int = 3) -> None:
    settings = get_settings()
    symbols = [s.strip() for s in settings.DATA_SYMBOLS.split(",") if s.strip()]
    loader = CryptoDataLoader(source=settings.DATA_SOURCE)
    loader.sync_symbols(symbols)

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    for symbol in symbols:
        for attempt in range(1, max_retries + 1):
            try:
                loader.collect_ohlcv(
                    symbols=[symbol],
                    interval=interval,
                    start=start,
                    end=end,
                )
                break
            except DataUnavailableError as exc:
                if attempt == max_retries:
                    print(f"[WARN] Skipping {symbol} after {max_retries} attempts: {exc}")
                else:
                    time.sleep(2 * attempt)


if __name__ == "__main__":
    collect_interval("5m", days=30, max_retries=5)
