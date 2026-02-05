"""
Database cleanup utilities.

Removes duplicate crypto candles by keeping the lowest id per key.
"""

from __future__ import annotations

from typing import Tuple

from sqlalchemy import text

from .database import get_db_session
from ..core.logger import get_logger


logger = get_logger(__name__)


def count_duplicate_candles() -> int:
    session = get_db_session()
    try:
        result = session.execute(
            text(
                """
                SELECT COALESCE(SUM(cnt - 1), 0) AS dupes
                FROM (
                    SELECT COUNT(*) AS cnt
                    FROM crypto_candles
                    GROUP BY source, symbol, interval, timestamp
                    HAVING COUNT(*) > 1
                ) AS grouped
                """
            )
        ).scalar_one()
        return int(result or 0)
    finally:
        session.close()


def dedupe_crypto_candles() -> int:
    session = get_db_session()
    try:
        result = session.execute(
            text(
                """
                DELETE FROM crypto_candles a
                USING crypto_candles b
                WHERE a.id > b.id
                  AND a.source = b.source
                  AND a.symbol = b.symbol
                  AND a.interval = b.interval
                  AND a.timestamp = b.timestamp
                """
            )
        )
        session.commit()
        deleted = result.rowcount or 0
        logger.info(f"Removed {deleted} duplicate candles")
        return int(deleted)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
