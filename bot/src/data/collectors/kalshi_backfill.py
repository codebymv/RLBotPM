"""
Kalshi Settled Market Backfill

Fetches settled (resolved) binary markets from Kalshi and stores them
in the database for RL training. Targets high-frequency series
(hourly crypto, S&P, forex) that produce hundreds of resolved
episodes per week.

Usage:
    python main.py kalshi backfill-settled --series KXBTC,KXETH --max-per-series 2000
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from ...core.logger import get_logger
from ...data.database import KalshiSettledMarket

logger = get_logger(__name__)

# High-frequency series worth backfilling for RL training
HOURLY_SERIES = [
    "KXBTC",    # Bitcoin range (hourly)
    "KXBTCD",   # Bitcoin above/below (hourly)
    "KXETH",    # Ethereum range (hourly)
    "KXETHD",   # Ethereum above/below (hourly)
    "KXSOL",    # Solana range (hourly)
    "KXSOLD",   # Solana above/below (hourly)
    "KXDOGE",   # Dogecoin range (hourly)
    "KXXRP",    # XRP range (hourly)
    "KXINXU",   # S&P 500 above/below (hourly)
    "INXI",     # S&P 500 hourly
    "KXEURUSDH",  # EUR/USD hourly
    "KXWTIH",   # WTI oil hourly
]

DAILY_SERIES = [
    "INX",      # S&P 500 daily range
    "KXLTCD",   # Litecoin daily
    "KXAVAXD",  # Avalanche daily
    "KXDOTD",   # Polkadot daily
    "TNOTED",   # Treasury 10Y daily yield
]


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _extract_series_ticker(event_ticker: str) -> str:
    """Extract the series ticker from an event ticker.
    
    e.g. 'KXBTC-26FEB1120' -> 'KXBTC'
    """
    parts = event_ticker.split("-")
    return parts[0] if parts else event_ticker


def _safe_float(val) -> Optional[float]:
    """Convert to float, treating empty strings and invalid values as None."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val, default: int = 0) -> int:
    """Convert to int, treating empty strings and invalid values as default."""
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def market_to_record(m: Dict) -> KalshiSettledMarket:
    """Convert a Kalshi API market dict to a KalshiSettledMarket ORM object."""
    result = m.get("result", "no")
    outcome = 1 if result == "yes" else 0

    event_ticker = m.get("event_ticker", "")
    series_ticker = _extract_series_ticker(event_ticker)

    return KalshiSettledMarket(
        ticker=m["ticker"],
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title=m.get("title", ""),
        subtitle=m.get("subtitle", ""),
        strike_type=m.get("strike_type", "") or "",
        floor_strike=_safe_float(m.get("floor_strike")),
        cap_strike=_safe_float(m.get("cap_strike")),
        result=result,
        outcome=outcome,
        expiration_value=_safe_float(m.get("expiration_value")),
        settlement_value=_safe_float(m.get("settlement_value")),
        last_price=_safe_int(m.get("last_price")),
        yes_bid=_safe_int(m.get("yes_bid")),
        yes_ask=_safe_int(m.get("yes_ask")),
        no_bid=_safe_int(m.get("no_bid")),
        no_ask=_safe_int(m.get("no_ask")),
        previous_price=_safe_int(m.get("previous_price")),
        volume=_safe_int(m.get("volume")),
        open_interest=_safe_int(m.get("open_interest")),
        liquidity=_safe_float(m.get("liquidity")),
        open_time=_parse_ts(m.get("open_time")),
        close_time=_parse_ts(m.get("close_time")),
        settlement_ts=_parse_ts(m.get("settlement_ts")),
        created_time=_parse_ts(m.get("created_time")),
        category=m.get("category", ""),
    )


def backfill_settled_series(
    adapter,
    session,
    series_ticker: str,
    max_markets: int = 5000,
    batch_size: int = 200,
    skip_existing: bool = True,
    verbose: bool = True,
) -> int:
    """
    Backfill all settled markets for a given series.

    Args:
        adapter: KalshiAdapter instance
        session: SQLAlchemy session
        series_ticker: e.g. 'KXBTC'
        max_markets: max markets to fetch
        batch_size: API page size (max 200)
        skip_existing: skip tickers already in DB
        verbose: print progress

    Returns:
        Number of rows inserted
    """
    # Get existing tickers to skip
    existing: Set[str] = set()
    if skip_existing:
        rows = (
            session.query(KalshiSettledMarket.ticker)
            .filter(KalshiSettledMarket.series_ticker == series_ticker)
            .all()
        )
        existing = {r[0] for r in rows}

    total_inserted = 0
    cursor = None
    fetched = 0

    while fetched < max_markets:
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": min(batch_size, max_markets - fetched),
        }
        if cursor:
            params["cursor"] = cursor

        try:
            data = adapter._request("GET", "/markets", params=params)
        except Exception as e:
            logger.error(f"API error fetching {series_ticker}: {e}")
            break

        markets = data.get("markets", [])
        cursor = data.get("cursor")

        if not markets:
            break

        to_insert = []
        for m in markets:
            ticker = m.get("ticker", "")
            if ticker in existing:
                continue
            existing.add(ticker)
            try:
                record = market_to_record(m)
                to_insert.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse {ticker}: {e}")

        if to_insert:
            session.add_all(to_insert)
            session.flush()
            total_inserted += len(to_insert)

        fetched += len(markets)

        if verbose and fetched % 1000 == 0:
            logger.info(f"  {series_ticker}: fetched {fetched}, inserted {total_inserted}")

        if not cursor:
            break

        # Rate limit
        time.sleep(0.1)

    return total_inserted


def backfill_all_series(
    adapter,
    session,
    series_list: Optional[List[str]] = None,
    max_per_series: int = 5000,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Backfill settled markets for all target series.

    Args:
        adapter: KalshiAdapter instance
        session: SQLAlchemy session
        series_list: list of series tickers (default: HOURLY_SERIES + DAILY_SERIES)
        max_per_series: max markets per series
        verbose: print progress

    Returns:
        Dict of series_ticker -> rows_inserted
    """
    if series_list is None:
        series_list = HOURLY_SERIES + DAILY_SERIES

    results = {}
    for series in series_list:
        if verbose:
            logger.info(f"Backfilling {series}...")
        n = backfill_settled_series(
            adapter, session, series,
            max_markets=max_per_series,
            verbose=verbose,
        )
        results[series] = n
        if verbose:
            logger.info(f"  {series}: +{n} settled markets")

    return results
