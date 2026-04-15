"""
Unified spot price feeds for multiple asset classes.

Provides a single get_spot_price(asset) interface backed by:
  - Coinbase for crypto (BTC, ETH, SOL, DOGE, XRP)
  - Yahoo Finance v8 for equities/indices (SPX via ^GSPC)
  - exchangerate.host / ECB for FX (EUR/USD)
  - Yahoo Finance for commodities (WTI via CL=F)
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import requests

from ...core.logger import get_logger

logger = get_logger(__name__)

_cache: Dict[str, Tuple[float, float]] = {}  # asset -> (price, timestamp)


def _cached(asset: str, max_age_s: int = 30) -> Optional[float]:
    entry = _cache.get(asset)
    if entry and (time.time() - entry[1]) <= max_age_s:
        return entry[0]
    return None


def _store(asset: str, price: float) -> float:
    _cache[asset] = (price, time.time())
    return price


def _get_json(url: str, params: Optional[dict] = None, timeout: int = 10) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=timeout, headers={
            "User-Agent": "RLBotPM/1.0",
        })
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("HTTP GET %s failed: %s", url, exc)
    return None


# ── Crypto via Coinbase ──────────────────────────────────────────────

_CRYPTO_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    "XRP": "XRP-USD",
}


def _fetch_crypto(asset: str) -> Optional[float]:
    symbol = _CRYPTO_SYMBOLS.get(asset)
    if not symbol:
        return None
    data = _get_json(f"https://api.exchange.coinbase.com/products/{symbol}/ticker")
    if data and data.get("price"):
        return float(data["price"])
    return None


# ── S&P 500 via Yahoo Finance chart endpoint ─────────────────────────

_INDEX_YAHOO_SYMBOLS = {
    "SPX": "^GSPC",
    "INXU": "^GSPC",
    "INX": "^GSPC",
}


def _fetch_index(asset: str) -> Optional[float]:
    symbol = _INDEX_YAHOO_SYMBOLS.get(asset)
    if not symbol:
        return None
    data = _get_json(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"interval": "1m", "range": "1d"},
    )
    if not data:
        return None
    try:
        meta = data["chart"]["result"][0]["meta"]
        return float(meta["regularMarketPrice"])
    except (KeyError, IndexError, TypeError):
        pass
    return None


# ── EUR/USD via ECB / exchangerate.host ──────────────────────────────

def _fetch_eurusd() -> Optional[float]:
    data = _get_json("https://open.er-api.com/v6/latest/EUR")
    if data and data.get("rates", {}).get("USD"):
        return float(data["rates"]["USD"])
    data = _get_json("https://api.exchangerate.host/latest", params={"base": "EUR", "symbols": "USD"})
    if data and data.get("rates", {}).get("USD"):
        return float(data["rates"]["USD"])
    return None


# ── WTI Oil via Yahoo Finance ────────────────────────────────────────

def _fetch_wti() -> Optional[float]:
    data = _get_json(
        "https://query1.finance.yahoo.com/v8/finance/chart/CL=F",
        params={"interval": "1m", "range": "1d"},
    )
    if not data:
        return None
    try:
        meta = data["chart"]["result"][0]["meta"]
        return float(meta["regularMarketPrice"])
    except (KeyError, IndexError, TypeError):
        pass
    return None


# ── Treasury 10Y Yield via Yahoo Finance ─────────────────────────────

def _fetch_tnote() -> Optional[float]:
    data = _get_json(
        "https://query1.finance.yahoo.com/v8/finance/chart/^TNX",
        params={"interval": "1m", "range": "1d"},
    )
    if not data:
        return None
    try:
        meta = data["chart"]["result"][0]["meta"]
        return float(meta["regularMarketPrice"])
    except (KeyError, IndexError, TypeError):
        pass
    return None


# ── Public API ───────────────────────────────────────────────────────

_ASSET_FETCHERS = {
    "BTC": _fetch_crypto,
    "ETH": _fetch_crypto,
    "SOL": _fetch_crypto,
    "DOGE": _fetch_crypto,
    "XRP": _fetch_crypto,
    "SPX": _fetch_index,
    "INXU": _fetch_index,
    "INX": _fetch_index,
    "EURUSD": _fetch_eurusd,
    "WTI": _fetch_wti,
    "TNOTE": _fetch_tnote,
}


def get_spot_price(asset: str, max_age_s: int = 30) -> Optional[float]:
    """
    Get the current spot price for an asset.

    Supported assets: BTC, ETH, SOL, DOGE, XRP, SPX, INXU, INX,
                      EURUSD, WTI, TNOTE
    """
    cached = _cached(asset, max_age_s)
    if cached is not None:
        return cached

    fetcher = _ASSET_FETCHERS.get(asset)
    if fetcher is None:
        return None

    try:
        if asset in ("EURUSD", "WTI", "TNOTE"):
            price = fetcher()
        else:
            price = fetcher(asset)
        if price is not None and price > 0:
            return _store(asset, price)
    except Exception as exc:
        logger.warning("Spot price fetch failed for %s: %s", asset, exc)

    return None


# ── Volatility estimates ─────────────────────────────────────────────

# Static annualized vol estimates. These are regime-dependent approximations
# and should be periodically re-calibrated against recent realized vol.
# Using stale vol during a regime shift will mis-price the lognormal model.
ANNUALIZED_VOL: Dict[str, float] = {
    "BTC": 0.56,
    "ETH": 0.70,
    "SOL": 0.74,
    "DOGE": 0.65,
    "XRP": 0.71,
    "SPX": 0.16,
    "INXU": 0.16,
    "INX": 0.16,
    "EURUSD": 0.08,
    "WTI": 0.35,
    "TNOTE": 0.15,
}


def get_annualized_vol(asset: str) -> float:
    """Return calibrated annualized vol for the asset, defaulting to 0.50.

    WARNING: These are static estimates. In high-vol regimes the model will
    understate tail probabilities, and in low-vol regimes it will overstate them.
    """
    return ANNUALIZED_VOL.get(asset, 0.50)
