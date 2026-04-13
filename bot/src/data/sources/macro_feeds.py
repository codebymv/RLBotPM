"""
Macro-economic data feeds for CPI, NFP, and Fed Funds Rate edge detection.

Data sources (all free, no API key required unless noted):
  - FRED API (requires FRED_API_KEY env var) for leading indicators
  - Cleveland Fed Inflation Nowcast (public RSS/JSON)
  - BLS latest release data
  - CME FedWatch implied probabilities (scraped from public page)

The main entry point is get_macro_nowcast(indicator) which returns a
probability distribution over possible outcome buckets.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from ...core.logger import get_logger

logger = get_logger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
_FRED_BASE = "https://api.stlouisfed.org/fred"

_cache: Dict[str, Tuple[object, float]] = {}


def _cached(key: str, max_age_s: int = 3600) -> Optional[object]:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) <= max_age_s:
        return entry[0]
    return None


def _store(key: str, value: object) -> object:
    _cache[key] = (value, time.time())
    return value


def _get_json(url: str, params: Optional[dict] = None, timeout: int = 15) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=timeout, headers={
            "User-Agent": "RLBotPM/1.0",
        })
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("Macro feed GET %s failed: %s", url, exc)
    return None


# ── FRED Series ──────────────────────────────────────────────────────

_CPI_LEADING_SERIES = {
    "CPIAUCSL": "CPI All Urban (seasonally adjusted)",
    "CPILFESL": "Core CPI (less food/energy)",
    "GASREGW": "Regular gasoline price (weekly proxy for energy CPI)",
    "ICSA": "Initial jobless claims (labor cost proxy)",
}

_NFP_LEADING_SERIES = {
    "ICSA": "Initial jobless claims",
    "ADPWNUSNERSA": "ADP private payrolls",
    "PAYEMS": "Total nonfarm payrolls",
    "UNRATE": "Unemployment rate",
}

_FED_SERIES = {
    "DFEDTARU": "Fed funds target upper",
    "DFEDTARL": "Fed funds target lower",
}


def get_fred_latest(series_id: str) -> Optional[float]:
    """Fetch the latest observation from FRED for a given series."""
    if not FRED_API_KEY:
        return None
    cached = _cached(f"fred:{series_id}", max_age_s=3600)
    if cached is not None:
        return cached

    data = _get_json(f"{_FRED_BASE}/series/observations", params={
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    })
    if not data:
        return None
    try:
        obs = data["observations"][0]
        val = float(obs["value"])
        _store(f"fred:{series_id}", val)
        return val
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def get_fred_recent(series_id: str, count: int = 12) -> List[float]:
    """Fetch the last N observations from FRED."""
    if not FRED_API_KEY:
        return []
    cache_key = f"fred_recent:{series_id}:{count}"
    cached = _cached(cache_key, max_age_s=3600)
    if cached is not None:
        return cached

    data = _get_json(f"{_FRED_BASE}/series/observations", params={
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": count,
    })
    if not data:
        return []
    try:
        vals = [float(o["value"]) for o in reversed(data["observations"]) if o["value"] != "."]
        _store(cache_key, vals)
        return vals
    except (KeyError, ValueError, TypeError):
        return []


# ── Cleveland Fed CPI Nowcast ────────────────────────────────────────

def get_cleveland_fed_nowcast() -> Optional[Dict[str, float]]:
    """
    Fetch the Cleveland Fed's inflation nowcast.

    Returns dict with keys like:
        cpi_mom: month-over-month CPI nowcast
        cpi_yoy: year-over-year CPI nowcast
        core_cpi_mom: core CPI month-over-month
    """
    cached = _cached("cleveland_nowcast", max_age_s=7200)
    if cached is not None:
        return cached

    data = _get_json("https://www.clevelandfed.org/api/InflationNowcasting/data")
    if not data:
        data = _get_json("https://www.clevelandfed.org/~/media/content/indicators/inflation-nowcasting/inflnowcast.json")
    if not data:
        return None

    try:
        result = {}
        if isinstance(data, dict):
            for key in ("cpi", "coreCpi", "pce", "corePce"):
                section = data.get(key, {})
                if isinstance(section, dict):
                    for sub_key, sub_val in section.items():
                        if isinstance(sub_val, (int, float)):
                            result[f"{key}_{sub_key}"] = float(sub_val)
        if result:
            _store("cleveland_nowcast", result)
            return result
    except Exception:
        pass
    return None


# ── Consensus & Historical Bias ──────────────────────────────────────

_HISTORICAL_CPI_MOM = [
    0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.3,
]

_HISTORICAL_NFP = [
    256, 307, 228, 206, 114, 179, 157, 144, 254, 212, 227, 143,
]


def get_cpi_consensus_range() -> Tuple[float, float, float]:
    """
    Return (low, consensus, high) for next CPI MoM reading.
    Uses FRED historical data if available, otherwise hardcoded recent values.
    """
    recent = get_fred_recent("CPIAUCSL", 12)
    if len(recent) >= 2:
        mom_changes = [
            (recent[i] - recent[i - 1]) / recent[i - 1] * 100.0
            for i in range(1, len(recent))
        ]
        if mom_changes:
            import numpy as np
            mean = float(np.mean(mom_changes))
            std = float(np.std(mom_changes)) if len(mom_changes) > 1 else 0.1
            return (mean - 1.5 * std, mean, mean + 1.5 * std)

    import numpy as np
    vals = _HISTORICAL_CPI_MOM
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    return (mean - 1.5 * std, mean, mean + 1.5 * std)


def get_nfp_consensus_range() -> Tuple[float, float, float]:
    """
    Return (low, consensus, high) for next NFP reading (thousands).
    """
    recent = get_fred_recent("PAYEMS", 12)
    if len(recent) >= 2:
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        if changes:
            import numpy as np
            mean = float(np.mean(changes))
            std = float(np.std(changes)) if len(changes) > 1 else 50.0
            return (mean - 1.5 * std, mean, mean + 1.5 * std)

    import numpy as np
    vals = _HISTORICAL_NFP
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    return (mean - 1.5 * std, mean, mean + 1.5 * std)


def get_fed_rate_current() -> Optional[Tuple[float, float]]:
    """Return current (lower, upper) fed funds rate target from FRED."""
    upper = get_fred_latest("DFEDTARU")
    lower = get_fred_latest("DFEDTARL")
    if upper is not None and lower is not None:
        return (lower, upper)
    return None


# ── Probability Distribution Builder ─────────────────────────────────

def build_normal_bucket_probs(
    mean: float,
    std: float,
    buckets: List[Tuple[Optional[float], Optional[float]]],
) -> List[float]:
    """
    Given a normal distribution (mean, std) and a list of (low, high) buckets,
    return the probability mass in each bucket.

    Bucket format: (low, high) where None means unbounded.
    """
    from scipy.stats import norm
    probs = []
    for lo, hi in buckets:
        if lo is None and hi is not None:
            probs.append(float(norm.cdf(hi, loc=mean, scale=max(std, 1e-6))))
        elif lo is not None and hi is None:
            probs.append(float(1.0 - norm.cdf(lo, loc=mean, scale=max(std, 1e-6))))
        elif lo is not None and hi is not None:
            p = float(norm.cdf(hi, loc=mean, scale=max(std, 1e-6)) -
                       norm.cdf(lo, loc=mean, scale=max(std, 1e-6)))
            probs.append(max(0.0, p))
        else:
            probs.append(1.0)
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    return probs
