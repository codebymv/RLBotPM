"""
Weather forecast data feeds for temperature-based prediction markets.

Uses Open-Meteo (free, no API key, CC BY 4.0 license) for ensemble
forecast data from multiple NWP models (GFS, ECMWF IFS, ICON, GEM).

The main entry point is get_temperature_forecast(lat, lon) which returns
a dict with mean, std, min, max, and ensemble_spread for the next 24h
high temperature at the given location.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np

from ...core.logger import get_logger

logger = get_logger(__name__)

_cache: Dict[str, Tuple[object, float]] = {}


def _cached(key: str, max_age_s: int = 1800) -> Optional[object]:
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
        logger.debug("Weather feed GET %s failed: %s", url, exc)
    return None


def get_temperature_forecast(
    lat: float,
    lon: float,
    forecast_days: int = 2,
) -> Optional[Dict[str, float]]:
    """
    Get ensemble temperature forecast for a location.

    Returns dict with:
        mean: ensemble mean high temp (°F)
        std: ensemble standard deviation (°F)
        min: ensemble minimum high temp (°F)
        max: ensemble maximum high temp (°F)
        ensemble_spread: max - min range (°F)
        members: number of ensemble members used
    """
    cache_key = f"weather:{lat:.2f},{lon:.2f}:{forecast_days}"
    cached = _cached(cache_key, max_age_s=1800)
    if cached is not None:
        return cached

    result = _fetch_open_meteo_ensemble(lat, lon, forecast_days)
    if result is not None:
        _store(cache_key, result)
        return result

    result = _fetch_open_meteo_deterministic(lat, lon, forecast_days)
    if result is not None:
        _store(cache_key, result)
        return result

    return None


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _fetch_open_meteo_ensemble(
    lat: float,
    lon: float,
    forecast_days: int,
) -> Optional[Dict[str, float]]:
    """Fetch ensemble forecast from Open-Meteo (multiple NWP models)."""
    data = _get_json("https://ensemble-api.open-meteo.com/v1/ensemble", params={
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "forecast_days": forecast_days,
        "models": "gfs_seamless,ecmwf_ifs025,icon_seamless,gem_global",
    })
    if not data:
        return None

    try:
        daily = data.get("daily", {})
        all_highs: List[float] = []
        for key, values in daily.items():
            if key.startswith("temperature_2m_max") and isinstance(values, list):
                for v in values:
                    if v is not None:
                        all_highs.append(_c_to_f(float(v)))

        if len(all_highs) < 2:
            return None

        arr = np.array(all_highs)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "ensemble_spread": float(np.max(arr) - np.min(arr)),
            "members": len(all_highs),
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Failed to parse Open-Meteo ensemble response: %s", exc)
        return None


def _fetch_open_meteo_deterministic(
    lat: float,
    lon: float,
    forecast_days: int,
) -> Optional[Dict[str, float]]:
    """Fallback: use deterministic GFS forecast with assumed uncertainty."""
    data = _get_json("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "forecast_days": forecast_days,
        "temperature_unit": "fahrenheit",
    })
    if not data:
        return None

    try:
        daily = data.get("daily", {})
        highs = [float(v) for v in (daily.get("temperature_2m_max") or []) if v is not None]
        if not highs:
            return None

        mean_high = float(np.mean(highs))
        assumed_std = 3.5

        return {
            "mean": mean_high,
            "std": assumed_std,
            "min": mean_high - 2 * assumed_std,
            "max": mean_high + 2 * assumed_std,
            "ensemble_spread": 4 * assumed_std,
            "members": 1,
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Failed to parse Open-Meteo deterministic response: %s", exc)
        return None


def get_multi_day_forecast(
    lat: float,
    lon: float,
    days: int = 7,
) -> Optional[List[Dict[str, float]]]:
    """
    Get daily temperature forecasts for multiple days ahead.

    Returns list of dicts (one per day), each containing mean/std/min/max
    for the daily high temperature.
    """
    data = _get_json("https://ensemble-api.open-meteo.com/v1/ensemble", params={
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "forecast_days": days,
        "models": "gfs_seamless,ecmwf_ifs025,icon_seamless,gem_global",
    })
    if not data:
        return None

    try:
        daily = data.get("daily", {})
        time_arr = daily.get("time", [])
        if not time_arr:
            return None

        model_keys = [k for k in daily if k.startswith("temperature_2m_max") and k != "time"]
        num_days = len(time_arr)
        results = []

        for day_idx in range(num_days):
            day_values: List[float] = []
            for mk in model_keys:
                vals = daily[mk]
                if day_idx < len(vals) and vals[day_idx] is not None:
                    day_values.append(_c_to_f(float(vals[day_idx])))

            if not day_values:
                results.append(None)
                continue

            arr = np.array(day_values)
            results.append({
                "date": time_arr[day_idx],
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)) if len(arr) > 1 else 3.5,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "ensemble_spread": float(np.max(arr) - np.min(arr)),
                "members": len(day_values),
            })

        return [r for r in results if r is not None]
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Failed to parse multi-day forecast: %s", exc)
        return None
