"""
Tests for CryptoTradingEnv with real-data enforcement.

These tests avoid any synthetic data creation.
"""

import pytest
import pandas as pd

from src.environment import CryptoTradingEnv
from src.data import CryptoCandle, get_db_session
from src.data.sources.base import DataUnavailableError


def test_environment_requires_real_dataset():
    """Environment should fail if dataset is missing."""
    with pytest.raises(DataUnavailableError):
        CryptoTradingEnv(dataset=None)

    with pytest.raises(DataUnavailableError):
        CryptoTradingEnv(dataset=pd.DataFrame())


def test_environment_with_db_data_if_available():
    """
    If DB has real candles, environment should initialize.
    Otherwise, skip the test (no synthetic fallback).
    """
    session = get_db_session()
    try:
        row = session.query(CryptoCandle).first()
        if row is None:
            pytest.skip("No real candles in DB; skipping.")
        data = session.query(CryptoCandle).limit(2000).all()
    finally:
        session.close()

    df = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in data
        ]
    )

    env = CryptoTradingEnv(dataset=df, interval="1h", max_steps=50)
    obs, info = env.reset()

    assert obs.shape == (42,), f"Expected observation shape (42,) but got {obs.shape}"
    assert "capital" in info
    assert info["capital"] == env.initial_capital
