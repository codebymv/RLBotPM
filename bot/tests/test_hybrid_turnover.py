"""Tests for the hybrid turnover strategy: sleeve classification, allocation, and ranking."""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

REPO_ROOT = str(Path(__file__).resolve().parents[2])
BOT_DIR = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

import pytest
from bot.src.strategies.paper_trader import (
    classify_sleeve,
    hours_to_close,
    FAST_SERIES,
    MACRO_SERIES_SET,
    HYBRID_FAST_HORIZON_HOURS,
)


# ---------------------------------------------------------------------------
# classify_sleeve
# ---------------------------------------------------------------------------

class TestClassifySleeve:
    def _market(self, series: str, hours_until_close=None):
        ct = None
        if hours_until_close is not None:
            ct = datetime.now(timezone.utc) + timedelta(hours=hours_until_close)
        return {"series_ticker": series, "close_time": ct}

    def test_crypto_daily_is_fast(self):
        assert classify_sleeve(self._market("KXBTCD", hours_until_close=6)) == "fast"

    def test_crypto_no_close_time_still_fast(self):
        assert classify_sleeve(self._market("KXBTCD")) == "fast"

    def test_eth_hourly_is_fast(self):
        assert classify_sleeve(self._market("KXETHD", hours_until_close=2)) == "fast"

    def test_sol_daily_is_fast(self):
        assert classify_sleeve(self._market("KXSOLD", hours_until_close=10)) == "fast"

    def test_index_within_horizon_is_fast(self):
        assert classify_sleeve(self._market("KXINXU", hours_until_close=48)) == "fast"

    def test_fx_hourly_is_fast(self):
        assert classify_sleeve(self._market("KXEURUSDH", hours_until_close=1)) == "fast"

    def test_commodity_hourly_is_fast(self):
        assert classify_sleeve(self._market("KXWTIH", hours_until_close=0.5)) == "fast"

    def test_cpi_is_macro(self):
        assert classify_sleeve(self._market("KXCPI", hours_until_close=24)) == "macro"

    def test_payrolls_is_macro(self):
        assert classify_sleeve(self._market("KXPAYROLLS")) == "macro"

    def test_nfp_is_macro(self):
        assert classify_sleeve(self._market("KXUSNFP")) == "macro"

    def test_fed_is_macro(self):
        assert classify_sleeve(self._market("KXFFR")) == "macro"

    def test_weather_is_macro(self):
        assert classify_sleeve(self._market("KXTEMP")) == "macro"

    def test_unknown_series_is_other(self):
        assert classify_sleeve(self._market("UNKNOWNSERIES")) == "other"

    def test_custom_horizon(self):
        m = self._market("KXBTCD", hours_until_close=5)
        assert classify_sleeve(m, fast_horizon_hours=3) == "fast"

    def test_macro_ignores_horizon(self):
        m = self._market("KXCPI", hours_until_close=1)
        assert classify_sleeve(m) == "macro"


# ---------------------------------------------------------------------------
# hours_to_close
# ---------------------------------------------------------------------------

class TestHoursToClose:
    def test_returns_hours(self):
        ct = datetime.now(timezone.utc) + timedelta(hours=5)
        m = {"close_time": ct}
        h = hours_to_close(m)
        assert h is not None
        assert abs(h - 5.0) < 0.1

    def test_none_when_missing(self):
        assert hours_to_close({}) is None
        assert hours_to_close({"close_time": None}) is None

    def test_zero_when_past(self):
        ct = datetime.now(timezone.utc) - timedelta(hours=1)
        h = hours_to_close({"close_time": ct})
        assert h == 0.0

    def test_string_close_time(self):
        ct = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
        h = hours_to_close({"close_time": ct})
        assert h is not None
        assert abs(h - 3.0) < 0.1


# ---------------------------------------------------------------------------
# Sleeve budget constants
# ---------------------------------------------------------------------------

class TestSleeveDefaults:
    def test_fast_series_contains_crypto(self):
        assert "KXBTCD" in FAST_SERIES
        assert "KXETHD" in FAST_SERIES
        assert "KXSOLD" in FAST_SERIES

    def test_fast_series_contains_index_fx(self):
        assert "KXINXU" in FAST_SERIES
        assert "KXEURUSDH" in FAST_SERIES
        assert "KXWTIH" in FAST_SERIES

    def test_macro_set_contains_macro(self):
        assert "KXCPI" in MACRO_SERIES_SET
        assert "KXPAYROLLS" in MACRO_SERIES_SET
        assert "KXUSNFP" in MACRO_SERIES_SET
        assert "KXFFR" in MACRO_SERIES_SET

    def test_macro_set_contains_weather(self):
        assert "KXTEMP" in MACRO_SERIES_SET
        assert "KXHMONTHRANGE" in MACRO_SERIES_SET

    def test_no_overlap(self):
        assert FAST_SERIES.isdisjoint(MACRO_SERIES_SET)

    def test_default_horizon(self):
        assert HYBRID_FAST_HORIZON_HOURS == 72


# ---------------------------------------------------------------------------
# Ranking preference: shorter-dated should score higher
# ---------------------------------------------------------------------------

class TestHybridRanking:
    """Verify that the scoring function used in hybrid mode favors
    shorter time-to-close when edge values are equal."""

    def _make_edge_like(self, edge_value, hours_left):
        """Minimal namespace that mimics an Edge for scoring purposes."""
        ct = datetime.now(timezone.utc) + timedelta(hours=hours_left)

        class _E:
            pass

        e = _E()
        e.edge_value = edge_value
        e.confidence = 1.0
        e.market_data = {"close_time": ct}
        return e

    def test_same_edge_shorter_wins(self):
        short = self._make_edge_like(0.05, hours_left=6)
        long_ = self._make_edge_like(0.05, hours_left=100)

        def _score(e):
            base = e.edge_value * e.confidence
            h = hours_to_close(e.market_data)
            if h is not None and h <= 24:
                base *= 1.5
            elif h is not None and h <= 72:
                base *= 1.2
            return base

        assert _score(short) > _score(long_)

    def test_24h_beats_72h(self):
        e24 = self._make_edge_like(0.05, hours_left=12)
        e72 = self._make_edge_like(0.05, hours_left=48)

        def _score(e):
            base = e.edge_value * e.confidence
            h = hours_to_close(e.market_data)
            if h is not None and h <= 24:
                base *= 1.5
            elif h is not None and h <= 72:
                base *= 1.2
            return base

        assert _score(e24) > _score(e72)
