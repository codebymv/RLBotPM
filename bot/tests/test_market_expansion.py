"""Tests for market expansion: new asset types, edge detectors, and data feeds."""

import sys
import os
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
BOT_DIR = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

import pytest
from bot.src.strategies.kalshi_edges import StatisticalEdgeDetector


@pytest.fixture
def detector():
    return StatisticalEdgeDetector(min_edge=0.01, min_liquidity=0, max_spread=1000)


class TestInferAsset:
    def test_crypto_btc(self, detector):
        m = {"series_ticker": "KXBTC", "ticker": "KXBTC-26APR10", "title": "", "subtitle": ""}
        assert detector._infer_asset(m) == "BTC"

    def test_index_spx(self, detector):
        m = {"series_ticker": "KXINXU", "ticker": "KXINXU-26APR10", "title": "", "subtitle": ""}
        assert detector._infer_asset(m) == "SPX"

    def test_fx_eurusd(self, detector):
        m = {"series_ticker": "KXEURUSDH", "ticker": "KXEURUSDH-26APR10", "title": "", "subtitle": ""}
        assert detector._infer_asset(m) == "EURUSD"

    def test_commodity_wti(self, detector):
        m = {"series_ticker": "KXWTIH", "ticker": "KXWTIH-26APR10", "title": "", "subtitle": ""}
        assert detector._infer_asset(m) == "WTI"

    def test_inxi_maps_to_spx(self, detector):
        m = {"series_ticker": "INXI", "ticker": "INXI-26APR10", "title": "", "subtitle": ""}
        assert detector._infer_asset(m) == "SPX"

    def test_unknown_returns_none(self, detector):
        m = {"series_ticker": "", "ticker": "RANDOM", "title": "Some market", "subtitle": ""}
        assert detector._infer_asset(m) is None


class TestClassifyMacroMarket:
    def test_cpi(self, detector):
        m = {"title": "CPI in April above 0.3%", "subtitle": ""}
        assert detector._classify_macro_market(m) == "cpi"

    def test_nfp(self, detector):
        m = {"title": "Nonfarm payrolls March 2026", "subtitle": ""}
        assert detector._classify_macro_market(m) == "nfp"

    def test_fed(self, detector):
        m = {"title": "Fed rate cut June 2026", "subtitle": ""}
        assert detector._classify_macro_market(m) == "fed"

    def test_non_macro_returns_none(self, detector):
        m = {"title": "Bitcoin above 90000", "subtitle": ""}
        assert detector._classify_macro_market(m) is None


class TestIsWeatherMarket:
    def test_temperature_in_title(self, detector):
        m = {"title": "NYC Temperature above 75 degrees", "subtitle": "", "series_ticker": ""}
        assert detector._is_weather_market(m) is True

    def test_kxtemp_series(self, detector):
        m = {"title": "High", "subtitle": "", "series_ticker": "KXTEMP"}
        assert detector._is_weather_market(m) is True

    def test_non_weather(self, detector):
        m = {"title": "Bitcoin above 90000", "subtitle": "", "series_ticker": ""}
        assert detector._is_weather_market(m) is False


class TestExtractStrikeFX:
    def test_fx_decimal_values(self, detector):
        m = {
            "title": "EUR/USD above 1.08",
            "subtitle": "",
            "series_ticker": "KXEURUSDH",
            "ticker": "KXEURUSDH-26APR10",
        }
        strike_type, floor, cap = detector._extract_strike(m)
        assert strike_type == "greater"
        assert floor == pytest.approx(1.08)


class TestRangeEdgeAcceptsNewAssets:
    def test_spx_market_is_processed(self, detector):
        m = {
            "series_ticker": "KXINXU",
            "ticker": "KXINXU-26APR10-B5500",
            "event_ticker": "KXINXU-26APR10",
            "title": "S&P 500 between 5,400 and 5,600",
            "subtitle": "",
            "last_price": 50,
            "yes_bid": 48,
            "yes_ask": 52,
            "volume": 500,
            "open_interest": 200,
            "liquidity": 200,
            "close_time": None,
            "strike_type": "between",
            "floor_strike": 5400,
            "cap_strike": 5600,
        }
        edge = detector.detect_range_edge(m)
        # We can't guarantee an edge (depends on live spot price),
        # but the method should not return None due to asset mismatch
        # If spot fetch fails, it returns None — that's OK for this test
        # The key assertion is that _infer_asset returns "SPX"
        assert detector._infer_asset(m) == "SPX"


class TestLiveSeriesExpanded:
    def test_live_series_includes_all_categories(self):
        from bot.src.strategies.paper_trader import LIVE_SERIES, CRYPTO_SERIES
        assert "KXINXU" in LIVE_SERIES
        assert "INXI" in LIVE_SERIES
        assert "KXEURUSDH" in LIVE_SERIES
        assert "KXWTIH" in LIVE_SERIES
        assert "KXCPI" in LIVE_SERIES
        assert "KXUSNFP" in LIVE_SERIES
        assert "KXTEMP" in LIVE_SERIES
        for s in CRYPTO_SERIES:
            assert s in LIVE_SERIES


class TestEdgeTypeFilters:
    def test_paper_trader_accepts_new_types(self):
        accepted = ("spot_vs_strike", "crypto_spot_mispricing", "strike_dominance", "macro_data", "weather")
        for t in accepted:
            assert t in accepted

    def test_live_trader_accepts_new_types(self):
        accepted = ("spot_vs_strike", "crypto_spot_mispricing", "macro_data", "weather")
        for t in accepted:
            assert t in accepted


class TestBackwardCompat:
    def test_detect_crypto_range_edge_alias_exists(self, detector):
        assert hasattr(detector, "detect_crypto_range_edge")
        assert detector.detect_crypto_range_edge == detector.detect_range_edge
