"""
Offline fixtures for H-KALS Σp scanner + Option B (--live) path guards.

No network. Synthetic markets only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "scanners" / "kals_001_probability_sum_scan.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("kals_001_scan", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kals_001_scan"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_mod()


def _mkt(
    *,
    ticker: str,
    event_ticker: str,
    strike_type: str,
    floor: float | None,
    cap: float | None,
    yes_bid: float,
    yes_ask: float,
    close_time: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        event_ticker=event_ticker,
        strike_type=strike_type,
        floor_strike=floor,
        cap_strike=cap,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        close_time=close_time or datetime(2026, 7, 23, tzinfo=timezone.utc),
    )


def test_default_output_paths_separate_demo_and_live(m):
    demo_b = m.default_output_path("001b", live=False)
    live_b = m.default_output_path("001b", live=True)
    demo_a = m.default_output_path("001", live=False)
    live_a = m.default_output_path("001", live=True)
    assert demo_b != live_b
    assert demo_a != live_a
    assert "H-KALS-001b-live" in str(live_b)
    assert "H-KALS-001-live" in str(live_a)
    assert "H-KALS-001b" in str(demo_b) and "live" not in demo_b.name


def test_refuse_live_into_demo_default_path(m):
    with pytest.raises(m.OutputModeConflict):
        m.assert_output_mode_compatible(m._DEFAULT_OUT_001B, live=True)
    with pytest.raises(m.OutputModeConflict):
        m.assert_output_mode_compatible(m._DEFAULT_OUT_001B_LIVE, live=False)
    # Override allowed
    m.assert_output_mode_compatible(
        m._DEFAULT_OUT_001B, live=True, allow_mixed_output=True
    )


def test_custom_output_path_allowed(m, tmp_path):
    custom = tmp_path / "custom_live.jsonl"
    m.assert_output_mode_compatible(custom, live=True)
    m.assert_output_mode_compatible(custom, live=False)


def test_rule_b_contiguous_ladder_under_violation(m):
    # Contiguous bins 100-110 / 110-120 with mids summing UNDER 0.95
    close = datetime(2026, 7, 23, tzinfo=timezone.utc)
    markets = [
        _mkt(
            ticker="A",
            event_ticker="EVT1",
            strike_type="between",
            floor=100,
            cap=110,
            yes_bid=20,
            yes_ask=30,  # mid 0.25
            close_time=close,
        ),
        _mkt(
            ticker="B",
            event_ticker="EVT1",
            strike_type="between",
            floor=110,
            cap=120,
            yes_bid=20,
            yes_ask=30,  # mid 0.25 → sum 0.50 UNDER
            close_time=close,
        ),
    ]
    row = m.build_scan_row(markets, variant="001b", demo=False, timestamp="t0")
    assert row["api_mode"] == "live"
    assert row["demo"] is False
    assert row["events_partition_candidates"] == 1
    assert row["violation_count"] == 1
    assert row["violations"][0]["kind"] == "UNDER"
    assert row["violations"][0]["sum_p"] == pytest.approx(0.5)


def test_rule_b_over_violation_and_gap_rejected(m):
    close = datetime(2026, 7, 23, tzinfo=timezone.utc)
    # Contiguous OVER ladder
    over = [
        _mkt(
            ticker="X1",
            event_ticker="EVT_OVER",
            strike_type="between",
            floor=0,
            cap=10,
            yes_bid=55,
            yes_ask=65,  # mid 0.60
            close_time=close,
        ),
        _mkt(
            ticker="X2",
            event_ticker="EVT_OVER",
            strike_type="between",
            floor=10,
            cap=20,
            yes_bid=50,
            yes_ask=60,  # mid 0.55 → sum 1.15 OVER
            close_time=close,
        ),
    ]
    # Gapped floors (not contiguous) — must not form a candidate
    gapped = [
        _mkt(
            ticker="G1",
            event_ticker="EVT_GAP",
            strike_type="between",
            floor=0,
            cap=10,
            yes_bid=10,
            yes_ask=20,
            close_time=close,
        ),
        _mkt(
            ticker="G2",
            event_ticker="EVT_GAP",
            strike_type="between",
            floor=50,
            cap=60,
            yes_bid=10,
            yes_ask=20,
            close_time=close,
        ),
    ]
    row = m.build_scan_row(over + gapped, variant="001b", demo=True)
    assert row["events_partition_candidates"] == 1
    assert row["violation_count"] == 1
    assert row["violations"][0]["kind"] == "OVER"
    assert row["violations"][0]["event_ticker"] == "EVT_OVER"


def test_audit_jsonl_rejects_mixed_modes(m, tmp_path):
    path = tmp_path / "scan_events.jsonl"
    path.write_text(
        json.dumps({"demo": True, "type": "kals_001b_scan"})
        + "\n"
        + json.dumps({"demo": False, "type": "kals_001b_scan"})
        + "\n",
        encoding="utf-8",
    )
    report = m.audit_jsonl_mode_purity(path)
    assert report["ok"] is False
    assert report["demo_lines"] == 1
    assert report["live_lines"] == 1
    assert any("mixed" in e for e in report["errors"])


def test_audit_jsonl_expect_live(m, tmp_path):
    path = tmp_path / "live.jsonl"
    path.write_text(
        json.dumps({"demo": False, "api_mode": "live"}) + "\n",
        encoding="utf-8",
    )
    assert m.audit_jsonl_mode_purity(path, expect_live=True)["ok"] is True
    bad = m.audit_jsonl_mode_purity(path, expect_live=False)
    assert bad["ok"] is False


def test_demo_jsonl_still_pure(m):
    demo_path = m._DEFAULT_OUT_001B
    if not demo_path.is_file():
        pytest.skip("demo scan_events.jsonl missing")
    report = m.audit_jsonl_mode_purity(demo_path, expect_live=False)
    assert report["ok"] is True
    assert report["live_lines"] == 0
    assert report["demo_lines"] == report["lines"]
