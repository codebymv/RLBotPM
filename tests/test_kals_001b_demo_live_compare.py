"""
Offline fixtures for H-KALS-001b demo vs live violation comparison.

No network. Synthetic JSONL only (plus optional real-path smoke).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "scanners" / "compare_kals_001b_demo_live.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("kals_001b_compare", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kals_001b_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_mod()


def _scan(*, demo: bool, events: list[tuple[str, str, float]]) -> dict:
    """events: (event_ticker, kind, sum_p)"""
    viols = [
        {
            "event_ticker": et,
            "kind": kind,
            "sum_p": sum_p,
            "tickers": [f"{et}-A", f"{et}-B"],
            "ps": [sum_p / 2, sum_p / 2],
            "sum_yes_ask_frac": 0.9,
            "sum_yes_bid_frac": 0.1,
            "toy_naive_long_ask_vs_par": 0.1,
            "toy_fee_illustrative_roundtrip": 0.04,
        }
        for et, kind, sum_p in events
    ]
    return {
        "type": "kals_001b_scan",
        "variant": "001b",
        "demo": demo,
        "api_mode": "demo" if demo else "live",
        "markets_fetched": 1000 if demo else 10000,
        "events_partition_candidates": len(events),
        "violation_count": len(viols),
        "violations": viols,
    }


def test_event_family(m):
    assert m.event_family("KXBTCY-27JAN0100") == "KXBTCY"
    assert m.event_family("KXHIGHCHI") == "KXHIGHCHI"


def test_structural_replication_no_identity(m):
    demo = [
        _scan(
            demo=True,
            events=[
                ("KXHIGHCHI-26APR26", "UNDER", 0.2),
                ("KXBTCY-27JAN0100", "UNDER", 0.5),
            ],
        )
    ]
    live = [
        _scan(
            demo=False,
            events=[
                ("KXXRP-26JUL2321", "UNDER", 0.3),
                ("KXDOGEY-27JAN0100", "OVER", 1.1),
            ],
        ),
        _scan(
            demo=False,
            events=[
                ("KXXRP-26JUL2321", "UNDER", 0.31),
                ("KXDOGEY-27JAN0100", "UNDER", 0.4),
            ],
        ),
    ]
    report = m.compare_demo_live(demo, live, purity_ok=True)
    assert report["overlap"]["exact_event_count"] == 0
    assert report["overlap"]["family_count"] == 0
    assert report["verdict"]["label"] == "STRUCTURAL_REPLICATION_PROMISING"
    assert report["verdict"]["promising_for_more_live_batches"] is True
    assert report["verdict"]["capital_pass"] is False
    assert report["verdict"]["do_not_promote"] is True
    assert report["verdict"]["g3_style_10_scan_freeze_ready"] is False
    assert report["live"]["stable_events"]["intersection_size"] == 2
    freeze = report["live_g3_freeze"]
    assert freeze["successful_live_scans"] == 2
    assert freeze["scans_remaining_to_freeze"] == 8
    assert freeze["existence_gate"] == "PENDING"
    assert freeze["addendum_ready"] is False
    assert freeze["capital_pass"] is False


def test_partial_identity_label(m):
    demo = [_scan(demo=True, events=[("KXBTCY-27JAN0100", "UNDER", 0.4)])]
    live = [_scan(demo=False, events=[("KXBTCY-27JAN0100", "UNDER", 0.45)])]
    report = m.compare_demo_live(demo, live, purity_ok=True)
    assert report["overlap"]["exact_event_count"] == 1
    assert report["verdict"]["label"] == "STRUCTURAL_AND_PARTIAL_IDENTITY"


def test_no_live_data(m):
    demo = [_scan(demo=True, events=[("A-1", "UNDER", 0.1)])]
    report = m.compare_demo_live(demo, [], purity_ok=True)
    assert report["verdict"]["label"] == "NO_LIVE_DATA"
    assert report["verdict"]["promising_for_more_live_batches"] is False
    assert report["live_g3_freeze"]["existence_gate"] == "PENDING"


def test_impure_blocks_structural_label(m):
    demo = [_scan(demo=True, events=[("A-1", "UNDER", 0.1)])]
    live = [_scan(demo=False, events=[("B-1", "UNDER", 0.2)])]
    report = m.compare_demo_live(demo, live, purity_ok=False)
    assert report["verdict"]["label"] == "PROVENANCE_IMPURE"
    assert report["verdict"]["promising_for_more_live_batches"] is False
    assert report["live_g3_freeze"]["existence_gate"] == "INCONCLUSIVE_DATA"
    assert report["live_g3_freeze"]["g3_style_10_scan_freeze_ready"] is False


def test_live_g3_freeze_observed_at_ten(m):
    demo = [_scan(demo=True, events=[("DEMO-1", "UNDER", 0.2)])]
    live = [
        _scan(demo=False, events=[(f"LIVE-{i}", "UNDER", 0.3)]) for i in range(10)
    ]
    report = m.compare_demo_live(demo, live, purity_ok=True)
    freeze = report["live_g3_freeze"]
    assert freeze["successful_live_scans"] == 10
    assert freeze["scans_remaining_to_freeze"] == 0
    assert freeze["g3_style_10_scan_freeze_ready"] is True
    assert freeze["existence_gate"] == "VIOLATIONS_OBSERVED"
    assert freeze["addendum_ready"] is True
    assert freeze["capital_pass"] is False
    assert report["verdict"]["g3_style_10_scan_freeze_ready"] is True
    assert report["verdict"]["capital_pass"] is False


def test_live_g3_freeze_zero_violation_fail(m):
    empty_live = {
        "type": "kals_001b_scan",
        "variant": "001b",
        "demo": False,
        "api_mode": "live",
        "markets_fetched": 100,
        "events_partition_candidates": 0,
        "violation_count": 0,
        "violations": [],
    }
    live = [dict(empty_live) for _ in range(10)]
    freeze = m.live_g3_freeze_status(live, purity_ok=True)
    assert freeze["existence_gate"] == "FAIL"
    assert freeze["addendum_ready"] is True
    assert freeze["capital_pass"] is False


def test_live_g3_freeze_window_ignores_post_freeze_append(m):
    """First-10 zero-violation FAIL must not flip when later scans violate."""
    empty_live = {
        "type": "kals_001b_scan",
        "variant": "001b",
        "demo": False,
        "api_mode": "live",
        "markets_fetched": 100,
        "events_partition_candidates": 0,
        "violation_count": 0,
        "violations": [],
    }
    late_violation = _scan(demo=False, events=[("LATE-1", "UNDER", 0.2)])
    live = [dict(empty_live) for _ in range(10)] + [late_violation]
    freeze = m.live_g3_freeze_status(live, purity_ok=True)
    assert freeze["successful_live_scans"] == 11
    assert freeze["freeze_window_scans"] == 10
    assert freeze["total_violation_rows"] == 0
    assert freeze["existence_gate"] == "FAIL"
    assert freeze["addendum_ready"] is True
    assert freeze["capital_pass"] is False


def test_cli_on_repo_jsonl(m):
    demo_path = m._DEFAULT_DEMO
    live_path = m._DEFAULT_LIVE
    if not demo_path.is_file() or not live_path.is_file():
        pytest.skip("repo demo/live JSONL missing")
    rc = m.main(["--demo", str(demo_path), "--live", str(live_path)])
    assert rc == 0
    report = m.compare_demo_live(
        m.load_scan_rows(demo_path),
        m.load_scan_rows(live_path),
        purity_ok=True,
    )
    assert report["demo"]["n_scans"] >= 1
    assert report["live"]["n_scans"] >= 1
    assert report["verdict"]["capital_pass"] is False
    assert report["live_g3_freeze"]["capital_pass"] is False
    assert report["live_g3_freeze"]["existence_gate"] == "PENDING"
    # Current production appends: structural, not identity copy of April demo set
    assert report["verdict"]["label"] in (
        "STRUCTURAL_REPLICATION_PROMISING",
        "STRUCTURAL_AND_PARTIAL_IDENTITY",
    )


def test_cli_rejects_impure_without_flag(m, tmp_path):
    demo_path = tmp_path / "demo.jsonl"
    live_path = tmp_path / "live.jsonl"
    demo_path.write_text(
        json.dumps(_scan(demo=True, events=[("A-1", "UNDER", 0.1)])) + "\n"
        + json.dumps(_scan(demo=False, events=[("BAD", "UNDER", 0.2)])) + "\n",
        encoding="utf-8",
    )
    live_path.write_text(
        json.dumps(_scan(demo=False, events=[("B-1", "UNDER", 0.2)])) + "\n",
        encoding="utf-8",
    )
    assert m.main(["--demo", str(demo_path), "--live", str(live_path)]) == 3
    assert (
        m.main(
            [
                "--demo",
                str(demo_path),
                "--live",
                str(live_path),
                "--allow-impure",
                "--json",
            ]
        )
        == 0
    )


def test_load_roundtrip(m, tmp_path):
    path = tmp_path / "x.jsonl"
    row = _scan(demo=False, events=[("E-1", "UNDER", 0.2)])
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = m.load_scan_rows(path)
    assert len(loaded) == 1
    assert m.union_event_tickers(loaded) == {"E-1"}
