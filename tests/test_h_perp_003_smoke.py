"""
Smoke test guarding the H-PERP-003 D1 (>= 365 calendar days) gate.

While the dataset is still under D1 (today: ~103 days from public OKX REST),
the backtest MUST report `INCONCLUSIVE_DATA` and never accidentally promote
to a PASS verdict. This test catches:

1. A future regression that lets the gate run on insufficient data.
2. A future "G6 fix" that accidentally re-introduces the multiset-invariant
   placebo (cumulative sum or Sharpe of within-segment shuffle), which would
   show up as `g6_placebo_frac_beats == 1.0`.

Once D1 is met (via daily_capture / ingest_external_csv), update or remove
this test alongside the new `07` results doc.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "backtests" / "H-PERP-003.py"
METRICS = ROOT / "research" / "backtests" / "H-PERP-003_metrics.json"
CSV_PATH = ROOT / "research" / "datasets" / "H-PERP-003" / "btc_hedged_panel_okx.csv"


def _run_backtest() -> dict:
    assert SCRIPT.exists(), f"missing backtest script {SCRIPT}"
    assert CSV_PATH.exists(), f"missing dataset CSV {CSV_PATH}"
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--bootstrap-trials", "200", "--seed", "42"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_h_perp_003_inconclusive_until_d1_met() -> None:
    metrics = _run_backtest()
    assert metrics["hypothesis"] == "H-PERP-003"
    if metrics.get("days_span", 0) < 365.0:
        assert metrics["verdict"] == "INCONCLUSIVE_DATA", (
            f"backtest must refuse to verdict on <365d data, got "
            f"verdict={metrics['verdict']!r} on {metrics.get('days_span', 0)}d"
        )
        assert not metrics["data_contract_ok"], "data_contract_ok must be False under D1 fail"
        assert not metrics["d1_calendar_depth_ge_365d"], "D1 must report False"


def test_h_perp_003_g6_placebo_not_degenerate() -> None:
    metrics = _run_backtest()
    if "g6_placebo_frac_beats" not in metrics:
        return
    frac = float(metrics["g6_placebo_frac_beats"])
    method = str(metrics.get("g6_method", ""))
    assert method == "random_sign_flip", (
        f"G6 must be the pre-registered random_sign_flip placebo (06 §5 amendment), "
        f"got method={method!r}. Permutation/multiset placebos are degenerate."
    )
    assert frac < 0.999, (
        f"G6 placebo frac_beats={frac} indicates a degenerate (multiset-invariant) "
        f"comparator. Re-read 06 §5 G6 amendment."
    )


def test_h_perp_003_csv_align_invariants() -> None:
    """The CSV must keep its append-only / align_ok contract."""
    import csv

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows, "panel CSV is empty"
    align_n = sum(1 for r in rows if int(r.get("align_ok") or 0) == 1)
    align_pct = 100.0 * align_n / len(rows)
    assert align_pct >= 99.0, (
        f"align_ok dropped to {align_pct:.2f}%; expected >=99% per 06 §1 (skew<=60s)."
    )
    ts = [int(r["fundingTime"]) for r in rows]
    assert ts == sorted(ts), "rows out of fundingTime order — append should preserve sort"
    assert len(set(ts)) == len(ts), "duplicate fundingTime — daily_capture dedup failed"
