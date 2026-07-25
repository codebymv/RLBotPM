"""
Self-tests for ``bot/scripts/check_h_perp_003_tracking.py``.

Phase 5 of H-PERP-003 turns the paper logger's `pnl_interval_usdt` series
into a PASS/FAIL signal. The tracking verifier is the only thing that
makes that signal trustworthy. Before we count any of the 30 days towards
fundability, we need to know:

1. When the paper log perfectly reproduces the offline rebuild (the
   no-drift case), the verifier returns PASS.
2. When the paper log diverges by more than the per-interval tolerance
   (e.g. an off-by-one fee, a drifting clock), the verifier returns FAIL.
3. Paper Sharpe uses the Phase 4 ``sharpe_8h`` annualization
   (``sqrt(365*3)``), not sample ``sqrt(n)`` — otherwise the drift gate
   compares incomparable quantities.
4. When sample ≥ ``--drift-min-intervals`` and annualized Sharpe is far
   from Phase 4 OOS, the verifier returns FAIL (honest promotion block).

Both formula tests synthesize the paper log from the offline panel itself,
so they do not depend on any wallclock paper data.
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "bot" / "scripts" / "check_h_perp_003_tracking.py"
PANEL_CSV = ROOT / "research" / "datasets" / "H-PERP-003" / "btc_hedged_panel_okx.csv"
PHASE4_METRICS = ROOT / "research" / "backtests" / "H-PERP-003_metrics.json"

NOTIONAL_USDT = 100.0
INTERVALS_PER_DAY = 3.0


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location("check_h_perp_003_tracking", VERIFIER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_panel() -> list[dict]:
    import csv

    rows: list[dict] = []
    with open(PANEL_CSV, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            try:
                ts = int(raw["fundingTime"])
            except (TypeError, ValueError, KeyError):
                continue
            rows.append({**raw, "fundingTime": ts})
    rows.sort(key=lambda r: r["fundingTime"])
    return rows


def _synthesize_paper_log(panel: list[dict], n_intervals: int = 12) -> list[dict]:
    """Build a fake paper-jsonl payload by replaying the last `n_intervals`
    aligned rows of the offline panel through the same formula the live
    paper logger uses.
    """
    valid: list[dict] = []
    for row in panel:
        try:
            if int(row["align_ok"]) != 1:
                continue
            mark = float(row["mark_close"])
            spot = float(row["spot_close"])
            fr = float(row["fundingRate"])
        except (TypeError, ValueError, KeyError):
            continue
        if mark <= 0 or spot <= 0:
            continue
        valid.append(
            {
                "fundingTime": int(row["fundingTime"]),
                "fundingRate": fr,
                "mark_close": mark,
                "spot_close": spot,
            }
        )

    tail = valid[-(n_intervals + 1) :]
    if len(tail) < 2:
        return []

    ev: list[dict] = []
    cum = 0.0
    prev = None
    for entry in tail:
        if prev is None:
            ev.append(
                {
                    "type": "research_h_perp_003",
                    "fundingTime": entry["fundingTime"],
                    "fundingRate": entry["fundingRate"],
                    "mark_close": entry["mark_close"],
                    "spot_close": entry["spot_close"],
                    "pnl_interval_usdt": None,
                    "cum_pnl_usdt": 0.0,
                    "align_ok": 1,
                }
            )
        else:
            pnl = NOTIONAL_USDT * entry["fundingRate"] + NOTIONAL_USDT * (
                math.log(entry["spot_close"] / prev["spot_close"])
                - math.log(entry["mark_close"] / prev["mark_close"])
            )
            cum += pnl
            ev.append(
                {
                    "type": "research_h_perp_003",
                    "fundingTime": entry["fundingTime"],
                    "fundingRate": entry["fundingRate"],
                    "mark_close": entry["mark_close"],
                    "spot_close": entry["spot_close"],
                    "pnl_interval_usdt": pnl,
                    "cum_pnl_usdt": cum,
                    "align_ok": 1,
                }
            )
        prev = entry
    return ev


def _write_log(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _run_verifier(paper_log: Path, extra_args: list[str] | None = None) -> dict:
    cmd = [
        sys.executable,
        str(VERIFIER),
        "--paper-log",
        str(paper_log),
        "--panel-csv",
        str(PANEL_CSV),
        "--phase4-metrics",
        str(PHASE4_METRICS),
    ]
    if extra_args:
        cmd.extend(extra_args)
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(res.stdout)
    payload["_exit_code"] = res.returncode
    return payload


def test_tracking_pass_on_synthesized_offline_replay() -> None:
    panel = _load_panel()
    log_rows = _synthesize_paper_log(panel, n_intervals=24)
    assert len(log_rows) > 2, "panel did not yield enough aligned tail rows"
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "paper_research_H-PERP-003.jsonl"
        _write_log(log_path, log_rows)
        result = _run_verifier(log_path)
    assert result["verdict"] == "PASS", json.dumps(result, indent=2, default=str)
    assert result["per_interval"]["pass"] is True
    assert result["daily_cum_pnl"]["pass"] is True
    assert result["per_interval"]["max_abs_diff_usdt"] < 1e-9
    assert result["_exit_code"] == 0


def test_tracking_fail_on_perturbed_paper_pnl() -> None:
    panel = _load_panel()
    log_rows = _synthesize_paper_log(panel, n_intervals=24)
    perturbed = False
    for row in log_rows:
        if row.get("pnl_interval_usdt") is not None:
            row["pnl_interval_usdt"] += 0.10
            perturbed = True
            break
    assert perturbed, "could not find a non-baseline row to perturb"
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "paper_research_H-PERP-003.jsonl"
        _write_log(log_path, log_rows)
        result = _run_verifier(log_path)
    assert result["verdict"] == "FAIL", json.dumps(result, indent=2, default=str)
    assert result["per_interval"]["pass"] is False
    assert result["per_interval"]["max_abs_diff_usdt"] >= 0.10 - 1e-9
    assert result["_exit_code"] == 1


def test_stats_sharpe_matches_phase4_annualization() -> None:
    """Regression: drift gate must not use sample sqrt(n) Sharpe."""
    mod = _load_verifier_module()
    series = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.008, 0.012, -0.002, 0.004]
    got = mod._stats(series)["sharpe"]
    mean = statistics.fmean(series)
    stdev = statistics.pstdev(series)
    expected_8h = (mean / stdev) * math.sqrt(365.0 * INTERVALS_PER_DAY)
    wrong_sqrt_n = (mean / stdev) * math.sqrt(len(series))
    assert got is not None
    assert abs(got - expected_8h) < 1e-12
    assert abs(got - wrong_sqrt_n) > 0.5  # formulas must stay distinct


def test_phase4_drift_gate_fails_on_low_sharpe_window() -> None:
    """Honest FAIL when annualized paper Sharpe is far below Phase 4 OOS."""
    panel = _load_panel()
    log_rows = _synthesize_paper_log(panel, n_intervals=100)
    assert len(log_rows) > 90, "need ≥90 intervals to evaluate drift gate"
    # Zero out interval PnL → annualized Sharpe collapses; formula match still holds
    # only if offline rebuild also sees zeros — so perturb paper stats only after
    # writing a matched log, by scaling paper pnl toward zero while keeping the
    # same fundingTimes (breaks formula gate). Instead: keep formula match and
    # point phase4 metrics at an unreachable Sharpe reference.
    phase4 = json.loads(PHASE4_METRICS.read_text(encoding="utf-8"))
    fake_phase4 = {**phase4, "sharpe_oos": 50.0, "profit_factor_oos": 50.0}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log_path = tmp_path / "paper_research_H-PERP-003.jsonl"
        metrics_path = tmp_path / "phase4.json"
        _write_log(log_path, log_rows)
        metrics_path.write_text(json.dumps(fake_phase4), encoding="utf-8")
        result = _run_verifier(
            log_path,
            extra_args=[
                "--phase4-metrics",
                str(metrics_path),
                "--drift-min-intervals",
                "90",
            ],
        )
    assert result["phase4_drift"]["evaluable"] is True
    assert result["phase4_drift"]["sharpe_within_drift"] is False
    assert result["verdict"] == "FAIL", json.dumps(result, indent=2, default=str)
    assert result["_exit_code"] == 1
    # Clear FAIL diagnostics: codes + relative drift + DO NOT PROMOTE line.
    assert "phase4_sharpe_drift" in result["fail_reasons"]
    assert "phase4_profit_factor_drift" in result["fail_reasons"]
    assert result["phase4_drift"]["relative_sharpe_drift"] is not None
    assert result["phase4_drift"]["relative_sharpe_drift"] > 0.30
    assert "DO NOT PROMOTE" in result["diagnosis"]
    assert "Sharpe drift" in result["diagnosis"]
    # Paper Sharpe reported by the verifier must be 8h-annualized (not ~sqrt(n)).
    paper_sharpe = result["phase4_drift"]["paper_sharpe"]
    assert paper_sharpe is not None
    series = [float(r["pnl_interval_usdt"]) for r in log_rows if r.get("pnl_interval_usdt") is not None]
    mean = statistics.fmean(series)
    stdev = statistics.pstdev(series)
    expected = (mean / stdev) * math.sqrt(365.0 * INTERVALS_PER_DAY)
    assert abs(paper_sharpe - expected) < 1e-9


def test_fail_reasons_on_per_interval_mismatch() -> None:
    panel = _load_panel()
    log_rows = _synthesize_paper_log(panel, n_intervals=24)
    for row in log_rows:
        if row.get("pnl_interval_usdt") is not None:
            row["pnl_interval_usdt"] += 0.10
            break
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "paper_research_H-PERP-003.jsonl"
        _write_log(log_path, log_rows)
        result = _run_verifier(log_path)
    assert result["verdict"] == "FAIL"
    # Perturbing an interval also breaks the daily cum gate; both must be named.
    assert result["fail_reasons"][0] == "per_interval_pnl_mismatch"
    assert "per_interval_pnl_mismatch" in result["fail_reasons"]
    assert "daily_cum_pnl_drift" in result["fail_reasons"]
    assert "per-interval PnL mismatch" in result["diagnosis"]
    assert "DO NOT PROMOTE" in result["diagnosis"]


def test_pass_has_empty_fail_reasons() -> None:
    panel = _load_panel()
    log_rows = _synthesize_paper_log(panel, n_intervals=24)
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "paper_research_H-PERP-003.jsonl"
        _write_log(log_path, log_rows)
        result = _run_verifier(log_path)
    assert result["verdict"] == "PASS"
    assert result["fail_reasons"] == []
    assert "passed" in result["diagnosis"].lower()


def test_drift_gate_honest_below_min_intervals() -> None:
    """n < drift-min must not FAIL via Phase 4 drift (INSUFFICIENT sample)."""
    panel = _load_panel()
    log_rows = _synthesize_paper_log(panel, n_intervals=24)
    phase4 = json.loads(PHASE4_METRICS.read_text(encoding="utf-8"))
    fake_phase4 = {**phase4, "sharpe_oos": 50.0, "profit_factor_oos": 50.0}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log_path = tmp_path / "paper_research_H-PERP-003.jsonl"
        metrics_path = tmp_path / "phase4.json"
        _write_log(log_path, log_rows)
        metrics_path.write_text(json.dumps(fake_phase4), encoding="utf-8")
        result = _run_verifier(
            log_path,
            extra_args=[
                "--phase4-metrics",
                str(metrics_path),
                "--drift-min-intervals",
                "90",
            ],
        )
    assert result["phase4_drift"]["evaluable"] is False
    assert result["verdict"] == "PASS", json.dumps(result, indent=2, default=str)
    assert result["fail_reasons"] == []
    assert result["_exit_code"] == 0


def test_missing_phase4_metrics_is_fail_not_silent_pass() -> None:
    """Missing Phase 4 metrics must not collapse to a silent PASS."""
    panel = _load_panel()
    log_rows = _synthesize_paper_log(panel, n_intervals=24)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log_path = tmp_path / "paper_research_H-PERP-003.jsonl"
        missing = tmp_path / "does_not_exist.json"
        _write_log(log_path, log_rows)
        result = _run_verifier(
            log_path,
            extra_args=["--phase4-metrics", str(missing)],
        )
    assert result["verdict"] == "FAIL"
    assert "phase4_metrics_missing" in result["fail_reasons"]
    assert result["_exit_code"] == 1
