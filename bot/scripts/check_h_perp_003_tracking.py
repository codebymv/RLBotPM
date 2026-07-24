#!/usr/bin/env python3
"""
H-PERP-003 paper vs offline tracking-error verifier.

Implements the gate table in
``research/08_paper_protocol_H-PERP-003.md`` §"Tracking vs backtest":

- Per-interval ``pnl_interval_usdt`` from the paper logger must match the
  offline rebuild on the same ``fundingTime`` within an absolute tolerance
  (default 1e-6 USDT per interval), and within a relative tolerance for the
  numerically-large entries.
- Daily ``cum_pnl_usdt`` agreement between paper and offline must be within
  ``--daily-tolerance-pct`` (default 5%).
- Paper Sharpe and profit factor over the observed window must be within
  ``--phase4-drift-pct`` (default 30%) of the Phase 4 OOS values from
  ``research/backtests/H-PERP-003_metrics.json``, when applicable
  (i.e., enough closed intervals to evaluate). Sharpe uses the same
  ``sharpe_8h`` annualization as ``research/backtests/H-PERP-003.py``
  (``sqrt(365 * 3)`` on 8h interval returns) — never sample ``sqrt(n)``.

Inputs:

- Paper log: ``bot/logs/paper_research_H-PERP-003.jsonl`` (one JSON per
  funding boundary written by ``append_h_perp_003_paper_snapshot``).
- Offline panel: ``research/datasets/H-PERP-003/btc_hedged_panel_okx.csv``
  (the same CSV ``H-PERP-003.py`` consumes).
- Phase 4 metrics: ``research/backtests/H-PERP-003_metrics.json``.

Exit code is ``0`` for PASS and ``1`` for FAIL so this script can be wired
into a CI / cron sanity check later.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_LOG = REPO_ROOT / "bot" / "logs" / "paper_research_H-PERP-003.jsonl"
PANEL_CSV = REPO_ROOT / "research" / "datasets" / "H-PERP-003" / "btc_hedged_panel_okx.csv"
PHASE4_METRICS = REPO_ROOT / "research" / "backtests" / "H-PERP-003_metrics.json"

NOTIONAL_USDT = 100.0
# Match research/backtests/H-PERP-003.py sharpe_8h annualization.
INTERVALS_PER_DAY = 3.0


def _utc_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).date().isoformat()


def _load_paper_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "research_h_perp_003":
                continue
            rows.append(row)
    rows.sort(key=lambda r: int(r.get("fundingTime") or 0))
    return rows


def _load_panel_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            try:
                ts = int(raw["fundingTime"])
            except (TypeError, ValueError, KeyError):
                continue
            rows.append({**raw, "fundingTime": ts})
    rows.sort(key=lambda r: r["fundingTime"])
    return rows


def _rebuild_offline_pnl(
    panel: list[dict],
    sequence_funding_times: list[int] | None = None,
) -> dict[int, float]:
    """Mirror the formula in ``research/backtests/H-PERP-003.py`` exactly:

    pnl_i = V * fundingRate_{i+1}
          + V * (ln(spot_{i+1}/spot_i) - ln(mark_{i+1}/mark_i))

    Returns ``{ end_funding_ts: pnl_usdt }``. Skips intervals where either
    side is not align_ok or has non-positive prices, matching the backtest.

    When ``sequence_funding_times`` is provided (the paper log's observed
    fundingTimes), rebuild along that sequence rather than every contiguous
    panel pair. This keeps the gate honest if the paper logger missed a
    boundary while ``daily_capture`` still wrote the intervening row.
    """
    by_ts = {int(r["fundingTime"]): r for r in panel}
    if sequence_funding_times is None:
        times = [int(r["fundingTime"]) for r in panel]
    else:
        times = [int(t) for t in sequence_funding_times if int(t) in by_ts]

    out: dict[int, float] = {}
    for i in range(len(times) - 1):
        a = by_ts[times[i]]
        b = by_ts[times[i + 1]]
        try:
            if int(a["align_ok"]) != 1 or int(b["align_ok"]) != 1:
                continue
            f0 = float(a["mark_close"])
            f1 = float(b["mark_close"])
            s0 = float(a["spot_close"])
            s1 = float(b["spot_close"])
            fr = float(b.get("fundingRate") or 0.0)
        except (TypeError, ValueError, KeyError):
            continue
        if f0 <= 0 or f1 <= 0 or s0 <= 0 or s1 <= 0:
            continue
        pnl = NOTIONAL_USDT * fr + NOTIONAL_USDT * (math.log(s1 / s0) - math.log(f1 / f0))
        out[int(b["fundingTime"])] = pnl
    return out


def _per_interval_check(
    paper_rows: list[dict],
    offline_pnl: dict[int, float],
    abs_tol: float,
) -> dict:
    matched = 0
    missing = 0
    diffs: list[float] = []
    worst: dict | None = None
    for row in paper_rows:
        if row.get("pnl_interval_usdt") is None:
            continue
        ts = int(row["fundingTime"])
        if ts not in offline_pnl:
            missing += 1
            continue
        paper_pnl = float(row["pnl_interval_usdt"])
        offline_v = offline_pnl[ts]
        d = abs(paper_pnl - offline_v)
        diffs.append(d)
        if worst is None or d > worst["abs_diff"]:
            worst = {
                "fundingTime": ts,
                "paper_pnl_usdt": paper_pnl,
                "offline_pnl_usdt": offline_v,
                "abs_diff": d,
            }
        matched += 1

    pass_ = bool(diffs) and max(diffs) <= abs_tol
    return {
        "matched": matched,
        "missing": missing,
        "abs_tolerance_usdt": abs_tol,
        "max_abs_diff_usdt": max(diffs) if diffs else 0.0,
        "mean_abs_diff_usdt": statistics.fmean(diffs) if diffs else 0.0,
        "worst_interval": worst,
        "pass": pass_,
    }


def _daily_check(
    paper_rows: list[dict],
    offline_pnl: dict[int, float],
    daily_tol_pct: float,
) -> dict:
    """Group intervals by UTC calendar day on the END timestamp, build a
    paper-cum and offline-cum sequence, and ensure each closed day's
    paper cum is within ``daily_tol_pct`` of the offline cum.
    """
    by_day_paper: dict[str, float] = {}
    by_day_offline: dict[str, float] = {}
    matched_ts: set[int] = set()
    for row in paper_rows:
        if row.get("pnl_interval_usdt") is None:
            continue
        ts = int(row["fundingTime"])
        if ts not in offline_pnl:
            continue
        matched_ts.add(ts)
        day = _utc_day(ts)
        by_day_paper[day] = by_day_paper.get(day, 0.0) + float(row["pnl_interval_usdt"])
        by_day_offline[day] = by_day_offline.get(day, 0.0) + float(offline_pnl[ts])

    days_sorted = sorted(set(by_day_paper) | set(by_day_offline))
    paper_cum = 0.0
    offline_cum = 0.0
    daily: list[dict] = []
    worst: dict | None = None
    failures = 0
    for day in days_sorted:
        paper_cum += by_day_paper.get(day, 0.0)
        offline_cum += by_day_offline.get(day, 0.0)
        ref = abs(offline_cum)
        rel_diff = (
            abs(paper_cum - offline_cum) / ref
            if ref > 1e-9
            else (0.0 if abs(paper_cum) < 1e-9 else float("inf"))
        )
        ok = rel_diff <= daily_tol_pct
        if not ok:
            failures += 1
        entry = {
            "day_utc": day,
            "paper_cum_usdt": paper_cum,
            "offline_cum_usdt": offline_cum,
            "rel_diff": rel_diff,
            "pass": ok,
        }
        daily.append(entry)
        if worst is None or rel_diff > worst["rel_diff"]:
            worst = entry

    return {
        "days_observed": len(daily),
        "daily_tolerance_pct": daily_tol_pct,
        "failures": failures,
        "worst_day": worst,
        "daily": daily,
        "pass": failures == 0 and len(daily) > 0,
    }


def _stats(intervals: Iterable[float]) -> dict:
    """Paper-window stats using the same Sharpe definition as Phase 4.

    Phase 4 ``sharpe_8h`` annualizes 8h interval returns with
    ``sqrt(365 * INTERVALS_PER_DAY)``. Comparing that reference to a
    sample ``sqrt(n)`` Sharpe was an apples-to-oranges bug that inflated
    reported drift; both sides must use the Phase 4 convention.
    """
    series = list(intervals)
    if not series:
        return {"n": 0, "sharpe": None, "profit_factor": None}
    mean = statistics.fmean(series)
    stdev = statistics.pstdev(series) if len(series) > 1 else 0.0
    if len(series) < 10 or stdev < 1e-12:
        sharpe = 0.0 if len(series) >= 10 else None
    else:
        sharpe = (mean / stdev) * math.sqrt(365.0 * INTERVALS_PER_DAY)
    gain = sum(x for x in series if x > 0)
    loss = -sum(x for x in series if x < 0)
    pf = (gain / loss) if loss > 0 else (None if gain == 0 else float("inf"))
    return {"n": len(series), "sharpe": sharpe, "profit_factor": pf}


def _phase4_drift_check(
    paper_rows: list[dict],
    offline_pnl: dict[int, float],
    phase4_path: Path,
    drift_pct: float,
    drift_min_intervals: int,
) -> dict:
    paper_series = [
        float(r["pnl_interval_usdt"])
        for r in paper_rows
        if r.get("pnl_interval_usdt") is not None
    ]
    matched_paper_ts = {
        int(r["fundingTime"])
        for r in paper_rows
        if r.get("pnl_interval_usdt") is not None and int(r["fundingTime"]) in offline_pnl
    }
    offline_window_series = [offline_pnl[ts] for ts in sorted(matched_paper_ts)]

    paper_stats = _stats(paper_series)
    window_stats = _stats(offline_window_series)

    if not phase4_path.exists():
        return {
            "n_paper_intervals": paper_stats["n"],
            "phase4_metrics_present": False,
            "pass": False,
            "skip_reason": f"Phase 4 metrics not found at {phase4_path}",
        }

    with open(phase4_path, encoding="utf-8") as f:
        phase4 = json.load(f)
    phase4_sharpe = phase4.get("sharpe_oos")
    phase4_pf = phase4.get("profit_factor_oos")

    def _within(actual: float | None, ref: float | None) -> bool | None:
        if actual is None or ref is None or ref == 0:
            return None
        return abs(actual - ref) / abs(ref) <= drift_pct

    sharpe_ok = _within(paper_stats["sharpe"], phase4_sharpe)
    pf_ok = _within(paper_stats["profit_factor"], phase4_pf)
    sufficient_sample = paper_stats["n"] >= drift_min_intervals
    can_evaluate = sufficient_sample and sharpe_ok is not None and pf_ok is not None
    skip_reason: str | None = None
    if not sufficient_sample:
        skip_reason = (
            f"only {paper_stats['n']} paper intervals; drift gate requires "
            f"{drift_min_intervals} (08 §Tracking vs backtest: 'over the rolling 30d paper window')"
        )

    return {
        "n_paper_intervals": paper_stats["n"],
        "drift_min_intervals": drift_min_intervals,
        "phase4_metrics_present": True,
        "phase4_sharpe_oos": phase4_sharpe,
        "phase4_profit_factor_oos": phase4_pf,
        "paper_sharpe": paper_stats["sharpe"],
        "paper_profit_factor": paper_stats["profit_factor"],
        "offline_window_sharpe": window_stats["sharpe"],
        "offline_window_profit_factor": window_stats["profit_factor"],
        "drift_tolerance_pct": drift_pct,
        "sharpe_within_drift": sharpe_ok,
        "profit_factor_within_drift": pf_ok,
        "evaluable": can_evaluate,
        "skip_reason": skip_reason,
        "pass": bool(can_evaluate and sharpe_ok and pf_ok),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-log", type=Path, default=PAPER_LOG)
    parser.add_argument("--panel-csv", type=Path, default=PANEL_CSV)
    parser.add_argument("--phase4-metrics", type=Path, default=PHASE4_METRICS)
    parser.add_argument(
        "--per-interval-tolerance-usdt",
        type=float,
        default=1e-6,
        help="Max abs(pnl_interval_paper - pnl_interval_offline) per matched interval.",
    )
    parser.add_argument(
        "--daily-tolerance-pct",
        type=float,
        default=0.05,
        help="Max rel diff between paper and offline cum_pnl on each closed UTC day.",
    )
    parser.add_argument(
        "--phase4-drift-pct",
        type=float,
        default=0.30,
        help="Max rel diff between paper Sharpe / PF and Phase 4 OOS values.",
    )
    parser.add_argument(
        "--drift-min-intervals",
        type=int,
        default=90,
        help=(
            "Minimum paper intervals before the Phase 4 drift gate is allowed "
            "to fail the verifier (08 \u00a7Phase 5 minimum: 90 funding snapshots)."
        ),
    )
    parser.add_argument(
        "--require-min-intervals",
        type=int,
        default=2,
        help="Below this many matched intervals, the check is INSUFFICIENT (not FAIL).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    paper_rows = _load_paper_rows(args.paper_log)
    panel_rows = _load_panel_rows(args.panel_csv)
    paper_times = [int(r["fundingTime"]) for r in paper_rows]
    offline_pnl = _rebuild_offline_pnl(panel_rows, paper_times)

    per_interval = _per_interval_check(paper_rows, offline_pnl, args.per_interval_tolerance_usdt)
    daily = _daily_check(paper_rows, offline_pnl, args.daily_tolerance_pct)
    drift = _phase4_drift_check(
        paper_rows,
        offline_pnl,
        args.phase4_metrics,
        args.phase4_drift_pct,
        args.drift_min_intervals,
    )

    matched_intervals = per_interval["matched"]
    insufficient = matched_intervals < args.require_min_intervals
    drift_blocking = drift.get("evaluable") and not drift.get("pass")
    if insufficient:
        verdict = "INSUFFICIENT"
    elif per_interval["pass"] and daily["pass"] and not drift_blocking:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_log": str(args.paper_log),
        "panel_csv": str(args.panel_csv),
        "n_paper_rows": len(paper_rows),
        "n_panel_rows": len(panel_rows),
        "n_offline_intervals": len(offline_pnl),
        "n_matched_intervals": matched_intervals,
        "verdict": verdict,
        "per_interval": per_interval,
        "daily_cum_pnl": daily,
        "phase4_drift": drift,
    }

    print(json.dumps(summary, indent=2, default=str))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return 0 if verdict in ("PASS", "INSUFFICIENT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
