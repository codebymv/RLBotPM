#!/usr/bin/env python3
"""
Funding-monitor snapshot — sketch backend for the C1 dashboard.

Reads the H-PERP-003 daily-capture CSV and pull log, and emits a single
JSON document conforming to the `funding-snapshot.v1` schema described in
[research/C1_FUNDING_DASHBOARD_SKETCH.md](../../research/C1_FUNDING_DASHBOARD_SKETCH.md).

This is intentionally tiny and dependency-light (pure stdlib) so it can
run inside the daily GitHub Actions cron and stash the result as a
release artifact, without pulling in pandas / numpy / FastAPI.

Usage:
  python bot/scripts/funding_dashboard_snapshot.py
  python bot/scripts/funding_dashboard_snapshot.py --out funding_snapshot.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    REPO_ROOT / "research" / "datasets" / "H-PERP-003" / "btc_hedged_panel_okx.csv"
)
DEFAULT_LOG = (
    REPO_ROOT / "research" / "datasets" / "H-PERP-003" / "pull_log.jsonl"
)

D1_TARGET_DAYS = 365
STALE_HOURS = 26  # one OKX 8h funding cycle + 2h cron buffer


def _read_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["fundingTime"] = int(row["fundingTime"])
                row["fundingRate"] = float(row["fundingRate"])
                row["align_ok"] = (str(row.get("align_ok", "")).lower() in ("true", "1"))
            except (KeyError, ValueError):
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["fundingTime"])
    return rows


def _read_log_tail(log_path: Path, n: int = 10) -> list[dict]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _data_health(rows: list[dict], log: list[dict]) -> dict:
    if not rows:
        return {
            "rows": 0,
            "days_covered": 0.0,
            "d1_target_days": D1_TARGET_DAYS,
            "d1_progress_pct": 0.0,
            "align_ok_pct": 0.0,
            "last_pull_at": None,
            "last_pull_rows_added": None,
            "last_pull_status": "stale",
            "consecutive_clean_days": 0,
        }
    span_ms = rows[-1]["fundingTime"] - rows[0]["fundingTime"]
    days_covered = max(0.0, span_ms / 86400000.0)
    align_ok_pct = sum(1 for r in rows if r["align_ok"]) / len(rows)

    last_pull = log[-1] if log else None
    # `timestamp` is the actual key written by daily_capture.py /
    # ingest_external_csv.py; `ts` is accepted as a fallback in case the
    # logger schema is renamed in the future.
    last_pull_at = (
        (last_pull.get("timestamp") or last_pull.get("ts")) if last_pull else None
    )
    last_pull_rows_added = last_pull.get("rows_added") if last_pull else None
    pull_status = "stale"
    if last_pull_at:
        try:
            ts = datetime.fromisoformat(last_pull_at.replace("Z", "+00:00"))
            age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if last_pull.get("ok") is False or last_pull.get("error"):
                pull_status = "error"
            elif age_h <= STALE_HOURS:
                pull_status = "ok"
            else:
                pull_status = "stale"
        except (ValueError, TypeError):
            pass

    # A1 acceptance test: 7 consecutive clean appends with align_ok >= 99%.
    # Loggers write `align_ok_pct` as a 0–100 percentage, not a fraction.
    clean_streak = 0
    for entry in reversed(log):
        if entry.get("ok") is False or entry.get("error"):
            break
        align = entry.get("align_ok_pct")
        if align is not None and align < 99.0:
            break
        clean_streak += 1

    return {
        "rows": len(rows),
        "days_covered": round(days_covered, 2),
        "d1_target_days": D1_TARGET_DAYS,
        "d1_progress_pct": round(100.0 * days_covered / D1_TARGET_DAYS, 2),
        "align_ok_pct": round(align_ok_pct, 4),
        "last_pull_at": last_pull_at,
        "last_pull_rows_added": last_pull_rows_added,
        "last_pull_status": pull_status,
        "consecutive_clean_days": clean_streak,
    }


def _funding_series(rows: list[dict]) -> list[dict]:
    series: list[dict] = []
    window: list[float] = []
    for row in rows:
        window.append(row["fundingRate"])
        # 24h rolling at 8h cadence == last 3 points.
        if len(window) > 3:
            window.pop(0)
        rolling = statistics.fmean(window) if window else None
        series.append(
            {
                "fundingTime": row["fundingTime"],
                "fundingRate": row["fundingRate"],
                "rolling_24h_mean": round(rolling, 8) if rolling is not None else None,
            }
        )
    return series


def _decile_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"deciles": [], "current_decile": 0}
    rates = sorted(r["fundingRate"] for r in rows)
    n = len(rates)
    if n < 11:
        # Not enough data to compute deciles meaningfully; fall back to range.
        return {
            "deciles": [rates[0], rates[-1]],
            "current_decile": 0,
        }
    deciles = [rates[int(i * (n - 1) / 10)] for i in range(11)]
    current = rows[-1]["fundingRate"]
    cd = 0
    for i in range(10):
        if deciles[i] <= current <= deciles[i + 1]:
            cd = i
            break
    return {
        "deciles": [round(d, 8) for d in deciles],
        "current_decile": cd,
    }


def build_snapshot(csv_path: Path, log_path: Path) -> dict:
    rows = _read_csv(csv_path)
    log = _read_log_tail(log_path, n=14)
    return {
        "schema": "funding-snapshot.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "H-PERP-003",
        "data_health": _data_health(rows, log),
        "funding_series": _funding_series(rows),
        "decile_summary": _decile_summary(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--out", default=None, help="If set, write JSON to this path; otherwise stdout")
    args = parser.parse_args()

    snapshot = build_snapshot(Path(args.csv), Path(args.log))
    text = json.dumps(snapshot, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
