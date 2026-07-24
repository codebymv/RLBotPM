#!/usr/bin/env python3
"""
Sync offline hedged-panel rows from the H-PERP-003 paper log.

Why this exists
---------------
``daily_capture.py`` re-pulls OKX candles at capture time. Candle closes for a
past ``fundingTime`` can differ from the closes observed by the paper logger
when that boundary was live (venue restatement / bar selection drift). Phase 5
tracking then FAIL even when the paper PnL formula is correct.

For any ``fundingTime`` already written to
``bot/logs/paper_research_H-PERP-003.jsonl``, the paper snapshot is the
authoritative Phase 5 observation. This script upserts those rows into
``btc_hedged_panel_okx.csv`` (keyed on ``fundingTime``) without touching
pre-paper history.

Usage:
  python research/datasets/H-PERP-003/sync_panel_from_paper_log.py
  python research/datasets/H-PERP-003/sync_panel_from_paper_log.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAPER_LOG = REPO / "bot" / "logs" / "paper_research_H-PERP-003.jsonl"
CSV_PATH = HERE / "btc_hedged_panel_okx.csv"
LOG_PATH = HERE / "pull_log.jsonl"

FIELDS = [
    "fundingTime",
    "fundingRate",
    "mark_candle_ts",
    "mark_close",
    "spot_candle_ts",
    "spot_close",
    "align_ok",
    "mark_skew_ms",
    "spot_skew_ms",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_panel(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ts = int(row["fundingTime"])
            except (TypeError, ValueError, KeyError):
                continue
            out[ts] = {k: row.get(k, "") for k in FIELDS}
    return out


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
    return rows


def _paper_to_panel_row(row: dict) -> dict:
    return {
        "fundingTime": int(row["fundingTime"]),
        "fundingRate": row.get("fundingRate", ""),
        "mark_candle_ts": "" if row.get("mark_candle_ts") is None else row.get("mark_candle_ts"),
        "mark_close": "" if row.get("mark_close") is None else row.get("mark_close"),
        "spot_candle_ts": "" if row.get("spot_candle_ts") is None else row.get("spot_candle_ts"),
        "spot_close": "" if row.get("spot_close") is None else row.get("spot_close"),
        "align_ok": row.get("align_ok", 0),
        "mark_skew_ms": "" if row.get("mark_skew_ms") is None else row.get("mark_skew_ms"),
        "spot_skew_ms": "" if row.get("spot_skew_ms") is None else row.get("spot_skew_ms"),
    }


def _materially_differs(old: dict, new: dict) -> bool:
    for key in ("fundingRate", "mark_close", "spot_close", "align_ok", "mark_candle_ts", "spot_candle_ts"):
        a = str(old.get(key, "")).strip()
        b = str(new.get(key, "")).strip()
        if a == b:
            continue
        try:
            af = float(a) if a != "" else None
            bf = float(b) if b != "" else None
        except ValueError:
            return True
        if af is None and bf is None:
            continue
        if af is None or bf is None:
            return True
        if abs(af - bf) > 1e-9:
            return True
    return False


def _write_csv(path: Path, rows: dict[int, dict]) -> None:
    ordered = [rows[ts] for ts in sorted(rows)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paper-log", type=Path, default=PAPER_LOG)
    parser.add_argument("--panel-csv", type=Path, default=CSV_PATH)
    args = parser.parse_args(argv)

    panel = _load_panel(args.panel_csv)
    paper_rows = _load_paper_rows(args.paper_log)
    added = 0
    updated = 0
    unchanged = 0

    for row in paper_rows:
        try:
            new_row = _paper_to_panel_row(row)
        except (KeyError, TypeError, ValueError):
            continue
        ts = int(new_row["fundingTime"])
        if ts not in panel:
            panel[ts] = new_row
            added += 1
            continue
        if _materially_differs(panel[ts], new_row):
            panel[ts] = new_row
            updated += 1
        else:
            unchanged += 1

    if not args.dry_run and (added or updated):
        _write_csv(args.panel_csv, panel)

    payload = {
        "timestamp": _now_iso(),
        "ok": True,
        "action": "sync_panel_from_paper_log",
        "dry_run": bool(args.dry_run),
        "paper_rows": len(paper_rows),
        "rows_added": added,
        "rows_updated": updated,
        "rows_unchanged": unchanged,
        "rows_total": len(panel),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
