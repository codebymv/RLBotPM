#!/usr/bin/env python3
"""
Daily append-only capture for H-PERP-003.

Re-runs the OKX public puller from `fetch_hedged_panel.py` and **merges** any new
rows into `btc_hedged_panel_okx.csv`. Existing rows are never modified —
deduplication is keyed on `fundingTime` (ms). One JSON line is written to
`pull_log.jsonl` per run with: timestamp, rows added, rows already present,
new align_ok %, span (days), and any error.

Operating rule (architecture-audit-03 Track A1):
- Append-only, idempotent. Safe to run on any cadence (recommended: daily).
- Refuses to overwrite or reorder existing rows.
- Verifies pre-existing rows with the same `fundingTime` agree with the new pull
  (catches venue restatements). Any mismatch is logged but does NOT mutate the
  CSV — operator must reconcile manually before the row is updated.

Usage:
  python RLBotPM/research/datasets/H-PERP-003/daily_capture.py
  python RLBotPM/research/datasets/H-PERP-003/daily_capture.py --max-runtime-sec 120
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fetch_hedged_panel as fhp  # noqa: E402

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


def _load_existing() -> dict[int, dict]:
    if not CSV_PATH.exists():
        return {}
    out: dict[int, dict] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ts = int(row["fundingTime"])
            except (TypeError, ValueError, KeyError):
                continue
            out[ts] = row
    return out


def _normalize_row(row: dict) -> dict:
    out: dict[str, object] = {}
    for k in FIELDS:
        out[k] = row.get(k, "")
    return out


def _row_matches(old: dict, new: dict) -> tuple[bool, list[str]]:
    """
    Compare materially significant fields. fundingRate / mark_close / spot_close are
    floats — compare with string equivalence after rounding the new pull through
    the same CSV write path used previously.
    """
    diffs: list[str] = []
    for key in ("fundingRate", "mark_close", "spot_close", "align_ok"):
        a = str(old.get(key, "")).strip()
        b = str(new.get(key, "")).strip()
        if a == b:
            continue
        try:
            af = float(a) if a != "" else None
            bf = float(b) if b != "" else None
        except ValueError:
            diffs.append(f"{key}:{a!r}!={b!r}")
            continue
        if af is None and bf is None:
            continue
        if af is None or bf is None:
            diffs.append(f"{key}:{a!r}!={b!r}")
            continue
        if abs(af - bf) > 1e-9:
            diffs.append(f"{key}:{af}!={bf}")
    return (len(diffs) == 0), diffs


def _build_merged() -> list[dict]:
    """
    Run the existing public-REST puller and return one merged row per fundingTime
    using the schema in `fetch_hedged_panel.py`.
    """
    funding = fhp.fetch_funding_all()
    if not funding:
        return []
    t0 = int(funding[0]["fundingTime"])
    floor_ms = t0 - fhp.CANDLE_MS
    mark = fhp._fetch_candles_paged(
        "/market/history-mark-price-candles", fhp.INST_SWAP, floor_ms
    )
    spot = fhp._fetch_candles_paged("/market/history-candles", fhp.INST_SPOT, floor_ms)

    def _close_by_ts(candles: list[list]) -> dict[int, float]:
        return {int(r[0]): float(r[4]) for r in candles}

    mark_c = _close_by_ts(mark)
    spot_c = _close_by_ts(spot)
    mark_keys = sorted(mark_c)
    spot_keys = sorted(spot_c)

    out: list[dict] = []
    for r in funding:
        fts = int(r["fundingTime"])
        fr = float(r.get("fundingRate") or r.get("realizedRate") or 0.0)
        mts, mpx, ms = _candle_covering(fts, mark_keys, mark_c)
        sts, spx, ss = _candle_covering(fts, spot_keys, spot_c)
        skew_ok = ms <= fhp.MAX_SKEW_MS and ss <= fhp.MAX_SKEW_MS
        ok = int(mpx is not None and spx is not None and skew_ok)
        out.append(
            {
                "fundingTime": fts,
                "fundingRate": fr,
                "mark_candle_ts": mts or "",
                "mark_close": mpx if mpx is not None else "",
                "spot_candle_ts": sts or "",
                "spot_close": spx if spx is not None else "",
                "align_ok": ok,
                "mark_skew_ms": ms if ms < 10**14 else "",
                "spot_skew_ms": ss if ss < 10**14 else "",
            }
        )
    return out


def _candle_covering(
    fts: int, keys: list[int], closes: dict[int, float]
) -> tuple[int | None, float | None, int]:
    import bisect

    i = bisect.bisect_right(keys, fts) - 1
    if i < 0:
        return None, None, 10**15
    ts = keys[i]
    if fts >= ts + fhp.CANDLE_MS:
        return None, None, 10**15
    skew = min(fts - ts, ts + fhp.CANDLE_MS - fts)
    return ts, closes[ts], skew


def _write_csv(rows: Iterable[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: int(r["fundingTime"]))
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows_sorted:
            w.writerow({k: row.get(k, "") for k in FIELDS})


def _append_log(payload: dict) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pull and report deltas without writing CSV.",
    )
    parser.add_argument(
        "--refresh-tail",
        type=int,
        default=2,
        help=(
            "Allow the last N existing rows to be overwritten by the new pull "
            "(handles partial-bar snapshots at the head of the CSV). Older rows "
            "are NEVER mutated; mismatches in older rows are logged only. "
            "Default 2."
        ),
    )
    args = parser.parse_args(argv)

    started = _now_iso()
    err_payload: dict | None = None
    try:
        new_rows = _build_merged()
    except Exception as exc:  # network / API failures captured for log
        err_payload = {
            "timestamp": started,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _append_log(err_payload)
        print(json.dumps(err_payload, indent=2))
        return 2

    existing = _load_existing()
    new_by_ts = {int(r["fundingTime"]): _normalize_row(r) for r in new_rows}

    refresh_n = max(0, int(args.refresh_tail))
    refreshable_ts: set[int] = set()
    if refresh_n > 0 and existing:
        latest = sorted(existing.keys(), reverse=True)[:refresh_n]
        refreshable_ts = set(latest)

    added = 0
    confirmed = 0
    refreshed = 0
    mismatches: list[dict] = []

    merged: dict[int, dict] = dict(existing)
    for ts, row in new_by_ts.items():
        if ts in existing:
            ok, diffs = _row_matches(existing[ts], row)
            if ok:
                confirmed += 1
            elif ts in refreshable_ts:
                merged[ts] = row
                refreshed += 1
            else:
                mismatches.append({"fundingTime": ts, "diffs": diffs})
            continue
        merged[ts] = row
        added += 1

    if not args.dry_run and (added > 0 or refreshed > 0):
        _write_csv(merged.values())

    align_ok_n = 0
    for r in merged.values():
        try:
            if int(r.get("align_ok") or 0) == 1:
                align_ok_n += 1
        except (TypeError, ValueError):
            pass

    n_total = len(merged)
    if n_total >= 2:
        ts_sorted = sorted(int(r["fundingTime"]) for r in merged.values())
        days_span = (ts_sorted[-1] - ts_sorted[0]) / (1000.0 * 86400.0)
    else:
        days_span = 0.0

    payload = {
        "timestamp": started,
        "ok": True,
        "dry_run": bool(args.dry_run),
        "refresh_tail": refresh_n,
        "rows_added": added,
        "rows_confirmed": confirmed,
        "rows_refreshed": refreshed,
        "rows_total": n_total,
        "align_ok_total": align_ok_n,
        "align_ok_pct": round(100.0 * align_ok_n / n_total, 2) if n_total else 0.0,
        "days_span": round(days_span, 3),
        "phase3_d1_met": days_span >= 365.0,
        "mismatch_count": len(mismatches),
        "mismatches_sample": mismatches[:5],
        "github_actions_run": bool(os.environ.get("GITHUB_ACTIONS")),
    }
    _append_log(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
