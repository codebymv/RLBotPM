#!/usr/bin/env python3
"""
Ingest an externally-sourced funding-rate CSV (OKX historical-data download,
Tardis.dev, Kaiko, Amberdata, etc.) and merge it into `btc_hedged_panel_okx.csv`
using the H-PERP-003 schema.

The 1H mark/spot closes are looked up from OKX public candle endpoints — those
reach back much further than the funding-rate-history endpoint (see
[AUTH_INVESTIGATION.md](AUTH_INVESTIGATION.md)). So an external file only needs
to provide `fundingTime` + `fundingRate`; the candle backfill is automatic.

Operating rule (Track A3):
- Append-only into the existing CSV via the same dedup logic as
  [daily_capture.py](daily_capture.py).
- Records source provenance to `pull_log.jsonl` AND a dedicated
  `INGEST_<file>.md` provenance stub.
- Does NOT mutate rows that already exist with conflicting values unless
  `--allow-overwrite` is passed (operator opt-in).

Usage:
  # OKX historical-data download (auto-detect schema):
  python research/datasets/H-PERP-003/ingest_external_csv.py path/to/okx-funding.csv

  # Manual column hints for an unknown vendor format:
  python research/datasets/H-PERP-003/ingest_external_csv.py vendor.csv \
      --funding-time-col timestamp \
      --funding-rate-col rate \
      --ts-units s
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

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

# Common alternative column names across vendors (lowercased).
TIME_COL_CANDIDATES = (
    "fundingtime",
    "funding_time",
    "funding time",
    "funding-time",
    "funding_time_utc",
    "funding time utc",
    "timestamp",
    "time",
    "ts",
    "datetime",
    "date",
    "open_time",
)
RATE_COL_CANDIDATES = (
    "fundingrate",
    "funding_rate",
    "funding rate",
    "funding-rate",
    "rate",
    "realized_rate",
    "realized funding rate",
    "realized_funding_rate",
    "fundingrateactual",
    "premium_index_funding_rate",
)


def _normalize_header(value: str) -> str:
    return value.strip().lower().replace("\ufeff", "")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_columns(headers: list[str]) -> tuple[str, str]:
    h_lower = {_normalize_header(h): h for h in headers}
    time_col = next((h_lower[c] for c in TIME_COL_CANDIDATES if c in h_lower), None)
    rate_col = next((h_lower[c] for c in RATE_COL_CANDIDATES if c in h_lower), None)
    if not time_col or not rate_col:
        raise SystemExit(
            f"Could not auto-detect funding-time / funding-rate columns in headers "
            f"{headers!r}. Pass --funding-time-col / --funding-rate-col explicitly."
        )
    return time_col, rate_col


def _parse_timestamp(value: str, units: str) -> int:
    """Return milliseconds since epoch."""
    if value is None or value == "":
        raise ValueError("empty timestamp")
    s = value.strip()
    if units == "ms":
        return int(float(s))
    if units == "s":
        return int(float(s) * 1000.0)
    if units == "iso":
        s_norm = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s_norm)
        except ValueError as exc:
            raise ValueError(f"unparseable iso timestamp {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if units == "auto":
        # try int first
        try:
            v = float(s)
            if v > 1e12:  # ms
                return int(v)
            if v > 1e9:  # s
                return int(v * 1000.0)
        except ValueError:
            pass
        return _parse_timestamp(s, "iso")
    raise SystemExit(f"unknown ts-units: {units}")


def _load_external(
    path: Path, time_col: str | None, rate_col: str | None, ts_units: str
) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header row")
        if time_col and time_col not in reader.fieldnames:
            raise SystemExit(f"{path} missing column {time_col!r}")
        if rate_col and rate_col not in reader.fieldnames:
            raise SystemExit(f"{path} missing column {rate_col!r}")
        if not time_col or not rate_col:
            time_col, rate_col = _detect_columns(list(reader.fieldnames))
        out: list[dict] = []
        for row in reader:
            try:
                ts = _parse_timestamp(row[time_col], ts_units)
                rate = float(row[rate_col])
            except (KeyError, ValueError) as exc:
                continue
            out.append({"fundingTime": ts, "fundingRate": rate})
    out.sort(key=lambda r: r["fundingTime"])
    return out


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


def _build_candle_index(funding_rows: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
    if not funding_rows:
        return {}, {}
    t0 = min(r["fundingTime"] for r in funding_rows)
    floor_ms = t0 - fhp.CANDLE_MS
    mark = fhp._fetch_candles_paged(
        "/market/history-mark-price-candles", fhp.INST_SWAP, floor_ms
    )
    spot = fhp._fetch_candles_paged("/market/history-candles", fhp.INST_SPOT, floor_ms)
    mark_c = {int(r[0]): float(r[4]) for r in mark}
    spot_c = {int(r[0]): float(r[4]) for r in spot}
    return mark_c, spot_c


def _candle_covering(
    fts: int, keys: list[int], closes: dict[int, float]
) -> tuple[int | None, float | None, int]:
    i = bisect.bisect_right(keys, fts) - 1
    if i < 0:
        return None, None, 10**15
    ts = keys[i]
    if fts >= ts + fhp.CANDLE_MS:
        return None, None, 10**15
    skew = min(fts - ts, ts + fhp.CANDLE_MS - fts)
    return ts, closes[ts], skew


def _attach_candles(funding_rows: list[dict]) -> list[dict]:
    mark_c, spot_c = _build_candle_index(funding_rows)
    mark_keys = sorted(mark_c)
    spot_keys = sorted(spot_c)
    out: list[dict] = []
    for r in funding_rows:
        fts = r["fundingTime"]
        mts, mpx, ms = _candle_covering(fts, mark_keys, mark_c)
        sts, spx, ss = _candle_covering(fts, spot_keys, spot_c)
        skew_ok = ms <= fhp.MAX_SKEW_MS and ss <= fhp.MAX_SKEW_MS
        ok = int(mpx is not None and spx is not None and skew_ok)
        out.append(
            {
                "fundingTime": fts,
                "fundingRate": float(r["fundingRate"]),
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


def _write_csv(rows_by_ts: dict[int, dict]) -> None:
    rows_sorted = [rows_by_ts[k] for k in sorted(rows_by_ts)]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows_sorted:
            w.writerow({k: row.get(k, "") for k in FIELDS})


def _row_matches(old: dict, new: dict, tol: float = 1e-7) -> bool:
    for key in ("fundingRate",):
        a = str(old.get(key, "")).strip()
        b = str(new.get(key, "")).strip()
        try:
            if abs(float(a) - float(b)) > tol:
                return False
        except ValueError:
            return a == b
    return True


def _append_log(payload: dict) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_provenance(src_path: Path, payload: dict) -> Path:
    stem = src_path.stem.replace(" ", "_")
    md_path = HERE / f"INGEST_{stem}.md"
    md_path.write_text(
        f"""# Ingest provenance — `{src_path.name}`

| Field | Value |
|-------|--------|
| Ingested (UTC) | {payload['timestamp']} |
| Source file | `{src_path}` |
| Detected funding-time column | `{payload['time_col']}` |
| Detected funding-rate column | `{payload['rate_col']}` |
| Time units | `{payload['ts_units']}` |
| External rows parsed | {payload['external_rows']} |
| Rows added to panel | {payload['rows_added']} |
| Rows confirmed (already matched) | {payload['rows_confirmed']} |
| Rows overwritten | {payload['rows_overwritten']} |
| Rows skipped (mismatch, no overwrite) | {payload['rows_mismatch_skipped']} |
| Panel rows after ingest | {payload['rows_total']} |
| Panel `align_ok` after ingest | {payload['align_ok_pct']:.2f}% |
| Span (days) | {payload['days_span']:.2f} |
| D1 met (≥365d) | {payload['phase3_d1_met']} |

## Reproduce

```bash
python research/datasets/H-PERP-003/ingest_external_csv.py {src_path} \\
    --funding-time-col {payload['time_col']} \\
    --funding-rate-col {payload['rate_col']} \\
    --ts-units {payload['ts_units']}
```
""",
        encoding="utf-8",
    )
    return md_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="External CSV path")
    p.add_argument("--funding-time-col", default=None)
    p.add_argument("--funding-rate-col", default=None)
    p.add_argument(
        "--ts-units",
        default="auto",
        choices=("auto", "ms", "s", "iso"),
        help="Timestamp units in the external CSV (default: auto-detect).",
    )
    p.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing rows when the new fundingRate disagrees. "
        "Default: skip (safer).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"source file not found: {args.source}")

    started = _now_iso()
    external = _load_external(
        args.source, args.funding_time_col, args.funding_rate_col, args.ts_units
    )
    if not external:
        raise SystemExit(f"no usable rows parsed from {args.source}")

    # Re-detect columns so they appear in the log even when auto-detected.
    with open(args.source, newline="", encoding="utf-8-sig") as f:
        headers = next(csv.reader(f))
    detected_time, detected_rate = (args.funding_time_col, args.funding_rate_col)
    if not detected_time or not detected_rate:
        detected_time, detected_rate = _detect_columns(headers)

    funding_rows = _attach_candles(external)
    new_by_ts = {r["fundingTime"]: r for r in funding_rows}

    existing = _load_existing()
    merged: dict[int, dict] = dict(existing)

    added = confirmed = overwritten = mismatch_skipped = 0
    for ts, row in new_by_ts.items():
        if ts in existing:
            if _row_matches(existing[ts], row):
                confirmed += 1
                continue
            if args.allow_overwrite:
                merged[ts] = row
                overwritten += 1
            else:
                mismatch_skipped += 1
            continue
        merged[ts] = row
        added += 1

    if not args.dry_run and (added > 0 or overwritten > 0):
        _write_csv(merged)

    align_ok_n = sum(1 for r in merged.values() if int(r.get("align_ok") or 0) == 1)
    n_total = len(merged)
    if n_total >= 2:
        ts_sorted = sorted(merged)
        days_span = (ts_sorted[-1] - ts_sorted[0]) / (1000.0 * 86400.0)
    else:
        days_span = 0.0

    payload = {
        "timestamp": started,
        "ok": True,
        "kind": "ingest_external_csv",
        "dry_run": bool(args.dry_run),
        "source": str(args.source),
        "time_col": detected_time,
        "rate_col": detected_rate,
        "ts_units": args.ts_units,
        "external_rows": len(external),
        "rows_added": added,
        "rows_confirmed": confirmed,
        "rows_overwritten": overwritten,
        "rows_mismatch_skipped": mismatch_skipped,
        "rows_total": n_total,
        "align_ok_total": align_ok_n,
        "align_ok_pct": round(100.0 * align_ok_n / n_total, 2) if n_total else 0.0,
        "days_span": round(days_span, 3),
        "phase3_d1_met": days_span >= 365.0,
    }
    _append_log(payload)
    if not args.dry_run:
        _write_provenance(args.source, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
