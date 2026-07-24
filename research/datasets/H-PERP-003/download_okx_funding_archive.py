#!/usr/bin/env python3
"""
Download first-party OKX historical funding-rate ZIPs and combine them into a
single CSV suitable for `ingest_external_csv.py`.

OKX's historical-data page is a React app, but the download button calls a
public JSON endpoint:

  POST /priapi/v5/broker/public/trade-data/download-link

This script uses that endpoint to obtain canonical `static.okx.com` ZIP URLs
for `BTC-USDT-SWAP` monthly funding-rate archives, then normalizes the rows to:

  fundingTime,fundingRate,source_file

Usage:

  python research/datasets/H-PERP-003/download_okx_funding_archive.py \
      --start 2024-05 --end 2026-05 --ingest

By default it writes the combined CSV only. Pass `--ingest` to invoke the
existing panel merge after download.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

HERE = Path(__file__).resolve().parent
OKX_API = "https://www.okx.com/priapi/v5/broker/public/trade-data/download-link"
DEFAULT_INST_ID = "BTC-USDT-SWAP"
USER_AGENT = "Mozilla/5.0"

TIME_COLS = (
    "fundingtime",
    "funding_time",
    "funding time",
    "funding-time",
    "fundingtimeutc",
    "funding_time_utc",
    "funding time utc",
)
RATE_COLS = (
    "fundingrate",
    "funding_rate",
    "funding rate",
    "funding-rate",
    "realizedfundingrate",
    "realized_funding_rate",
    "realized funding rate",
)
FILENAME_MONTH_MARKER = "-fundingrates-"


def _month_start(month: str) -> datetime:
    try:
        return datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemExit(f"invalid month {month!r}; expected YYYY-MM") from exc


def _month_after(dt: datetime) -> datetime:
    year = dt.year + (1 if dt.month == 12 else 0)
    month = 1 if dt.month == 12 else dt.month + 1
    return dt.replace(year=year, month=month, day=1)


def _month_end_ms(dt: datetime) -> str:
    return str(int(_month_after(dt).timestamp() * 1000) - 1)


def _ms(dt: datetime) -> str:
    return str(int(dt.timestamp() * 1000))


def _inst_family(inst_id: str) -> str:
    if not inst_id.endswith("-SWAP"):
        raise SystemExit(f"expected an OKX swap instrument ending in -SWAP: {inst_id}")
    return inst_id[: -len("-SWAP")]


def _normalize_header(value: str) -> str:
    return value.strip().lower().replace("\ufeff", "").replace("_", "").replace("-", "").replace(" ", "")


def _find_col(headers: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    by_norm = {_normalize_header(h): h for h in headers}
    for candidate in candidates:
        found = by_norm.get(_normalize_header(candidate))
        if found:
            return found
    return None


def _okx_download_links(inst_id: str, start_month: datetime, end_month: datetime) -> list[dict]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Referer": "https://www.okx.com/en-us/historical-data",
            "Origin": "https://www.okx.com",
            "Content-Type": "application/json",
        }
    )
    payload = {
        "module": "3",  # FUNDING_RATES in the OKX historical-data bundle.
        "instType": "SWAP",
        "instQueryParam": {"instFamilyList": [_inst_family(inst_id)]},
        "dateQuery": {
            "dateAggrType": "monthly",
            "begin": _ms(start_month),
            "end": _month_end_ms(end_month),
        },
    }
    resp = session.post(OKX_API, json=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if str(body.get("code")) != "0":
        raise SystemExit(f"OKX download-link failed: {json.dumps(body, indent=2)}")

    links: list[dict] = []
    for detail in body.get("data", {}).get("details", []):
        for group in detail.get("groupDetails", []):
            filename = group.get("filename") or ""
            if group.get("url") and filename and _filename_month_in_range(filename, start_month, end_month):
                links.append(group)
    links.sort(key=lambda item: item.get("filename", ""))
    return links


def _filename_month_in_range(filename: str, start_month: datetime, end_month: datetime) -> bool:
    """OKX may return the following month; keep only the requested range."""
    try:
        marker_start = filename.index(FILENAME_MONTH_MARKER) + len(FILENAME_MONTH_MARKER)
        month = datetime.strptime(filename[marker_start: marker_start + 7], "%Y-%m").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, IndexError):
        return True
    return start_month <= month <= end_month


def _parse_zip_csv(content: bytes, source_file: str) -> list[dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"{source_file} has no CSV member")
        for name in csv_names:
            with zf.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text)
                if not reader.fieldnames:
                    raise ValueError(f"{source_file}:{name} has no header row")
                time_col = _find_col(reader.fieldnames, TIME_COLS)
                rate_col = _find_col(reader.fieldnames, RATE_COLS)
                if not time_col or not rate_col:
                    raise ValueError(
                        f"{source_file}:{name} missing funding columns; "
                        f"headers={reader.fieldnames!r}"
                    )
                for row in reader:
                    fts = (row.get(time_col) or "").strip()
                    rate = (row.get(rate_col) or "").strip()
                    if not fts or not rate:
                        continue
                    rows.append(
                        {
                            "fundingTime": fts,
                            "fundingRate": rate,
                            "source_file": source_file,
                        }
                    )
    return rows


def _download_and_combine(links: list[dict], out_path: Path, raw_dir: Path | None) -> dict:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows: list[dict] = []
    downloaded: list[str] = []
    for link in links:
        url = link["url"]
        filename = link["filename"]
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        content = resp.content
        downloaded.append(filename)
        if raw_dir:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / filename).write_bytes(content)
        rows.extend(_parse_zip_csv(content, filename))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fundingTime", "fundingRate", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "ok": True,
        "archives": len(downloaded),
        "rows": len(rows),
        "output": str(out_path),
        "downloaded": downloaded,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inst-id", default=DEFAULT_INST_ID)
    parser.add_argument("--start", required=True, help="Start month, YYYY-MM")
    parser.add_argument("--end", required=True, help="End month, YYYY-MM, inclusive")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Combined funding CSV output path.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Optional directory for downloaded OKX ZIPs.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print OKX archive URLs without downloading ZIP contents.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="After writing the combined CSV, run ingest_external_csv.py on it.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Pass --allow-overwrite through to ingest_external_csv.py.",
    )
    args = parser.parse_args(argv)

    start_month = _month_start(args.start)
    end_month = _month_start(args.end)
    if end_month < start_month:
        raise SystemExit("--end must be >= --start")

    out_path = args.out or HERE / f"okx_{args.inst_id}_funding_{args.start}_to_{args.end}.csv"
    links = _okx_download_links(args.inst_id, start_month, end_month)
    if not links:
        raise SystemExit("OKX returned no archive links for the requested range")

    if args.list_only:
        print(json.dumps({"ok": True, "archives": links}, indent=2))
        return 0

    result = _download_and_combine(links, out_path, args.raw_dir)
    print(json.dumps(result, indent=2))

    if args.ingest:
        cmd = [
            sys.executable,
            str(HERE / "ingest_external_csv.py"),
            str(out_path),
            "--funding-time-col",
            "fundingTime",
            "--funding-rate-col",
            "fundingRate",
            "--ts-units",
            "auto",
        ]
        if args.allow_overwrite:
            cmd.append("--allow-overwrite")
        raise SystemExit(subprocess.call(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
