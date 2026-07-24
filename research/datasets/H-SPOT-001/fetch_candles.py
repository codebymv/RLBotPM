#!/usr/bin/env python3
"""Download Coinbase Exchange BTC-USD daily candles (public, no key).

First request returns the latest ~350 rows (exchange cap). Subsequent requests
use explicit ``start``/``end`` windows stepping backward 300 days at a time so
we can exceed 12 months of history.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "btcusd_daily_coinbase.csv"
PROV = Path(__file__).resolve().parent / "PROVENANCE.md"
BASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
DAY = 86400
PAGE = 300  # max candles per Coinbase request


def main() -> None:
    by_t: dict[int, list] = {}

    # Page 0: latest chunk (no start/end — exchange returns max available window)
    r = requests.get(BASE, params={"granularity": str(DAY)}, timeout=60)
    r.raise_for_status()
    for row in r.json():
        by_t[int(row[0])] = row
    oldest = min(by_t)
    time.sleep(0.11)

    # Walk backward in 300-day windows until 900+ calendar days or empty/stale
    target_days = 900
    pages = 0
    while pages < 40 and len(by_t) * (DAY / DAY) < target_days:
        end_ts = oldest - 1
        start_ts = end_ts - PAGE * DAY
        r = requests.get(
            BASE,
            params={
                "granularity": str(DAY),
                "start": str(start_ts),
                "end": str(end_ts),
            },
            timeout=60,
        )
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        before = len(by_t)
        for row in chunk:
            by_t[int(row[0])] = row
        new_oldest = min(int(c[0]) for c in chunk)
        if new_oldest >= oldest and len(by_t) == before:
            break
        oldest = min(oldest, new_oldest)
        pages += 1
        time.sleep(0.11)

    asc = sorted(by_t.values(), key=lambda c: int(c[0]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "low", "high", "open", "close", "volume"])
        for c in asc:
            w.writerow(c)

    first = datetime.fromtimestamp(int(asc[0][0]), tz=timezone.utc)
    last = datetime.fromtimestamp(int(asc[-1][0]), tz=timezone.utc)
    span_days = (int(asc[-1][0]) - int(asc[0][0])) / DAY
    PROV.write_text(
        f"""# Dataset provenance — H-SPOT-001

| Field | Value |
|-------|--------|
| Source | Coinbase Exchange REST `GET /products/BTC-USD/candles` |
| URL | {BASE} |
| Granularity | 86400 (1 day) |
| Retrieved (UTC) | {datetime.now(timezone.utc).isoformat()} |
| Rows | {len(asc)} |
| Span (days) | ~{span_days:.0f} |
| First candle (UTC) | {first.isoformat()} |
| Last candle (UTC) | {last.isoformat()} |

Schema: time (epoch s), low, high, open, close, volume (Coinbase native order).
Pagination: initial unbounded request + backward `{PAGE}`-day `start`/`end` windows.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(asc),
                "span_days": round(span_days, 1),
                "first": first.isoformat(),
                "last": last.isoformat(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
