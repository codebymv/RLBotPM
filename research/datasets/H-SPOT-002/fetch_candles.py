#!/usr/bin/env python3
"""Download Coinbase Exchange BTC-USD + ETH-USD daily candles (public, no key).

Writes an aligned inner-join CSV for H-SPOT-002 and a PROVENANCE.md.
Pagination mirrors H-SPOT-001 (unbounded latest chunk + backward 300-day windows).
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "btc_eth_daily_coinbase.csv"
PROV = OUT_DIR / "PROVENANCE.md"
BASE = "https://api.exchange.coinbase.com/products/{product}/candles"
PRODUCTS = ("BTC-USD", "ETH-USD")
DAY = 86400
PAGE = 300
TARGET_DAYS = 900


def _fetch_product(product: str) -> dict[int, list]:
    url = BASE.format(product=product)
    by_t: dict[int, list] = {}

    r = requests.get(url, params={"granularity": str(DAY)}, timeout=60)
    r.raise_for_status()
    for row in r.json():
        by_t[int(row[0])] = row
    oldest = min(by_t)
    time.sleep(0.11)

    pages = 0
    while pages < 40 and (max(by_t) - min(by_t)) / DAY < TARGET_DAYS:
        end_ts = oldest - 1
        start_ts = end_ts - PAGE * DAY
        r = requests.get(
            url,
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

    return by_t


def main() -> None:
    series = {p: _fetch_product(p) for p in PRODUCTS}
    common = sorted(set(series["BTC-USD"]) & set(series["ETH-USD"]))
    if not common:
        raise SystemExit("no overlapping candle timestamps between BTC-USD and ETH-USD")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "btc_close", "eth_close", "btc_volume", "eth_volume"])
        for ts in common:
            b = series["BTC-USD"][ts]
            e = series["ETH-USD"][ts]
            # Coinbase candle: [time, low, high, open, close, volume]
            w.writerow([ts, b[4], e[4], b[5], e[5]])

    first = datetime.fromtimestamp(common[0], tz=timezone.utc)
    last = datetime.fromtimestamp(common[-1], tz=timezone.utc)
    span_days = (common[-1] - common[0]) / DAY
    PROV.write_text(
        f"""# Dataset provenance — H-SPOT-002

| Field | Value |
|-------|--------|
| Source | Coinbase Exchange REST `GET /products/{{BTC-USD,ETH-USD}}/candles` |
| Granularity | 86400 (1 day) |
| Retrieved (UTC) | {datetime.now(timezone.utc).isoformat()} |
| Aligned rows | {len(common)} |
| Span (days) | ~{span_days:.0f} |
| First candle (UTC) | {first.isoformat()} |
| Last candle (UTC) | {last.isoformat()} |
| BTC-only rows (dropped) | {len(series["BTC-USD"]) - len(common)} |
| ETH-only rows (dropped) | {len(series["ETH-USD"]) - len(common)} |

Schema: time (epoch s), btc_close, eth_close, btc_volume, eth_volume.
Join: inner join on candle timestamp. Pagination: initial unbounded request +
backward `{PAGE}`-day `start`/`end` windows per product.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aligned_rows": len(common),
                "span_days": round(span_days, 1),
                "first": first.isoformat(),
                "last": last.isoformat(),
                "csv": str(OUT_CSV),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
