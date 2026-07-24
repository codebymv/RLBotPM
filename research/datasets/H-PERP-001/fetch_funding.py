#!/usr/bin/env python3
"""
Download OKX BTC-USDT-SWAP funding history into CSV (public API, no key).

Bybit `api.bybit.com` and Binance `fapi.binance.com` returned HTTP 403 / 451
from this environment on 2026-04-19; OKX public endpoint responded 200.
See DATA_SOURCE_SUBSTITUTION.md.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "btcusdt_swap_funding_okx.csv"
PROV = Path(__file__).resolve().parent / "PROVENANCE.md"
BASE = "https://www.okx.com/api/v5/public/funding-rate-history"


def main() -> None:
    rows: list[dict] = []
    # OKX pagination: pass `after` = oldest fundingTime from the previous page
    # to retrieve the next (older) chunk. Verified empirically 2026-04-19.
    after_cursor: str | None = None
    for page in range(120):
        params: dict = {"instId": "BTC-USDT-SWAP", "limit": "100"}
        if after_cursor is not None:
            params["after"] = after_cursor
        r = requests.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        if js.get("code") != "0":
            raise RuntimeError(js)
        chunk = js.get("data") or []
        if not chunk:
            break
        rows.extend(chunk)
        oldest = min(int(x["fundingTime"]) for x in chunk)
        after_cursor = str(oldest)
        time.sleep(0.12)
        if len(rows) >= 5000:
            break

    by_ts = {int(r["fundingTime"]): r for r in rows}
    rows_sorted = sorted(by_ts.values(), key=lambda x: int(x["fundingTime"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["fundingTime", "fundingRate", "instId", "realizedRate"],
        )
        w.writeheader()
        for r in rows_sorted:
            w.writerow(
                {
                    "fundingTime": r["fundingTime"],
                    "fundingRate": r.get("fundingRate", r.get("realizedRate", "")),
                    "instId": r.get("instId", "BTC-USDT-SWAP"),
                    "realizedRate": r.get("realizedRate", ""),
                }
            )

    first = datetime.fromtimestamp(
        int(rows_sorted[0]["fundingTime"]) / 1000, tz=timezone.utc
    )
    last = datetime.fromtimestamp(
        int(rows_sorted[-1]["fundingTime"]) / 1000, tz=timezone.utc
    )
    PROV.write_text(
        f"""# Dataset provenance — H-PERP-001

| Field | Value |
|-------|--------|
| Primary source (attempted) | Bybit `GET /v5/market/funding/history` — **HTTP 403** from fetch environment (2026-04-19) |
| **Substitution** | OKX public `GET /api/v5/public/funding-rate-history` |
| URL | {BASE} |
| Instrument | `BTC-USDT-SWAP` |
| Retrieved (UTC) | {datetime.now(timezone.utc).isoformat()} |
| Rows | {len(rows_sorted)} |
| First timestamp | {first.isoformat()} |
| Last timestamp | {last.isoformat()} |
| Output CSV | `{OUT.name}` |

## Sign convention

OKX publishes `fundingRate` / `realizedRate` per interval. **Backtest code re-validates**
direction against OKX documentation before interpreting long/short cashflows.

## Reproducibility

```bash
python fetch_funding.py
```
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows_sorted),
                "first_ts_utc": first.isoformat(),
                "last_ts_utc": last.isoformat(),
                "csv": OUT.name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
