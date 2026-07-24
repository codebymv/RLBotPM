#!/usr/bin/env python3
"""
Pull OKX BTC-USDT-SWAP funding + 1H mark candles + 1H BTC-USDT spot candles,
merge at each fundingTime per `06_backtest_design_H-PERP-003.md` (1H bar containing
`fundingTime`, skew-to-boundary ≤ 60s).

Candles: public REST returns ~300 bars per request; older history is paginated with
`after=<min_open_ts - 1>` (OKX returns newest-first; `before=` does not walk history
reliably on these endpoints).

Public API only. Run from repo anywhere:
  python RLBotPM/research/datasets/H-PERP-003/fetch_hedged_panel.py
"""
from __future__ import annotations

import bisect
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

CANDLE_MS = 3600 * 1000  # 1H bar length
MAX_SKEW_MS = 60_000  # D1/D4: drop intervals with skew-to-boundary > 60s (06 §1)
CANDLE_LIMIT = "300"
# align_ok: both legs have a 1h candle [ts, ts+1h) containing fts, each skew ≤ MAX_SKEW_MS

SESSION = requests.Session()
BASE = "https://www.okx.com/api/v5"
INST_SWAP = "BTC-USDT-SWAP"
INST_SPOT = "BTC-USDT"
# OKX mark-price history does not accept 8H bars; 1H close is sampled per funding row.
BAR = "1H"
SLEEP = 0.12
HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "btc_hedged_panel_okx.csv"
PROV = HERE / "PROVENANCE_PULL.md"


def _get(path: str, params: dict) -> dict:
    r = SESSION.get(f"{BASE}{path}", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    if js.get("code") != "0":
        raise RuntimeError(js)
    return js


def fetch_funding_all() -> list[dict]:
    """Paginate funding older until empty or cap."""
    rows: list[dict] = []
    after_cursor: str | None = None
    for _ in range(500):
        params: dict = {"instId": INST_SWAP, "limit": "100"}
        if after_cursor is not None:
            params["after"] = after_cursor
        js = _get("/public/funding-rate-history", params)
        chunk = js.get("data") or []
        if not chunk:
            break
        rows.extend(chunk)
        oldest = min(int(x["fundingTime"]) for x in chunk)
        after_cursor = str(oldest)
        time.sleep(SLEEP)
    by_ts = {int(r["fundingTime"]): r for r in rows}
    return sorted(by_ts.values(), key=lambda x: int(x["fundingTime"]))


def _fetch_candles_paged(path: str, inst_id: str, floor_ts_ms: int) -> list[list]:
    """
    Walk from newest toward older using `after=<previous_page_min_open - 1>`.
    Dedup by candle open time. Stop when oldest open is at/before `floor_ts_ms`, the
    API returns empty, or pagination stops making progress.
    """
    by_ts: dict[int, list] = {}
    after: str | None = None
    prev_min_open: int | None = None
    for _ in range(5000):
        params: dict = {"instId": inst_id, "bar": BAR, "limit": CANDLE_LIMIT}
        if after is not None:
            params["after"] = after
        js = _get(path, params)
        chunk = js.get("data") or []
        if not chunk:
            break
        min_open = min(int(x[0]) for x in chunk if x and not isinstance(x, str))
        for row in chunk:
            if not row or isinstance(row, str):
                continue
            by_ts[int(row[0])] = row
        time.sleep(SLEEP)
        if min_open <= floor_ts_ms:
            break
        if prev_min_open is not None and min_open >= prev_min_open:
            break
        prev_min_open = min_open
        after = str(min_open - 1)
    return [by_ts[k] for k in sorted(by_ts)]


def main() -> None:
    funding = fetch_funding_all()
    if not funding:
        raise SystemExit("No funding rows")

    t0 = int(funding[0]["fundingTime"])
    floor_ms = t0 - CANDLE_MS
    mark_candles = _fetch_candles_paged("/market/history-mark-price-candles", INST_SWAP, floor_ms)
    spot_candles = _fetch_candles_paged("/market/history-candles", INST_SPOT, floor_ms)

    # index candle close by open ts (ms)
    def close_by_ts(candles: list[list]) -> dict[int, float]:
        out: dict[int, float] = {}
        for row in candles:
            ts = int(row[0])
            out[ts] = float(row[4])
        return out

    mark_c = close_by_ts(mark_candles)
    spot_c = close_by_ts(spot_candles)
    mark_ts_sorted = sorted(mark_c)
    spot_ts_sorted = sorted(spot_c)

    def candle_covering(
        fts: int, keys: list[int], closes: dict[int, float]
    ) -> tuple[int | None, float | None, int]:
        """
        1H candle [ts, ts+1h) containing fts; skew = distance to nearer boundary (for D4).
        """
        i = bisect.bisect_right(keys, fts) - 1
        if i < 0:
            return None, None, 10**15
        ts = keys[i]
        if fts >= ts + CANDLE_MS:
            return None, None, 10**15
        skew = min(fts - ts, ts + CANDLE_MS - fts)
        return ts, closes[ts], skew

    merged: list[dict] = []
    for r in funding:
        fts = int(r["fundingTime"])
        fr = float(r.get("fundingRate") or r.get("realizedRate") or 0.0)
        mts, mpx, ms = candle_covering(fts, mark_ts_sorted, mark_c)
        sts, spx, ss = candle_covering(fts, spot_ts_sorted, spot_c)
        skew_ok = ms <= MAX_SKEW_MS and ss <= MAX_SKEW_MS
        ok = int(mpx is not None and spx is not None and skew_ok)
        merged.append(
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

    first = datetime.fromtimestamp(merged[0]["fundingTime"] / 1000, tz=timezone.utc)
    last = datetime.fromtimestamp(merged[-1]["fundingTime"] / 1000, tz=timezone.utc)
    days = (merged[-1]["fundingTime"] - merged[0]["fundingTime"]) / (1000 * 86400)
    align_pct = 100.0 * sum(1 for x in merged if x["align_ok"]) / len(merged)

    HERE.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fundingTime",
                "fundingRate",
                "mark_candle_ts",
                "mark_close",
                "spot_candle_ts",
                "spot_close",
                "align_ok",
                "mark_skew_ms",
                "spot_skew_ms",
            ],
        )
        w.writeheader()
        for row in merged:
            w.writerow({k: row[k] for k in w.fieldnames})

    PROV.write_text(
        f"""# H-PERP-003 — pull provenance

| Field | Value |
|-------|--------|
| Retrieved (UTC) | {datetime.now(timezone.utc).isoformat()} |
| Script | `fetch_hedged_panel.py` |
| Base URL | `{BASE}` |
| Swap | `{INST_SWAP}` funding + mark **1H** (OKX rejects `8H` mark candles) |
| Spot | `{INST_SPOT}` candles **1H** |
| Funding rows | {len(merged)} |
| First funding (UTC) | {first.isoformat()} |
| Last funding (UTC) | {last.isoformat()} |
| Approx. span | {days:.1f} calendar days |
| `align_ok` (both legs in 1h bar + skew-to-boundary ≤ 60s each) | {align_pct:.2f}% |
| Unique 1H candle opens (mark / spot) | {len(mark_candles)} / {len(spot_candles)} |
| Output | `{OUT_CSV.name}` |

## Funding sign (short)

OKX **positive** `fundingRate` ⇒ **long pays short** ⇒ a **short** receives **+rate × notional**
in the usual linear USDT convention (verify against [OKX funding](https://www.okx.com/help/funding-fee) before live use).
Backtest [H-PERP-003.py](../../backtests/H-PERP-003.py) uses `b_i = fundingRate` as **USDT per $1 notional per interval**
only if that matches your spot check; flip sign in one place in code if docs disagree.

## Reproduce

```bash
python fetch_hedged_panel.py
```
""",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "rows": len(merged),
                "days_span": round(days, 2),
                "align_ok_pct": round(align_pct, 2),
                "csv": str(OUT_CSV.name),
                "phase3_d1_met": days >= 365.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
