#!/usr/bin/env python3
"""
Pre-registered backtest for H-SPOT-002 (see ../06_backtest_design_H-SPOT-002.md).

Run from anywhere:
  python RLBotPM/research/backtests/H-SPOT-002.py

Without the aligned CSV (or with D1 unmet), verdict is INCONCLUSIVE_DATA —
never a silent PASS.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets" / "H-SPOT-002" / "btc_eth_daily_coinbase.csv"
OUT_JSON = ROOT / "backtests" / "H-SPOT-002_metrics.json"

W = 30
Z_ENTER = 2.0
Z_EXIT = 0.5
TAKER = 0.006  # per side; fee uses 2 * TAKER * |Δpos|
D1_MIN_DAYS = 730.0
DAY = 86400


def load_series(path: Path) -> tuple[list[int], list[float], list[float]]:
    times: list[int] = []
    btc: list[float] = []
    eth: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times.append(int(row["time"]))
            btc.append(float(row["btc_close"]))
            eth.append(float(row["eth_close"]))
    return times, btc, eth


def log_spread(btc: list[float], eth: list[float]) -> list[float]:
    return [math.log(b) - math.log(e) for b, e in zip(btc, eth)]


def zscores(spread: list[float], w: int = W) -> list[float | None]:
    n = len(spread)
    out: list[float | None] = [None] * n
    for t in range(w - 1, n):
        window = spread[t - w + 1 : t + 1]
        mu = mean(window)
        sd = pstdev(window)
        if sd < 1e-12:
            out[t] = None
        else:
            out[t] = (spread[t] - mu) / sd
    return out


def build_pos_raw(z: list[float | None]) -> list[int | None]:
    """Stateful band trade per 06 §2."""
    n = len(z)
    pos: list[int | None] = [None] * n
    state = 0
    for t in range(n):
        zt = z[t]
        if zt is None:
            pos[t] = None
            continue
        if state == 0:
            if zt > Z_ENTER:
                state = -1
            elif zt < -Z_ENTER:
                state = 1
        elif state == 1:
            if zt > Z_ENTER:
                state = -1
            elif abs(zt) < Z_EXIT:
                state = 0
        elif state == -1:
            if zt < -Z_ENTER:
                state = 1
            elif abs(zt) < Z_EXIT:
                state = 0
        pos[t] = state
    return pos


def daily_returns(
    btc: list[float], eth: list[float], pos_raw: list[int | None]
) -> tuple[list[float], list[float]]:
    n = len(btc)
    rg = [0.0] * n
    rn = [0.0] * n
    for t in range(W, n):
        p0, p1 = pos_raw[t - 1], pos_raw[t]
        if p0 is None or p1 is None:
            continue
        r_ratio = (btc[t] / btc[t - 1] - 1.0) - (eth[t] / eth[t - 1] - 1.0)
        g = p0 * r_ratio
        fee = 2.0 * TAKER * abs(p1 - p0)
        rg[t] = g
        rn[t] = g - fee
    return rg, rn


def segment_indices(n: int) -> list[tuple[int, int]]:
    lo = W
    length = n - lo
    q = length // 3
    out: list[tuple[int, int]] = []
    for k in range(3):
        a = lo + k * q
        b = lo + (k + 1) * q if k < 2 else n
        out.append((a, b))
    return out


def sharpe(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    if len(xs) < 10 or pstdev(xs) < 1e-12:
        return 0.0
    return (mean(xs) / pstdev(xs)) * math.sqrt(365)


def profit_factor(xs: list[float]) -> float:
    pos = sum(x for x in xs if x > 0)
    neg = sum(-x for x in xs if x < 0)
    if neg < 1e-12:
        return 99.0 if pos > 0 else 0.0
    return pos / neg


def week_missing_breach(times: list[int]) -> bool:
    """D4: any ISO-ish week bucket with >50% missing vs expected daily grid."""
    if len(times) < 2:
        return True
    # Bucket by floor(ts / (7*DAY)); count observed days; expect ~7 if dense.
    buckets: dict[int, int] = {}
    for ts in times:
        buckets[ts // (7 * DAY)] = buckets.get(ts // (7 * DAY), 0) + 1
    # Skip first/last partial weeks (boundary).
    keys = sorted(buckets)
    if len(keys) <= 2:
        return False
    for k in keys[1:-1]:
        if buckets[k] < 3.5:  # >50% missing of 7
            return True
    return False


def evaluate(
    times: list[int],
    btc: list[float],
    eth: list[float],
    *,
    trials: int = 500,
    seed: int = 42,
) -> dict:
    n = len(times)
    span_days = (times[-1] - times[0]) / DAY if n >= 2 else 0.0
    d1 = span_days >= D1_MIN_DAYS
    d4_ok = not week_missing_breach(times)
    data_ok = d1 and d4_ok and n > W + 30

    if not data_ok:
        return {
            "hypothesis": "H-SPOT-002",
            "n_rows": n,
            "days_span": round(span_days, 2),
            "d1_calendar_depth_ge_730d": d1,
            "d4_week_density_ok": d4_ok,
            "data_contract_ok": False,
            "verdict": "INCONCLUSIVE_DATA",
            "g6_method": "random_sign_flip",
        }

    spread = log_spread(btc, eth)
    z = zscores(spread)
    pos_raw = build_pos_raw(z)
    _, r_net = daily_returns(btc, eth, pos_raw)
    segs = segment_indices(n)
    oos = [r_net[t] for a, b in segs for t in range(a, b)]
    seg_means = [mean([r_net[t] for t in range(a, b)]) for a, b in segs]

    cum = sum(oos)
    s = sharpe(oos)
    pf = profit_factor(oos)

    pos_sum = sum(x for x in oos if x > 0)
    best_day = max((x for x in oos if x > 0), default=0.0)
    g5 = True if pos_sum <= 1e-12 else (best_day / pos_sum) <= 0.25

    # 2x fee stress
    r_net_2x: list[float] = []
    for a, b in segs:
        for t in range(a, b):
            p0, p1 = pos_raw[t - 1], pos_raw[t]
            if p0 is None or p1 is None:
                continue
            r_ratio = (btc[t] / btc[t - 1] - 1.0) - (eth[t] / eth[t - 1] - 1.0)
            g = p0 * r_ratio
            fee = 2.0 * (2.0 * TAKER) * abs(p1 - p0)
            r_net_2x.append(g - fee)
    cum_2x = sum(r_net_2x)

    rng = random.Random(seed)
    beat = 0
    for _ in range(trials):
        shuf_sum = sum(rng.choice((-1.0, 1.0)) * x for x in oos)
        if shuf_sum >= cum:
            beat += 1
    g6 = (beat / trials) <= 0.05

    g1 = s >= 1.5
    g2 = pf >= 1.4
    g3 = cum_2x > 0
    g4 = sum(1 for m in seg_means if m > 0) >= 3

    verdict = "PASS" if all([g1, g2, g3, g4, g5, g6]) else "FAIL"
    n_signals = sum(
        1
        for t in range(1, n)
        if pos_raw[t] is not None
        and pos_raw[t - 1] is not None
        and pos_raw[t] != pos_raw[t - 1]
        and pos_raw[t] != 0
    )

    return {
        "hypothesis": "H-SPOT-002",
        "n_rows": n,
        "days_span": round(span_days, 2),
        "oos_days": len(oos),
        "n_entries": n_signals,
        "sharpe_oos": round(s, 4),
        "profit_factor_oos": round(pf, 4),
        "cum_pnl_oos": round(cum, 6),
        "cum_pnl_oos_2x_fee": round(cum_2x, 6),
        "seg_means": [round(x, 6) for x in seg_means],
        "d1_calendar_depth_ge_730d": d1,
        "d4_week_density_ok": d4_ok,
        "data_contract_ok": True,
        "g1_sharpe": g1,
        "g2_pf": g2,
        "g3_2x_fee": g3,
        "g4_segpos": g4,
        "g5_concentration": g5,
        "g6_placebo_frac_beats": round(beat / trials, 4),
        "g6_method": "random_sign_flip",
        "g6_pass": g6,
        "verdict": verdict,
        "params": {
            "W": W,
            "Z_ENTER": Z_ENTER,
            "Z_EXIT": Z_EXIT,
            "taker_per_side": TAKER,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()

    if not args.csv.exists():
        out = {
            "hypothesis": "H-SPOT-002",
            "n_rows": 0,
            "days_span": 0.0,
            "d1_calendar_depth_ge_730d": False,
            "d4_week_density_ok": False,
            "data_contract_ok": False,
            "verdict": "INCONCLUSIVE_DATA",
            "g6_method": "random_sign_flip",
            "note": f"missing dataset CSV: {args.csv}",
        }
    else:
        times, btc, eth = load_series(args.csv)
        out = evaluate(
            times, btc, eth, trials=args.bootstrap_trials, seed=args.seed
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
