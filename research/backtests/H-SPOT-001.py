#!/usr/bin/env python3
"""
Pre-registered backtest for H-SPOT-001 (see ../06_backtest_design_H-SPOT-001.md).

Run from anywhere:
  python RLBotPM/research/backtests/H-SPOT-001.py
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets" / "H-SPOT-001" / "btcusd_daily_coinbase.csv"
OUT_JSON = ROOT / "backtests" / "H-SPOT-001_metrics.json"

SLOW, FAST = 120, 20
TAKER = 0.006  # 0.6% per side when position changes by 1 unit


def sma(series: list[float], t: int, n: int) -> float:
    return sum(series[t - n + 1 : t + 1]) / n


def load_closes() -> list[float]:
    closes: list[float] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            closes.append(float(row["close"]))
    return closes


def build_pos_raw(closes: list[float]) -> list[int | None]:
    n = len(closes)
    pos: list[int | None] = [None] * n
    start = SLOW - 1
    for t in range(start, n):
        s120 = sma(closes, t, SLOW)
        s20 = sma(closes, t, FAST)
        c = closes[t]
        pos[t] = 1 if (c > s120 and s20 > s120) else 0
    return pos


def daily_returns(closes: list[float], pos_raw: list[int | None]) -> tuple[list[float], list[float]]:
    """Return (r_gross, r_net) aligned index t for return from t-1 -> t."""
    n = len(closes)
    rg = [0.0] * n
    rn = [0.0] * n
    for t in range(SLOW, n):
        if pos_raw[t - 1] is None:
            continue
        p0, p1 = pos_raw[t - 1], pos_raw[t]
        if p0 is None or p1 is None:
            continue
        g = p0 * (closes[t] / closes[t - 1] - 1.0)
        fee = TAKER * abs(p1 - p0)
        rg[t] = g
        rn[t] = g - fee
    return rg, rn


def segment_indices(n: int) -> list[tuple[int, int]]:
    """Three equal contiguous OOS ranges [lo, hi) after warm-up (see design doc)."""
    lo = SLOW
    length = n - lo
    q = length // 3
    out = []
    for k in range(3):
        a = lo + k * q
        b = lo + (k + 1) * q if k < 2 else n
        out.append((a, b))
    return out


def oos_mask(n: int, segs: list[tuple[int, int]]) -> list[bool]:
    mask = [False] * n
    for a, b in segs:
        for t in range(a, b):
            mask[t] = True
    return mask


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


def main() -> None:
    closes = load_closes()
    pos_raw = build_pos_raw(closes)
    _, r_net = daily_returns(closes, pos_raw)
    n = len(closes)
    segs = segment_indices(n)
    mask = oos_mask(n, segs)

    oos = [r_net[t] for t in range(n) if mask[t]]
    seg_means = []
    for a, b in segs:
        seg_means.append(mean([r_net[t] for t in range(a, b)]))

    cum = sum(oos)
    s = sharpe(oos)
    pf = profit_factor(oos)
    # G5: no single *winning* day contributes >25% of total *gross* OOS profit
    pos_sum = sum(x for x in oos if x > 0)
    best_day = max((x for x in oos if x > 0), default=0.0)
    if pos_sum <= 1e-12:
        g5 = True
    else:
        g5 = (best_day / pos_sum) <= 0.25

    # 2x fee stress
    r_net_2x = []
    for t in range(n):
        if not mask[t]:
            continue
        if pos_raw[t] is None or pos_raw[t - 1] is None:
            continue
        p0, p1 = pos_raw[t - 1], pos_raw[t]
        g = p0 * (closes[t] / closes[t - 1] - 1.0)
        fee = 2 * TAKER * abs(p1 - p0)
        r_net_2x.append(g - fee)
    cum_2x = sum(r_net_2x)

    # Placebo: permute within each OOS segment independently
    rng = random.Random(42)
    beat = 0
    trials = 500
    for _ in range(trials):
        shuf: list[float] = []
        for a, b in segs:
            block = [r_net[t] for t in range(a, b)]
            rng.shuffle(block)
            shuf.extend(block)
        if sum(shuf) >= cum:
            beat += 1
    g6 = (beat / trials) <= 0.05

    g1 = s >= 1.5
    g2 = pf >= 1.4
    g3 = cum_2x > 0
    g4 = sum(1 for m in seg_means if m > 0) >= 3

    verdict = "PASS" if all([g1, g2, g3, g4, g5, g6]) else "FAIL"

    out = {
        "hypothesis": "H-SPOT-001",
        "n_closes": n,
        "oos_days": len(oos),
        "sharpe_oos": round(s, 4),
        "profit_factor_oos": round(pf, 4),
        "cum_pnl_oos": round(cum, 6),
        "cum_pnl_oos_2x_fee": round(cum_2x, 6),
        "seg_means": [round(x, 6) for x in seg_means],
        "g1_sharpe": g1,
        "g2_pf": g2,
        "g3_2x_fee": g3,
        "g4_segpos": g4,
        "g5_concentration": g5,
        "g6_placebo_frac_beats": round(beat / trials, 4),
        "g6_pass": g6,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
