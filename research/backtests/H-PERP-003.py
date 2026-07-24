#!/usr/bin/env python3
"""
Pre-registered evaluation for H-PERP-003 (see ../06_backtest_design_H-PERP-003.md).

Run from anywhere:
  python RLBotPM/research/backtests/H-PERP-003.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets" / "H-PERP-003" / "btc_hedged_panel_okx.csv"
OUT_JSON = ROOT / "backtests" / "H-PERP-003_metrics.json"
V = 100.0
FEE_RT = 0.0011  # one-way on V at each full-window edge (06 §3)
INTERVALS_PER_DAY = 3.0  # ~8h funding; Sharpe annualization matches H-PERP-001 spirit


def load_rows(csv_path: Path | None = None) -> list[dict]:
    path = Path(csv_path) if csv_path else CSV_PATH
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: int(r["fundingTime"]))
    return rows


def sharpe_8h(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    if len(xs) < 10 or pstdev(xs) < 1e-12:
        return 0.0
    return (mean(xs) / pstdev(xs)) * math.sqrt(365.0 * INTERVALS_PER_DAY)


def profit_factor(xs: list[float]) -> float:
    pos = sum(x for x in xs if x > 0)
    neg = sum(-x for x in xs if x < 0)
    if neg < 1e-12:
        return 99.0 if pos > 0 else 0.0
    return pos / neg


def week_key_utc(ts_ms: int) -> tuple[int, int]:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    y, w, _ = dt.isocalendar()
    return (y, w)


def d4_bad_weeks(rows: list[dict]) -> tuple[int, list[tuple[int, int, float]]]:
    """
    D4: any ISO week (UTC) with >50% funding rows not align_ok → INCONCLUSIVE_DATA.
    Returns (count_bad_weeks, detail sample).
    """
    by_w: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in rows:
        ok = int(r.get("align_ok") or 0)
        by_w[week_key_utc(int(r["fundingTime"]))].append(ok)
    bad: list[tuple[int, int, float]] = []
    for wk, oks in by_w.items():
        n = len(oks)
        if n <= 0:
            continue
        frac_ok = sum(oks) / n
        if frac_ok < 0.5:
            bad.append((wk[0], wk[1], round(frac_ok, 4)))
    return len(bad), bad[:12]


def segment_id(ts: int, t0: int, t1: int, k: int = 4) -> int:
    if t1 <= t0:
        return 0
    span = t1 - t0
    x = (ts - t0) / span
    seg = int(x * k)
    if seg >= k:
        seg = k - 1
    return seg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bootstrap-trials",
        type=int,
        default=500,
        help="G6 sign-flip placebo trials (default 500). Frozen in 06 §5 amendment.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="G6 placebo RNG seed (default 42)."
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Path to a hedged-panel CSV (schema = btc_hedged_panel_okx.csv). "
            "Defaults to research/datasets/H-PERP-003/btc_hedged_panel_okx.csv. "
            "Required by C3 backtest-as-a-service sketch."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Path to write the verdict JSON. Defaults to "
            "research/backtests/H-PERP-003_metrics.json."
        ),
    )
    args = p.parse_args(argv)
    csv_path = Path(args.csv) if args.csv else CSV_PATH
    out_path = Path(args.out) if args.out else OUT_JSON
    rows = load_rows(csv_path)
    if len(rows) < 8:
        out = {
            "hypothesis": "H-PERP-003",
            "verdict": "INCONCLUSIVE_DATA",
            "reason": f"panel has {len(rows)} rows; need merged CSV from fetch_hedged_panel.py",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 0

    t_first = int(rows[0]["fundingTime"])
    t_last = int(rows[-1]["fundingTime"])
    days_span = (t_last - t_first) / (1000.0 * 86400.0)
    d1_ok = days_span >= 365.0
    d4_n_bad, d4_detail = d4_bad_weeks(rows)
    d4_ok = d4_n_bad == 0
    data_contract_ok = d1_ok and d4_ok

    intervals_pnl: list[tuple[int, float]] = []  # (end_funding_ts, pnl_usdt)
    for i in range(len(rows) - 1):
        a, nxt = rows[i], rows[i + 1]
        if int(a["align_ok"]) != 1 or int(nxt["align_ok"]) != 1:
            continue
        try:
            f0 = float(a["mark_close"])
            f1 = float(nxt["mark_close"])
            s0 = float(a["spot_close"])
            s1 = float(nxt["spot_close"])
        except (TypeError, ValueError):
            continue
        if f0 <= 0 or f1 <= 0 or s0 <= 0 or s1 <= 0:
            continue
        r_f = math.log(f1 / f0)
        r_s = math.log(s1 / s0)
        fr = float(nxt.get("fundingRate") or 0.0)
        pnl = V * fr + V * (r_s - r_f)
        intervals_pnl.append((int(nxt["fundingTime"]), pnl))

    if not intervals_pnl:
        out = {
            "hypothesis": "H-PERP-003",
            "verdict": "INCONCLUSIVE_DATA",
            "reason": "no valid consecutive aligned intervals",
            "days_span": round(days_span, 3),
            "d1_calendar_depth_ge_365d": d1_ok,
            "d4_weeks_gt_half_missing": d4_n_bad,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 0

    t0, t1 = t_first, t_last
    oos_pnls: list[float] = []
    seg_pnls: dict[int, list[float]] = defaultdict(list)
    for ts_end, pnl in intervals_pnl:
        seg = segment_id(ts_end, t0, t1, 4)
        seg_pnls[seg].append(pnl)
        if seg >= 1:
            oos_pnls.append(pnl)

    seg_means = [mean(seg_pnls[k]) if seg_pnls[k] else 0.0 for k in range(4)]
    oos_seg_means = [mean(seg_pnls[k]) if seg_pnls[k] else 0.0 for k in (1, 2, 3)]

    cum_oos_gross = sum(oos_pnls)
    cum_with_fee = cum_oos_gross - 2.0 * FEE_RT * V
    cum_2x_fee = cum_oos_gross - 2.0 * (2.0 * FEE_RT) * V

    s = sharpe_8h(oos_pnls)
    pf = profit_factor(oos_pnls)
    pos_sum = sum(x for x in oos_pnls if x > 0)
    best_iv = max((x for x in oos_pnls if x > 0), default=0.0)
    if pos_sum <= 1e-12:
        g5 = True
    else:
        g5 = (best_iv / pos_sum) <= 0.25

    # G6 (06 §5 amendment 2026-05-04): random sign-flip placebo.
    # Permutation-of-multiset placebos are degenerate (sum and Sharpe are both
    # multiset invariants), so we randomize the *sign* of each interval instead
    # — the cumulative OOS sum then varies across trials around 0 under the null.
    rng = random.Random(args.seed)
    trials = max(1, int(args.bootstrap_trials))
    beat = 0
    shuffle_sums: list[float] = []
    for _ in range(trials):
        shuf_sum = 0.0
        for x in oos_pnls:
            sign = 1.0 if rng.random() >= 0.5 else -1.0
            shuf_sum += sign * x
        shuffle_sums.append(shuf_sum)
        if shuf_sum >= cum_oos_gross:
            beat += 1
    g6 = (beat / trials) <= 0.05

    g1 = s >= 1.5
    g2 = pf >= 1.4
    g3 = cum_2x_fee > 0.0
    g4 = sum(1 for m in oos_seg_means if m > 0.0) >= 3
    gates = [g1, g2, g3, g4, g5, g6]

    if not data_contract_ok:
        verdict = "INCONCLUSIVE_DATA"
    elif all(gates):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    align_n = sum(int(r["align_ok"]) for r in rows)
    out = {
        "hypothesis": "H-PERP-003",
        "n_funding_rows": len(rows),
        "n_intervals_used": len(intervals_pnl),
        "n_oos_intervals": len(oos_pnls),
        "days_span": round(days_span, 3),
        "d1_calendar_depth_ge_365d": d1_ok,
        "d4_weeks_gt_half_missing": d4_n_bad,
        "d4_bad_weeks_sample": d4_detail,
        "data_contract_ok": data_contract_ok,
        "align_ok_frac": round(align_n / len(rows), 4),
        "sharpe_oos": round(s, 4),
        "profit_factor_oos": round(pf, 4),
        "cum_oos_gross_usdt": round(cum_oos_gross, 6),
        "cum_oos_after_window_fees_usdt": round(cum_with_fee, 6),
        "cum_oos_2x_window_fees_usdt": round(cum_2x_fee, 6),
        "seg_means_all_four": [round(x, 6) for x in seg_means],
        "oos_seg_means": [round(x, 6) for x in oos_seg_means],
        "g1_sharpe": g1,
        "g2_pf": g2,
        "g3_2x_fee": g3,
        "g4_segpos": g4,
        "g5_concentration": g5,
        "g6_method": "random_sign_flip",
        "g6_placebo_trials": trials,
        "g6_placebo_seed": args.seed,
        "g6_placebo_frac_beats": round(beat / trials, 4),
        "g6_placebo_sum_mean": round(sum(shuffle_sums) / trials, 6) if trials else 0.0,
        "g6_pass": g6,
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
