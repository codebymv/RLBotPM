#!/usr/bin/env python3
"""
Position review: rank every open Kalshi position by time-to-exit and
estimated exit value so we can decide whether to wait, hold, or unwind.
"""

import os
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

from src.execution.kalshi_client import KalshiExecutionClient

MONTH_MAP = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
             "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def parse_expiry(ticker: str):
    m = re.search(r"26(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(\d{2})", ticker)
    if m:
        try:
            return datetime(2026, MONTH_MAP[m.group(1)], int(m.group(2)),
                            int(m.group(3)), 0, 0, tzinfo=timezone.utc)
        except ValueError:
            return None
    m2 = re.search(r"26(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?:-|$)", ticker)
    if m2:
        try:
            return datetime(2026, MONTH_MAP[m2.group(1)], 28, 23, 59,
                            tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def main():
    client = KalshiExecutionClient(demo=False)
    avail, total = client.get_balance()
    positions = [p for p in client.get_positions(strict=True) if p.position != 0]

    print("=" * 82)
    print("  POSITION REVIEW — FREE-CASH TIMELINE")
    print("=" * 82)
    # Kalshi /portfolio/balance returns portfolio_value as positions-only
    # mark-to-market, NOT cash+positions.
    positions_value = total
    total_wealth = avail + positions_value
    deployed_cost = sum(p.total_cost for p in positions)
    print(f"  Cash:               ${avail:.2f}")
    print(f"  Positions value:    ${positions_value:.2f}  (Kalshi mark-to-market)")
    print(f"  Total wealth:       ${total_wealth:.2f}  (cash + positions)")
    print(f"  Deployed cost:      ${deployed_cost:.2f}")
    print(f"  Unrealized vs cost: ${positions_value - deployed_cost:+.2f}")

    now = datetime.now(timezone.utc)
    rows = []

    for p in positions:
        exp = parse_expiry(p.ticker)
        hours_to_exp = (exp - now).total_seconds() / 3600.0 if exp else None
        per_contract_cost = p.total_cost / max(abs(p.position), 1)
        # Payout per contract = $1 if correct
        max_gain = abs(p.position) * (1.0 - per_contract_cost)
        max_loss = p.total_cost
        rows.append({
            "ticker": p.ticker,
            "side": "NO" if p.position < 0 else "YES",
            "qty": abs(p.position),
            "cost": p.total_cost,
            "cost_per": per_contract_cost,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "expiry": exp,
            "hours": hours_to_exp,
        })

    rows.sort(key=lambda r: (r["hours"] is None, r["hours"] or 1e9))

    print(f"\n  {'Ticker':<40} {'Side':<3} {'Qty':>3} {'Cost':>6} {'PerCt':>6} "
          f"{'MaxGain':>7} {'Expires':>18} {'Days':>5}")
    print("  " + "-" * 78)

    buckets = {"<24h": [], "24-72h": [], "3-30d": [], "30d+": [], "unknown": []}
    for r in rows:
        if r["hours"] is None:
            bucket = "unknown"
            exp_str = "?"
            days_str = "?"
        elif r["hours"] < 24:
            bucket = "<24h"
            exp_str = r["expiry"].strftime("%m/%d %H:%M")
            days_str = f"{r['hours']/24:.1f}"
        elif r["hours"] < 72:
            bucket = "24-72h"
            exp_str = r["expiry"].strftime("%m/%d %H:%M")
            days_str = f"{r['hours']/24:.1f}"
        elif r["hours"] < 24 * 30:
            bucket = "3-30d"
            exp_str = r["expiry"].strftime("%m/%d")
            days_str = f"{r['hours']/24:.0f}"
        else:
            bucket = "30d+"
            exp_str = r["expiry"].strftime("%Y-%m-%d")
            days_str = f"{r['hours']/24:.0f}"
        buckets[bucket].append(r)

        print(f"  {r['ticker']:<40} {r['side']:<3} {r['qty']:>3} "
              f"${r['cost']:>5.2f} ${r['cost_per']:>4.2f} "
              f"${r['max_gain']:>6.2f} {exp_str:>18} {days_str:>5}")

    print(f"\n  {'Bucket':<10} {'Count':>6} {'CostSum':>10} {'MaxGainSum':>12}")
    print("  " + "-" * 40)
    for name in ("<24h", "24-72h", "3-30d", "30d+", "unknown"):
        items = buckets[name]
        c = sum(x["cost"] for x in items)
        g = sum(x["max_gain"] for x in items)
        print(f"  {name:<10} {len(items):>6} ${c:>9.2f} ${g:>11.2f}")

    print(f"\n{'=' * 82}")
    print("  FREE-CASH RECOVERY OUTLOOK")
    print(f"{'=' * 82}")
    soon = sum(x["cost"] for x in buckets["<24h"])
    short = sum(x["cost"] for x in buckets["24-72h"])
    med = sum(x["cost"] for x in buckets["3-30d"])
    late = sum(x["cost"] for x in buckets["30d+"])
    print(f"  Cost recoverable <24h:    ${soon:.2f}")
    print(f"  Cost recoverable 24-72h:  ${short:.2f}")
    print(f"  Cost recoverable 3-30d:   ${med:.2f}")
    print(f"  Cost recoverable 30d+:    ${late:.2f}")

    if avail < 1.0:
        print(f"\n  Cash is below the $1.00 min trade cost.")
        print(f"  Bot cannot place new fast-sleeve trades until either:")
        print(f"    (a) a position settles and returns cash, or")
        print(f"    (b) you manually unwind a position by selling it back")

    print(f"\n{'=' * 82}")
    print("  RECOMMENDED ACTION RANKING")
    print(f"{'=' * 82}")
    # Recommend based on days to expiry and sleeve
    for i, r in enumerate(rows, 1):
        if r["hours"] is None:
            rec = "UNKNOWN — inspect manually"
        elif r["hours"] < 24:
            rec = "HOLD (settles within a day, exits itself)"
        elif r["hours"] < 72:
            rec = "HOLD (settles within 3 days)"
        elif r["hours"] < 24 * 30:
            rec = "CONSIDER SELL (trapped weeks)"
        else:
            rec = "SELL (trapped months, major opportunity cost)"
        print(f"  {i:>2}. {r['ticker']:<40} {rec}")


if __name__ == "__main__":
    main()
