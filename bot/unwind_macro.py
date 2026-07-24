#!/usr/bin/env python3
"""
Controlled macro unwind.

Sells back only the macro positions that expire more than 30 days out,
so fast-sleeve capital can be freed without losing the near-term macro
positions that are already close to settlement.

Defaults to DRY RUN so you can see the plan before anything executes.

Usage:
    python unwind_macro.py                # dry run (no orders placed)
    python unwind_macro.py --execute      # actually place the sell orders
    python unwind_macro.py --execute --aggressive  # cross spread for faster fills

Pricing:
    - default: place a limit sell 1 cent better than mid to encourage a fill
      without crossing the spread immediately
    - --aggressive: place a limit sell at the bid (equivalent of a mid-cross
      for the NO side) so the order fills quickly
"""

import os
import re
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

from src.execution.kalshi_client import KalshiExecutionClient
from src.data.sources.kalshi import KalshiAdapter


MONTH_MAP = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
             "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

MACRO_PREFIXES = {"KXCPI", "KXUSNFP", "KXPAYROLLS", "KXFFR", "KXTEMP", "KXHMONTHRANGE"}

UNWIND_MIN_DAYS = 30   # only unwind positions further out than this


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


def classify_macro(ticker: str) -> bool:
    series = ticker.split("-")[0] if ticker else ""
    return series in MACRO_PREFIXES


@click.command()
@click.option("--execute", is_flag=True, help="Actually place the sell orders (default: dry run).")
@click.option("--aggressive", is_flag=True, help="Cross the spread for faster fills.")
@click.option("--min-days", default=UNWIND_MIN_DAYS, help="Only unwind positions expiring in more than this many days.")
@click.option("--log", default="logs/unwind.jsonl", help="JSONL log path for unwind actions.")
@click.option("--cancel-first", is_flag=True, help="Cancel all resting orders on target tickers before re-posting.")
def main(execute, aggressive, min_days, log, cancel_first):
    client = KalshiExecutionClient(demo=False)
    adapter = KalshiAdapter(demo=False)

    avail, total = client.get_balance()
    positions = [p for p in client.get_positions(strict=True) if p.position != 0]

    print("=" * 78)
    print(f"  MACRO UNWIND {'(EXECUTE)' if execute else '(DRY RUN)'}")
    print("=" * 78)
    print(f"  Cash before:   ${avail:.2f}")
    print(f"  Portfolio:     ${total:.2f}")
    print(f"  Positions:     {len(positions)}")
    print(f"  Rule:          unwind macro positions expiring > {min_days} days out")
    print(f"  Pricing:       {'aggressive (cross)' if aggressive else 'passive (mid-1c)'}")

    now = datetime.now(timezone.utc)
    unwind_plan = []

    for p in positions:
        if not classify_macro(p.ticker):
            continue
        exp = parse_expiry(p.ticker)
        if exp is None:
            continue
        days = (exp - now).total_seconds() / 86400.0
        if days <= min_days:
            continue
        unwind_plan.append({
            "ticker": p.ticker,
            "side": "YES" if p.position > 0 else "NO",
            "qty": abs(p.position),
            "cost": p.total_cost,
            "cost_per_cents": round(p.total_cost / max(abs(p.position), 1) * 100, 0),
            "days": days,
            "expiry": exp.isoformat(),
        })

    if not unwind_plan:
        print(f"\n  No positions match the unwind rule. Nothing to do.")
        return

    print(f"\n  Positions to unwind: {len(unwind_plan)}")
    total_cost = sum(x["cost"] for x in unwind_plan)
    print(f"  Total cost basis:    ${total_cost:.2f}")

    # Build the plan with live bid/ask for each market
    print(f"\n  {'Ticker':<32} {'Side':<4} {'Qty':>3} {'CostPer':>8} "
          f"{'YesBid':>6} {'YesAsk':>6} {'ExitPx':>7} {'EstVal':>8} {'Days':>5}")
    print("  " + "-" * 76)

    enriched = []
    for x in unwind_plan:
        try:
            m = adapter.get_market(x["ticker"])
            yes_bid = float(m.yes_bid or 0)
            yes_ask = float(m.yes_ask or 100)
        except Exception as e:
            print(f"  {x['ticker']}: could not fetch market data ({e})")
            continue

        spread = yes_ask - yes_bid

        # In Kalshi, for a YES/NO market:
        #   no_bid = 100 - yes_ask   (best price a buyer will pay for NO)
        #   no_ask = 100 - yes_bid   (best price a seller is asking for NO)
        # sell_position posts limit_price as yes_price for YES positions and
        # as no_price for NO positions, so limit_px must be in the right
        # ccy-side.
        if x["side"] == "NO":
            no_bid = max(0, 100 - yes_ask)
            no_ask = max(0, 100 - yes_bid)
            if aggressive:
                # hit the bid to fill immediately
                limit_px = int(round(max(1, min(99, no_bid if no_bid > 0 else (no_ask - 1)))))
            else:
                # undercut best ask by 1c to encourage a passive fill
                limit_px = int(round(max(1, min(99, (no_ask - 1) if no_ask > 1 else no_bid))))
            est_val_per_cent = limit_px  # selling NO at limit_px cents -> receive limit_px cents
        else:
            if aggressive:
                limit_px = int(round(max(1, min(99, yes_bid if yes_bid > 0 else (yes_ask - 1)))))
            else:
                limit_px = int(round(max(1, min(99, (yes_ask - 1) if yes_ask > 1 else yes_bid))))
            est_val_per_cent = limit_px

        est_value = est_val_per_cent * x["qty"] / 100.0
        enriched.append({
            **x,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "spread": spread,
            "limit_px": limit_px,
            "est_value": est_value,
            "est_realized": est_value - x["cost"],
        })

        print(f"  {x['ticker']:<32} {x['side']:<4} {x['qty']:>3} "
              f"{x['cost_per_cents']:>7.0f}c "
              f"{yes_bid:>6.0f} {yes_ask:>6.0f} "
              f"{limit_px:>6d}c ${est_value:>7.2f} {x['days']:>5.0f}")

    total_est_value = sum(e["est_value"] for e in enriched)
    total_est_realized = sum(e["est_realized"] for e in enriched)
    print(f"\n  Estimated total received: ${total_est_value:.2f}")
    print(f"  Estimated realized P&L:   ${total_est_realized:+.2f}")
    print(f"  (cost basis ${total_cost:.2f} -> estimated ${total_est_value:.2f})")

    if not execute:
        print(f"\n  DRY RUN only. Pass --execute to actually place sell orders.")
        print(f"  After execute, re-run health_check.py to confirm freed cash.")
        return

    # Cancel first if requested — Kalshi reserves inventory for resting sells,
    # so we have to clear them before re-posting with new prices or we get
    # "insufficient_balance" errors.
    if cancel_first:
        print(f"\n  Cancelling existing orders on {len(enriched)} tickers...")
        total_cancelled = 0
        for e in enriched:
            try:
                open_orders = client.get_open_orders(ticker=e["ticker"])
                for o in open_orders:
                    if client.cancel_order(o.order_id):
                        total_cancelled += 1
                if open_orders:
                    print(f"    {e['ticker']}: cancelled {len(open_orders)} resting order(s)")
            except Exception as exc:
                print(f"    cancel fail {e['ticker']}: {exc}")
        print(f"  Cancelled {total_cancelled} resting order(s) total")
        time.sleep(2.0)

    # Execute
    print(f"\n  EXECUTING {len(enriched)} sell orders...")
    log_path = Path(log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for e in enriched:
        try:
            order = client.sell_position(
                ticker=e["ticker"],
                contracts=e["qty"],
                use_market=False,
                limit_price=e["limit_px"],
            )
            success = order is not None
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": e["ticker"],
                "side": e["side"],
                "qty": e["qty"],
                "limit_px": e["limit_px"],
                "est_value": e["est_value"],
                "cost": e["cost"],
                "success": success,
                "order_id": order.order_id if order else None,
            }
        except Exception as exc:
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": e["ticker"],
                "error": str(exc),
                "success": False,
            }
        results.append(result)
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        status = "OK" if result.get("success") else "FAIL"
        print(f"  [{status}] {e['ticker']} sell {e['qty']}@{e['limit_px']}c")
        time.sleep(0.25)

    # Snapshot after
    time.sleep(1.5)
    avail_after, total_after = client.get_balance()
    print(f"\n  Cash after:        ${avail_after:.2f}  (was ${avail:.2f})")
    print(f"  Portfolio after:   ${total_after:.2f}  (was ${total:.2f})")
    ok_count = sum(1 for r in results if r.get("success"))
    print(f"  Orders placed:     {ok_count}/{len(results)} successful")
    print(f"\n  Note: these are LIMIT orders; some may rest unfilled. Check")
    print(f"  Kalshi portfolio. To be more aggressive, re-run with --aggressive.")


if __name__ == "__main__":
    main()
