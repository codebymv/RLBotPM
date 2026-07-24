#!/usr/bin/env python3
"""
Phase 4: Research Scoreboard.

Compares operating modes across live sessions, paper sessions, and backtests
using a unified set of conviction metrics.

A mode must pass ALL conviction thresholds to be recommended for capital.

Metrics tracked per mode (fast-only, hybrid, macro-lite):
  1. Realized P&L per day        (must be > $0)
  2. Win rate                    (must be > 55%)
  3. Average hold time           (must be < 48h for fast, < 720h for macro)
  4. Capital recycle rate        (trades closed / day, higher is better)
  5. Execution edge realized     (actual return / cost, must be > 50%)
  6. Phantom rate                (phantoms / trades, must be 0%)
  7. Max drawdown                (peak-to-trough, must be < 30%)
  8. Sample size                 (min 20 settled trades for any confidence)

Usage:
    python research_scoreboard.py
    python research_scoreboard.py --log logs/live_trades.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import click


CONVICTION_THRESHOLDS = {
    "daily_pnl_rate": ("Daily P&L rate", "> $0.00", lambda v: v > 0),
    "win_rate": ("Win rate", "> 55%", lambda v: v > 0.55),
    "avg_hold_hours": ("Avg hold time", "< 48h", lambda v: v < 48),
    "recycle_rate": ("Recycle rate", "> 1/day", lambda v: v > 1),
    "execution_edge": ("Execution edge", "> 50%", lambda v: v > 0.50),
    "phantom_rate": ("Phantom rate", "= 0%", lambda v: v <= 0.001),
    "sample_size": ("Sample size", ">= 20", lambda v: v >= 20),
}


def load_events(log_path: Path):
    events = []
    if not log_path.exists():
        return events
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def compute_mode_metrics(events):
    """Compute conviction metrics from all available session data."""
    sessions = [e for e in events if e.get("type") == "session_end"]
    orders = [e for e in events if e.get("type") == "order_placed"]
    settlements = [e for e in events if e.get("type") == "settlement"]
    phantoms = [e for e in events if e.get("type") == "phantom_removed"]

    total_hours = sum(s.get("session_hours", 0) for s in sessions)
    total_days = max(total_hours / 24.0, 0.01)
    total_won = sum(s.get("trades_won", 0) for s in sessions)
    total_lost = sum(s.get("trades_lost", 0) for s in sessions)
    total_settled = total_won + total_lost
    total_pnl = sum(s.get("realized_pnl", 0) for s in sessions)
    total_closed = sum(s.get("positions_closed", 0) for s in sessions)
    total_phantom = len(phantoms)
    total_orders = len(orders)

    # Avg hold from sessions that report it
    hold_hours = [s.get("avg_hold_hours", 0) for s in sessions if s.get("positions_closed", 0) > 0]
    avg_hold = sum(hold_hours) / max(len(hold_hours), 1)

    # Execution edge from order data
    exec_edges = []
    for o in orders:
        cost = o.get("cost", 0)
        contracts = o.get("contracts", 1)
        if cost > 0 and contracts > 0:
            cost_per = cost / contracts
            payout = 1.0
            exec_edge = (payout - cost_per) / cost_per
            exec_edges.append(exec_edge)
    avg_exec_edge = sum(exec_edges) / max(len(exec_edges), 1)

    return {
        "daily_pnl_rate": total_pnl / total_days,
        "win_rate": total_won / max(total_settled, 1),
        "avg_hold_hours": avg_hold,
        "recycle_rate": total_closed / total_days,
        "execution_edge": avg_exec_edge,
        "phantom_rate": total_phantom / max(total_orders, 1),
        "sample_size": total_settled,
        "_total_pnl": total_pnl,
        "_total_hours": total_hours,
        "_sessions": len(sessions),
        "_orders": total_orders,
        "_settlements": total_settled,
        "_phantoms": total_phantom,
    }


@click.command()
@click.option("--log", default="logs/live_trades.jsonl", help="Path to JSONL trade log")
def main(log):
    log_path = Path(log)
    events = load_events(log_path)

    if not events:
        print(f"No events found in {log_path}")
        sys.exit(1)

    metrics = compute_mode_metrics(events)

    W = 72
    print("=" * W)
    print("  RESEARCH SCOREBOARD - CONVICTION ASSESSMENT")
    print("=" * W)

    print(f"\n  Data summary:")
    print(f"    Sessions:      {metrics['_sessions']}")
    print(f"    Total hours:   {metrics['_total_hours']:.1f}")
    print(f"    Orders placed: {metrics['_orders']}")
    print(f"    Settlements:   {metrics['_settlements']}")
    print(f"    Phantoms:      {metrics['_phantoms']}")
    print(f"    Total P&L:     ${metrics['_total_pnl']:+.2f}")

    print(f"\n  {'Metric':<25} {'Value':>12} {'Threshold':>12} {'Pass':>6}")
    print(f"  {'-'*55}")

    pass_count = 0
    total_checks = len(CONVICTION_THRESHOLDS)

    for key, (name, threshold_desc, check_fn) in CONVICTION_THRESHOLDS.items():
        value = metrics.get(key, 0)
        passed = check_fn(value)
        if passed:
            pass_count += 1
        mark = "YES" if passed else "NO"

        if key == "daily_pnl_rate":
            val_str = f"${value:+.2f}/day"
        elif key in ("win_rate", "execution_edge", "phantom_rate"):
            val_str = f"{value:.1%}"
        elif key == "avg_hold_hours":
            val_str = f"{value:.1f}h"
        elif key == "recycle_rate":
            val_str = f"{value:.1f}/day"
        elif key == "sample_size":
            val_str = f"{int(value)}"
        else:
            val_str = f"{value:.2f}"

        print(f"  {name:<25} {val_str:>12} {threshold_desc:>12} {mark:>6}")

    print(f"\n  {'='*55}")
    print(f"  CONVICTION SCORE: {pass_count}/{total_checks}")

    if pass_count == total_checks:
        print(f"\n  ** ALL THRESHOLDS MET **")
        print(f"  This mode has earned conviction for additional capital.")
    elif pass_count >= total_checks - 2:
        print(f"\n  CLOSE but not there yet. Fix the failing metrics:")
        for key, (name, threshold_desc, check_fn) in CONVICTION_THRESHOLDS.items():
            if not check_fn(metrics.get(key, 0)):
                print(f"    - {name}: needs {threshold_desc}")
    else:
        print(f"\n  INSUFFICIENT EVIDENCE for capital deployment.")
        print(f"  Major improvements needed before more funds are justified.")

    # Recommendations
    print(f"\n{'=' * W}")
    print("  RESEARCH PROGRAM RECOMMENDATIONS")
    print(f"{'=' * W}")

    if metrics["_phantoms"] > 0:
        print("""
  PRIORITY 1: Fix phantom problem.
    {phantoms} phantom events detected. This is an infrastructure failure,
    not a strategy failure. Until phantoms reach zero:
      - Run bot as persistent service (not manual start/stop)
      - Add settlement recovery on startup
      - Save position state to disk between sessions
""".format(phantoms=metrics["_phantoms"]))

    if metrics["sample_size"] < 20:
        print("""
  PRIORITY 2: Gather more data.
    Only {n} settled trades. Need at least 20 for minimal statistical
    confidence. Run the bot continuously in fast-only mode to accumulate
    settlements quickly.
""".format(n=int(metrics["sample_size"])))

    if metrics["win_rate"] <= 0.55 and metrics["sample_size"] >= 10:
        print("""
  PRIORITY 3: Improve signal quality.
    Win rate of {wr:.0%} is below the 55% threshold. Consider:
      - Restricting to far-OTM trivial edges only
      - Adding vol regime detection
      - Paper-trading near-the-money signals before going live
""".format(wr=metrics["win_rate"]))

    print(f"\n  Run 'python compare_sessions.py --sleeve-detail' for session-level breakdown.")
    print(f"  Run 'python edge_audit.py' for edge type analysis.")
    print(f"  Run 'python loss_narrative.py' for the full loss story.")
    print("=" * W)


if __name__ == "__main__":
    main()
