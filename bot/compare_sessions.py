#!/usr/bin/env python3
"""
Compare live trading sessions to decide which operating mode is best.

Reads session_end events from logs/live_trades.jsonl and computes
turnover-aware metrics for each session, then ranks them.

Compatible with both old-format (no sleeves) and enriched session_end events.

Decision criteria (in order of importance):
  1. Daily realized P&L rate ($/day)
  2. Win rate (must be > 50% to be sustainable)
  3. Capital recycle time (avg hours per round-trip)
  4. Phantom count (infrastructure health indicator)

Promotion criteria — a mode is promoted to default if it meets ALL of:
  1. Daily P&L rate > $0.00 (profitable)
  2. Win rate > 50%
  3. Average hold time < 48 hours
  4. At least 10 completed round-trips
  5. No kill-switch triggers
  6. Zero phantom removals (infrastructure must be stable)

Usage:
    python compare_sessions.py
    python compare_sessions.py --log logs/live_trades.jsonl
    python compare_sessions.py --sleeve-detail
"""

import json
import sys
from pathlib import Path

import click


def load_sessions(log_path: Path):
    sessions = []
    if not log_path.exists():
        return sessions
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "session_end":
                sessions.append(evt)
    return sessions


def _fmt_rate(val):
    if val is None or val == 0:
        return "  n/a"
    return f"${val:+.2f}"


@click.command()
@click.option("--log", default="logs/live_trades.jsonl", help="Path to JSONL trade log")
@click.option("--sleeve-detail", is_flag=True, help="Show per-sleeve breakdown for each session")
def main(log, sleeve_detail):
    log_path = Path(log)
    sessions = load_sessions(log_path)

    if not sessions:
        print(f"No session_end events found in {log_path}")
        sys.exit(1)

    # Filter out empty sessions (no trades, no positions)
    meaningful = [s for s in sessions
                  if s.get("trades_taken", 0) > 0 or s.get("positions_open", 0) > 0]

    print("=" * 90)
    print("  SESSION COMPARISON - DECISION GATE")
    print("=" * 90)

    if not meaningful:
        print("\n  No meaningful sessions found (all had 0 trades and 0 positions).")
        print("  Run the bot with different modes to populate this table.")
        return

    ranked = []
    for s in meaningful:
        sid = s.get("session_id", "?")
        hours = s.get("session_hours", 0)
        won = s.get("trades_won", 0)
        lost = s.get("trades_lost", 0)
        total = won + lost
        win_rate = won / max(total, 1)
        pnl = s.get("realized_pnl", 0)
        daily_rate = s.get("daily_pnl_rate", pnl / max(hours / 24, 0.01))
        closed = s.get("positions_closed", 0)
        still_open = s.get("positions_open", 0)
        avg_hold = s.get("avg_hold_hours", 0)
        killed = s.get("killed", False)
        phantoms = s.get("phantom_count", 0)
        open_cap = s.get("open_capital", 0)
        rest_cap = s.get("resting_capital", 0)
        sleeves = s.get("sleeves", {})
        taken = s.get("trades_taken", 0)

        ranked.append({
            "id": sid,
            "hours": hours,
            "trades": total,
            "taken": taken,
            "win_rate": win_rate,
            "pnl": pnl,
            "daily_rate": daily_rate,
            "closed": closed,
            "open": still_open,
            "avg_hold": avg_hold,
            "killed": killed,
            "phantoms": phantoms,
            "open_capital": open_cap,
            "resting_capital": rest_cap,
            "sleeves": sleeves,
            "kill_reason": s.get("kill_reason", ""),
        })

    ranked.sort(key=lambda x: x["daily_rate"], reverse=True)

    hdr = (f"  {'Session':<28} {'Hrs':>5} {'Trades':>6} {'WR':>5} "
           f"{'P&L':>8} {'$/Day':>8} {'Closed':>6} {'Open':>4} "
           f"{'Hold':>6} {'Phntm':>5} {'Kill':>5}")
    print(f"\n{hdr}")
    print("  " + "-" * 86)

    for r in ranked:
        sid = r["id"]
        if len(sid) > 26:
            sid = sid[:26] + ".."
        kill_mark = "YES" if r["killed"] else "-"
        print(
            f"  {sid:<28} {r['hours']:>5.1f} {r['trades']:>6} "
            f"{r['win_rate']:>4.0%} {r['pnl']:>+8.2f} "
            f"{_fmt_rate(r['daily_rate']):>8} {r['closed']:>6} "
            f"{r['open']:>4} {r['avg_hold']:>5.1f}h "
            f"{r['phantoms']:>5} {kill_mark:>5}"
        )

    # Sleeve detail
    if sleeve_detail:
        print(f"\n{'=' * 90}")
        print("  SLEEVE BREAKDOWN PER SESSION")
        print(f"{'=' * 90}")
        for r in ranked:
            sleeves = r.get("sleeves", {})
            if not sleeves:
                continue
            print(f"\n  {r['id']}")
            for sl_name in ("fast", "macro", "other"):
                sl = sleeves.get(sl_name, {})
                wins = sl.get("wins", 0)
                losses = sl.get("losses", 0)
                pnl = sl.get("pnl", 0)
                cost = sl.get("cost", 0)
                op = sl.get("open", 0)
                if wins or losses or op or cost > 0:
                    wr = wins / max(wins + losses, 1) * 100
                    print(f"    {sl_name:6s}: {wins}W/{losses}L ({wr:.0f}% WR) "
                          f"pnl=${pnl:+.2f} cost=${cost:.2f} open={op}")

    # Promotion evaluation
    print(f"\n{'=' * 90}")
    print("  PROMOTION CRITERIA")
    print(f"{'=' * 90}")
    print("""
  A mode is promoted to default if it meets ALL of:
    1. Daily P&L rate > $0.00 (profitable)
    2. Win rate > 50%
    3. Average hold time < 48 hours (capital recycling)
    4. At least 10 completed round-trips (statistical significance)
    5. No kill-switch triggers during the session
    6. Zero phantom removals (infrastructure stability)
""")

    best = ranked[0] if ranked else None
    if best:
        issues = []
        if best["daily_rate"] <= 0:
            issues.append("Daily P&L rate is not positive")
        if best["win_rate"] <= 0.50:
            issues.append(f"Win rate {best['win_rate']:.0%} is below 50%")
        if best["avg_hold"] > 48:
            issues.append(f"Avg hold {best['avg_hold']:.1f}h exceeds 48h target")
        if best["trades"] < 10:
            issues.append(f"Only {best['trades']} trades (need 10+ for significance)")
        if best["killed"]:
            issues.append(f"Session killed: {best['kill_reason']}")
        if best["phantoms"] > 0:
            issues.append(f"{best['phantoms']} phantom removals (infrastructure issue)")

        if not issues:
            print(f"  VERDICT: Session {best['id']} meets all promotion criteria!")
            print(f"  This mode can be considered for additional capital.")
        else:
            print(f"  VERDICT: Best session ({best['id']}) has issues:")
            for i in issues:
                print(f"    - {i}")
            print("  Continue running sessions to gather more evidence.")

    print()


if __name__ == "__main__":
    main()
