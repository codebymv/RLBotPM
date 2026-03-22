"""
Paper trading promotion gate check.

Evaluates readiness for real-money trading based on lifetime paper stats.
Run from repo root: python bot/scripts/paper_promotion_check.py

Usage:
    python bot/scripts/paper_promotion_check.py
    python bot/scripts/paper_promotion_check.py --log bot/logs/paper_trades.jsonl
"""

from __future__ import annotations

import argparse
import sys
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


# Promotion gate targets
GATE_MIN_SETTLED = 200
GATE_MIN_SESSIONS = 10
GATE_MIN_DAYS = 14
GATE_MIN_WIN_RATE = 0.85
GATE_WORST_SESSION_PNL = -5.0  # worst session must be > -$5


def _parse_args() -> argparse.Namespace:
    # Resolve log path relative to repo root (parent of bot/)
    _repo_root = Path(__file__).resolve().parent.parent.parent
    _default_log = _repo_root / "bot" / "logs" / "paper_trades.jsonl"
    parser = argparse.ArgumentParser(description="Check paper trading promotion readiness.")
    parser.add_argument(
        "--log",
        default=str(_default_log),
        help="Path to paper_trades.jsonl",
    )
    return parser.parse_args()


def _iter_events(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _compute_session_stats(events: List[Dict[str, Any]]) -> tuple[List[Dict], Dict]:
    """Compute per-session stats and lifetime totals."""
    sessions: List[Dict] = []
    cur_id: str | None = None
    cur_start: str | None = None
    cur_pnl = 0.0
    cur_settled = 0
    cur_wins = 0
    cur_losses = 0

    for e in events:
        t = e.get("type")
        if t == "session_start":
            if cur_id is not None:
                sessions.append({
                    "session_id": cur_id,
                    "started_at": cur_start,
                    "pnl": cur_pnl,
                    "settled": cur_settled,
                    "wins": cur_wins,
                    "losses": cur_losses,
                })
            cur_id = e.get("session_id")
            cur_start = e.get("timestamp")
            cur_pnl = 0.0
            cur_settled = 0
            cur_wins = 0
            cur_losses = 0
        elif t == "settlement" and cur_id and e.get("session_id") == cur_id:
            pnl = float(e.get("pnl") or 0)
            cur_pnl += pnl
            cur_settled += 1
            if pnl > 0:
                cur_wins += 1
            else:
                cur_losses += 1

    if cur_id is not None:
        sessions.append({
            "session_id": cur_id,
            "started_at": cur_start,
            "pnl": cur_pnl,
            "settled": cur_settled,
            "wins": cur_wins,
            "losses": cur_losses,
        })

    total_settled = sum(s["settled"] for s in sessions)
    total_wins = sum(s["wins"] for s in sessions)
    total_losses = sum(s["losses"] for s in sessions)
    total_pnl = sum(s["pnl"] for s in sessions)
    worst_pnl = min(s["pnl"] for s in sessions) if sessions else 0.0
    win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.0

    lifetime = {
        "sessions": len(sessions),
        "total_settled": total_settled,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "worst_session_pnl": worst_pnl,
    }
    return sessions, lifetime


def _days_span(events: List[Dict]) -> int:
    """Approximate span in days from first to last session_start."""
    from datetime import datetime

    starts = [e.get("timestamp") for e in events if e.get("type") == "session_start" and e.get("timestamp")]
    if len(starts) < 2:
        return 0
    try:
        first = datetime.fromisoformat(starts[0].replace("Z", "+00:00"))
        last = datetime.fromisoformat(starts[-1].replace("Z", "+00:00"))
        return max(0, (last - first).days)
    except (ValueError, TypeError):
        return 0


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        sys.exit(1)

    events = list(_iter_events(log_path))
    sessions, lifetime = _compute_session_stats(events)
    days_span = _days_span(events)

    # Evaluate gates
    g1_settled = lifetime["total_settled"] >= GATE_MIN_SETTLED
    g1_sessions = lifetime["sessions"] >= GATE_MIN_SESSIONS
    g1_days = days_span >= GATE_MIN_DAYS
    g2_wr = lifetime["win_rate"] >= GATE_MIN_WIN_RATE
    g2_pnl = lifetime["total_pnl"] > 0
    g2_worst = lifetime["worst_session_pnl"] > GATE_WORST_SESSION_PNL

    print("=" * 50)
    print("Paper Trading Promotion Gate Check")
    print("=" * 50)
    print()
    print("Lifetime stats:")
    print(f"  Sessions: {lifetime['sessions']}")
    print(f"  Settled trades: {lifetime['total_settled']} ({lifetime['total_wins']}W / {lifetime['total_losses']}L)")
    print(f"  Win rate: {lifetime['win_rate']:.1%}")
    print(f"  Lifetime PnL: ${lifetime['total_pnl']:+.2f}")
    print(f"  Worst session PnL: ${lifetime['worst_session_pnl']:.2f}")
    print(f"  Days observed: {days_span}")
    print()
    print("Per-session PnL (last 10):")
    for s in sessions[-10:]:
        sid = (s["session_id"] or "?")[:24]
        print(f"  {sid}... | settled {s['settled']} | PnL ${s['pnl']:+.2f}")
    print()
    print("Gate evaluation:")
    print(f"  [{'PASS' if g1_settled else 'FAIL'}] Settled >= {GATE_MIN_SETTLED}")
    print(f"  [{'PASS' if g1_sessions else 'FAIL'}] Sessions >= {GATE_MIN_SESSIONS}")
    print(f"  [{'PASS' if g1_days else 'FAIL'}] Days >= {GATE_MIN_DAYS}")
    print(f"  [{'PASS' if g2_wr else 'FAIL'}] Win rate >= {GATE_MIN_WIN_RATE:.0%}")
    print(f"  [{'PASS' if g2_pnl else 'FAIL'}] Lifetime PnL > 0")
    print(f"  [{'PASS' if g2_worst else 'FAIL'}] Worst session > ${GATE_WORST_SESSION_PNL}")
    print()

    all_pass = g1_settled and g1_sessions and g1_days and g2_wr and g2_pnl and g2_worst
    if all_pass:
        print("Result: READY for micro-live phase")
    else:
        print("Result: NOT READY — keep paper trading until all gates pass")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
