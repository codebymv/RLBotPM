"""
RL Crypto Bot - Paper trading promotion gate check.

Evaluates readiness for real-money trading based on RLCryptoTrade DB records.
Mirrors Kalshi's paper_promotion_check pattern for consistency.

Run from repo root: python bot/scripts/rl_promotion_check.py

Usage:
    python bot/scripts/rl_promotion_check.py
    python bot/scripts/rl_promotion_check.py --database-url postgres://...
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Promotion gate targets (aligned with LIVE_TRADING_READINESS.md)
GATE_MIN_CLOSED_TRADES = 100
GATE_MIN_SESSIONS = 5
GATE_MIN_DAYS = 14
GATE_MIN_WIN_RATE = 0.45       # Realistic for RL (vs Kalshi 0.85)
GATE_MIN_PROFIT_FACTOR = 1.3
GATE_MIN_TOTAL_RETURN_PCT = 0.5   # +0.5% minimum
GATE_WORST_SESSION_PNL = -50.0    # Worst session must be > -$50 (vs Kalshi -$5)


def _parse_args() -> argparse.Namespace:
    _repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        description="Check RL Crypto Bot paper trading promotion readiness."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection string (default: DATABASE_URL env)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-session breakdown",
    )
    return parser.parse_args()


def _get_session_pnls(db_url: str) -> tuple[list[dict], dict]:
    """Query RLCryptoTrade for closed paper trades, grouped by session."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("Error: sqlalchemy required. pip install sqlalchemy psycopg2-binary")
        sys.exit(2)

    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT session_id, pnl, pnl_pct, closed_at, symbol
                FROM rl_crypto_trades
                WHERE mode = 'paper' AND pnl IS NOT NULL
                ORDER BY closed_at ASC
            """)
        ).fetchall()

    if not rows:
        return [], {
            "sessions": 0,
            "total_closed": 0,
            "total_wins": 0,
            "total_losses": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "worst_session_pnl": 0.0,
            "profit_factor": 0.0,
            "total_return_pct": 0.0,
        }

    # Group by session_id (fallback to date if missing)
    by_session: dict[str, list[tuple]] = {}
    for r in rows:
        sid = r.session_id or (r.closed_at.strftime("%Y-%m-%d") if r.closed_at else "unknown")
        by_session.setdefault(sid, []).append(r)

    sessions = []
    for sid, trades in by_session.items():
        pnls = [float(t.pnl) for t in trades if t.pnl is not None]
        pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        closed_at = max((t.closed_at for t in trades if t.closed_at), default=None)
        sessions.append({
            "session_id": sid,
            "closed_at": closed_at,
            "settled": len(pnls),
            "pnl": pnl,
            "wins": wins,
            "losses": losses,
        })

    sessions.sort(key=lambda s: s["closed_at"] or datetime.min)

    total_closed = sum(s["settled"] for s in sessions)
    total_wins = sum(s["wins"] for s in sessions)
    total_losses = sum(s["losses"] for s in sessions)
    total_pnl = sum(s["pnl"] for s in sessions)
    worst_pnl = min(s["pnl"] for s in sessions) if sessions else 0.0
    win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.0

    all_pnls = [float(r.pnl) for r in rows if r.pnl is not None]
    gross_profit = sum(p for p in all_pnls if p > 0)
    gross_loss = abs(sum(p for p in all_pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    # Approximate total return (assume $1000 initial; rough)
    total_return_pct = (total_pnl / 1000.0) * 100 if total_closed > 0 else 0.0

    lifetime = {
        "sessions": len(sessions),
        "total_closed": total_closed,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "worst_session_pnl": worst_pnl,
        "profit_factor": profit_factor,
        "total_return_pct": total_return_pct,
    }
    return sessions, lifetime


def _days_span(sessions: list[dict]) -> int:
    dates = [s["closed_at"] for s in sessions if s.get("closed_at")]
    if len(dates) < 2:
        return 0
    first = min(dates)
    last = max(dates)
    return max(0, (last - first).days)


def main() -> int:
    args = _parse_args()
    db_url = args.database_url
    if not db_url:
        print("Error: DATABASE_URL not set and --database-url not provided")
        return 1

    try:
        sessions, lifetime = _get_session_pnls(db_url)
    except Exception as e:
        print(f"Error querying database: {e}")
        print("  (Ensure DATABASE_URL is set and rl_crypto_trades table exists)")
        return 1

    days_span = _days_span(sessions)

    # Evaluate gates
    g1_trades = lifetime["total_closed"] >= GATE_MIN_CLOSED_TRADES
    g1_sessions = lifetime["sessions"] >= GATE_MIN_SESSIONS
    g1_days = days_span >= GATE_MIN_DAYS
    g2_wr = lifetime["win_rate"] >= GATE_MIN_WIN_RATE
    g2_pnl = lifetime["total_pnl"] > 0
    g2_return = lifetime["total_return_pct"] >= GATE_MIN_TOTAL_RETURN_PCT
    g2_pf = lifetime["profit_factor"] >= GATE_MIN_PROFIT_FACTOR
    g2_worst = lifetime["worst_session_pnl"] > GATE_WORST_SESSION_PNL

    print("=" * 50)
    print("RL Crypto Bot - Paper Trading Promotion Gate Check")
    print("=" * 50)
    print()
    print("Lifetime stats (from rl_crypto_trades, mode=paper):")
    print(f"  Sessions: {lifetime['sessions']}")
    print(f"  Closed trades: {lifetime['total_closed']} ({lifetime['total_wins']}W / {lifetime['total_losses']}L)")
    print(f"  Win rate: {lifetime['win_rate']:.1%}")
    print(f"  Profit factor: {lifetime['profit_factor']:.2f}")
    print(f"  Lifetime PnL: ${lifetime['total_pnl']:+.2f}")
    print(f"  Approx return: {lifetime['total_return_pct']:+.2f}%")
    print(f"  Worst session PnL: ${lifetime['worst_session_pnl']:.2f}")
    print(f"  Days observed: {days_span}")
    print()

    if args.verbose and sessions:
        print("Per-session PnL (last 10):")
        for s in sessions[-10:]:
            sid = (s.get("session_id") or "?")[:24]
            print(f"  {sid}... | closed {s['settled']} | PnL ${s['pnl']:+.2f}")
        print()

    print("Gate evaluation:")
    print(f"  [{'PASS' if g1_trades else 'FAIL'}] Closed trades >= {GATE_MIN_CLOSED_TRADES}")
    print(f"  [{'PASS' if g1_sessions else 'FAIL'}] Sessions >= {GATE_MIN_SESSIONS}")
    print(f"  [{'PASS' if g1_days else 'FAIL'}] Days >= {GATE_MIN_DAYS}")
    print(f"  [{'PASS' if g2_wr else 'FAIL'}] Win rate >= {GATE_MIN_WIN_RATE:.0%}")
    print(f"  [{'PASS' if g2_pnl else 'FAIL'}] Lifetime PnL > 0")
    print(f"  [{'PASS' if g2_return else 'FAIL'}] Return >= {GATE_MIN_TOTAL_RETURN_PCT}%")
    print(f"  [{'PASS' if g2_pf else 'FAIL'}] Profit factor >= {GATE_MIN_PROFIT_FACTOR}")
    print(f"  [{'PASS' if g2_worst else 'FAIL'}] Worst session > ${GATE_WORST_SESSION_PNL}")
    print()

    all_pass = g1_trades and g1_sessions and g1_days and g2_wr and g2_pnl and g2_return and g2_pf and g2_worst
    if all_pass:
        print("Result: READY for live trading")
        return 0
    else:
        print("Result: NOT READY — keep paper trading until all gates pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
