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

# Promotion gate targets (aligned with RL_PROFITABILITY_AUDIT.md Stage 2)
GATE_MIN_CLOSED_TRADES = 50
GATE_MIN_SESSIONS = 5
GATE_MIN_DAYS = 14
GATE_MIN_WIN_RATE = 0.45
GATE_MIN_PROFIT_FACTOR = 1.2
GATE_MIN_TOTAL_RETURN_PCT = 0.0   # Must be net profitable (> 0%)
GATE_WORST_SESSION_PNL = -30.0    # Worst session must be > -$30 on $1000 capital
GATE_MAX_FEE_DRAG_PCT = 25.0      # Fees must be < 25% of gross profit
GATE_MAX_SINGLE_TRADE_LOSS_PCT = 3.0  # No single trade > -3% of capital

# Architecture-audit-03 §B3: a walk-forward result file is required before
# promotion, even if every paper-trading gate passes. The model is expected to
# have positive mean Sharpe across the OOS folds — anything below this is
# evidence of a one-fold fluke or training-window overfit. The runner reads
# `bot/models/walk_forward_run_<id>.json` (one per training run) produced by
# `bot/scripts/run_walk_forward.py` (or the equivalent invocation that
# pickles `WalkForwardResult` objects).
GATE_MIN_WALK_FORWARD_FOLDS = 3
GATE_MIN_WALK_FORWARD_MEAN_SHARPE = 1.0
GATE_MIN_WALK_FORWARD_FOLDS_POSITIVE = 2  # at least N of K folds must be > 0 return


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
    parser.add_argument(
        "--walk-forward",
        type=Path,
        default=_repo_root / "bot" / "models",
        help=(
            "Directory or JSON file containing walk-forward results. Promotion "
            "requires a `walk_forward_run_<id>.json` file with mean Sharpe "
            ">= GATE_MIN_WALK_FORWARD_MEAN_SHARPE across at least "
            "GATE_MIN_WALK_FORWARD_FOLDS folds. Default: bot/models/."
        ),
    )
    return parser.parse_args()


def _load_walk_forward(path: Path) -> dict:
    """
    Returns {'present': bool, 'folds': int, 'mean_sharpe': float, 'positive_folds': int, 'source': str}
    """
    import json
    import glob

    candidates: list[Path] = []
    if path.is_dir():
        candidates = sorted(path.glob("walk_forward_run_*.json"))
    elif path.is_file():
        candidates = [path]

    if not candidates:
        return {
            "present": False,
            "folds": 0,
            "mean_sharpe": 0.0,
            "positive_folds": 0,
            "source": str(path),
        }

    # Use the most recently modified file
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "present": False,
            "folds": 0,
            "mean_sharpe": 0.0,
            "positive_folds": 0,
            "source": f"{latest} (parse error: {exc})",
        }

    folds = data.get("folds") or data.get("results") or []
    if not isinstance(folds, list):
        folds = []
    sharpes = [float(f.get("sharpe_ratio", 0.0)) for f in folds]
    returns = [float(f.get("total_return", 0.0)) for f in folds]
    mean_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
    positive = sum(1 for r in returns if r > 0)
    return {
        "present": True,
        "folds": len(folds),
        "mean_sharpe": mean_sharpe,
        "positive_folds": positive,
        "source": str(latest),
    }


def _get_session_pnls(db_url: str) -> tuple[list[dict], dict]:
    """Query RLCryptoTrade for closed paper trades, grouped by session."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("Error: sqlalchemy required. pip install sqlalchemy psycopg2-binary")
        sys.exit(2)

    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        # `fee_usdt`, `cumulative_equity`, `peak_equity` were added in
        # bot/scripts/migrate_rl_crypto_trades_b4.py. They may be null on
        # rows from before the migration — fee_drag_pct below treats null as
        # "unknown" and falls back to 0 for that row only.
        rows = conn.execute(
            text("""
                SELECT session_id, pnl, pnl_pct, closed_at, symbol,
                       fee_usdt, cumulative_equity, peak_equity
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
            "fee_drag_pct": 0.0,
            "max_single_trade_loss_pct": 0.0,
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

    # Fee drag (audit-03 §B4): fees as % of gross profit, computed from the
    # `fee_usdt` column added in migrate_rl_crypto_trades_b4.py. Rows with
    # null `fee_usdt` (pre-migration) contribute 0 to the numerator only and
    # do not bias fee_drag downward — they are excluded from the denominator
    # too via `gross_profit_with_fee_data`.
    fees_known = [
        (float(r.fee_usdt), float(r.pnl))
        for r in rows
        if getattr(r, "fee_usdt", None) is not None and r.pnl is not None
    ]
    total_fees_known = sum(fee for fee, _ in fees_known)
    gross_profit_with_fee_data = sum(p for _, p in fees_known if p > 0)
    if gross_profit_with_fee_data > 0:
        fee_drag_pct = 100.0 * total_fees_known / gross_profit_with_fee_data
    else:
        fee_drag_pct = 0.0  # no fee data yet — gate evaluates as PASS-by-vacuity

    # Max single-trade loss as % of capital ($1000 assumed)
    all_pnl_pcts = [float(r.pnl_pct) for r in rows if r.pnl_pct is not None]
    max_single_trade_loss_pct = abs(min(all_pnl_pcts)) if all_pnl_pcts else 0.0

    # Approximate total return (assume $1000 initial)
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
        "fee_drag_pct": fee_drag_pct,
        "max_single_trade_loss_pct": max_single_trade_loss_pct,
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
    wf = _load_walk_forward(args.walk_forward)

    # Evaluate gates
    g_trades = lifetime["total_closed"] >= GATE_MIN_CLOSED_TRADES
    g_sessions = lifetime["sessions"] >= GATE_MIN_SESSIONS
    g_days = days_span >= GATE_MIN_DAYS
    g_wr = lifetime["win_rate"] >= GATE_MIN_WIN_RATE
    g_pnl = lifetime["total_pnl"] > 0
    g_return = lifetime["total_return_pct"] >= GATE_MIN_TOTAL_RETURN_PCT
    g_pf = lifetime["profit_factor"] >= GATE_MIN_PROFIT_FACTOR
    g_worst = lifetime["worst_session_pnl"] > GATE_WORST_SESSION_PNL
    g_fee = lifetime["fee_drag_pct"] < GATE_MAX_FEE_DRAG_PCT
    g_single = lifetime["max_single_trade_loss_pct"] < GATE_MAX_SINGLE_TRADE_LOSS_PCT
    # Walk-forward gate (audit-03 §B3): must exist + cross fold + Sharpe thresholds.
    g_wf = (
        wf["present"]
        and wf["folds"] >= GATE_MIN_WALK_FORWARD_FOLDS
        and wf["mean_sharpe"] >= GATE_MIN_WALK_FORWARD_MEAN_SHARPE
        and wf["positive_folds"] >= GATE_MIN_WALK_FORWARD_FOLDS_POSITIVE
    )

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
    print(f"  Max single-trade loss: {lifetime['max_single_trade_loss_pct']:.2f}%")
    print(f"  Days observed: {days_span}")
    print()

    if args.verbose and sessions:
        print("Per-session PnL (last 10):")
        for s in sessions[-10:]:
            sid = (s.get("session_id") or "?")[:24]
            print(f"  {sid}... | closed {s['settled']} | PnL ${s['pnl']:+.2f}")
        print()

    print("Gate evaluation:")
    print(f"  [{'PASS' if g_trades else 'FAIL'}] Closed trades >= {GATE_MIN_CLOSED_TRADES}")
    print(f"  [{'PASS' if g_sessions else 'FAIL'}] Sessions >= {GATE_MIN_SESSIONS}")
    print(f"  [{'PASS' if g_days else 'FAIL'}] Days >= {GATE_MIN_DAYS}")
    print(f"  [{'PASS' if g_wr else 'FAIL'}] Win rate >= {GATE_MIN_WIN_RATE:.0%}")
    print(f"  [{'PASS' if g_pnl else 'FAIL'}] Lifetime PnL > 0")
    print(f"  [{'PASS' if g_return else 'FAIL'}] Return >= {GATE_MIN_TOTAL_RETURN_PCT}%")
    print(f"  [{'PASS' if g_pf else 'FAIL'}] Profit factor >= {GATE_MIN_PROFIT_FACTOR}")
    print(f"  [{'PASS' if g_worst else 'FAIL'}] Worst session > ${GATE_WORST_SESSION_PNL}")
    print(f"  [{'PASS' if g_fee else 'FAIL'}] Fee drag < {GATE_MAX_FEE_DRAG_PCT}% of gross profit")
    print(f"  [{'PASS' if g_single else 'FAIL'}] No single trade loss > {GATE_MAX_SINGLE_TRADE_LOSS_PCT}%")
    if not wf["present"]:
        wf_label = "missing"
    else:
        wf_label = (
            f"folds={wf['folds']} mean_sharpe={wf['mean_sharpe']:.2f} "
            f"positive_folds={wf['positive_folds']} (source: {wf['source']})"
        )
    print(
        f"  [{'PASS' if g_wf else 'FAIL'}] Walk-forward (audit-03 §B3): {wf_label}"
    )
    print()

    all_pass = all([g_trades, g_sessions, g_days, g_wr, g_pnl, g_return, g_pf, g_worst, g_fee, g_single, g_wf])
    if all_pass:
        print("Result: READY for live trading")
        return 0
    else:
        print("Result: NOT READY — keep paper trading until all gates pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
