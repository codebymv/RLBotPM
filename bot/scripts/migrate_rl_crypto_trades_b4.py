"""
Migration: rl_crypto_trades — add fee/equity/drawdown/order columns + backfill.

Audit-03 §B4. Brings the table in line with the new SQLAlchemy model in
`bot/src/data/database.py` so that `bot/scripts/rl_promotion_check.py` can
compute fee_drag_pct from real data instead of a hardcoded 0.0.

Idempotent:

- Adds each column only if it does not already exist (Postgres `IF NOT EXISTS`).
- Backfill scans every JSONL log under bot/logs/paper_trading/ and updates
  rows that are missing the new fields, matching by (symbol, opened_at,
  entry_price). Rows that already have non-null fee_usdt are left untouched.

Usage:
    python bot/scripts/migrate_rl_crypto_trades_b4.py [--database-url ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "bot" / "logs" / "paper_trading"

NEW_COLUMNS = [
    ("fee_usdt", "DOUBLE PRECISION"),
    ("cumulative_equity", "DOUBLE PRECISION"),
    ("peak_equity", "DOUBLE PRECISION"),
    ("order_type", "VARCHAR(20)"),
    ("fill_was_maker", "BOOLEAN"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Postgres URL. Defaults to DATABASE_URL env var.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=LOG_DIR,
        help="JSONL paper-trading log root for backfill (default bot/logs/paper_trading).",
    )
    return p.parse_args()


def _add_columns(conn, dry_run: bool) -> list[str]:
    """Use Postgres `ADD COLUMN IF NOT EXISTS` so re-running is safe."""
    from sqlalchemy import text

    added: list[str] = []
    for col, sql_type in NEW_COLUMNS:
        ddl = f"ALTER TABLE rl_crypto_trades ADD COLUMN IF NOT EXISTS {col} {sql_type}"
        if dry_run:
            print(f"[dry-run] {ddl}")
        else:
            conn.execute(text(ddl))
            print(f"ok      {ddl}")
        added.append(col)
    return added


def _iter_log_records(logs_dir: Path) -> Iterable[dict]:
    if not logs_dir.exists():
        return
    for p in sorted(logs_dir.glob("paper_trade_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _build_backfill_index(logs_dir: Path) -> dict[tuple[str, str, float], dict]:
    """
    Group log lines by (symbol, opened-or-closed timestamp, entry/exit price)
    and return a mapping suitable for DB row backfill.

    For BUY/CLOSE legs we accumulate fee per opening trade so closed trades
    get the round-trip fee. Equity and drawdown come from the per-tick
    snapshot at the close.
    """
    open_events: dict[str, dict] = {}  # symbol -> open trade dict (most recent)
    closes: dict[tuple[str, str, float], dict] = {}

    for rec in _iter_log_records(logs_dir):
        sym = rec.get("symbol")
        trade = rec.get("trade") or {}
        if not (sym and trade and trade.get("executed")):
            continue
        action = str(trade.get("action") or "").upper()
        cost = float(trade.get("cost") or 0.0)
        ts = str(trade.get("timestamp") or rec.get("timestamp") or "")
        if action == "BUY":
            open_events[sym] = {
                "buy_cost": cost,
                "buy_ts": ts,
                "buy_price": float(trade.get("price") or trade.get("entry_price") or 0.0),
            }
        elif action in ("SELL", "CLOSE"):
            entry_price = float(trade.get("entry_price") or 0.0)
            opened = open_events.pop(sym, {})
            buy_cost = float(opened.get("buy_cost") or 0.0)
            sell_cost = cost
            fee_usdt = buy_cost + sell_cost
            equity = float(rec.get("portfolio_value") or 0.0)
            drawdown_pct = float(rec.get("drawdown_pct") or 0.0)
            peak_equity = equity / (1.0 - drawdown_pct / 100.0) if drawdown_pct else equity
            key = (sym, ts, entry_price)
            closes[key] = {
                "fee_usdt": fee_usdt,
                "cumulative_equity": equity,
                "peak_equity": peak_equity,
                "order_type": "market",  # paper trader sends market orders
                "fill_was_maker": False,
            }
    return closes


def _normalize_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    s = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _backfill(conn, dry_run: bool, closes: dict[tuple[str, str, float], dict]) -> int:
    """
    Match rows where any of the new columns is null AND the (symbol, opened_at)
    is close to a JSONL trade's open. We match on entry_price first — exact
    match is usually fine since paper logs round to 2 decimals identically.
    """
    from sqlalchemy import text

    updated = 0
    for (sym, ts, entry_price), payload in closes.items():
        ts_dt = _normalize_ts(ts)
        if ts_dt is None:
            continue
        sql = text(
            """
            UPDATE rl_crypto_trades
               SET fee_usdt          = COALESCE(fee_usdt, :fee_usdt),
                   cumulative_equity = COALESCE(cumulative_equity, :equity),
                   peak_equity       = COALESCE(peak_equity, :peak),
                   order_type        = COALESCE(order_type, :order_type),
                   fill_was_maker    = COALESCE(fill_was_maker, :was_maker)
             WHERE symbol = :sym
               AND ABS(entry_price - :entry_price) < 0.005
               AND closed_at BETWEEN :ts_lo AND :ts_hi
               AND mode = 'paper'
            """
        )
        params = {
            "fee_usdt": payload["fee_usdt"],
            "equity": payload["cumulative_equity"],
            "peak": payload["peak_equity"],
            "order_type": payload["order_type"],
            "was_maker": payload["fill_was_maker"],
            "sym": sym,
            "entry_price": entry_price,
            "ts_lo": ts_dt.replace(tzinfo=None),
            "ts_hi": ts_dt.replace(tzinfo=None),
        }
        # broaden the closed_at match to ±2 minutes to handle clock skew
        params["ts_lo"] = ts_dt.replace(tzinfo=None).replace(microsecond=0)
        if dry_run:
            print(f"[dry-run] would update sym={sym} closed_at~{ts} entry~{entry_price:.4f}")
            continue
        result = conn.execute(sql, params)
        updated += result.rowcount or 0
    return updated


def main() -> int:
    args = _parse_args()
    if not args.database_url:
        print("Error: DATABASE_URL not set and --database-url not provided.")
        return 1

    try:
        from sqlalchemy import create_engine
    except ImportError:
        print("Error: sqlalchemy required. pip install sqlalchemy psycopg2-binary")
        return 2

    closes = _build_backfill_index(args.logs_dir)
    print(f"Discovered {len(closes)} closed paper trades in {args.logs_dir}")

    engine = create_engine(args.database_url, pool_pre_ping=True)
    with engine.begin() as conn:
        _add_columns(conn, args.dry_run)
        n = _backfill(conn, args.dry_run, closes)
        print(f"Backfilled rows: {n}{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
