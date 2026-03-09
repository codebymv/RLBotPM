"""
Analyze paper-trading loss clusters from paper_trades.jsonl.

Usage:
    python bot/scripts/analyze_paper_losses.py
    python bot/scripts/analyze_paper_losses.py --log bot/logs/paper_trades.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class LossRecord:
    timestamp: str
    ticker: str
    side: str
    pnl: float
    edge: float | None
    asset: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze loss clustering in paper trading logs.")
    parser.add_argument(
        "--log",
        default="bot/logs/paper_trades.jsonl",
        help="Path to paper_trades.jsonl",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top buckets to print per view",
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


def _asset_from_ticker(ticker: str) -> str:
    if ticker.startswith(("KXBTC", "KXBTCD")):
        return "BTC"
    if ticker.startswith(("KXETH", "KXETHD")):
        return "ETH"
    if ticker.startswith("KXSOLD"):
        return "SOL"
    if ticker.startswith("KXDOGE"):
        return "DOGE"
    if ticker.startswith("KXXRP"):
        return "XRP"
    return ticker.split("-", 1)[0]


def _edge_bucket(edge: float | None) -> str:
    if edge is None:
        return "unknown"
    pct = edge * 100
    if pct < 2:
        return "<2%"
    if pct < 3:
        return "2-3%"
    if pct < 4:
        return "3-4%"
    if pct < 5:
        return "4-5%"
    return ">=5%"


def _hour_bucket(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    return f"{dt.hour:02d}:00-{dt.hour:02d}:59 UTC"


def _print_top(title: str, counter: Counter, top: int) -> None:
    print(f"\n{title}")
    if not counter:
        print("  none")
        return
    for key, count in counter.most_common(top):
        print(f"  {key}: {count}")


def main() -> int:
    args = _parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return 1

    open_by_ticker: Dict[str, Dict[str, Any]] = {}
    losses: List[LossRecord] = []
    settled = 0
    wins = 0

    for event in _iter_events(log_path):
        etype = event.get("type")
        if etype == "open_position":
            open_by_ticker[event.get("ticker", "")] = event
            continue
        if etype != "settlement":
            continue

        settled += 1
        pnl = float(event.get("pnl", 0.0))
        if pnl > 0:
            wins += 1
            continue

        ticker = str(event.get("ticker", ""))
        open_event = open_by_ticker.get(ticker, {})
        edge_val = open_event.get("edge")
        edge = float(edge_val) if isinstance(edge_val, (float, int)) else None
        losses.append(
            LossRecord(
                timestamp=str(event.get("timestamp", "")),
                ticker=ticker,
                side=str(event.get("side", "unknown")),
                pnl=pnl,
                edge=edge,
                asset=_asset_from_ticker(ticker),
            )
        )

    loss_count = len(losses)
    print("Paper Trading Loss Cluster Analysis")
    print("=" * 38)
    print(f"Log file: {log_path}")
    print(f"Settled trades: {settled}")
    print(f"Wins: {wins}")
    print(f"Losses: {loss_count}")
    print(f"Settled win rate: {(wins / settled * 100) if settled else 0:.1f}%")

    by_asset = Counter(r.asset for r in losses)
    by_hour = Counter(_hour_bucket(r.timestamp) for r in losses)
    by_edge = Counter(_edge_bucket(r.edge) for r in losses)
    by_side = Counter(r.side for r in losses)

    _print_top("Losses by asset", by_asset, args.top)
    _print_top("Losses by hour bucket", by_hour, args.top)
    _print_top("Losses by edge bucket", by_edge, args.top)
    _print_top("Losses by side", by_side, args.top)

    by_day: Dict[str, List[LossRecord]] = defaultdict(list)
    for row in losses:
        day = row.timestamp[:10] if row.timestamp else "unknown"
        by_day[day].append(row)
    worst_day = max(by_day.items(), key=lambda kv: len(kv[1]), default=(None, []))
    if worst_day[0]:
        worst_day_pnl = sum(r.pnl for r in worst_day[1])
        print(f"\nWorst loss day: {worst_day[0]} ({len(worst_day[1])} losses, ${worst_day_pnl:+.2f})")

    print("\nSuggested next-run cap tuning:")
    if by_asset:
        heaviest_asset, heaviest_count = by_asset.most_common(1)[0]
        print(f"- Reduce per-asset cap for {heaviest_asset} first (highest loss count: {heaviest_count}).")
    print("- Keep BUY_YES disabled unless running isolated yes-only experiments.")
    print("- Set a session loss stop near one worst-day drawdown.")
    print("- Re-check clusters after each 24h checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

