#!/usr/bin/env python3
"""
Kalshi Bot Health Check & Daily Report

Run this before starting the bot, or anytime to audit current state.
Covers: balance, positions, settlements, concentration, and readiness.

Usage:
    python health_check.py
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

from src.execution.kalshi_client import KalshiExecutionClient
from src.strategies.paper_trader import _extract_asset, classify_sleeve, hours_to_close


def run_health_check():
    print("=" * 60)
    print("  KALSHI BOT HEALTH CHECK")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    client = KalshiExecutionClient(demo=False)

    # 1. Balance + Positions (fetch together so balance section can reference positions)
    avail, total = client.get_balance()
    positions = client.get_positions(strict=True)
    active = [p for p in positions if p.position != 0]
    settled_with_pnl = [p for p in positions if p.position == 0 and abs(p.realized_pnl or 0) > 1e-9]
    total_cost = sum(p.total_cost for p in active)
    total_realized = sum(p.realized_pnl for p in positions)

    deployed_cost = total_cost
    implied_free = total - deployed_cost if total > deployed_cost else 0

    print(f"\n1. BALANCE")
    print(f"   Exchange cash:   ${avail:.2f}")
    print(f"   Portfolio total: ${total:.2f}")
    print(f"   Deployed cost:   ${deployed_cost:.2f}")
    print(f"   Implied free:    ${implied_free:.2f}")
    if abs(avail - implied_free) > 1.0:
        print(f"   !! MISMATCH: exchange cash vs implied free differs by ${abs(avail - implied_free):.2f}")
    if avail < 1.0:
        print("   !! WARNING: Cash below $1.00 minimum trade cost")

    print(f"\n2. POSITIONS")
    print(f"   Active:     {len(active)} (${total_cost:.2f} cost basis)")
    print(f"   Settled:    {len(settled_with_pnl)} with non-zero P&L")
    print(f"   Realized:   ${total_realized:+.2f}")

    # 3. Sleeve breakdown
    asset_map = defaultdict(lambda: {"count": 0, "cost": 0.0, "tickers": []})
    fast_count = macro_count = other_count = 0
    fast_cost = macro_cost = other_cost = 0.0
    settling_24h = settling_72h = 0

    for p in active:
        asset = _extract_asset(p.ticker)
        asset_map[asset]["count"] += abs(p.position)
        asset_map[asset]["cost"] += p.total_cost
        asset_map[asset]["tickers"].append(p.ticker)

        mkt = {"series_ticker": p.ticker.split("-")[0], "close_time": None}
        sleeve = classify_sleeve(mkt)
        if sleeve == "fast":
            fast_count += 1
            fast_cost += p.total_cost
        elif sleeve == "macro":
            macro_count += 1
            macro_cost += p.total_cost
        else:
            other_count += 1
            other_cost += p.total_cost

    print(f"\n3. SLEEVE BREAKDOWN")
    print(f"   Fast:   {fast_count} positions (${fast_cost:.2f})")
    print(f"   Macro:  {macro_count} positions (${macro_cost:.2f})")
    print(f"   Other:  {other_count} positions (${other_cost:.2f})")

    # 4. Concentration
    print(f"\n4. CONCENTRATION")
    for asset, info in sorted(asset_map.items(), key=lambda x: -x[1]["cost"]):
        pct = (info["cost"] / total_cost * 100) if total_cost > 0 else 0
        flag = " !! OVER 30%" if pct > 30 else ""
        print(f"   {asset:8s}: {info['count']:3d} contracts  ${info['cost']:6.2f} ({pct:4.1f}%){flag}")

    macro_cluster = {"CPI", "NFP", "FED"}
    cluster_cost = sum(v["cost"] for k, v in asset_map.items() if k in macro_cluster)
    if total_cost > 0:
        cluster_pct = cluster_cost / total_cost * 100
        flag = " !! OVER 50%" if cluster_pct > 50 else ""
        print(f"   Macro cluster (CPI+NFP+FED): ${cluster_cost:.2f} ({cluster_pct:.1f}%){flag}")

    # 5. Active positions detail
    print(f"\n5. POSITIONS DETAIL")
    for p in sorted(active, key=lambda x: x.ticker):
        side = "NO" if p.position < 0 else "YES"
        print(f"   {p.ticker}: {abs(p.position)} {side} | cost=${p.total_cost:.2f} | rpnl=${p.realized_pnl:.2f}")

    # 6. Recent settlements
    if settled_with_pnl:
        print(f"\n6. SETTLED POSITIONS (non-zero P&L)")
        wins = losses = 0
        for p in settled_with_pnl:
            tag = "WIN" if p.realized_pnl > 0 else "LOSS"
            if p.realized_pnl > 0:
                wins += 1
            else:
                losses += 1
            print(f"   {p.ticker}: ${p.realized_pnl:+.2f} ({tag})")
        print(f"   Record: {wins}W-{losses}L | Total: ${sum(p.realized_pnl for p in settled_with_pnl):+.2f}")
    else:
        print(f"\n6. No settled positions with non-zero P&L found")

    # 7. Readiness check
    print(f"\n7. READINESS CHECK")
    issues = []
    if avail < 1.0:
        issues.append("Cash below $1.00 -- bot cannot place trades")
    if total_cost > 0:
        for asset, info in asset_map.items():
            pct = info["cost"] / total_cost * 100
            if pct > 30:
                issues.append(f"{asset} concentration at {pct:.1f}% (limit 30%)")
        if cluster_cost / total_cost * 100 > 50:
            issues.append(f"Macro cluster at {cluster_cost/total_cost*100:.1f}% (limit 50%)")
    if len(active) >= 10:
        issues.append(f"Position count at {len(active)}/10 -- no room for new trades")

    if issues:
        for issue in issues:
            print(f"   !! {issue}")
    else:
        print("   All checks passed -- bot is ready to trade")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    run_health_check()
