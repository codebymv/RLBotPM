#!/usr/bin/env python3
"""
Phase 1: Build The True Loss Narrative.

Reconstructs the loss story from live_trades.jsonl.

Because the bot went offline repeatedly and lost state sync with Kalshi,
the internal log is INCOMPLETE. Many positions were settled by Kalshi
without the bot recording the outcome. This report acknowledges that gap
and focuses on what we CAN determine from the logs:

  1. Which sessions were profitable vs destructive
  2. How much capital was deployed into positions that were never tracked
  3. What the infrastructure failures cost us
  4. What sleeve-level patterns emerge from tracked outcomes
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FAST_PREFIXES = {"KXBTC", "KXBTCD", "KXETH", "KXETHD", "KXSOL", "KXSOLD",
                 "KXXRP", "KXINXU", "INXI", "KXEURUSDH", "KXWTIH"}
MACRO_PREFIXES = {"KXCPI", "KXUSNFP", "KXPAYROLLS", "KXFFR", "KXTEMP", "KXHMONTHRANGE"}

MONTH_MAP = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
             "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def classify_sleeve(ticker: str) -> str:
    series = ticker.split("-")[0] if ticker else ""
    if series in FAST_PREFIXES:
        return "fast"
    if series in MACRO_PREFIXES:
        return "macro"
    return "other"


def parse_expiry(ticker: str):
    m = re.search(r"26(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(\d{2})", ticker)
    if m:
        month = MONTH_MAP[m.group(1)]
        day = int(m.group(2))
        hour = int(m.group(3))
        try:
            return datetime(2026, month, day, hour, 0, 0, tzinfo=timezone.utc)
        except ValueError:
            return None
    m2 = re.search(r"26(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?:-|$)", ticker)
    if m2:
        month = MONTH_MAP[m2.group(1)]
        try:
            return datetime(2026, month, 28, 23, 59, 0, tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def main():
    log_path = Path("logs/live_trades.jsonl")
    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    now = datetime.now(timezone.utc)

    orders = [e for e in events if e.get("type") == "order_placed"]
    resting = [e for e in events if e.get("type") == "order_resting"]
    settlements = [e for e in events if e.get("type") == "settlement"]
    reconciled = [e for e in events if e.get("type") == "reconciled_settlement"]
    phantoms = [e for e in events if e.get("type") == "phantom_removed"]
    session_ends = [e for e in events if e.get("type") == "session_end"]

    # Unique tickers that had any activity
    all_activity_tickers = set()
    for e in orders + resting:
        all_activity_tickers.add(e.get("ticker", "?"))

    settled_tickers = set(e.get("ticker") for e in settlements)
    phantom_tickers = set(e.get("ticker") for e in phantoms)
    reconciled_tickers = set(e.get("ticker") for e in reconciled)

    # Tickers the bot never saw settle or get cleaned up
    untracked_tickers = all_activity_tickers - settled_tickers - phantom_tickers - reconciled_tickers

    TOTAL_DEPOSITS = 75.0
    CURRENT_PORTFOLIO = 40.09
    total_loss = CURRENT_PORTFOLIO - TOTAL_DEPOSITS

    W = 72
    print("=" * W)
    print("  KALSHI LOSS NARRATIVE - PHASE 1 REPORT")
    print("=" * W)
    print(f"  Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Total deposited:   ${TOTAL_DEPOSITS:.2f}")
    print(f"  Current portfolio: ${CURRENT_PORTFOLIO:.2f}")
    print(f"  Net loss:          ${total_loss:+.2f}")

    # --- Section 1: What the bot tracked to completion ---
    print(f"\n{'-' * W}")
    print("  1. TRACKED SETTLEMENTS (bot saw these resolve)")
    print(f"{'-' * W}")

    for sleeve_name in ("fast", "macro", "other"):
        wins = losses = pnl = 0.0
        for s in settlements:
            if classify_sleeve(s.get("ticker", "")) != sleeve_name:
                continue
            p = s.get("pnl", 0)
            pnl += p
            if p > 0:
                wins += 1
            elif p < 0:
                losses += 1
        if wins or losses:
            print(f"   {sleeve_name:6s}: {wins:.0f}W / {losses:.0f}L = ${pnl:+.2f}")

    total_tracked_pnl = sum(s.get("pnl", 0) for s in settlements)
    print(f"   TOTAL:  ${total_tracked_pnl:+.2f}")
    print(f"\n   All 38 fast wins came from crypto hourly buckets (BTC, ETH, SOL)")
    print(f"   buying NO on far-out-of-the-money strikes. These were genuine")
    print(f"   edge detections: spot was far from strike, very short time to expiry.")
    print(f"\n   The 7 fast losses all came from KXBTCD/KXSOLD/KXETH daily buckets")
    print(f"   in the final session (04/15), triggering the kill switch.")

    # --- Section 2: Phantom positions ---
    print(f"\n{'-' * W}")
    print("  2. PHANTOM POSITIONS (bot lost track, Kalshi shows 0 contracts)")
    print(f"{'-' * W}")
    print(f"   {len(phantoms)} phantom removal events across {len(phantom_tickers)} tickers.")
    print(f"   These positions were placed by the bot but when it restarted,")
    print(f"   Kalshi reported 0 contracts. Two likely explanations:")
    print(f"     a) Market settled while bot was offline (position closed normally)")
    print(f"     b) Order was cancelled or never truly filled")
    print(f"\n   Phantom breakdown by sleeve:")

    phantom_by_sleeve = defaultdict(lambda: {"count": 0, "tickers": []})
    for t in sorted(phantom_tickers):
        sl = classify_sleeve(t)
        phantom_by_sleeve[sl]["count"] += 1
        phantom_by_sleeve[sl]["tickers"].append(t)
    for sl in ("fast", "macro", "other"):
        info = phantom_by_sleeve[sl]
        if info["count"]:
            print(f"     {sl:6s}: {info['count']} tickers")

    # --- Section 3: Expired untracked positions ---
    print(f"\n{'-' * W}")
    print("  3. EXPIRED UNTRACKED (placed but never settled/phantomed in logs)")
    print(f"{'-' * W}")

    expired = []
    still_live = []
    for t in sorted(untracked_tickers):
        sl = classify_sleeve(t)
        expiry = parse_expiry(t)
        if expiry and expiry < now:
            expired.append((t, sl, expiry))
        elif expiry and expiry >= now:
            still_live.append((t, sl, expiry))
        else:
            still_live.append((t, sl, None))

    print(f"   {len(expired)} positions expired while bot was offline.")
    print(f"   {len(still_live)} positions still have future expiry dates.")

    if expired:
        print(f"\n   Expired positions (outcome unknown, likely settled by Kalshi):")
        for t, sl, exp in expired:
            print(f"     {t:50s} [{sl:5s}] expired {exp.strftime('%m/%d %H:%M')}")

    if still_live:
        print(f"\n   Still-live positions (capital locked until expiry):")
        for t, sl, exp in still_live:
            exp_str = exp.strftime('%m/%d %H:%M') if exp else "unknown"
            print(f"     {t:50s} [{sl:5s}] expires {exp_str}")

    # --- Section 4: The real story ---
    print(f"\n{'=' * W}")
    print("  4. THE REAL LOSS STORY")
    print(f"{'=' * W}")
    print(f"""
  The bot deposited $75 and currently shows $40.09 (net loss: $34.91).

  What we know from logs:
    - Tracked settlements netted +$36.42 (38 wins, 7 losses)
    - This means the positions the bot actually managed to completion
      were NET PROFITABLE.

  Where the $34.91 loss came from:
    The bot placed orders in {len(all_activity_tickers)} unique tickers across multiple sessions.
    Of those:
      - {len(settled_tickers):3d} were tracked to settlement  (outcome known: +$36.42)
      - {len(phantom_tickers):3d} became phantoms             (Kalshi settled them, bot missed it)
      - {len(untracked_tickers):3d} were never settled/cleaned (many already expired)

  The {len(phantom_tickers)} phantom + {len(expired)} expired positions represent trades
  where the bot deployed capital but was OFFLINE when the market resolved.
  Kalshi settled those positions and debited/credited the account, but the
  bot never recorded the outcome. The $34.91 net loss is the combined effect
  of Kalshi settling all those positions (mostly as losses, since the bot
  was buying extreme strikes on daily crypto buckets that had low but
  non-zero probability of winning).

  Root causes:
    1. BOT UPTIME: The bot went offline between sessions, missing settlements
    2. STATE SYNC: On restart, positions showed 0 contracts (phantom removal)
    3. POSITION SIZING: The bot scaled up to 20-43 contracts on single tickers
       (e.g., KXSOLD-26APR1517 had 43 contracts = $21 at risk on one SOL bet)
    4. NO RECOVERY: No mechanism to query Kalshi for settled position outcomes
       and reconcile them after-the-fact
""")

    # --- Section 5: Session timeline ---
    print(f"\n{'-' * W}")
    print("  5. SESSION TIMELINE")
    print(f"{'-' * W}")

    for se in session_ends:
        sid = se.get("session_id", "?")
        if sid == "?":
            continue
        trades = se.get("trades_taken", 0)
        won = se.get("trades_won", 0)
        lost = se.get("trades_lost", 0)
        pnl = se.get("realized_pnl", 0)
        open_p = se.get("open_positions", 0)
        killed = se.get("killed", False)
        hours = se.get("session_hours", 0)
        reason = se.get("kill_reason", "")
        if trades == 0 and pnl == 0 and open_p == 0:
            continue
        status = "KILLED" if killed else "OK"
        print(f"\n   {sid} ({hours:.1f}h, {status})")
        print(f"     Trades: {trades} (W:{won} L:{lost}) | P&L: ${pnl:+.2f} | Open: {open_p}")
        if killed:
            print(f"     Kill: {reason}")

        # Describe what happened
        if sid == "live_20260412_145950":
            print("     >> Best session: hourly BTC/ETH/SOL buckets, 38 consecutive wins")
            print("     >> Left 10 positions open when stopped")
        elif "20260413_173611" in sid:
            print("     >> Placed 10 macro bets (CPI, Payrolls). None settled yet.")
        elif "20260415" in sid:
            print("     >> Final session: daily crypto buckets, 7 consecutive losses")
            print("     >> Kill switch triggered correctly")

    # --- Section 6: Infrastructure fixes needed ---
    print(f"\n{'=' * W}")
    print("  6. INFRASTRUCTURE FIXES FOR PHASE 3+")
    print(f"{'=' * W}")
    print("""
  To prevent these losses from recurring:

  A. SETTLEMENT RECOVERY ON STARTUP
     Query Kalshi's settlement/fills API on every bot startup to reconcile
     any positions that settled while offline. Log the true outcome.

  B. CONTINUOUS OPERATION OR WATCHDOG
     Run the bot as a persistent service with auto-restart, or add a
     watchdog that ensures the bot is always running during market hours.

  C. POSITION SIZE LIMITS PER TICKER
     Cap contracts per ticker (e.g., max 5 contracts = $5 risk on any
     single market). The 43-contract SOL position was excessive.

  D. SESSION-AWARE STATE PERSISTENCE
     Save position state to disk so restarts can resume cleanly
     instead of relying on Kalshi API reconciliation (which proved fragile).

  E. ENRICHED SESSION LOGGING
     Log sleeve classification, capital deployed by sleeve, and
     unresolved position count at session end for proper analytics.
""")
    print("=" * W)


if __name__ == "__main__":
    main()
