#!/usr/bin/env python3
"""
Phase 2: Audit The Edge Claim, Not Just The Code.

Separates three questions currently blurred in the strategy:
  Q1: Is the probability model calibrated? (model risk)
  Q2: Is the signal tradeable after spread/fill? (execution risk)
  Q3: Is the bankroll usage worth the holding period? (capital risk)

For each edge type (spot_vs_strike, macro_data, weather), documents:
  - The model assumption
  - The failure mode
  - Falsifiable hypothesis
  - Required evidence threshold

Also analyzes the live trade log to show how each assumption played out.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path("logs/live_trades.jsonl")


def load_events():
    events = []
    if not LOG_PATH.exists():
        return events
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def analyze_edge_quality(events):
    """Analyze how well reported edges predicted outcomes."""
    orders = [e for e in events if e.get("type") == "order_placed"]
    settlements = {e.get("ticker"): e for e in events if e.get("type") == "settlement"}

    edge_analysis = defaultdict(lambda: {
        "count": 0, "settled_count": 0, "wins": 0, "losses": 0,
        "edges": [], "costs": [], "pnls": [],
        "avg_reported_edge": 0.0,
    })

    for o in orders:
        etype = o.get("edge_type", "unknown")
        edge = o.get("edge", 0)
        cost = o.get("cost", 0)
        ticker = o.get("ticker", "")
        info = edge_analysis[etype]
        info["count"] += 1
        info["edges"].append(edge)
        info["costs"].append(cost)

        if ticker in settlements:
            s = settlements[ticker]
            pnl = s.get("pnl", 0)
            info["settled_count"] += 1
            info["pnls"].append(pnl)
            if pnl > 0:
                info["wins"] += 1
            elif pnl < 0:
                info["losses"] += 1

    return edge_analysis


def main():
    events = load_events()
    edge_analysis = analyze_edge_quality(events)

    W = 72
    print("=" * W)
    print("  EDGE CLAIM AUDIT - PHASE 2 REPORT")
    print("=" * W)

    # Live performance by edge type
    print(f"\n{'-' * W}")
    print("  A. LIVE EDGE PERFORMANCE BY TYPE")
    print(f"{'-' * W}")
    for etype, info in sorted(edge_analysis.items()):
        avg_edge = sum(info["edges"]) / max(len(info["edges"]), 1)
        total_cost = sum(info["costs"])
        total_pnl = sum(info["pnls"])
        wr = info["wins"] / max(info["wins"] + info["losses"], 1)
        print(f"\n   {etype}:")
        print(f"     Orders placed:   {info['count']}")
        print(f"     Avg reported edge: {avg_edge:.1%}")
        print(f"     Settled:         {info['settled_count']} (W:{info['wins']} L:{info['losses']}, {wr:.0%} WR)")
        print(f"     Settled P&L:     ${total_pnl:+.2f}")
        print(f"     Total cost:      ${total_cost:.2f}")
        unresolved = info["count"] - info["settled_count"]
        if unresolved > 0:
            print(f"     UNRESOLVED:      {unresolved} ({unresolved/info['count']:.0%} of orders)")

    # Q1: Model calibration
    print(f"\n{'=' * W}")
    print("  Q1: IS THE PROBABILITY MODEL CALIBRATED?")
    print(f"{'=' * W}")
    print("""
  SPOT_VS_STRIKE MODEL:
    Assumption: Lognormal diffusion with STATIC annualized vol.
    Formula:    P(S_T > K) = 1 - Phi((ln(K/S) + 0.5*v^2*T) / (v*sqrt(T)))

    Failure modes:
    a) Static vol is wrong. BTC vol was set to 56%, but realized vol
       during this period may have been higher or lower. In a volatility
       spike, the model underestimates tail probabilities (making it
       sell tails too cheaply). In a vol crush, the opposite.

    b) Lognormal assumption ignores jumps. Crypto has fat tails.
       The model says "0% chance BTC hits $82K in 1 hour" when BTC
       is at $70K. If BTC jumps 15% intraday, the model is wrong.

    c) Time-to-expiry is very short (often < 1 hour). The lognormal
       model becomes essentially binary at T -> 0: either the spot
       is already past the strike or it isn't. For hourly buckets
       near the money, the model CAN'T distinguish 40% from 60%.

    Live evidence:
      - The 38 hourly wins were all far-OTM: spot was $70K, strikes
        were at $81K+. The model correctly identified these as ~0%
        probability. These wins are TRIVIAL and didn't need a model.
      - The 7 daily losses were closer to the money. The model
        claimed 50%+ edge, but the outcomes went against it.

    HYPOTHESIS: The lognormal model only produces genuine edges
    when the spot is very far from the strike (>15% away) AND
    time-to-expiry is short (<2h). In these cases, ANY model
    would say "buy NO" — the edge is trivial, not model-dependent.

    EVIDENCE THRESHOLD: The model must demonstrate profitability
    on NON-TRIVIAL edges (within 10% of strike, 4h+ to expiry)
    to prove it adds value beyond common sense.

  MACRO_DATA MODEL:
    Assumption: Normal distribution around FRED consensus estimate
    with historically-calibrated standard deviation.

    Failure modes:
    a) Consensus estimates are already priced in. Kalshi's crowd
       pricing likely reflects the same FRED data.
    b) Macro data has regime shifts the normal distribution misses.
    c) Holding period is months, so even a correct probability
       estimate has poor capital efficiency.

    Live evidence:
      - ZERO macro settlements to date. All macro positions either
        became phantoms or are still locked.
      - No evidence for or against the model's calibration.

    HYPOTHESIS: Macro edges only exist when the model's std estimate
    is meaningfully tighter than the market-implied std. If Kalshi
    prices already embed FRED consensus, the raw signal is noise.

    EVIDENCE THRESHOLD: Must show positive realized P&L across at
    least 5 independent macro settlements before being trusted.

  WEATHER MODEL:
    Not yet tested in live trading. No live evidence to audit.
""")

    # Q2: Execution reality
    print(f"{'=' * W}")
    print("  Q2: IS THE SIGNAL TRADEABLE AFTER SPREAD/FILL?")
    print(f"{'=' * W}")
    print("""
  HEADLINE EDGE vs EXECUTION EDGE:
    The detector reports "59% edge" meaning model says 0% but market
    prices at 59%. But the actual trade pays:
      - BUY_NO cost = (100 - yes_ask) cents per contract
      - If NO wins, return = $1.00 per contract
      - Execution edge = ($1.00 - cost) / cost

    For the 38 winning hourly trades:
      - Most were bought at 50c (NO cost = 50c)
      - Return per win = $1.00 on $0.50 invested = 100% return
      - But the "model edge" was reported as 50% (fair=0c vs mkt=50c)
      - The execution edge was actually just buying a coin flip
        that happened to always land in our favor because we only
        bought extremely OTM strikes

    For the 7 losing daily trades:
      - Costs ranged from 4c to 96c
      - Returns were -100% on each (total loss of invested amount)
      - The "model edge" of 50%+ was overconfident for daily markets

  SPREAD IMPACT:
    The bot uses mid-price for edge calculation but trades at bid/ask.
    In low-liquidity markets, spread can be 20-40c, eating most of
    the reported edge.

    HYPOTHESIS: True execution edge = reported_edge - (spread/2)/100
    Any trade where (spread/2) > reported_edge*100 has negative
    expected value after execution.

    EVIDENCE THRESHOLD: Track execution edge (cost vs payout) separately
    from model edge (fair vs mid). Only trades with positive execution
    edge should be taken.
""")

    # Q3: Capital usage
    print(f"{'=' * W}")
    print("  Q3: IS THE BANKROLL USAGE WORTH THE HOLDING PERIOD?")
    print(f"{'=' * W}")
    print("""
  CAPITAL VELOCITY:
    - Hourly crypto: 1h hold, can recycle 24x per day
    - Daily crypto: 24h hold, can recycle 1x per day
    - Macro (CPI/NFP): 30-180 day hold, cannot recycle

    $1 in hourly crypto at 100% return = $24/day potential
    $1 in daily crypto at 100% return = $1/day potential
    $1 in macro at 100% return = $0.006/day potential (180d hold)

  OPPORTUNITY COST:
    Every dollar locked in a macro bet is a dollar that can't earn
    hourly crypto returns. At the bot's proven hourly win rate, the
    opportunity cost of macro lock-up is massive.

  ACTUAL EXPERIENCE:
    - $8.82 deployed into macro positions on 04/13 (CPI, Payrolls)
    - ZERO have settled as of 04/16
    - That $8.82 could have been recycled 72+ times in hourly markets
    - Assuming 50c cost and $1 return per win at 100% win rate,
      that's potentially $8.82 * 72 = $635 in missed hourly trades

  HYPOTHESIS: Macro bets are only justified if:
    1. Expected edge exceeds 30% (to compensate for 100+ day lock-up)
    2. Maximum 5% of bankroll is allocated to macro sleeve
    3. The specific macro indicator has demonstrated historical
       miscalibration between FRED and Kalshi (evidence required)

  EVIDENCE THRESHOLD: No macro bet should be taken until fast-sleeve
  profitability is proven and excess capital exists beyond what the
  fast sleeve can deploy.
""")

    # Summary of falsifiable hypotheses
    print(f"{'=' * W}")
    print("  FALSIFIABLE HYPOTHESES FOR PHASE 3 TRADE RULES")
    print(f"{'=' * W}")
    print("""
  H1 (TRIVIAL EDGE): When spot is >15% from strike AND T<2h,
      buying NO on OTM crypto buckets is profitable regardless
      of model quality. This is the only proven edge.
      TEST: Continue running hourly-only with >15% OTM filter.

  H2 (MODEL EDGE): When spot is within 15% of strike OR T>4h,
      the lognormal model does NOT produce tradeable edge after
      execution friction. The 7 consecutive losses support this.
      TEST: Paper-trade "near-the-money" edges for 50+ trades
      and measure realized win rate.

  H3 (STATIC VOL): Static volatility estimates are adequate for
      far-OTM hourly trades but NOT for any trade where the model
      probability is between 20% and 80%.
      TEST: Compare static vol to trailing 7-day realized vol.
      If they differ by >20%, flag all trades in the 20-80% zone.

  H4 (MACRO WORTHLESS): Macro bets have zero proven edge and
      massive opportunity cost. They should be disabled until
      at least 5 settlements show positive realized P&L.
      TEST: Track all macro settlements. Re-enable only after
      passing the evidence threshold.

  H5 (EXECUTION FILTER): Only take trades where:
      execution_edge = (payout - cost) / cost > 50%
      AND spread < 20c
      AND volume + OI > 5
      TEST: Apply this filter retroactively to all 145 historical
      orders and measure hypothetical P&L.
""")
    print("=" * W)


if __name__ == "__main__":
    main()
