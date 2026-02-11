"""
Practical Edge Finder for Kalshi Markets

This module demonstrates how to find profitable prediction market opportunities
by comparing market prices against external probability estimates.

THE KEY INSIGHT:
================
Market price = crowd's probability estimate
Your edge = (Your probability - Market price)

If you think an event is 60% likely but market says 40%, you have +20% edge.

WHERE EDGE COMES FROM:
======================
1. Information Advantage - You know something others don't
2. Model Advantage - Your model is better calibrated than crowd
3. Speed Advantage - You react to news faster
4. Arbitrage - Same event priced differently across markets
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sources.kalshi import KalshiAdapter


@dataclass
class Opportunity:
    """A detected trading opportunity."""
    market_ticker: str
    market_title: str
    market_price: float      # Current YES price (0-1)
    our_estimate: float      # Our probability estimate (0-1)
    edge: float              # our_estimate - market_price
    confidence: str          # "high", "medium", "low"
    edge_source: str         # Where our edge comes from
    recommended_action: str  # "BUY YES", "BUY NO", "WAIT"
    bet_size: float          # Fraction of bankroll (Kelly)
    notes: str


class EdgeSource(Enum):
    """Types of edges we can exploit."""
    
    # Cross-market arbitrage
    ARBITRAGE = "arbitrage"
    
    # News hasn't been priced in yet
    NEWS_ALPHA = "news_alpha"
    
    # Fundamental analysis differs from market
    FUNDAMENTAL = "fundamental"
    
    # Market is inefficient (low volume, wide spread)
    INEFFICIENCY = "market_inefficiency"
    
    # Historical pattern suggests mispricing
    STATISTICAL = "statistical_edge"


def find_arbitrage_opportunities(
    kalshi: KalshiAdapter,
    polymarket_prices: Dict[str, float] = None,
) -> List[Opportunity]:
    """
    Find arbitrage between Kalshi and Polymarket.
    
    Same event, different prices = free money.
    
    Example:
    - Kalshi: "Will BTC hit $100k by Dec?" at 45%
    - Polymarket: Same question at 55%
    
    Buy YES on Kalshi @45%, Buy NO on Polymarket @45% (100%-55%)
    Guaranteed profit regardless of outcome.
    """
    opportunities = []
    
    if not polymarket_prices:
        # Would fetch from Polymarket API
        polymarket_prices = {}
    
    # Get all Kalshi markets
    markets_resp = kalshi.get_markets(limit=200)
    if "markets" not in markets_resp:
        return opportunities
    
    for market in markets_resp["markets"]:
        ticker = market.get("ticker", "")
        title = market.get("title", "").lower()
        kalshi_price = market.get("last_price", 0) / 100
        
        # Check if we have matching Polymarket price
        for poly_key, poly_price in polymarket_prices.items():
            if _markets_match(title, poly_key):
                diff = abs(kalshi_price - poly_price)
                if diff > 0.05:  # 5% arbitrage threshold
                    opportunities.append(Opportunity(
                        market_ticker=ticker,
                        market_title=title,
                        market_price=kalshi_price,
                        our_estimate=poly_price,
                        edge=diff,
                        confidence="high",
                        edge_source="Cross-platform arbitrage",
                        recommended_action="BUY YES" if kalshi_price < poly_price else "BUY NO",
                        bet_size=0.1,  # Size limited by liquidity
                        notes=f"Polymarket: {poly_price:.0%} vs Kalshi: {kalshi_price:.0%}"
                    ))
    
    return opportunities


def find_mispriced_parlays(
    kalshi: KalshiAdapter,
    team_win_probs: Dict[str, float],
) -> List[Opportunity]:
    """
    Find mispriced sports parlays.
    
    Parlays multiply probabilities, so small errors compound.
    
    Example 3-team parlay:
    - Market price: 5% (implied: all three teams are ~37% each)
    - Our estimate: Team A 50%, Team B 55%, Team C 45%
    - True probability: 0.50 * 0.55 * 0.45 = 12.4%
    
    Edge = 12.4% - 5% = +7.4%
    """
    opportunities = []
    
    markets = kalshi.get_markets(limit=500, status="open")
    if "markets" not in markets:
        return opportunities
    
    for market in markets.get("markets", []):
        title = market.get("title", "").lower()
        
        # Check if it's a parlay
        if "parlay" not in title and "multi" not in title:
            continue
        
        market_price = market.get("last_price", 0) / 100
        if market_price == 0:
            continue
        
        # Find teams mentioned
        teams_in_market = []
        for team, prob in team_win_probs.items():
            if team.lower() in title:
                teams_in_market.append((team, prob))
        
        if len(teams_in_market) >= 2:
            # Calculate parlay probability
            our_prob = 1.0
            for team, prob in teams_in_market:
                our_prob *= prob
            
            edge = our_prob - market_price
            
            if abs(edge) > 0.03:  # 3% edge threshold
                opportunities.append(Opportunity(
                    market_ticker=market.get("ticker"),
                    market_title=title,
                    market_price=market_price,
                    our_estimate=our_prob,
                    edge=edge,
                    confidence="medium",
                    edge_source="Parlay mispricing",
                    recommended_action="BUY YES" if edge > 0 else "BUY NO",
                    bet_size=calculate_kelly(edge, our_prob) * 0.25,
                    notes=f"Teams: {[t[0] for t in teams_in_market]}, Individual probs: {[f'{t[1]:.0%}' for t in teams_in_market]}"
                ))
    
    return opportunities


def find_news_alpha(
    kalshi: KalshiAdapter,
    recent_news: List[Dict],
) -> List[Opportunity]:
    """
    Find markets that haven't priced in recent news.
    
    News creates alpha when:
    1. News is recent (< 1 hour)
    2. News is relevant to market
    3. Market price hasn't moved yet
    
    Example:
    - Market: "Will Fed raise rates in March?"
    - News: "Inflation report comes in hot at 3.5%"
    - If market is still at 40%, but news implies 60%, buy YES
    """
    opportunities = []
    
    # This would analyze news feed and match to markets
    # Implementation depends on news API (Bloomberg, Reuters, etc.)
    
    return opportunities


def find_inefficient_markets(
    kalshi: KalshiAdapter,
) -> List[Opportunity]:
    """
    Find structurally inefficient markets.
    
    Signs of inefficiency:
    - Very low volume (few participants)
    - Wide bid-ask spread (>10%)
    - Price hasn't moved in days
    - Contradictory prices in related markets
    """
    opportunities = []
    
    markets = kalshi.get_markets(limit=500, status="open")
    if "markets" not in markets:
        return opportunities
    
    for market in markets.get("markets", []):
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 100)
        volume = market.get("volume", 0)
        
        spread = yes_ask - yes_bid
        mid_price = (yes_bid + yes_ask) / 2 / 100
        
        # Flag wide spreads in low-volume markets
        if spread > 15 and volume < 1000:
            opportunities.append(Opportunity(
                market_ticker=ticker,
                market_title=title,
                market_price=mid_price,
                our_estimate=mid_price,  # No edge, just noting inefficiency
                edge=0,
                confidence="low",
                edge_source="Market inefficiency",
                recommended_action="RESEARCH",  # Need more analysis
                bet_size=0,
                notes=f"Wide spread: {spread}%, Low volume: {volume}"
            ))
    
    return opportunities


def calculate_kelly(edge: float, probability: float) -> float:
    """
    Kelly criterion: optimal bet size.
    
    f* = edge / (1 - probability) for favorable bets
    
    Returns fraction of bankroll to bet (0-1).
    """
    if edge <= 0:
        return 0
    
    # For binary outcomes
    # f* = (p*b - q) / b where p=prob, q=1-p, b=odds
    b = (1 - probability) / probability if probability > 0 else 0
    q = 1 - probability
    
    if b > 0:
        kelly = (probability * b - q) / b
        return max(0, min(kelly, 0.5))  # Cap at 50%
    
    return 0


def _markets_match(title1: str, title2: str) -> bool:
    """Check if two market titles refer to same event."""
    # Simple keyword matching - would need NLP for production
    title1_words = set(title1.lower().split())
    title2_words = set(title2.lower().split())
    
    overlap = len(title1_words & title2_words)
    min_len = min(len(title1_words), len(title2_words))
    
    return overlap / min_len > 0.5 if min_len > 0 else False


# ==========================================================================
# PRACTICAL EXAMPLE: Finding Edge with Real Data
# ==========================================================================

def demo_edge_finding():
    """
    Demonstrate edge-finding with example data.
    """
    print("=" * 60)
    print("EDGE FINDING DEMONSTRATION")
    print("=" * 60)
    
    # Example: You have external probability estimates for NBA teams
    # These could come from ESPN BPI, 538, your own model, etc.
    nba_team_probs = {
        "Atlanta": 0.45,      # Hawks game tonight
        "Indiana": 0.52,      # Pacers favored
        "Phoenix": 0.48,      # Suns at home
        "Boston": 0.72,       # Celtics heavy favorite
        "Cleveland": 0.58,    # Cavs solid
        "Oklahoma City": 0.68, # Thunder strong
    }
    
    print("\n📊 Your NBA Win Probability Estimates:")
    for team, prob in sorted(nba_team_probs.items(), key=lambda x: -x[1]):
        print(f"  {team}: {prob:.0%}")
    
    # Connect to Kalshi and find mispriced parlays
    try:
        kalshi = KalshiAdapter()
        print("\n🔍 Scanning Kalshi for mispriced parlays...")
        
        opportunities = find_mispriced_parlays(kalshi, nba_team_probs)
        
        if opportunities:
            print(f"\n✅ Found {len(opportunities)} potential opportunities:\n")
            for opp in sorted(opportunities, key=lambda x: -abs(x.edge)):
                print(f"Market: {opp.market_title[:60]}...")
                print(f"  Ticker: {opp.market_ticker}")
                print(f"  Market Price: {opp.market_price:.1%}")
                print(f"  Our Estimate: {opp.our_estimate:.1%}")
                print(f"  Edge: {opp.edge:+.1%}")
                print(f"  Action: {opp.recommended_action}")
                print(f"  Bet Size: {opp.bet_size:.1%} of bankroll")
                print(f"  Notes: {opp.notes}")
                print()
        else:
            print("\n❌ No opportunities found with current estimates.")
            print("   This could mean:")
            print("   1. Markets are efficiently priced")
            print("   2. Need better/different probability estimates")
            print("   3. Need to scan more markets")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Show the math
    print("\n" + "=" * 60)
    print("THE MATH: Why Parlays Create Edge Opportunities")
    print("=" * 60)
    
    print("""
Example 3-team parlay:

Market implies each team has ~37% win probability:
  0.37 ^ 3 = 5.1% parlay probability

But if you estimate:
  Team A: 50% (better than market thinks)
  Team B: 55% (better than market thinks)  
  Team C: 45% (worse than market thinks)

Your probability: 0.50 × 0.55 × 0.45 = 12.4%

EDGE = 12.4% - 5.1% = +7.3%

Expected Value per $1 bet:
  Win: 12.4% × ($1 / 0.051 - $1) = $2.18
  Lose: 87.6% × (-$1) = -$0.88
  EV = $2.18 - $0.88 = +$1.30 per $1 bet

This is 130% expected return!
    """)
    
    print("=" * 60)
    print("TO MAKE MONEY, YOU NEED:")
    print("=" * 60)
    print("""
1. EXTERNAL DATA SOURCE
   - Sports: ESPN BPI, FiveThirtyEight, odds APIs
   - Elections: Poll aggregators, prediction models
   - Economics: FRED data, nowcasting models
   - Crypto: On-chain metrics, sentiment analysis

2. BETTER MODEL THAN CROWD
   - If markets are efficient, no edge
   - Your model must be more accurate than consensus
   
3. CAPITAL TO DEPLOY
   - Kalshi has position limits
   - Use Kelly sizing to manage risk
   
4. PATIENCE
   - Positive EV doesn't guarantee wins
   - Need many bets for edge to manifest
    """)


if __name__ == "__main__":
    demo_edge_finding()
