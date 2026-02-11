"""
Sports data signal sources for Kalshi prediction markets.

Integrates with free sports APIs to generate probability estimates
for NBA, NFL, MLB, etc. markets.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import requests

from ..core.logger import get_logger
from .kalshi_signals import SignalSource, Signal, MarketCategory

logger = get_logger(__name__)


class NBAStatsSignal(SignalSource):
    """
    NBA game predictions using public stats.
    
    Data sources:
    - balldontlie.io (free NBA stats)
    - ESPN API (public)
    - Basketball Reference
    """
    
    BASE_URL = "https://api.balldontlie.io/v1"
    
    TEAM_ABBREVS = {
        "atlanta": "ATL", "boston": "BOS", "brooklyn": "BKN", "charlotte": "CHA",
        "chicago": "CHI", "cleveland": "CLE", "dallas": "DAL", "denver": "DEN",
        "detroit": "DET", "golden state": "GSW", "houston": "HOU", "indiana": "IND",
        "la clippers": "LAC", "la lakers": "LAL", "memphis": "MEM", "miami": "MIA",
        "milwaukee": "MIL", "minnesota": "MIN", "new orleans": "NOP", "new york": "NYK",
        "oklahoma city": "OKC", "orlando": "ORL", "philadelphia": "PHI", "phoenix": "PHX",
        "portland": "POR", "sacramento": "SAC", "san antonio": "SAS", "toronto": "TOR",
        "utah": "UTA", "washington": "WAS",
    }
    
    def __init__(self):
        self.api_key = os.getenv("BALLDONTLIE_API_KEY", "")
    
    def supports_category(self, category: MarketCategory) -> bool:
        return category == MarketCategory.SPORTS
    
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Generate win probability for NBA game."""
        title = market.get("title", "").lower()
        
        # Check if NBA-related
        nba_keywords = list(self.TEAM_ABBREVS.keys())
        teams_found = [team for team in nba_keywords if team in title]
        
        if not teams_found:
            return None
        
        # For a parlay like "yes Atlanta, yes Indiana, yes Phoenix"
        # We need to estimate probability of ALL winning
        signals = []
        combined_prob = 1.0
        
        for team in teams_found:
            team_prob = self._get_team_win_probability(team)
            if team_prob:
                combined_prob *= team_prob
                signals.append(f"{team}:{team_prob:.0%}")
        
        if not signals:
            return None
        
        return Signal(
            source="nba_stats",
            probability=combined_prob,
            confidence=0.6,  # Sports models are uncertain
            timestamp=datetime.now(timezone.utc),
            metadata={"teams": teams_found, "individual_probs": signals}
        )
    
    def _get_team_win_probability(self, team_name: str) -> Optional[float]:
        """
        Estimate team's win probability based on recent performance.
        
        Simple model: win rate over last 10 games adjusted for opponent.
        """
        try:
            # Get team stats (would need API key for balldontlie)
            # Fallback: use general NBA home team advantage + win rate
            
            # Without API, return estimated probability based on general stats
            # In production, this would query real stats
            
            # Placeholder: assume average team has ~50% win rate
            # Better teams: 55-65%, worse teams: 35-45%
            
            # This should be replaced with real API data
            return 0.50  # Default to 50%
            
        except Exception as e:
            logger.debug(f"Error getting team stats: {e}")
            return None
    
    def get_team_stats(self, team_abbrev: str) -> Optional[Dict]:
        """Fetch team stats from API."""
        if not self.api_key:
            return None
        
        try:
            headers = {"Authorization": self.api_key}
            resp = requests.get(
                f"{self.BASE_URL}/teams",
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                teams = resp.json().get("data", [])
                for team in teams:
                    if team.get("abbreviation") == team_abbrev:
                        return team
        except Exception as e:
            logger.debug(f"API error: {e}")
        
        return None


class OddsAPISignal(SignalSource):
    """
    Cross-reference with sportsbook odds via The Odds API.
    
    If Kalshi price differs significantly from Vegas consensus,
    there may be an arbitrage opportunity.
    
    API: https://the-odds-api.com/ (free tier: 500 requests/month)
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self):
        self.api_key = os.getenv("ODDS_API_KEY", "")
    
    def supports_category(self, category: MarketCategory) -> bool:
        return category == MarketCategory.SPORTS
    
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Compare Kalshi price to Vegas odds consensus."""
        if not self.api_key:
            return None
        
        title = market.get("title", "").lower()
        
        # Parse teams from title
        # Would need to match Kalshi market to specific game
        # Then compare implied probabilities
        
        return None  # Needs implementation
    
    def get_nba_odds(self) -> List[Dict]:
        """Fetch current NBA odds from multiple books."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/sports/basketball_nba/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": "h2h",  # Moneyline
                    "oddsFormat": "american",
                },
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Odds API error: {e}")
        
        return []
    
    def american_to_probability(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


class ESPNSignal(SignalSource):
    """
    Use ESPN's public API for game predictions and power rankings.
    
    ESPN BPI (Basketball Power Index) provides team strength ratings
    that can be converted to win probabilities.
    """
    
    BASE_URL = "https://site.api.espn.com/apis/site/v2"
    
    def supports_category(self, category: MarketCategory) -> bool:
        return category == MarketCategory.SPORTS
    
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Get ESPN game prediction if available."""
        return None  # Needs implementation
    
    def get_nba_scoreboard(self) -> Dict:
        """Get today's NBA games with predictions."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/sports/basketball/nba/scoreboard",
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"ESPN API error: {e}")
        
        return {}
    
    def get_team_bpi(self, team_id: int) -> Optional[float]:
        """Get team's Basketball Power Index rating."""
        # ESPN BPI can be used to estimate win probability
        # Higher BPI = stronger team
        return None


def calculate_parlay_probability(individual_probs: List[float]) -> float:
    """
    Calculate parlay probability (all legs must win).
    
    For independent events: P(A and B and C) = P(A) * P(B) * P(C)
    
    Note: In reality, legs may be correlated (e.g., if Team A wins big,
    player props are more likely). We assume independence for simplicity.
    """
    result = 1.0
    for prob in individual_probs:
        result *= prob
    return result


def calculate_parlay_ev(
    individual_probs: List[float],
    market_price: float,  # 0-1
) -> Dict[str, float]:
    """
    Calculate expected value of a parlay bet.
    
    Returns:
        edge: Our probability - market price
        ev: Expected $ profit per $1 bet
        kelly: Optimal bet fraction
    """
    our_prob = calculate_parlay_probability(individual_probs)
    edge = our_prob - market_price
    
    # EV = (probability * payout) - (1 - probability) * stake
    # For Kalshi: win $1 - price paid, lose price paid
    payout = 1.0 - market_price  # Win amount per contract
    ev = (our_prob * payout) - ((1 - our_prob) * market_price)
    
    # Kelly criterion: f* = edge / odds
    # For binary: f* = (p*b - q) / b where b = payout/stake
    b = payout / market_price if market_price > 0 else 0
    kelly = (our_prob * b - (1 - our_prob)) / b if b > 0 else 0
    kelly = max(0, kelly)  # Don't bet negative
    
    return {
        "our_probability": our_prob,
        "market_probability": market_price,
        "edge": edge,
        "expected_value": ev,
        "kelly_fraction": kelly,
        "quarter_kelly": kelly * 0.25,
    }
