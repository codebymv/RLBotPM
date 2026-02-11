"""
Kalshi Prediction Market Signal Aggregator.

Generates probability estimates for various market types by combining
multiple data sources and models. The goal is to find markets where
our estimated probability differs significantly from the market price.

Key Principle: The market price IS a probability. We profit when our
probability estimate is MORE ACCURATE than the crowd's.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable, TYPE_CHECKING
from enum import Enum

import requests
import numpy as np

from ..core.logger import get_logger

if TYPE_CHECKING:
    from ..execution.kalshi_rl_bridge import KalshiRLBridge

logger = get_logger(__name__)


class MarketCategory(Enum):
    """Categories of prediction markets."""
    ELECTIONS = "elections"
    ECONOMICS = "economics"  # CPI, jobs, GDP
    FED = "fed"  # FOMC decisions
    CRYPTO = "crypto"  # BTC/ETH price ranges
    WEATHER = "weather"
    SPORTS = "sports"
    OTHER = "other"


@dataclass
class Signal:
    """A probability signal from a data source."""
    source: str
    probability: float  # 0-1
    confidence: float  # 0-1, how confident in this signal
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}


@dataclass
class MarketAnalysis:
    """Analysis of a prediction market opportunity."""
    ticker: str
    title: str
    market_price: float  # Current YES price (0-1)
    our_probability: float  # Our estimated probability (0-1)
    edge: float  # our_probability - market_price
    confidence: float  # How confident in our estimate
    signals: List[Signal]
    recommendation: str  # "BUY_YES", "BUY_NO", "HOLD"
    position_size_pct: float  # Recommended position size
    reasoning: str


class SignalSource(ABC):
    """Abstract base class for signal sources."""
    
    @abstractmethod
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Generate a signal for a market."""
        pass
    
    @abstractmethod
    def supports_category(self, category: MarketCategory) -> bool:
        """Check if this source supports a market category."""
        pass


class PolymarketSignal(SignalSource):
    """
    Cross-reference with Polymarket for arbitrage opportunities.
    If same event trades at different prices, there's an edge.
    """
    
    def __init__(self):
        self.base_url = "https://gamma-api.polymarket.com"
    
    def supports_category(self, category: MarketCategory) -> bool:
        return category in [MarketCategory.ELECTIONS, MarketCategory.CRYPTO, MarketCategory.OTHER]
    
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Find matching Polymarket market and compare prices."""
        # Search for similar market on Polymarket
        title = market.get("title", "")
        
        try:
            # Search Polymarket for similar markets
            resp = requests.get(
                f"{self.base_url}/markets",
                params={"_limit": 100, "active": True},
                timeout=10
            )
            if resp.status_code != 200:
                return None
            
            poly_markets = resp.json()
            
            # Find best match by title similarity
            best_match = None
            best_score = 0
            
            for pm in poly_markets:
                pm_title = pm.get("question", "")
                score = self._title_similarity(title, pm_title)
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = pm
            
            if best_match:
                # Get price from Polymarket
                poly_price = float(best_match.get("outcomePrices", [0.5, 0.5])[0])
                
                return Signal(
                    source="polymarket",
                    probability=poly_price,
                    confidence=best_score,  # Higher match = higher confidence
                    timestamp=datetime.now(timezone.utc),
                    metadata={"polymarket_id": best_match.get("id")}
                )
                
        except Exception as e:
            logger.debug(f"Polymarket signal error: {e}")
        
        return None
    
    def _title_similarity(self, s1: str, s2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 or not words2:
            return 0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)


class FedWatchSignal(SignalSource):
    """
    CME FedWatch tool probabilities for FOMC decisions.
    Professional futures market is very efficient - trust it.
    """
    
    def supports_category(self, category: MarketCategory) -> bool:
        return category == MarketCategory.FED
    
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Get Fed funds futures implied probabilities."""
        title = market.get("title", "").lower()
        
        # Check if this is a Fed-related market
        fed_keywords = ["fed", "fomc", "rate", "powell", "interest rate"]
        if not any(kw in title for kw in fed_keywords):
            return None
        
        # In production, scrape CME FedWatch or use futures data
        # For now, return None (could integrate with Fed futures API)
        logger.debug("Fed market detected - would query FedWatch")
        return None


class PollAggregatorSignal(SignalSource):
    """
    Aggregate polling data for election markets.
    Uses 538-style methodology: weight by recency, sample size, pollster rating.
    """
    
    def supports_category(self, category: MarketCategory) -> bool:
        return category == MarketCategory.ELECTIONS
    
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Aggregate polls for election markets."""
        title = market.get("title", "").lower()
        
        # Check if election-related
        election_keywords = ["election", "president", "senate", "house", "governor", "vote", "win"]
        if not any(kw in title for kw in election_keywords):
            return None
        
        # In production: query RealClearPolitics, 538, polling APIs
        # Weight polls by: recency, sample size, pollster rating, LV vs RV
        logger.debug("Election market detected - would aggregate polls")
        return None


class EconomicNowcastSignal(SignalSource):
    """
    Nowcasting models for economic data releases (CPI, jobs, GDP).
    Uses leading indicators to predict before official release.
    """
    
    def supports_category(self, category: MarketCategory) -> bool:
        return category == MarketCategory.ECONOMICS
    
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Generate nowcast for economic data."""
        title = market.get("title", "").lower()
        
        # Check if economic data market
        econ_keywords = ["cpi", "inflation", "jobs", "unemployment", "gdp", "payroll", "pce"]
        matched_keyword = None
        for kw in econ_keywords:
            if kw in title:
                matched_keyword = kw
                break
        
        if not matched_keyword:
            return None
        
        # In production: Use Fed nowcasting models, Cleveland Fed inflation nowcast,
        # Atlanta Fed GDPNow, survey data, high-frequency indicators
        logger.debug(f"Economic market detected ({matched_keyword}) - would run nowcast")
        return None


class LineMovementSignal(SignalSource):
    """
    Track price movements to detect sharp money.
    If price moved significantly, smart money may know something.
    """
    
    def __init__(self):
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def supports_category(self, category: MarketCategory) -> bool:
        return True  # Works for all categories
    
    def get_signal(self, market: Dict) -> Optional[Signal]:
        """Analyze line movement for momentum signal."""
        ticker = market.get("ticker", "")
        current_price = market.get("yes_price", 50) / 100.0
        
        # Record price
        now = datetime.now(timezone.utc)
        if ticker not in self.price_history:
            self.price_history[ticker] = []
        self.price_history[ticker].append((now, current_price))
        
        # Keep last 24 hours
        cutoff = now - timedelta(hours=24)
        self.price_history[ticker] = [
            (t, p) for t, p in self.price_history[ticker] if t > cutoff
        ]
        
        history = self.price_history[ticker]
        if len(history) < 2:
            return None
        
        # Calculate momentum
        oldest_price = history[0][1]
        price_change = current_price - oldest_price
        
        # Strong movement = potential signal
        if abs(price_change) > 0.05:  # 5% move
            # Momentum: follow the move (sharp money moving the line)
            probability = min(0.95, max(0.05, current_price + price_change * 0.3))
            
            return Signal(
                source="line_movement",
                probability=probability,
                confidence=min(0.7, abs(price_change) * 5),  # Higher move = more confident
                timestamp=now,
                metadata={"price_change": price_change, "hours": len(history)}
            )
        
        return None


class PPOSignalSource(SignalSource):
    """
    Signal source that uses the crypto PPO model for Kalshi crypto markets.

    Requires a KalshiRLBridge and an optional get_obs callable. When get_obs
    is set and returns an observation, the PPO recommendation is converted
    to a probability signal (BUY_YES -> high prob, BUY_NO -> low prob).
    """

    def __init__(
        self,
        bridge: "KalshiRLBridge",
        get_obs: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ):
        self.bridge = bridge
        self.get_obs = get_obs

    def supports_category(self, category: MarketCategory) -> bool:
        return category == MarketCategory.CRYPTO

    def get_signal(self, market: Dict) -> Optional[Signal]:
        if self.get_obs is None:
            return None
        try:
            obs = self.get_obs()
            if obs is None:
                return None
            recommendation, confidence = self.bridge.recommend_from_observation(obs)
            if recommendation == "HOLD":
                return None
            # Map BUY_YES -> probability 0.7, BUY_NO -> 0.3 (market-relative)
            probability = 0.7 if recommendation == "BUY_YES" else 0.3
            return Signal(
                source="ppo",
                probability=probability,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc),
                metadata={"recommendation": recommendation},
            )
        except Exception as e:
            logger.debug(f"PPO signal error: {e}")
            return None


class KalshiSignalAggregator:
    """
    Combines multiple signal sources to generate trading recommendations.
    
    The key insight: different sources have different strengths.
    - Polymarket: Good for popular markets (arbitrage opportunity)
    - Polls: Good for elections
    - Nowcasts: Good for economic data
    - Line movement: Detects smart money
    - PPO: Crypto RL model for crypto prediction markets (when use_ppo_signal and get_obs set)
    
    We weight each signal by confidence and source reliability.
    """
    
    def __init__(
        self,
        use_ppo_signal: bool = False,
        ppo_model_path: Optional[str] = None,
        get_ppo_obs: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ):
        self.sources: List[SignalSource] = [
            PolymarketSignal(),
            FedWatchSignal(),
            PollAggregatorSignal(),
            EconomicNowcastSignal(),
            LineMovementSignal(),
        ]
        
        if use_ppo_signal and ppo_model_path and get_ppo_obs:
            from ..execution.kalshi_rl_bridge import KalshiRLBridge
            bridge = KalshiRLBridge(ppo_model_path=ppo_model_path)
            if bridge.load_model():
                self.sources.append(PPOSignalSource(bridge, get_obs=get_ppo_obs))
                logger.info("KalshiSignalAggregator: PPO signal source enabled")
        
        # Source reliability weights (learned over time)
        self.source_weights = {
            "polymarket": 0.8,  # Cross-market arbitrage is strong
            "fedwatch": 0.9,   # Futures market is efficient
            "polls": 0.6,      # Polls have error
            "nowcast": 0.7,    # Models are decent
            "line_movement": 0.4,  # Noisy but useful
            "ppo": 0.75,       # RL model for crypto
        }
        
        # Minimum edge to trade
        self.min_edge = 0.10  # 10%
        self.min_confidence = 0.5
    
    def categorize_market(self, market: Dict) -> MarketCategory:
        """Determine market category from title/metadata."""
        title = market.get("title", "").lower()
        category = market.get("category", "").lower()
        
        if any(kw in title or kw in category for kw in ["election", "president", "senate", "governor"]):
            return MarketCategory.ELECTIONS
        if any(kw in title for kw in ["fed", "fomc", "rate hike", "rate cut"]):
            return MarketCategory.FED
        if any(kw in title for kw in ["cpi", "inflation", "jobs", "gdp", "unemployment"]):
            return MarketCategory.ECONOMICS
        if any(kw in title for kw in ["btc", "bitcoin", "eth", "ethereum", "crypto"]):
            return MarketCategory.CRYPTO
        if any(kw in title for kw in ["weather", "temperature", "hurricane"]):
            return MarketCategory.WEATHER
        if any(kw in title for kw in ["nba", "nfl", "mlb", "nhl", "game", "match"]):
            return MarketCategory.SPORTS
        
        return MarketCategory.OTHER
    
    def analyze_market(self, market: Dict) -> MarketAnalysis:
        """
        Generate a complete analysis and recommendation for a market.
        
        Returns probability estimate, edge, and position sizing.
        """
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        market_price = market.get("yes_price", 50) / 100.0
        
        category = self.categorize_market(market)
        
        # Collect signals from all applicable sources
        signals: List[Signal] = []
        for source in self.sources:
            if source.supports_category(category):
                try:
                    signal = source.get_signal(market)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.debug(f"Signal source error: {e}")
        
        # Aggregate signals into single probability estimate
        our_probability, confidence = self._aggregate_signals(signals, market_price)
        
        # Calculate edge
        edge = our_probability - market_price
        
        # Generate recommendation
        recommendation, reasoning = self._get_recommendation(
            edge, confidence, market_price, signals
        )
        
        # Position sizing based on edge and confidence (Kelly-inspired)
        position_size = self._calculate_position_size(edge, confidence)
        
        return MarketAnalysis(
            ticker=ticker,
            title=title,
            market_price=market_price,
            our_probability=our_probability,
            edge=edge,
            confidence=confidence,
            signals=signals,
            recommendation=recommendation,
            position_size_pct=position_size,
            reasoning=reasoning,
        )
    
    def _aggregate_signals(
        self, 
        signals: List[Signal], 
        market_price: float
    ) -> Tuple[float, float]:
        """
        Combine multiple signals into single probability estimate.
        
        Uses confidence-weighted average with market price as prior.
        """
        if not signals:
            # No signals - use market price as best estimate
            return market_price, 0.3  # Low confidence
        
        # Weighted average of signals
        total_weight = 0
        weighted_sum = 0
        
        for signal in signals:
            source_weight = self.source_weights.get(signal.source, 0.5)
            weight = signal.confidence * source_weight
            weighted_sum += signal.probability * weight
            total_weight += weight
        
        # Include market price as a signal (wisdom of crowd)
        market_weight = 0.3  # 30% weight to market consensus
        weighted_sum += market_price * market_weight
        total_weight += market_weight
        
        our_probability = weighted_sum / total_weight if total_weight > 0 else market_price
        
        # Confidence based on signal agreement and count
        if len(signals) >= 2:
            # Multiple signals agreeing = higher confidence
            signal_variance = sum(
                (s.probability - our_probability) ** 2 for s in signals
            ) / len(signals)
            agreement_confidence = max(0, 1 - signal_variance * 4)
            confidence = min(0.9, (sum(s.confidence for s in signals) / len(signals)) * agreement_confidence)
        else:
            confidence = signals[0].confidence * 0.7 if signals else 0.3
        
        return our_probability, confidence
    
    def _get_recommendation(
        self,
        edge: float,
        confidence: float,
        market_price: float,
        signals: List[Signal],
    ) -> Tuple[str, str]:
        """Generate trading recommendation with reasoning."""
        
        # Build reasoning
        reasons = []
        
        if abs(edge) < self.min_edge:
            return "HOLD", f"Edge too small ({edge:.1%}). Need >{self.min_edge:.0%} edge."
        
        if confidence < self.min_confidence:
            return "HOLD", f"Confidence too low ({confidence:.1%}). Need >{self.min_confidence:.0%}."
        
        for signal in signals:
            reasons.append(f"{signal.source}: {signal.probability:.0%} ({signal.confidence:.0%} conf)")
        
        if edge > self.min_edge:
            reasoning = f"BUY YES: Market={market_price:.0%}, Model={market_price+edge:.0%}, Edge={edge:.1%}. " + "; ".join(reasons)
            return "BUY_YES", reasoning
        elif edge < -self.min_edge:
            reasoning = f"BUY NO: Market={market_price:.0%}, Model={market_price+edge:.0%}, Edge={abs(edge):.1%}. " + "; ".join(reasons)
            return "BUY_NO", reasoning
        
        return "HOLD", "No actionable edge found."
    
    def _calculate_position_size(self, edge: float, confidence: float) -> float:
        """
        Calculate position size using fractional Kelly.
        
        Kelly formula: f* = edge / odds
        We use quarter-Kelly for safety.
        """
        if abs(edge) < self.min_edge or confidence < self.min_confidence:
            return 0.0
        
        # Simplified Kelly: bet proportional to edge * confidence
        kelly_fraction = 0.25  # Quarter Kelly
        raw_size = abs(edge) * confidence * kelly_fraction
        
        # Cap at 5% per position
        return min(0.05, raw_size)
    
    def find_opportunities(
        self, 
        markets: List[Dict],
        min_edge: float = 0.10,
    ) -> List[MarketAnalysis]:
        """
        Scan all markets and return sorted opportunities.
        
        Returns markets with edge, sorted by expected value.
        """
        opportunities = []
        
        for market in markets:
            try:
                analysis = self.analyze_market(market)
                
                if abs(analysis.edge) >= min_edge and analysis.confidence >= self.min_confidence:
                    opportunities.append(analysis)
                    
            except Exception as e:
                logger.debug(f"Error analyzing {market.get('ticker')}: {e}")
        
        # Sort by expected value (edge * confidence)
        opportunities.sort(
            key=lambda a: abs(a.edge) * a.confidence,
            reverse=True
        )
        
        return opportunities
