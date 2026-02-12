"""
Hybrid Kalshi trading engine: Statistical edges + RL timing + Kelly sizing.

Combines:
- StatisticalEdgeDetector: finds mispriced markets (alpha)
- RL model: decides execution timing (microstructure)
- Kelly criterion: position sizing (risk management)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..agents import PPOAgent
from ..strategies.kalshi_edges import StatisticalEdgeDetector, Edge
from ..core.logger import get_logger


logger = get_logger(__name__)


@dataclass
class TradeSignal:
    """Final trade recommendation combining edge + timing + sizing."""
    ticker: str
    action: str  # 'BUY_YES', 'BUY_NO', 'HOLD'
    contracts: int  # Position size
    edge: Edge  # The detected edge
    rl_confidence: float  # RL timing score (0-1)
    kelly_fraction: float  # Kelly-optimal bet size
    expected_value: float  # edge * kelly_fraction
    reasoning: str


def kelly_criterion(
    edge: float,
    win_prob: float,
    bankroll: float,
    max_fraction: float = 0.25,
) -> float:
    """
    Kelly criterion for binary bets.
    
    Args:
        edge: Expected profit margin (e.g., 0.10 = 10%)
        win_prob: Probability of winning (0-1)
        bankroll: Total capital
        max_fraction: Max fraction of bankroll to risk
    
    Returns:
        Dollar amount to bet
    """
    if edge <= 0 or bankroll <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0

    # Kelly formula for binary outcomes: f = (p*b - q) / b
    # where p=win_prob, q=1-p, b=odds
    # For prediction markets: b = (100-price)/price for YES, price/(100-price) for NO
    
    # Simplified: f = edge / variance
    # For binary: variance ≈ 1 (max variance)
    kelly_f = edge
    
    # Cap at max_fraction to avoid over-betting
    kelly_f = min(kelly_f, max_fraction)
    
    # Half-Kelly for safety
    kelly_f *= 0.5
    
    return kelly_f * bankroll


class HybridKalshiEngine:
    """
    Hybrid trading engine combining statistical edges with RL execution timing.
    
    Flow:
    1. Statistical detector finds edges (mispricing, arb, etc.)
    2. RL model scores each edge for timing/microstructure
    3. Kelly criterion sizes the position
    4. Execute if RL confidence > threshold
    """

    def __init__(
        self,
        rl_model_path: Optional[str] = None,
        rl_threshold: float = 0.6,
        min_edge: float = 0.05,
        bankroll: float = 25.0,
        max_kelly: float = 0.25,
    ):
        """
        Args:
            rl_model_path: Path to trained RL model (None = no RL filtering)
            rl_threshold: Min RL confidence to execute (0-1)
            min_edge: Min statistical edge to consider
            bankroll: Total capital
            max_kelly: Max Kelly fraction (0.25 = 25% of bankroll)
        """
        self.rl_threshold = rl_threshold
        self.bankroll = bankroll
        self.max_kelly = max_kelly

        self.edge_detector = StatisticalEdgeDetector(
            min_edge=min_edge,
            min_liquidity=100,
            max_spread=10,
        )

        # Load RL model if provided
        self.rl_model = None
        if rl_model_path:
            try:
                from ..environment.kalshi_env import KalshiEventEnv
                import pandas as pd
                
                # Create dummy env for model loading
                dummy_df = pd.DataFrame([{
                    "ticker": "DUMMY", "event_ticker": "DUMMY",
                    "series_ticker": "DUMMY", "last_price": 50,
                    "yes_bid": 49, "yes_ask": 51, "volume": 100,
                    "open_interest": 200, "liquidity": 1000,
                    "result": "yes", "outcome": 1,
                    "expiration_value": 50000, "strike_type": "greater",
                    "floor_strike": 49000, "cap_strike": None,
                    "open_time": None, "close_time": None,
                }])
                dummy_env = KalshiEventEnv(settled_markets=dummy_df, initial_capital=25.0)
                
                self.rl_model = PPOAgent(env=dummy_env, use_gpu=False)
                self.rl_model.load(rl_model_path)
                logger.info(f"Loaded RL model from {rl_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load RL model: {e}")
                self.rl_model = None

    def _market_to_obs(self, market: Dict) -> np.ndarray:
        """Convert market dict to RL observation vector (17-dim)."""
        obs = np.zeros(17, dtype=np.float32)
        
        yes_price = market.get("last_price", 50)
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 100)
        spread = max(0, yes_ask - yes_bid)
        volume = market.get("volume", 0)
        oi = market.get("open_interest", 0)
        liquidity = market.get("liquidity", 0)
        prev_price = market.get("previous_price", yes_price)

        obs[0] = np.clip(yes_price / 100.0, 0, 1)
        obs[1] = np.clip(spread / 100.0, 0, 1)
        obs[2] = np.clip(np.log1p(volume) / 10.0, 0, 1)
        obs[3] = np.clip(np.log1p(oi) / 10.0, 0, 1)
        obs[4] = 0.5  # time_to_expiry unknown
        obs[5] = 0.0  # strike_dist unknown
        obs[6] = 0.0  # strike_dir
        obs[7] = np.clip(yes_price / 100.0, 0, 1)
        obs[8] = np.clip((yes_price - prev_price) / 100.0, -1, 1)
        obs[9] = np.clip(np.log1p(liquidity) / 15.0, 0, 1)
        obs[10:] = 0.0  # Portfolio state (empty for live)

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _get_rl_confidence(self, market: Dict) -> float:
        """
        Get RL model's confidence score for this market.
        
        Returns 1.0 if no RL model (always trade edges).
        """
        if self.rl_model is None:
            return 1.0  # No RL filter = always confident

        try:
            obs = self._market_to_obs(market)
            action, _states = self.rl_model.predict(obs, deterministic=True)
            
            # Get action probabilities for confidence
            policy_output = self.rl_model.model.policy.forward(
                self.rl_model.model.policy.obs_to_tensor(obs.reshape(1, -1))[0],
                deterministic=False
            )
            action_probs = policy_output.distribution.distribution.probs.detach().cpu().numpy()[0]
            
            # If action is HOLD (0), return low confidence
            # If action is BUY (1 or 2), return that action's probability
            if action == 0:  # HOLD
                return 0.0
            return float(action_probs[action])
        except Exception as e:
            logger.warning(f"RL confidence failed: {e}")
            return 0.5  # Fallback

    def evaluate_edge(self, edge: Edge) -> Optional[TradeSignal]:
        """
        Evaluate a detected edge and return trade signal if conditions met.
        
        Combines:
        - Statistical edge (from detector)
        - RL timing (microstructure favorability)
        - Kelly sizing (risk management)
        """
        # Get RL confidence score
        rl_conf = self._get_rl_confidence(edge.market_data)

        # Filter: RL must agree this is a good time to trade
        if rl_conf < self.rl_threshold:
            return None

        # Position sizing via Kelly
        # Very rough mapping: turn edge magnitude into a slightly-shifted win prob.
        win_prob = float(np.clip(0.5 + (edge.edge_value * edge.confidence), 0.01, 0.99))
        bet_amount = kelly_criterion(
            edge=edge.edge_value,
            win_prob=win_prob,
            bankroll=self.bankroll,
            max_fraction=self.max_kelly,
        )

        if bet_amount < 0.10:  # Less than 10 cents = skip
            return None

        # Convert $ to contracts (each contract costs price/100 dollars)
        price = edge.market_price / 100.0
        if price <= 0:
            return None
        contracts = int(bet_amount / price)
        contracts = max(1, min(contracts, 100))  # 1-100 contracts

        # Map edge recommendation to action
        if edge.recommended_side == "yes":
            action = "BUY_YES"
        elif edge.recommended_side == "no":
            action = "BUY_NO"
        else:
            action = "HOLD"  # Arb or other complex trades

        return TradeSignal(
            ticker=edge.ticker,
            action=action,
            contracts=contracts,
            edge=edge,
            rl_confidence=rl_conf,
            kelly_fraction=bet_amount / self.bankroll,
            expected_value=float(edge.edge_value * edge.confidence * rl_conf),
            reasoning=f"{edge.edge_type}: {edge.reasoning} | RL timing={rl_conf:.0%} | Kelly=${bet_amount:.2f}",
        )

    def scan_and_rank(
        self,
        markets: List[Dict],
        top_n: int = 5,
    ) -> List[TradeSignal]:
        """
        Scan markets, detect edges, filter by RL, size by Kelly, return top signals.
        
        This is the main entry point for the hybrid strategy.
        """
        # Step 1: Statistical edge detection
        edges = self.edge_detector.scan_series(markets, top_n=20)
        
        if not edges:
            logger.info("No statistical edges found")
            return []

        logger.info(f"Found {len(edges)} statistical edges")

        # Step 2: RL filtering + Kelly sizing
        signals = []
        for edge in edges:
            signal = self.evaluate_edge(edge)
            if signal:
                signals.append(signal)

        # Step 3: Rank by expected value
        signals.sort(key=lambda s: s.expected_value, reverse=True)
        
        logger.info(f"Generated {len(signals)} tradeable signals (after RL filter)")
        return signals[:top_n]
