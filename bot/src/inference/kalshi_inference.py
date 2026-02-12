"""
Kalshi live inference: run trained model against current open markets.

Usage:
    python -m src.inference.kalshi_inference --model models/best_model_run_162
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from ..agents import PPOAgent
from ..data.sources.kalshi import KalshiAdapter
from ..environment.kalshi_env import KalshiEventEnv
from ..core.logger import get_logger


logger = get_logger(__name__)


@dataclass
class MarketSignal:
    """Recommended action for a Kalshi market."""
    ticker: str
    event_ticker: str
    action: str  # 'BUY_YES', 'BUY_NO', 'HOLD'
    confidence: float  # 0-1
    yes_price: float
    market_data: Dict


class KalshiInference:
    """
    Run trained RL model on live Kalshi markets and generate trading signals.
    """

    def __init__(
        self,
        model_path: str,
        adapter: KalshiAdapter,
        policy_type: str = "MlpPolicy",
        confidence_threshold: float = 0.6,
    ):
        """
        Args:
            model_path: Path to trained .zip model
            adapter: KalshiAdapter instance for live data
            policy_type: MlpPolicy or MlpLstmPolicy
            confidence_threshold: Min confidence to trade (0-1)
        """
        self.model_path = model_path
        self.adapter = adapter
        self.confidence_threshold = confidence_threshold

        # Create a dummy env for model loading (we'll use custom obs)
        # In production, you'd want to pass live market data through an env wrapper
        dummy_env = self._create_dummy_env()
        self.agent = PPOAgent(env=dummy_env, use_gpu=False, policy_type=policy_type)
        self.agent.load(model_path)
        logger.info(f"Loaded model from {model_path}")

    def _create_dummy_env(self) -> KalshiEventEnv:
        """Create minimal env for model loading."""
        import pandas as pd
        # Minimal dummy data — will be replaced by live obs
        dummy_df = pd.DataFrame([{
            "ticker": "DUMMY-1",
            "event_ticker": "DUMMY-EVT",
            "series_ticker": "DUMMY",
            "last_price": 50,
            "yes_bid": 49,
            "yes_ask": 51,
            "volume": 100,
            "open_interest": 200,
            "liquidity": 1000,
            "result": "yes",
            "outcome": 1,
            "expiration_value": 50000,
            "strike_type": "greater",
            "floor_strike": 49000,
            "cap_strike": None,
            "open_time": None,
            "close_time": None,
        }])
        return KalshiEventEnv(settled_markets=dummy_df, initial_capital=25.0)

    def _market_to_obs(self, market: Dict) -> np.ndarray:
        """
        Convert live Kalshi market dict to model observation.
        
        This mimics KalshiEventEnv._get_obs() but for live data.
        """
        obs = np.zeros(17, dtype=np.float32)
        
        yes_price = market.get("last_price", 50)
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 100)
        spread = max(0, yes_ask - yes_bid)
        volume = market.get("volume", 0)
        oi = market.get("open_interest", 0)
        liquidity = market.get("liquidity", 0)
        prev_price = market.get("previous_price", yes_price)

        # For live markets, we don't have expiration_value yet
        # Use mid-strike as proxy
        floor_s = market.get("floor_strike")
        cap_s = market.get("cap_strike")
        st = market.get("strike_type", "")
        floor_ok = floor_s is not None and np.isfinite(floor_s)
        cap_ok = cap_s is not None and np.isfinite(cap_s)
        
        if st == "greater" and floor_ok:
            strike_ref = floor_s
        elif st == "less" and cap_ok:
            strike_ref = cap_s
        elif floor_ok and cap_ok:
            strike_ref = (floor_s + cap_s) / 2
        else:
            strike_ref = 50000  # fallback

        # Live markets: set strike_dist=0, strike_dir based on type
        strike_dir = 1.0 if st == "greater" else -1.0 if st == "less" else 0.0

        # Fill observation vector
        obs[0] = np.clip(yes_price / 100.0, 0, 1)
        obs[1] = np.clip(spread / 100.0, 0, 1)
        obs[2] = np.clip(np.log1p(volume) / 10.0, 0, 1)
        obs[3] = np.clip(np.log1p(oi) / 10.0, 0, 1)
        obs[4] = 0.5  # time_to_expiry: unknown for live, use midpoint
        obs[5] = 0.0  # strike_dist: unknown without settlement
        obs[6] = strike_dir
        obs[7] = np.clip(yes_price / 100.0, 0, 1)
        obs[8] = np.clip((yes_price - prev_price) / 100.0, -1, 1)
        obs[9] = np.clip(np.log1p(liquidity) / 15.0, 0, 1)
        obs[10] = 0.0  # contract_index: single market inference
        obs[11] = 1.0  # capital_ratio: assume full capital
        obs[12] = 0.0  # num_positions: assume none
        obs[13] = 0.0  # total_exposure
        obs[14] = 0.0  # unrealized_edge
        obs[15] = 0.0  # win_rate: no history
        obs[16] = 0.0  # episode_return: fresh

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def analyze_market(self, market: Dict) -> MarketSignal:
        """
        Analyze a single market and return trading signal.
        
        Args:
            market: Market dict from KalshiAdapter.get_market(ticker)
        
        Returns:
            MarketSignal with recommended action and confidence
        """
        obs = self._market_to_obs(market)
        action, _states = self.agent.predict(obs, deterministic=True)
        
        # Get action probabilities for confidence
        # (In SB3, the model returns logits; we can get distribution)
        try:
            # This is a bit hacky but works for MaskablePPO
            policy_output = self.agent.model.policy.forward(
                self.agent.model.policy.obs_to_tensor(obs.reshape(1, -1))[0],
                deterministic=False
            )
            action_probs = policy_output.distribution.distribution.probs.detach().cpu().numpy()[0]
            confidence = float(action_probs[action])
        except Exception:
            confidence = 0.7  # fallback

        action_map = {0: "HOLD", 1: "BUY_YES", 2: "BUY_NO"}
        action_name = action_map.get(int(action), "HOLD")

        return MarketSignal(
            ticker=market["ticker"],
            event_ticker=market.get("event_ticker", ""),
            action=action_name,
            confidence=confidence,
            yes_price=market.get("last_price", 50),
            market_data=market,
        )

    def scan_series(
        self,
        series_ticker: str,
        status: str = "open",
        limit: int = 20,
    ) -> List[MarketSignal]:
        """
        Scan all open markets in a series and generate signals.
        
        Args:
            series_ticker: e.g. 'KXBTC', 'KXETH'
            status: 'open' or 'closed'
            limit: max markets to scan
        
        Returns:
            List of MarketSignal sorted by confidence (descending)
        """
        try:
            response = self.adapter._request("GET", "/markets", params={
                "series_ticker": series_ticker,
                "status": status,
                "limit": limit,
            })
            markets = response.get("markets", [])
        except Exception as e:
            logger.error(f"Failed to fetch {series_ticker} markets: {e}")
            return []

        signals = []
        for market in markets:
            try:
                signal = self.analyze_market(market)
                if signal.action != "HOLD" and signal.confidence >= self.confidence_threshold:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Failed to analyze {market.get('ticker', '?')}: {e}")

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
