"""
Statistical edge detection for Kalshi prediction markets.

Finds mispriced markets using fundamental analysis, arbitrage,
and market microstructure signals — no RL needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
import time
import numpy as np

from ..core.logger import get_logger


logger = get_logger(__name__)


@dataclass
class Edge:
    """A detected pricing inefficiency in a Kalshi market."""
    ticker: str
    event_ticker: str
    edge_type: str  # 'mispricing', 'spread_arb', 'momentum', 'liquidity'
    edge_value: float  # Expected profit margin (-1 to 1)
    confidence: float  # How confident in this edge (0-1)
    recommended_side: str  # 'yes' or 'no'
    market_price: float  # Current YES price (0-100)
    fair_price: Optional[float]  # Our estimate of fair price
    reasoning: str  # Human-readable explanation
    market_data: Dict  # Full market snapshot


class StatisticalEdgeDetector:
    """
    Detect edges in Kalshi markets using statistical models.
    
    This is the "alpha" layer — finds where the market is wrong.
    RL will be the "execution" layer — decides when to trade it.
    """

    def __init__(
        self,
        min_edge: float = 0.05,
        min_liquidity: float = 100,
        max_spread: float = 10,
    ):
        """
        Args:
            min_edge: Minimum edge required to signal (5% = 0.05)
            min_liquidity: Min liquidity to consider tradeable
            max_spread: Max bid/ask spread to trade (cents)
        """
        self.min_edge = min_edge
        self.min_liquidity = min_liquidity
        self.max_spread = max_spread

        self._spot_cache: Dict[str, Tuple[float, float]] = {}  # asset -> (spot_price, ts)

    def _effective_liquidity(self, market: Dict) -> float:
        liquidity = float(market.get("liquidity", 0) or 0)
        if liquidity > 0:
            return liquidity
        open_interest = float(market.get("open_interest", 0) or 0)
        if open_interest > 0:
            return open_interest
        volume = float(market.get("volume", 0) or 0)
        return volume

    def _infer_asset(self, market: Dict) -> Optional[str]:
        haystack = " ".join(
            str(x)
            for x in (
                market.get("series_ticker", ""),
                market.get("ticker", ""),
                market.get("title", ""),
                market.get("subtitle", ""),
            )
        ).upper()
        for asset in ("BTC", "ETH", "SOL", "DOGE", "XRP"):
            if asset in haystack:
                return asset
        return None

    def _parse_ts(self, value: object) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except Exception:
                return None
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            try:
                # Accept "Z" suffix.
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    def _time_to_expiry_years(self, market: Dict) -> float:
        close_ts = self._parse_ts(market.get("close_time")) or self._parse_ts(market.get("expiration_time"))
        if close_ts is None:
            return 1.0 / 365.0  # assume ~1 day if unknown
        now = datetime.now(timezone.utc)
        seconds = max(60.0, (close_ts - now).total_seconds())
        return seconds / (365.0 * 24.0 * 3600.0)

    def _normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _estimate_vol(self, asset: str) -> float:
        # Annualized vol calibrated from historical Kalshi settlement data.
        return {
            "BTC": 0.56,
            "ETH": 0.70,
            "SOL": 0.74,
            "DOGE": 0.65,
            "XRP": 0.71,
            "INXU": 0.11,
            "EURUSD": 0.12,
        }.get(asset, 0.70)

    def _get_spot_price(self, asset: str, max_age_s: int = 30) -> Optional[float]:
        cached = self._spot_cache.get(asset)
        if cached and (time.time() - cached[1]) <= max_age_s:
            return cached[0]

        symbol = f"{asset}-USD"
        try:
            from ..data.sources.coinbase import CoinbaseAdapter

            spot = float(CoinbaseAdapter().get_latest_price(symbol))
            self._spot_cache[asset] = (spot, time.time())
            return spot
        except Exception as e:
            logger.warning(f"Failed to fetch spot for {asset}: {e}")
            return None

    def _extract_strike(self, market: Dict) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        strike_type = market.get("strike_type")
        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")

        if strike_type and (floor_strike is not None or cap_strike is not None):
            try:
                return (
                    str(strike_type),
                    float(floor_strike) if floor_strike is not None else None,
                    float(cap_strike) if cap_strike is not None else None,
                )
            except Exception:
                pass

        title = str(market.get("title", ""))
        subtitle = str(market.get("subtitle", ""))
        text = f"{title} {subtitle}".lower()

        # Extract numbers with commas: 95,000 -> 95000
        raw_numbers = [
            float(n.replace(",", ""))
            for n in re.findall(r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)", text)
        ]
        # Filter out obvious date/time/year tokens (e.g., "Feb 12", "2026", "5pm").
        numbers = [n for n in raw_numbers if n >= 1000 and not (2000 <= n <= 2100)]
        if not numbers:
            return None, None, None

        if any(k in text for k in ("between", "from")) and any(k in text for k in ("and", "to")) and len(numbers) >= 2:
            lo, hi = sorted(numbers[:2])
            return "between", lo, hi

        if any(k in text for k in ("above", "greater", ">")):
            return "greater", numbers[0], None
        if any(k in text for k in ("below", "less", "<")):
            return "less", numbers[0], None

        # Fallback: treat first number as strike for "greater".
        return "greater", numbers[0], None

    def _enrich_crypto_strikes(self, markets: List[Dict]) -> None:
        """Populate strike fields for KX* crypto markets by parsing Kalshi tickers."""
        by_event: Dict[str, List[Dict]] = {}
        for m in markets:
            series = str(m.get("series_ticker", "") or "")
            if not series.startswith("KX"):
                continue
            evt = str(m.get("event_ticker", "") or "")
            if not evt:
                evt = str(m.get("ticker", "")).rsplit("-", 1)[0]
                m["event_ticker"] = evt
            by_event.setdefault(evt, []).append(m)

        for evt, group in by_event.items():
            parsed: List[Tuple[Dict, str, float]] = []
            for m in group:
                suffix = str(m.get("ticker", "")).split("-")[-1]
                if not suffix:
                    continue
                kind = suffix[0].upper()
                if kind not in {"T", "B"}:
                    continue
                try:
                    val = float(suffix[1:])
                except Exception:
                    continue
                parsed.append((m, kind, val))

            if not parsed:
                continue

            b_vals = sorted(v for (_m, k, v) in parsed if k == "B")
            step = None
            if len(b_vals) >= 2:
                diffs = [b_vals[i + 1] - b_vals[i] for i in range(len(b_vals) - 1) if (b_vals[i + 1] - b_vals[i]) > 0]
                if diffs:
                    step = float(np.median(diffs))

            half = (step / 2.0) if (step and step > 0) else None
            min_b = min(b_vals) if b_vals else None
            max_b = max(b_vals) if b_vals else None

            for m, kind, val in parsed:
                # Don't overwrite if already present.
                if m.get("strike_type") and (m.get("floor_strike") is not None or m.get("cap_strike") is not None):
                    continue

                if kind == "B":
                    if half is None:
                        continue
                    m["strike_type"] = "between"
                    m["floor_strike"] = float(val - half)
                    # Kalshi buckets usually look like [x, x+step) with 0.01 granularity; approximate.
                    m["cap_strike"] = float(val + half - 0.01)
                    continue

                # Tail markets: infer less/greater using B ladder if available.
                if kind == "T":
                    if min_b is not None and val < min_b:
                        m["strike_type"] = "less"
                        m["floor_strike"] = None
                        m["cap_strike"] = float(val)
                    elif max_b is not None and val > max_b:
                        m["strike_type"] = "greater"
                        m["floor_strike"] = float(val)
                        m["cap_strike"] = None
                    else:
                        # If we can't place it, default to greater.
                        m["strike_type"] = "greater"
                        m["floor_strike"] = float(val)
                        m["cap_strike"] = None

    def detect_spread_arbitrage(self, market: Dict) -> Optional[Edge]:
        """
        Detect arbitrage: YES + NO prices should sum to 100.
        
        If YES bid=45, NO bid=60, you can buy both for 105 and guarantee
        payout of 100 → -5 loss. But if someone else has YES ask=40, NO ask=50,
        that's 90 total → guaranteed 10 profit.
        """
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 100)
        no_bid = market.get("no_bid", 0)
        no_ask = market.get("no_ask", 100)

        # Can we buy YES + NO for < 100?
        buy_both = yes_ask + no_ask
        if buy_both < 95:  # 5 cent profit minimum
            edge_value = (100 - buy_both) / 100.0
            return Edge(
                ticker=market["ticker"],
                event_ticker=market.get("event_ticker", ""),
                edge_type="spread_arb",
                edge_value=edge_value,
                confidence=1.0,  # Arb is guaranteed
                recommended_side="both",  # Buy both YES and NO
                market_price=yes_ask,
                fair_price=50.0,
                reasoning=f"Buy YES@{yes_ask} + NO@{no_ask} = {buy_both} < 100 → ${100-buy_both} profit",
                market_data=market,
            )

        # Can we sell YES + NO for > 100?
        sell_both = yes_bid + no_bid
        if sell_both > 105:  # 5 cent profit minimum
            edge_value = (sell_both - 100) / 100.0
            return Edge(
                ticker=market["ticker"],
                event_ticker=market.get("event_ticker", ""),
                edge_type="spread_arb",
                edge_value=edge_value,
                confidence=1.0,
                recommended_side="both_short",
                market_price=yes_bid,
                fair_price=50.0,
                reasoning=f"Sell YES@{yes_bid} + NO@{no_bid} = {sell_both} > 100 → ${sell_both-100} profit",
                market_data=market,
            )

        return None

    def detect_mean_reversion(self, market: Dict) -> Optional[Edge]:
        """
        Detect extreme prices that will likely revert.
        
        For range markets (BTC between 95K-100K), if YES is trading at 95,
        that's overpriced unless we're very confident BTC stays in range.
        Use simple heuristic: extreme prices (>85 or <15) often revert.
        """
        series = str(market.get("series_ticker", "") or "")
        if series.startswith("KX"):
            return None

        last_price = market.get("last_price", 50)
        spread = market.get("yes_ask", 100) - market.get("yes_bid", 0)

        if spread > self.max_spread:
            return None  # Too wide to trade

        # Overpriced YES (likely to drop)
        if last_price > 85:
            edge_value = (last_price - 75) / 100.0  # How much likely to drop
            return Edge(
                ticker=market["ticker"],
                event_ticker=market.get("event_ticker", ""),
                edge_type="mean_reversion",
                edge_value=edge_value,
                confidence=0.6,  # Medium confidence
                recommended_side="no",
                market_price=last_price,
                fair_price=75.0,
                reasoning=f"YES@{last_price} is extreme, likely overpriced → fade it",
                market_data=market,
            )

        # Underpriced YES (likely to rise)
        if last_price < 15:
            edge_value = (25 - last_price) / 100.0
            return Edge(
                ticker=market["ticker"],
                event_ticker=market.get("event_ticker", ""),
                edge_type="mean_reversion",
                edge_value=edge_value,
                confidence=0.6,
                recommended_side="yes",
                market_price=last_price,
                fair_price=25.0,
                reasoning=f"YES@{last_price} is extreme, likely underpriced → buy it",
                market_data=market,
            )

        return None

    def detect_liquidity_edge(self, market: Dict) -> Optional[Edge]:
        """
        Detect markets with very tight spreads + high volume = market maker opportunity.
        
        If spread is 1 cent and volume is high, you can provide liquidity
        and capture spread. This is more of a market making signal.
        """
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 100)
        volume = market.get("volume", 0)
        liquidity = market.get("liquidity", 0)

        spread = yes_ask - yes_bid

        if spread <= 2 and liquidity > 1000 and volume > 500:
            # Tight spread + high activity = good market to make
            mid_price = (yes_bid + yes_ask) / 2
            
            # Slight edge toward the side with more depth
            if yes_bid > 50:
                side = "no"
                fair = yes_bid - 1  # Sell NO just inside bid
            else:
                side = "yes"
                fair = yes_ask + 1  # Buy YES just inside ask

            return Edge(
                ticker=market["ticker"],
                event_ticker=market.get("event_ticker", ""),
                edge_type="liquidity",
                edge_value=spread / 100.0,  # Capture spread
                confidence=0.7,
                recommended_side=side,
                market_price=mid_price,
                fair_price=fair,
                reasoning=f"Tight spread ({spread}¢) + volume ({volume}) = market making opportunity",
                market_data=market,
            )

        return None

    def detect_crypto_range_edge(self, market: Dict) -> Optional[Edge]:
        """
        Crypto-specific: spot-vs-strike mispricing.

        Uses Coinbase spot price + a simple lognormal model to estimate a fair probability
        that the event condition is true at expiry, then compares to Kalshi YES price.

        This is intentionally lightweight (no heavy feature engineering) and is meant
        to catch the "obvious" mispricings in crypto range/threshold markets.
        """
        asset = self._infer_asset(market)
        if asset is None:
            return None

        strike_type, floor_strike, cap_strike = self._extract_strike(market)
        if strike_type is None or (floor_strike is None and cap_strike is None):
            return None

        yes_bid = float(market.get("yes_bid", 0) or 0)
        yes_ask = float(market.get("yes_ask", 100) or 100)
        spread = yes_ask - yes_bid
        if spread > self.max_spread:
            return None

        # Be careful: last_price=0 is a real value (no trades / worthless),
        # not "missing". Only default to 50 when the key is truly absent.
        raw_price = market.get("last_price")
        if raw_price is None:
            raw_price = market.get("yes_price")
        if raw_price is None:
            return None  # Can't evaluate without a market price
        last_price = float(raw_price)
        market_prob = float(np.clip(last_price / 100.0, 0.0, 1.0))

        spot = self._get_spot_price(asset)
        if spot is None or spot <= 0:
            return None

        t = self._time_to_expiry_years(market)
        vol = self._estimate_vol(asset)
        sigma_sqrt_t = max(1e-6, vol * math.sqrt(max(t, 1e-6)))

        def prob_above(strike: float) -> float:
            k = max(1e-9, float(strike))
            z = (math.log(k / spot) + 0.5 * (vol**2) * t) / sigma_sqrt_t
            return float(1.0 - self._normal_cdf(z))

        def prob_below(strike: float) -> float:
            k = max(1e-9, float(strike))
            z = (math.log(k / spot) + 0.5 * (vol**2) * t) / sigma_sqrt_t
            return float(self._normal_cdf(z))

        if strike_type == "greater" and floor_strike is not None:
            fair_prob = prob_above(floor_strike)
            strike_desc = f"> {floor_strike:,.0f}"
        elif strike_type == "less" and floor_strike is not None:
            fair_prob = prob_below(floor_strike)
            strike_desc = f"< {floor_strike:,.0f}"
        elif strike_type == "between" and floor_strike is not None and cap_strike is not None:
            lo, hi = sorted((float(floor_strike), float(cap_strike)))
            fair_prob = max(0.0, min(1.0, prob_below(hi) - prob_below(lo)))
            strike_desc = f"between {lo:,.0f} and {hi:,.0f}"
        else:
            return None

        fair_price = fair_prob * 100.0
        edge_value = fair_prob - market_prob
        if abs(edge_value) < self.min_edge:
            return None

        recommended_side = "yes" if edge_value > 0 else "no"
        liquidity_score = float(np.clip(self._effective_liquidity(market) / max(self.min_liquidity, 1.0), 0.0, 2.0))
        confidence = float(np.clip(min(1.0, abs(edge_value) / 0.20) * 0.7 + 0.15 * liquidity_score, 0.1, 0.95))

        return Edge(
            ticker=market.get("ticker", ""),
            event_ticker=market.get("event_ticker", ""),
            edge_type="crypto_spot_mispricing",
            edge_value=float(abs(edge_value)),
            confidence=confidence,
            recommended_side=recommended_side,
            market_price=last_price,
            fair_price=fair_price,
            reasoning=(
                f"{asset} spot=${spot:,.0f}, strike {strike_desc}, T≈{t*365.0:.1f}d, vol≈{vol:.0%}: "
                f"fair={fair_price:.0f}¢ vs mkt={last_price:.0f}¢"
            ),
            market_data=market,
        )

    def scan_market(self, market: Dict) -> List[Edge]:
        """
        Run all edge detectors on a single market.
        
        Returns all detected edges, sorted by edge_value * confidence.
        """
        liquidity = self._effective_liquidity(market)
        if liquidity < self.min_liquidity:
            return []

        edges = []
        
        # Run all detectors
        for detector in [
            self.detect_spread_arbitrage,
            self.detect_mean_reversion,
            self.detect_liquidity_edge,
            self.detect_crypto_range_edge,
        ]:
            try:
                edge = detector(market)
                if edge and edge.edge_value >= self.min_edge:
                    edges.append(edge)
            except Exception as e:
                logger.warning(f"Edge detector {detector.__name__} failed: {e}")

        # Sort by expected value (edge * confidence)
        edges.sort(key=lambda e: e.edge_value * e.confidence, reverse=True)
        return edges

    def scan_series(
        self,
        markets: List[Dict],
        top_n: int = 10,
    ) -> List[Edge]:
        """
        Scan a list of markets and return top edges.
        
        Args:
            markets: List of market dicts from Kalshi API
            top_n: Return top N edges
        
        Returns:
            List of Edge objects, sorted by edge_value * confidence
        """
        # Preprocess crypto markets so strike fields exist even for live API objects.
        self._enrich_crypto_strikes(markets)

        all_edges = []
        for market in markets:
            edges = self.scan_market(market)
            all_edges.extend(edges)

        # Cross-market detectors that require seeing the full ladder
        all_edges.extend(self.detect_strike_dominance(markets))

        all_edges.sort(key=lambda e: e.edge_value * e.confidence, reverse=True)
        return all_edges[:top_n]

    # ------------------------------------------------------------------
    # Cross-market detectors (require seeing the full event ladder)
    # ------------------------------------------------------------------

    def detect_strike_dominance(self, markets: List[Dict]) -> List[Edge]:
        """
        Detect strike-dominance violations within the same event.

        Rule: for "greater" markets in the same event,
          price("BTC > 65K") must be >= price("BTC > 68K").
        A violation is a structural arbitrage.

        Similarly, within "between" buckets the prices should roughly form
        a probability distribution (sum ≈ 100).  Huge deviations are edges.
        """
        edges: List[Edge] = []

        # Group by event
        by_event: Dict[str, List[Dict]] = {}
        for m in markets:
            evt = m.get("event_ticker", "")
            if not evt:
                continue
            by_event.setdefault(evt, []).append(m)

        for evt, group in by_event.items():

            # --- Greater-than dominance ---
            gt_markets = [
                m for m in group
                if str(m.get("strike_type", "")).startswith("greater")
                and m.get("floor_strike") is not None
            ]
            gt_markets.sort(key=lambda m: float(m["floor_strike"]))

            for i in range(len(gt_markets) - 1):
                lo = gt_markets[i]   # lower strike → should be MORE expensive
                hi = gt_markets[i + 1]  # higher strike → should be LESS expensive

                lo_price = float(lo.get("last_price") if lo.get("last_price") is not None else -1)
                hi_price = float(hi.get("last_price") if hi.get("last_price") is not None else -1)
                if lo_price < 0 or hi_price < 0:
                    continue

                # Violation: higher strike priced MORE than lower strike
                if hi_price > lo_price + 1:  # 1¢ tolerance
                    violation = (hi_price - lo_price) / 100.0
                    lo_strike = float(lo["floor_strike"])
                    hi_strike = float(hi["floor_strike"])
                    edges.append(Edge(
                        ticker=hi["ticker"],
                        event_ticker=evt,
                        edge_type="strike_dominance",
                        edge_value=violation,
                        confidence=0.95,
                        recommended_side="no",  # Sell the overpriced higher strike
                        market_price=hi_price,
                        fair_price=lo_price,
                        reasoning=(
                            f"'{evt}' > {hi_strike:,.0f} @ {hi_price:.0f}¢ but "
                            f"> {lo_strike:,.0f} @ {lo_price:.0f}¢ — dominance violation"
                        ),
                        market_data=hi,
                    ))
                    edges.append(Edge(
                        ticker=lo["ticker"],
                        event_ticker=evt,
                        edge_type="strike_dominance",
                        edge_value=violation,
                        confidence=0.95,
                        recommended_side="yes",  # Buy the underpriced lower strike
                        market_price=lo_price,
                        fair_price=hi_price,
                        reasoning=(
                            f"'{evt}' > {lo_strike:,.0f} @ {lo_price:.0f}¢ but "
                            f"> {hi_strike:,.0f} @ {hi_price:.0f}¢ — dominance violation"
                        ),
                        market_data=lo,
                    ))

            # --- Between-bucket sum check (disabled: 7% win rate in backtest) ---
            # bw_markets = [...] 
            # Ladder overpriced detection has no predictive power on historical data.

        return edges
