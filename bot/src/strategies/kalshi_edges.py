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
        min_liquidity: float = 0,
        max_spread: float = 50,
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

    def _effective_liquidity(self, market: Dict) -> float:
        liquidity = float(market.get("liquidity", 0) or 0)
        if liquidity > 0:
            return liquidity
        open_interest = float(market.get("open_interest", 0) or 0)
        if open_interest > 0:
            return open_interest
        volume = float(market.get("volume", 0) or 0)
        return volume

    _ASSET_KEYWORDS: List[Tuple[str, List[str]]] = [
        ("BTC", ["BTC", "BITCOIN"]),
        ("ETH", ["ETH", "ETHEREUM"]),
        ("SOL", ["KXSOL", " SOL "]),
        ("DOGE", ["DOGE"]),
        ("XRP", ["XRP"]),
        ("SPX", ["KXINXU", "KXINX-", "INXI", " S&P ", "S&P 500", "SP500"]),
        ("EURUSD", ["KXEURUSD", "EUR/USD", "EURUSD"]),
        ("WTI", ["KXWTIH", "WTI", "CRUDE OIL"]),
        ("TNOTE", ["TNOTED", "TREASURY", "10Y YIELD", "10-YEAR"]),
    ]

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
        for asset, keywords in self._ASSET_KEYWORDS:
            for kw in keywords:
                if kw in haystack:
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
        from ..data.sources.spot_feeds import get_annualized_vol
        return get_annualized_vol(asset)

    def _get_spot_price(self, asset: str, max_age_s: int = 30) -> Optional[float]:
        from ..data.sources.spot_feeds import get_spot_price
        return get_spot_price(asset, max_age_s=max_age_s)

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

        raw_numbers = [
            float(n.replace(",", ""))
            for n in re.findall(r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)", text)
        ]

        asset = self._infer_asset(market)
        if asset in ("EURUSD", "WTI", "TNOTE"):
            numbers = [n for n in raw_numbers if n > 0 and not (2000 <= n <= 2100)]
        else:
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

    def _enrich_kx_strikes(self, markets: List[Dict]) -> None:
        """Populate strike fields for KX* markets (crypto, index, FX, commodity) by parsing Kalshi tickers."""
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

    def detect_range_edge(self, market: Dict) -> Optional[Edge]:
        """
        Spot-vs-strike mispricing for any asset with a live price feed.

        Uses a lognormal model to estimate a fair probability that the event
        condition is true at expiry, then compares to the Kalshi YES price.

        Supports crypto (BTC, ETH, SOL, DOGE, XRP), indices (S&P 500),
        FX (EUR/USD), and commodities (WTI oil).
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

        volume = float(market.get("volume", 0) or 0)
        open_interest = float(market.get("open_interest", 0) or 0)

        # Stale market detection: if no volume AND no open interest AND
        # bid/ask is the default 0/100 spread, the "price" is meaningless.
        if volume == 0 and open_interest == 0 and yes_bid == 0 and yes_ask == 100:
            return None

        raw_price = market.get("last_price")
        if raw_price is None:
            raw_price = market.get("yes_price")
        if raw_price is None:
            return None
        last_price = float(raw_price)

        # Use mid-price from bid/ask when available and there's actual
        # liquidity, rather than trusting last_price which can be stale.
        if yes_bid > 0 and yes_ask < 100:
            mid_price = (yes_bid + yes_ask) / 2.0
        else:
            mid_price = last_price

        market_prob = float(np.clip(mid_price / 100.0, 0.0, 1.0))

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

        # Penalize edges from low-activity markets: scale confidence down
        # so high-liquidity markets rank above ghost markets.
        activity = volume + open_interest
        liquidity_penalty = float(np.clip(activity / 50.0, 0.1, 1.0))

        recommended_side = "yes" if edge_value > 0 else "no"
        liquidity_score = float(np.clip(self._effective_liquidity(market) / max(self.min_liquidity, 1.0), 0.0, 2.0))
        confidence = float(np.clip(
            min(1.0, abs(edge_value) / 0.20) * 0.7 + 0.15 * liquidity_score,
            0.1, 0.95,
        )) * liquidity_penalty

        return Edge(
            ticker=market.get("ticker", ""),
            event_ticker=market.get("event_ticker", ""),
            edge_type="spot_vs_strike",
            edge_value=float(abs(edge_value)),
            confidence=confidence,
            recommended_side=recommended_side,
            market_price=mid_price,
            fair_price=fair_price,
            reasoning=(
                f"{asset} spot=${spot:,.0f}, strike {strike_desc}, T≈{t*365.0:.1f}d, vol≈{vol:.0%}: "
                f"fair={fair_price:.0f}¢ vs mkt={mid_price:.0f}¢ (bid={yes_bid:.0f} ask={yes_ask:.0f} vol={volume:.0f})"
            ),
            market_data=market,
        )

    # Backward-compatible alias
    detect_crypto_range_edge = detect_range_edge

    # ------------------------------------------------------------------
    # Macro data edge detector (CPI, NFP, Fed)
    # ------------------------------------------------------------------

    _MACRO_INDICATORS = {
        "cpi": {
            "keywords": ["cpi", "inflation", "consumer price"],
            "unit": "MoM %",
        },
        "nfp": {
            "keywords": ["nonfarm", "payroll", "jobs", "employment"],
            "unit": "K jobs",
        },
        "fed": {
            "keywords": ["fed", "fomc", "rate cut", "rate hike", "interest rate", "federal funds"],
            "unit": "bps",
        },
    }

    def _classify_macro_market(self, market: Dict) -> Optional[str]:
        title = (market.get("title", "") + " " + market.get("subtitle", "")).lower()
        for indicator, meta in self._MACRO_INDICATORS.items():
            if any(kw in title for kw in meta["keywords"]):
                return indicator
        return None

    def _parse_macro_bucket(self, market: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Parse bucket bounds from a macro market title/strike fields."""
        strike_type, floor_strike, cap_strike = self._extract_strike(market)
        if strike_type == "between" and floor_strike is not None and cap_strike is not None:
            return (float(floor_strike), float(cap_strike))
        if strike_type == "greater" and floor_strike is not None:
            return (float(floor_strike), None)
        if strike_type == "less" and (cap_strike is not None or floor_strike is not None):
            bound = float(cap_strike) if cap_strike is not None else float(floor_strike)
            return (None, bound)

        title = (market.get("title", "") + " " + market.get("subtitle", "")).lower()
        numbers = [
            float(n.replace(",", ""))
            for n in re.findall(r"(-?\d+(?:\.\d+)?)", title)
            if not (2000 <= float(n.replace(",", "")) <= 2100)
        ]
        if not numbers:
            return (None, None)

        if any(k in title for k in ("between", "from")) and len(numbers) >= 2:
            return (min(numbers[:2]), max(numbers[:2]))
        if any(k in title for k in ("above", "more than", "at least", "over", ">")):
            return (numbers[0], None)
        if any(k in title for k in ("below", "less than", "under", "fewer", "<")):
            return (None, numbers[0])
        return (numbers[0], None)

    def detect_macro_data_edge(self, market: Dict) -> Optional[Edge]:
        """
        Detect mispricings in macro data release markets (CPI, NFP, Fed).

        Uses FRED leading indicators and historical distributions to estimate
        a fair probability for each outcome bucket, then compares to the
        Kalshi market price.
        """
        indicator = self._classify_macro_market(market)
        if indicator is None:
            return None

        raw_price = market.get("last_price") or market.get("yes_price")
        if raw_price is None:
            return None
        last_price = float(raw_price)
        market_prob = float(np.clip(last_price / 100.0, 0.0, 1.0))

        lo, hi = self._parse_macro_bucket(market)
        if lo is None and hi is None:
            return None

        try:
            from ..data.sources.macro_feeds import (
                get_cpi_consensus_range,
                get_nfp_consensus_range,
                get_fed_rate_current,
                build_normal_bucket_probs,
            )
        except ImportError:
            return None

        mean: Optional[float] = None
        std: Optional[float] = None

        if indicator == "cpi":
            low, consensus, high = get_cpi_consensus_range()
            mean = consensus
            std = max(0.01, (high - low) / 3.0)
        elif indicator == "nfp":
            low, consensus, high = get_nfp_consensus_range()
            mean = consensus
            std = max(5.0, (high - low) / 3.0)
        elif indicator == "fed":
            rate = get_fed_rate_current()
            if rate is not None:
                mean = (rate[0] + rate[1]) / 2.0
                std = 0.25
            else:
                return None

        if mean is None or std is None:
            return None

        bucket = (lo, hi)
        probs = build_normal_bucket_probs(mean, std, [bucket])
        if not probs:
            return None
        fair_prob = probs[0]
        fair_price = fair_prob * 100.0

        edge_value = fair_prob - market_prob
        if abs(edge_value) < self.min_edge:
            return None

        recommended_side = "yes" if edge_value > 0 else "no"
        confidence = float(np.clip(min(1.0, abs(edge_value) / 0.15) * 0.65, 0.1, 0.85))

        bucket_desc = ""
        if lo is not None and hi is not None:
            bucket_desc = f"between {lo} and {hi}"
        elif lo is not None:
            bucket_desc = f"> {lo}"
        elif hi is not None:
            bucket_desc = f"< {hi}"

        return Edge(
            ticker=market.get("ticker", ""),
            event_ticker=market.get("event_ticker", ""),
            edge_type="macro_data",
            edge_value=float(abs(edge_value)),
            confidence=confidence,
            recommended_side=recommended_side,
            market_price=last_price,
            fair_price=fair_price,
            reasoning=(
                f"{indicator.upper()} nowcast: mean={mean:.2f}, bucket {bucket_desc}: "
                f"fair={fair_price:.0f}¢ vs mkt={last_price:.0f}¢"
            ),
            market_data=market,
        )

    # ------------------------------------------------------------------
    # Weather / Temperature edge detector
    # ------------------------------------------------------------------

    _WEATHER_KEYWORDS = [
        "temperature", "temp", "degrees", "°f", "°c",
        "weather", "climate", "heat", "cold", "high temp", "low temp",
    ]

    def _is_weather_market(self, market: Dict) -> bool:
        title = (market.get("title", "") + " " + market.get("subtitle", "")).lower()
        series = str(market.get("series_ticker", "") or "").upper()
        if "KXTEMP" in series or "KXHMONTH" in series or "KXWEATHER" in series:
            return True
        return any(kw in title for kw in self._WEATHER_KEYWORDS)

    def _parse_temp_bucket(self, market: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Parse temperature bucket bounds from market title/strikes."""
        strike_type, floor_strike, cap_strike = self._extract_strike(market)
        if strike_type == "between" and floor_strike is not None and cap_strike is not None:
            return (float(floor_strike), float(cap_strike))
        if strike_type == "greater" and floor_strike is not None:
            return (float(floor_strike), None)
        if strike_type == "less" and (cap_strike is not None or floor_strike is not None):
            bound = float(cap_strike) if cap_strike is not None else float(floor_strike)
            return (None, bound)

        title = (market.get("title", "") + " " + market.get("subtitle", "")).lower()
        numbers = [
            float(n)
            for n in re.findall(r"(-?\d+(?:\.\d+)?)", title)
            if -50 <= float(n) <= 150
        ]
        if not numbers:
            return (None, None)
        if any(k in title for k in ("between", "from")) and len(numbers) >= 2:
            return (min(numbers[:2]), max(numbers[:2]))
        if any(k in title for k in ("above", "over", "higher", "at least", ">")):
            return (numbers[0], None)
        if any(k in title for k in ("below", "under", "lower", "less", "<")):
            return (None, numbers[0])
        return (numbers[0], None)

    def _extract_location(self, market: Dict) -> Optional[Tuple[float, float]]:
        """Try to extract lat/lon from market title for weather lookups."""
        _CITY_COORDS = {
            "new york": (40.7128, -74.0060),
            "nyc": (40.7128, -74.0060),
            "los angeles": (34.0522, -118.2437),
            "chicago": (41.8781, -87.6298),
            "houston": (29.7604, -95.3698),
            "phoenix": (33.4484, -112.0740),
            "miami": (25.7617, -80.1918),
            "dallas": (32.7767, -96.7970),
            "denver": (39.7392, -104.9903),
            "seattle": (47.6062, -122.3321),
            "san francisco": (37.7749, -122.4194),
            "atlanta": (33.7490, -84.3880),
            "boston": (42.3601, -71.0589),
            "washington": (38.9072, -77.0369),
            "dc": (38.9072, -77.0369),
            "las vegas": (36.1699, -115.1398),
        }
        title = (market.get("title", "") + " " + market.get("subtitle", "")).lower()
        for city, coords in _CITY_COORDS.items():
            if city in title:
                return coords
        return None

    def detect_weather_edge(self, market: Dict) -> Optional[Edge]:
        """
        Detect mispricings in weather/temperature markets.

        Uses Open-Meteo (free, no API key) ensemble forecast data to compute
        a probability distribution over temperature buckets, then compares to
        the Kalshi market price.
        """
        if not self._is_weather_market(market):
            return None

        raw_price = market.get("last_price") or market.get("yes_price")
        if raw_price is None:
            return None
        last_price = float(raw_price)
        market_prob = float(np.clip(last_price / 100.0, 0.0, 1.0))

        lo, hi = self._parse_temp_bucket(market)
        if lo is None and hi is None:
            return None

        location = self._extract_location(market)
        if location is None:
            return None
        lat, lon = location

        try:
            from ..data.sources.weather_feeds import get_temperature_forecast
        except ImportError:
            return None

        forecast = get_temperature_forecast(lat, lon)
        if forecast is None:
            return None

        mean_temp = forecast.get("mean")
        std_temp = forecast.get("std")
        if mean_temp is None or std_temp is None:
            return None

        from ..data.sources.macro_feeds import build_normal_bucket_probs
        probs = build_normal_bucket_probs(mean_temp, std_temp, [(lo, hi)])
        if not probs:
            return None
        fair_prob = probs[0]
        fair_price = fair_prob * 100.0

        edge_value = fair_prob - market_prob
        if abs(edge_value) < self.min_edge:
            return None

        recommended_side = "yes" if edge_value > 0 else "no"
        spread_factor = float(np.clip(forecast.get("ensemble_spread", 5.0) / 10.0, 0.3, 1.0))
        confidence = float(np.clip(
            min(1.0, abs(edge_value) / 0.15) * 0.6 * (1.0 / spread_factor),
            0.1, 0.85,
        ))

        bucket_desc = ""
        if lo is not None and hi is not None:
            bucket_desc = f"between {lo:.0f}°F and {hi:.0f}°F"
        elif lo is not None:
            bucket_desc = f"> {lo:.0f}°F"
        elif hi is not None:
            bucket_desc = f"< {hi:.0f}°F"

        return Edge(
            ticker=market.get("ticker", ""),
            event_ticker=market.get("event_ticker", ""),
            edge_type="weather",
            edge_value=float(abs(edge_value)),
            confidence=confidence,
            recommended_side=recommended_side,
            market_price=last_price,
            fair_price=fair_price,
            reasoning=(
                f"Weather forecast: mean={mean_temp:.1f}°F ±{std_temp:.1f}, "
                f"bucket {bucket_desc}: fair={fair_price:.0f}¢ vs mkt={last_price:.0f}¢"
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

        volume = float(market.get("volume", 0) or 0)
        oi = float(market.get("open_interest", 0) or 0)
        yb = float(market.get("yes_bid", 0) or 0)
        ya = float(market.get("yes_ask", 100) or 100)
        if volume == 0 and oi == 0 and yb == 0 and ya == 100:
            return []

        edges = []
        
        detectors = [
            self.detect_spread_arbitrage,
            self.detect_mean_reversion,
            self.detect_liquidity_edge,
            self.detect_range_edge,
            self.detect_macro_data_edge,
            self.detect_weather_edge,
        ]
        for detector in detectors:
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
        self._enrich_kx_strikes(markets)

        all_edges = []
        _alive = 0
        for market in markets:
            vol = float(market.get("volume", 0) or 0)
            oi = float(market.get("open_interest", 0) or 0)
            yb = float(market.get("yes_bid", 0) or 0)
            ya = float(market.get("yes_ask", 100) or 100)
            if not (vol == 0 and oi == 0 and yb == 0 and ya == 100):
                _alive += 1
            edges = self.scan_market(market)
            all_edges.extend(edges)
        if _alive < len(markets):
            logger.info(
                f"scan_series: {len(markets)} total, {_alive} alive "
                f"({len(markets) - _alive} stale filtered)"
            )

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
