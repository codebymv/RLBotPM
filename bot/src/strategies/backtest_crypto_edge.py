"""
Historical backtest of the crypto spot-vs-strike edge detector.

Uses settled Kalshi markets with known outcomes to validate whether
the lognormal fair-price model can profitably identify mispriced markets.

Key design choice:
  We simulate seeing spot price ~1h before settlement by adding noise
  to expiration_value. This avoids look-ahead bias while testing
  whether the model's probability estimates lead to profitable trades.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from ..data.database import get_db_session
from ..environment.kalshi_env import load_kalshi_settled_markets
from .kalshi_edges import StatisticalEdgeDetector, Edge

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    ticker: str
    event_ticker: str
    series_ticker: str
    edge_type: str        # 'crypto_spot_mispricing', 'strike_dominance', 'ladder_overpriced'
    action: str           # 'BUY_YES' or 'BUY_NO'
    entry_price: float    # Kalshi price in cents (0-100)
    fair_price: float     # Our model's fair price in cents
    edge: float           # abs(fair - market) / 100
    outcome: int          # 1 = YES settled, 0 = NO settled
    pnl: float            # Per-contract PnL in dollars
    won: bool
    spot: float           # Spot price used
    strike_desc: str      # Human-readable strike description


@dataclass
class BacktestReport:
    total_markets_scanned: int = 0
    markets_with_edge: int = 0
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    win_rate: float = 0.0
    avg_edge: float = 0.0
    avg_pnl_per_trade: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    by_series: Dict[str, Dict] = field(default_factory=dict)
    by_edge_bucket: Dict[str, Dict] = field(default_factory=dict)
    by_edge_type: Dict[str, Dict] = field(default_factory=dict)
    trades: List[TradeRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CRYPTO EDGE DETECTOR - HISTORICAL BACKTEST",
            "=" * 60,
            f"Markets scanned        : {self.total_markets_scanned:,}",
            f"Markets with edge      : {self.markets_with_edge:,}",
            f"Trades taken           : {self.trades_taken:,}",
            f"  Won                  : {self.trades_won} ({self.win_rate:.1%})",
            f"  Lost                 : {self.trades_lost}",
            f"Avg edge (taken)       : {self.avg_edge:.1%}",
            f"Total P&L              : ${self.total_pnl:+.2f}",
            f"  Gross profit         : ${self.gross_profit:.2f}",
            f"  Gross loss           : ${self.gross_loss:.2f}",
            f"Avg P&L per trade      : ${self.avg_pnl_per_trade:+.4f}",
            f"Sharpe (per-trade)     : {self.sharpe:.2f}",
            f"Max drawdown           : {self.max_drawdown:.1%}",
        ]
        if self.by_series:
            lines.append("\n--- By Series ---")
            for s, d in sorted(self.by_series.items()):
                lines.append(
                    f"  {s:12s}  trades={d['n']:4d}  "
                    f"win={d['win_rate']:.1%}  "
                    f"PnL=${d['pnl']:+.2f}  "
                    f"avg_edge={d['avg_edge']:.1%}"
                )
        if self.by_edge_bucket:
            lines.append("\n--- By Edge Bucket ---")
            for b, d in sorted(self.by_edge_bucket.items()):
                lines.append(
                    f"  {b:12s}  trades={d['n']:4d}  "
                    f"win={d['win_rate']:.1%}  "
                    f"PnL=${d['pnl']:+.2f}"
                )
        if self.by_edge_type:
            lines.append("\n--- By Edge Type ---")
            for t, d in sorted(self.by_edge_type.items()):
                lines.append(
                    f"  {t:28s}  trades={d['n']:4d}  "
                    f"win={d['win_rate']:.1%}  "
                    f"PnL=${d['pnl']:+.2f}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)


def _simulate_spot(expiration_value: float, vol_annual: float, hours_before: float = 1.0) -> float:
    """
    Simulate what spot price we would have observed ~hours_before settlement.

    Adds random noise consistent with the asset's volatility to avoid look-ahead bias.
    Keeps the same sign of the move (so outcome direction is preserved
    with high probability for short horizons).
    """
    if expiration_value <= 0:
        return expiration_value
    t = hours_before / (365.0 * 24.0)
    sigma = vol_annual * math.sqrt(t)
    # Small perturbation - roughly 0.3-0.5% for 1h with 60% annual vol
    noise = np.random.normal(0, sigma)
    return expiration_value * math.exp(noise)


def run_backtest(
    min_edge: float = 0.03,
    min_tradeable_price: int = 1,
    max_tradeable_price: int = 99,
    hours_before: float = 1.0,
    seed: int = 42,
    series_filter: Optional[List[str]] = None,
    max_events: Optional[int] = None,
) -> BacktestReport:
    """
    Backtest the crypto edge detector on settled Kalshi markets.

    For each event:
      1. Take all markets in the event (the strike ladder)
      2. Simulate spot price = expiration_value + noise
      3. Run StatisticalEdgeDetector with that spot
      4. For markets where it finds an edge AND last_price is tradeable,
         simulate buying at last_price and settling at outcome
      5. Record P&L

    Args:
        min_edge: Minimum edge to take a trade (e.g. 0.03 = 3%)
        min_tradeable_price: Skip markets with last_price below this (can't buy at 0)
        max_tradeable_price: Skip markets with last_price above this
        hours_before: Simulated hours before settlement for spot noise
        seed: Random seed for reproducibility
        series_filter: Only test these series (e.g. ['KXBTC','KXETH'])
        max_events: Cap number of events to test (for speed)
    """
    np.random.seed(seed)

    logger.info("Loading settled markets...")
    session = get_db_session()
    df = load_kalshi_settled_markets(session)
    session.close()

    crypto = df[df["series_ticker"].str.startswith("KX", na=False)].copy()
    if series_filter:
        crypto = crypto[crypto["series_ticker"].isin(series_filter)]
    logger.info(f"Crypto markets: {len(crypto)} across {crypto['series_ticker'].nunique()} series")

    # Vol lookup - calibrated from historical settlement data
    vol_map = {
        "KXBTC": 0.56, "KXBTCD": 0.56,
        "KXETH": 0.70, "KXETHD": 0.70,
        "KXSOLD": 0.74,
        "KXDOGE": 0.65,
        "KXXRP": 0.71,
        "KXINXU": 0.11,
        "KXEURUSDH": 0.12,
    }

    # Group by event
    events = crypto.groupby("event_ticker")
    event_keys = list(events.groups.keys())
    if max_events:
        event_keys = event_keys[:max_events]

    report = BacktestReport()
    all_trades: List[TradeRecord] = []

    # Create detector - we'll monkey-patch spot prices per event
    detector = StatisticalEdgeDetector(
        min_edge=min_edge,
        min_liquidity=0,   # Don't filter liquidity in backtest
        max_spread=1000,   # Don't filter spread in backtest
    )

    for evt_key in event_keys:
        evt_df = events.get_group(evt_key)
        series = evt_df["series_ticker"].iloc[0]
        vol = vol_map.get(series, 0.80)

        # All markets in this event share the same expiration_value
        exp_val = evt_df["expiration_value"].iloc[0]
        if pd.isna(exp_val) or exp_val <= 0:
            continue

        # Simulate spot ~1h before settlement
        simulated_spot = _simulate_spot(exp_val, vol, hours_before)

        # Patch the detector's spot cache so it uses our simulated spot
        asset = None
        for a in ("BTC", "ETH", "SOL", "DOGE", "XRP", "INXU", "EURUSD"):
            if a in series.upper():
                asset = a
                break
        if asset is None:
            continue
        detector._spot_cache[asset] = (simulated_spot, time.time())

        # Convert event markets to dicts
        market_dicts = []
        for _, row in evt_df.iterrows():
            market_dicts.append({
                "ticker": row["ticker"],
                "event_ticker": row["event_ticker"],
                "series_ticker": row["series_ticker"],
                "title": str(row.get("title", "")),
                "subtitle": str(row.get("subtitle", "")),
                "strike_type": row["strike_type"],
                "floor_strike": row["floor_strike"] if pd.notna(row["floor_strike"]) else None,
                "cap_strike": row["cap_strike"] if pd.notna(row["cap_strike"]) else None,
                "last_price": row["last_price"],
                "yes_price": row["last_price"],
                "yes_bid": row.get("yes_bid", 0) or 0,
                "yes_ask": row.get("yes_ask", 100) or 100,
                "volume": row.get("volume", 0) or 0,
                "open_interest": row.get("open_interest", 0) or 0,
                "liquidity": row.get("liquidity", 0) or 0,
                "close_time": row.get("close_time"),
            })

        report.total_markets_scanned += len(market_dicts)

        # Run edge detection
        edges = detector.scan_series(market_dicts, top_n=100)
        report.markets_with_edge += len(edges)

        # Simulate trades on edges with tradeable prices
        for edge in edges:
            if edge.edge_type not in ("crypto_spot_mispricing", "strike_dominance", "ladder_overpriced"):
                continue  # Only test crypto-relevant edges

            price = edge.market_price
            if price < min_tradeable_price or price > max_tradeable_price:
                continue  # Can't trade at 0 or 100

            # Find outcome
            matching = evt_df[evt_df["ticker"] == edge.ticker]
            if matching.empty:
                continue
            outcome = int(matching.iloc[0]["outcome"])

            # Calculate P&L
            if edge.recommended_side == "yes":
                cost = price / 100.0
                payout = 1.0 if outcome == 1 else 0.0
                action = "BUY_YES"
            elif edge.recommended_side == "no":
                cost = (100 - price) / 100.0
                payout = 1.0 if outcome == 0 else 0.0
                action = "BUY_NO"
            else:
                continue

            pnl = payout - cost
            won = pnl > 0

            # Build strike description
            if edge.market_data.get("strike_type") == "between":
                f = edge.market_data.get("floor_strike", 0)
                c = edge.market_data.get("cap_strike", 0)
                strike_desc = f"between {f:,.0f}-{c:,.0f}"
            elif edge.market_data.get("strike_type") == "greater":
                f = edge.market_data.get("floor_strike", 0)
                strike_desc = f"> {f:,.0f}"
            elif edge.market_data.get("strike_type") == "less":
                c = edge.market_data.get("cap_strike", 0)
                strike_desc = f"< {c:,.0f}"
            else:
                strike_desc = str(edge.market_data.get("strike_type", "?"))

            trade = TradeRecord(
                ticker=edge.ticker,
                event_ticker=evt_key,
                series_ticker=series,
                edge_type=edge.edge_type,
                action=action,
                entry_price=price,
                fair_price=edge.fair_price or 0,
                edge=edge.edge_value,
                outcome=outcome,
                pnl=pnl,
                won=won,
                spot=simulated_spot,
                strike_desc=strike_desc,
            )
            all_trades.append(trade)

    # Compute metrics
    report.trades = all_trades
    report.trades_taken = len(all_trades)
    if not all_trades:
        logger.warning("No trades generated - try lowering min_edge or min_tradeable_price")
        return report

    pnls = np.array([t.pnl for t in all_trades])
    edges_taken = np.array([t.edge for t in all_trades])

    report.trades_won = sum(1 for t in all_trades if t.won)
    report.trades_lost = report.trades_taken - report.trades_won
    report.win_rate = report.trades_won / report.trades_taken
    report.total_pnl = float(pnls.sum())
    report.gross_profit = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    report.gross_loss = float(pnls[pnls < 0].sum()) if (pnls < 0).any() else 0.0
    report.avg_edge = float(edges_taken.mean())
    report.avg_pnl_per_trade = float(pnls.mean())
    report.sharpe = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0

    # Drawdown
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    report.max_drawdown = float(dd.min()) if len(dd) > 0 else 0.0

    # By series
    for t in all_trades:
        s = t.series_ticker
        if s not in report.by_series:
            report.by_series[s] = {"n": 0, "won": 0, "pnl": 0.0, "edges": []}
        report.by_series[s]["n"] += 1
        report.by_series[s]["won"] += int(t.won)
        report.by_series[s]["pnl"] += t.pnl
        report.by_series[s]["edges"].append(t.edge)
    for s, d in report.by_series.items():
        d["win_rate"] = d["won"] / d["n"] if d["n"] > 0 else 0
        d["avg_edge"] = float(np.mean(d["edges"])) if d["edges"] else 0
        del d["edges"]

    # By edge type
    for t in all_trades:
        et = t.edge_type
        if et not in report.by_edge_type:
            report.by_edge_type[et] = {"n": 0, "won": 0, "pnl": 0.0}
        report.by_edge_type[et]["n"] += 1
        report.by_edge_type[et]["won"] += int(t.won)
        report.by_edge_type[et]["pnl"] += t.pnl
    for et, d in report.by_edge_type.items():
        d["win_rate"] = d["won"] / d["n"] if d["n"] > 0 else 0

    # By edge bucket
    for t in all_trades:
        if t.edge < 0.05:
            bucket = "0-5%"
        elif t.edge < 0.10:
            bucket = "5-10%"
        elif t.edge < 0.20:
            bucket = "10-20%"
        elif t.edge < 0.40:
            bucket = "20-40%"
        else:
            bucket = "40%+"
        if bucket not in report.by_edge_bucket:
            report.by_edge_bucket[bucket] = {"n": 0, "won": 0, "pnl": 0.0}
        report.by_edge_bucket[bucket]["n"] += 1
        report.by_edge_bucket[bucket]["won"] += int(t.won)
        report.by_edge_bucket[bucket]["pnl"] += t.pnl
    for b, d in report.by_edge_bucket.items():
        d["win_rate"] = d["won"] / d["n"] if d["n"] > 0 else 0

    return report


if __name__ == "__main__":
    report = run_backtest(
        min_edge=0.03,
        min_tradeable_price=1,
        max_tradeable_price=99,
        hours_before=1.0,
    )
    print(report.summary())
