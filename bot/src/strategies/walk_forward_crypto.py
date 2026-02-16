"""
Walk-forward backtest for the crypto edge detector.

Splits settled markets chronologically into windows and runs the edge
detector on each window independently. This validates that the edge
persists across time (not just in aggregate) and catches regime changes.

Usage:
    python main.py kalshi walk-forward --windows 10 --min-edge 0.02 --max-edge 0.05
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from ..data.database import get_db_session
from ..environment.kalshi_env import load_kalshi_settled_markets
from .kalshi_edges import StatisticalEdgeDetector, Edge
from .backtest_crypto_edge import _simulate_spot, TradeRecord

logger = get_logger(__name__)

VOL_MAP = {
    "KXBTC": 0.56, "KXBTCD": 0.56,
    "KXETH": 0.70, "KXETHD": 0.70,
    "KXSOLD": 0.74,
    "KXDOGE": 0.65,
    "KXXRP": 0.71,
    "KXINXU": 0.11,
    "KXEURUSDH": 0.12,
}


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_id: int
    start_date: str
    end_date: str
    n_events: int
    n_markets: int
    n_trades: int
    n_won: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    sharpe: float
    avg_edge: float
    by_side: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class WalkForwardReport:
    """Aggregate walk-forward results."""
    n_windows: int = 0
    total_trades: int = 0
    total_won: int = 0
    total_pnl: float = 0.0
    overall_win_rate: float = 0.0
    overall_sharpe: float = 0.0
    profitable_windows: int = 0
    min_window_pnl: float = 0.0
    max_window_pnl: float = 0.0
    windows: List[WindowResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "WALK-FORWARD BACKTEST — CRYPTO EDGE DETECTOR",
            "=" * 70,
            f"Windows              : {self.n_windows}",
            f"Profitable windows   : {self.profitable_windows}/{self.n_windows} "
            f"({self.profitable_windows/self.n_windows:.0%})" if self.n_windows > 0 else "",
            f"Total trades         : {self.total_trades:,}",
            f"Total won            : {self.total_won} ({self.overall_win_rate:.1%})",
            f"Total P&L            : ${self.total_pnl:+.2f}",
            f"Overall Sharpe       : {self.overall_sharpe:.2f}",
            f"Window P&L range     : ${self.min_window_pnl:+.2f} to ${self.max_window_pnl:+.2f}",
            "",
            f"{'Win':>3s}  {'Dates':28s}  {'Events':>6s}  {'Trades':>6s}  "
            f"{'Win%':>6s}  {'PnL':>8s}  {'Sharpe':>6s}  {'AvgEdge':>7s}",
            "-" * 80,
        ]
        for w in self.windows:
            lines.append(
                f"{w.window_id:3d}  {w.start_date} → {w.end_date}  "
                f"{w.n_events:6d}  {w.n_trades:6d}  "
                f"{w.win_rate:6.1%}  ${w.total_pnl:+7.2f}  "
                f"{w.sharpe:6.2f}  {w.avg_edge:6.1%}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


def _run_window(
    events_df: pd.DataFrame,
    detector: StatisticalEdgeDetector,
    window_id: int,
    hours_before: float,
    min_tradeable_price: int,
    max_tradeable_price: int,
    max_edge: float,
    edge_types: List[str],
) -> WindowResult:
    """Run edge detection on a single chronological window of events."""

    events = events_df.groupby("event_ticker")
    all_trades: List[TradeRecord] = []
    n_events = 0

    for evt_key, evt_df in events:
        n_events += 1
        series = evt_df["series_ticker"].iloc[0]
        vol = VOL_MAP.get(series, 0.70)

        exp_val = evt_df["expiration_value"].iloc[0]
        if pd.isna(exp_val) or exp_val <= 0:
            continue

        simulated_spot = _simulate_spot(exp_val, vol, hours_before)

        # Infer asset for spot cache
        asset = None
        for a in ("BTC", "ETH", "SOL", "DOGE", "XRP", "INXU", "EURUSD"):
            if a in series.upper():
                asset = a
                break
        if asset is None:
            continue
        detector._spot_cache[asset] = (simulated_spot, time.time())

        # Convert to dicts
        market_dicts = []
        for _, row in evt_df.iterrows():
            market_dicts.append({
                "ticker": row["ticker"],
                "event_ticker": row["event_ticker"],
                "series_ticker": row["series_ticker"],
                "title": str(row.get("title", "")),
                "subtitle": str(row.get("subtitle", "")),
                "strike_type": row["strike_type"],
                "floor_strike": row["floor_strike"] if pd.notna(row.get("floor_strike")) else None,
                "cap_strike": row["cap_strike"] if pd.notna(row.get("cap_strike")) else None,
                "last_price": row["last_price"],
                "yes_price": row["last_price"],
                "yes_bid": row.get("yes_bid", 0) or 0,
                "yes_ask": row.get("yes_ask", 100) or 100,
                "volume": row.get("volume", 0) or 0,
                "open_interest": row.get("open_interest", 0) or 0,
                "liquidity": row.get("liquidity", 0) or 0,
                "close_time": row.get("close_time"),
            })

        edges = detector.scan_series(market_dicts, top_n=100)

        for edge in edges:
            if edge.edge_type not in edge_types:
                continue
            if edge.edge_value > max_edge:
                continue

            price = edge.market_price
            if price < min_tradeable_price or price > max_tradeable_price:
                continue

            matching = evt_df[evt_df["ticker"] == edge.ticker]
            if matching.empty:
                continue
            outcome = int(matching.iloc[0]["outcome"])

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

    # Compute window stats
    n_trades = len(all_trades)
    n_won = sum(1 for t in all_trades if t.won)
    pnls = np.array([t.pnl for t in all_trades]) if all_trades else np.array([0.0])
    edges_arr = np.array([t.edge for t in all_trades]) if all_trades else np.array([0.0])

    # Determine date range from close_time
    close_times = events_df["close_time"].dropna()
    if len(close_times) > 0:
        start_date = str(pd.to_datetime(close_times.min()).date())
        end_date = str(pd.to_datetime(close_times.max()).date())
    else:
        start_date = "?"
        end_date = "?"

    # By side
    by_side = {}
    for side in ["BUY_YES", "BUY_NO"]:
        side_trades = [t for t in all_trades if t.action == side]
        if side_trades:
            sp = np.array([t.pnl for t in side_trades])
            by_side[side] = {
                "n": len(side_trades),
                "won": sum(1 for t in side_trades if t.won),
                "win_rate": sum(1 for t in side_trades if t.won) / len(side_trades),
                "pnl": float(sp.sum()),
            }

    return WindowResult(
        window_id=window_id,
        start_date=start_date,
        end_date=end_date,
        n_events=n_events,
        n_markets=len(events_df),
        n_trades=n_trades,
        n_won=n_won,
        win_rate=n_won / n_trades if n_trades > 0 else 0.0,
        total_pnl=float(pnls.sum()),
        avg_pnl=float(pnls.mean()) if n_trades > 0 else 0.0,
        sharpe=float(pnls.mean() / pnls.std()) if n_trades > 1 and pnls.std() > 0 else 0.0,
        avg_edge=float(edges_arr.mean()) if n_trades > 0 else 0.0,
        by_side=by_side,
    )


def run_walk_forward(
    n_windows: int = 10,
    min_edge: float = 0.02,
    max_edge: float = 0.05,
    min_tradeable_price: int = 1,
    max_tradeable_price: int = 15,
    hours_before: float = 1.0,
    seed: int = 42,
    series_filter: Optional[List[str]] = None,
    edge_types: Optional[List[str]] = None,
) -> WalkForwardReport:
    """
    Walk-forward backtest: split data chronologically, test each window.

    Args:
        n_windows: Number of chronological windows
        min_edge: Min edge to trade
        max_edge: Max edge to trade (cap false-positives)
        min_tradeable_price: Min price in cents
        max_tradeable_price: Max price in cents
        hours_before: Spot noise hours
        seed: Random seed
        series_filter: Optional series filter
        edge_types: Edge types to include (default: crypto_spot_mispricing only)
    """
    np.random.seed(seed)

    if edge_types is None:
        edge_types = ["crypto_spot_mispricing"]

    logger.info("Loading settled markets for walk-forward...")
    session = get_db_session()
    df = load_kalshi_settled_markets(session)
    session.close()

    crypto = df[df["series_ticker"].str.startswith("KX", na=False)].copy()
    if series_filter:
        crypto = crypto[crypto["series_ticker"].isin(series_filter)]

    # Sort by close_time for chronological splitting
    crypto["close_time_dt"] = pd.to_datetime(crypto["close_time"], errors="coerce")
    crypto = crypto.dropna(subset=["close_time_dt"]).sort_values("close_time_dt")

    logger.info(f"Total crypto markets: {len(crypto)} | splitting into {n_windows} windows")

    # Split events chronologically (not markets — keep whole events together)
    event_times = crypto.groupby("event_ticker")["close_time_dt"].min().sort_values()
    event_list = event_times.index.tolist()
    window_size = len(event_list) // n_windows

    detector = StatisticalEdgeDetector(
        min_edge=min_edge,
        min_liquidity=0,
        max_spread=1000,
    )

    report = WalkForwardReport(n_windows=n_windows)
    all_trade_pnls = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i < n_windows - 1 else len(event_list)
        window_events = event_list[start_idx:end_idx]

        window_df = crypto[crypto["event_ticker"].isin(window_events)]

        result = _run_window(
            events_df=window_df,
            detector=detector,
            window_id=i + 1,
            hours_before=hours_before,
            min_tradeable_price=min_tradeable_price,
            max_tradeable_price=max_tradeable_price,
            max_edge=max_edge,
            edge_types=edge_types,
        )

        report.windows.append(result)
        report.total_trades += result.n_trades
        report.total_won += result.n_won
        report.total_pnl += result.total_pnl

        if result.total_pnl > 0:
            report.profitable_windows += 1

        # Collect per-trade PnLs for overall Sharpe
        # (we don't store them in WindowResult to save memory, so recompute)
        all_trade_pnls.extend([result.avg_pnl] * result.n_trades if result.n_trades > 0 else [])

    # Overall stats
    report.overall_win_rate = report.total_won / report.total_trades if report.total_trades > 0 else 0
    pnl_arr = np.array([w.total_pnl for w in report.windows])
    report.min_window_pnl = float(pnl_arr.min()) if len(pnl_arr) > 0 else 0
    report.max_window_pnl = float(pnl_arr.max()) if len(pnl_arr) > 0 else 0
    report.overall_sharpe = float(pnl_arr.mean() / pnl_arr.std()) if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0

    return report
