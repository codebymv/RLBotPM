"""
Paper trading loop for Kalshi crypto edge detector.

Polls live markets on a schedule, detects edges, logs hypothetical trades,
and tracks simulated portfolio performance — all without placing real orders.

Usage:
    python main.py kalshi paper-trade --interval 300 --bankroll 100

Logs are written to bot/logs/paper_trades.jsonl for audit.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..core.logger import get_logger
from ..strategies.kalshi_edges import StatisticalEdgeDetector, Edge

logger = get_logger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

CRYPTO_SERIES = ["KXBTC", "KXBTCD", "KXETH", "KXETHD", "KXSOLD", "KXDOGE", "KXXRP"]

# Backtest-validated parameters
DEFAULT_MIN_EDGE = 0.01      # 1% minimum edge
DEFAULT_MAX_EDGE = 0.20      # Skip large "edge" (12% win rate at 40%+)
DEFAULT_MIN_PRICE = 1        # Skip 0-priced markets
DEFAULT_MAX_PRICE = 50       # Low-price markets have best win rate (86.9% at 1-5¢)


@dataclass
class PaperPosition:
    """A hypothetical position opened by the paper trader."""
    ticker: str
    event_ticker: str
    series_ticker: str
    side: str               # 'yes' or 'no'
    entry_price_cents: float
    fair_price_cents: float
    edge_value: float
    edge_type: str
    contracts: int
    cost_dollars: float
    opened_at: str           # ISO timestamp
    reasoning: str
    # Filled on settlement
    settled: bool = False
    outcome: Optional[str] = None  # 'yes' or 'no'
    pnl: float = 0.0
    settled_at: Optional[str] = None


@dataclass
class PaperPortfolio:
    """Simulated portfolio state."""
    initial_capital: float = 100.0
    cash: float = 100.0
    realized_pnl: float = 0.0
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    open_positions: Dict[str, PaperPosition] = field(default_factory=dict)
    closed_positions: List[PaperPosition] = field(default_factory=list)
    scan_count: int = 0

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return self.trades_won / total if total > 0 else 0.0

    @property
    def total_value(self) -> float:
        open_cost = sum(p.cost_dollars for p in self.open_positions.values())
        return self.cash + open_cost  # conservative: value open positions at cost

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "PAPER TRADING PORTFOLIO",
            "=" * 50,
            f"Scans completed  : {self.scan_count}",
            f"Initial capital  : ${self.initial_capital:.2f}",
            f"Cash             : ${self.cash:.2f}",
            f"Realized P&L     : ${self.realized_pnl:+.2f}",
            f"Total value      : ${self.total_value:.2f}",
            f"Trades taken     : {self.trades_taken}",
            f"  Won            : {self.trades_won} ({self.win_rate:.1%})",
            f"  Lost           : {self.trades_lost}",
            f"Open positions   : {len(self.open_positions)}",
            f"Closed positions : {len(self.closed_positions)}",
            "=" * 50,
        ]
        if self.open_positions:
            lines.append("\nOpen:")
            for t, p in self.open_positions.items():
                lines.append(f"  {t}  {p.side.upper()} @ {p.entry_price_cents:.0f}¢  edge={p.edge_value:.1%}  cost=${p.cost_dollars:.2f}")
        return "\n".join(lines)


def _log_event(log_path: Path, event: Dict):
    """Append a JSON event to the paper trade log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _fetch_live_markets(adapter, series_list: List[str]) -> List[Dict]:
    """Fetch live markets from Kalshi API and convert to detector format."""
    all_markets = []
    for st in series_list:
        try:
            markets = adapter.get_markets(series_ticker=st, status="open", limit=100)
        except Exception as e:
            logger.warning(f"Failed to fetch {st}: {e}")
            continue

        for m in markets:
            all_markets.append({
                "ticker": m.ticker,
                "event_ticker": getattr(m, "event_ticker", None) or m.ticker.rsplit("-", 1)[0],
                "series_ticker": getattr(m, "series_ticker", None) or st,
                "title": getattr(m, "title", ""),
                "subtitle": getattr(m, "subtitle", ""),
                "category": getattr(m, "category", ""),
                "close_time": getattr(m, "close_time", None),
                "expiration_time": getattr(m, "expiration_time", None),
                "last_price": m.yes_price,
                "yes_bid": m.yes_bid,
                "yes_ask": m.yes_ask,
                "no_bid": max(0.0, 100.0 - float(m.yes_ask)),
                "no_ask": max(0.0, 100.0 - float(m.yes_bid)),
                "volume": m.volume,
                "open_interest": m.open_interest or 0,
                "liquidity": float(m.open_interest or 0) or float(m.volume or 0),
                "previous_price": m.yes_price,
            })
    return all_markets


def _check_settlements(adapter, portfolio: PaperPortfolio, log_path: Path):
    """Check if any open positions have settled."""
    settled_tickers = []
    for ticker, pos in portfolio.open_positions.items():
        try:
            market = adapter.get_market(ticker)
        except Exception:
            continue

        if market.status != "settled" or market.result is None:
            continue

        # Market has settled
        pos.settled = True
        pos.outcome = market.result
        pos.settled_at = datetime.now(timezone.utc).isoformat()

        if pos.side == "yes":
            payout = 1.0 if market.result == "yes" else 0.0
        else:
            payout = 1.0 if market.result == "no" else 0.0

        pos.pnl = (payout * pos.contracts) - pos.cost_dollars
        portfolio.realized_pnl += pos.pnl
        portfolio.cash += payout * pos.contracts

        if pos.pnl > 0:
            portfolio.trades_won += 1
        else:
            portfolio.trades_lost += 1

        settled_tickers.append(ticker)

        _log_event(log_path, {
            "type": "settlement",
            "timestamp": pos.settled_at,
            "ticker": ticker,
            "side": pos.side,
            "entry_price": pos.entry_price_cents,
            "outcome": market.result,
            "pnl": pos.pnl,
            "cumulative_pnl": portfolio.realized_pnl,
        })
        logger.info(
            f"SETTLED {ticker}: {market.result} → "
            f"{'WIN' if pos.pnl > 0 else 'LOSS'} ${pos.pnl:+.2f}  "
            f"(cumulative: ${portfolio.realized_pnl:+.2f})"
        )

    for t in settled_tickers:
        portfolio.closed_positions.append(portfolio.open_positions.pop(t))


def run_paper_trading(
    interval_seconds: int = 300,
    bankroll: float = 100.0,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_edge: float = DEFAULT_MAX_EDGE,
    min_price: int = DEFAULT_MIN_PRICE,
    max_price: int = DEFAULT_MAX_PRICE,
    max_contracts_per_trade: int = 10,
    max_open_positions: int = 20,
    series: Optional[List[str]] = None,
    demo: bool = True,
    max_scans: Optional[int] = None,
) -> PaperPortfolio:
    """
    Run the paper trading loop.

    Args:
        interval_seconds: Seconds between scans (default 5 min)
        bankroll: Starting capital
        min_edge: Minimum edge to open a position
        max_edge: Maximum edge (large "edges" are often wrong)
        min_price: Minimum market price in cents to trade
        max_price: Maximum market price in cents to trade
        max_contracts_per_trade: Max contracts per trade
        max_open_positions: Max simultaneous open positions
        series: Kalshi series to scan (default: all crypto)
        demo: Use Kalshi demo API
        max_scans: Stop after N scans (None = run forever)
    """
    from ..data.sources.kalshi import KalshiAdapter

    series_list = series or CRYPTO_SERIES
    log_path = LOG_DIR / "paper_trades.jsonl"

    adapter = KalshiAdapter(demo=demo)
    detector = StatisticalEdgeDetector(
        min_edge=min_edge,
        min_liquidity=0,
        max_spread=1000,
    )
    portfolio = PaperPortfolio(initial_capital=bankroll, cash=bankroll)

    logger.info(f"Paper trading started | bankroll=${bankroll} | interval={interval_seconds}s")
    logger.info(f"Series: {', '.join(series_list)}")
    logger.info(f"Edge range: {min_edge:.1%} – {max_edge:.1%} | Price range: {min_price}–{max_price}¢")
    logger.info(f"Log: {log_path}")

    _log_event(log_path, {
        "type": "session_start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bankroll": bankroll,
        "min_edge": min_edge,
        "max_edge": max_edge,
        "series": series_list,
    })

    scan_n = 0
    try:
        while True:
            scan_n += 1
            portfolio.scan_count = scan_n

            if max_scans and scan_n > max_scans:
                logger.info(f"Reached max_scans={max_scans}, stopping.")
                break

            scan_ts = datetime.now(timezone.utc).isoformat()

            # 1. Check settlements on open positions
            if portfolio.open_positions:
                _check_settlements(adapter, portfolio, log_path)

            # 2. Fetch live markets
            try:
                markets = _fetch_live_markets(adapter, series_list)
            except Exception as e:
                logger.error(f"Scan {scan_n}: fetch failed — {e}")
                time.sleep(interval_seconds)
                continue

            if not markets:
                logger.info(f"Scan {scan_n}: no markets found")
                time.sleep(interval_seconds)
                continue

            # 3. Run edge detection
            edges = detector.scan_series(markets, top_n=50)

            # 4. Filter and open new positions
            new_trades = 0
            for edge in edges:
                if edge.edge_type not in ("crypto_spot_mispricing", "strike_dominance"):
                    continue

                # Skip if already have a position in this market
                if edge.ticker in portfolio.open_positions:
                    continue

                # Edge bounds
                if edge.edge_value < min_edge or edge.edge_value > max_edge:
                    continue

                # Price bounds
                price = edge.market_price
                if price < min_price or price > max_price:
                    continue

                # Position limit
                if len(portfolio.open_positions) >= max_open_positions:
                    break

                # Size the trade
                if edge.recommended_side == "yes":
                    cost_per_contract = price / 100.0
                elif edge.recommended_side == "no":
                    cost_per_contract = (100 - price) / 100.0
                else:
                    continue

                # Simple sizing: fixed number of contracts, capped by available cash
                contracts = min(
                    max_contracts_per_trade,
                    int(portfolio.cash / cost_per_contract) if cost_per_contract > 0 else 0,
                )
                if contracts <= 0:
                    continue

                total_cost = contracts * cost_per_contract
                if total_cost > portfolio.cash:
                    continue

                # Open position
                portfolio.cash -= total_cost
                portfolio.trades_taken += 1

                pos = PaperPosition(
                    ticker=edge.ticker,
                    event_ticker=edge.event_ticker,
                    series_ticker=edge.market_data.get("series_ticker", ""),
                    side=edge.recommended_side,
                    entry_price_cents=price,
                    fair_price_cents=edge.fair_price or 0,
                    edge_value=edge.edge_value,
                    edge_type=edge.edge_type,
                    contracts=contracts,
                    cost_dollars=total_cost,
                    opened_at=scan_ts,
                    reasoning=edge.reasoning,
                )
                portfolio.open_positions[edge.ticker] = pos
                new_trades += 1

                _log_event(log_path, {
                    "type": "open_position",
                    "timestamp": scan_ts,
                    "ticker": edge.ticker,
                    "event_ticker": edge.event_ticker,
                    "side": edge.recommended_side,
                    "entry_price": price,
                    "fair_price": edge.fair_price,
                    "edge": edge.edge_value,
                    "edge_type": edge.edge_type,
                    "contracts": contracts,
                    "cost": total_cost,
                    "reasoning": edge.reasoning,
                })

                logger.info(
                    f"OPEN {edge.ticker}: "
                    f"BUY_{edge.recommended_side.upper()} {contracts}@{price:.0f}¢ "
                    f"edge={edge.edge_value:.1%} fair={edge.fair_price:.0f}¢ "
                    f"cost=${total_cost:.2f}"
                )

            # 5. Summary
            logger.info(
                f"Scan {scan_n}: {len(markets)} markets | "
                f"{len(edges)} edges | {new_trades} new trades | "
                f"open={len(portfolio.open_positions)} | "
                f"P&L=${portfolio.realized_pnl:+.2f} | "
                f"cash=${portfolio.cash:.2f}"
            )

            _log_event(log_path, {
                "type": "scan_summary",
                "timestamp": scan_ts,
                "scan": scan_n,
                "markets_scanned": len(markets),
                "edges_found": len(edges),
                "new_trades": new_trades,
                "open_positions": len(portfolio.open_positions),
                "realized_pnl": portfolio.realized_pnl,
                "cash": portfolio.cash,
                "total_value": portfolio.total_value,
            })

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Paper trading interrupted by user.")

    _log_event(log_path, {
        "type": "session_end",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scans": scan_n,
        "trades_taken": portfolio.trades_taken,
        "trades_won": portfolio.trades_won,
        "trades_lost": portfolio.trades_lost,
        "realized_pnl": portfolio.realized_pnl,
        "cash": portfolio.cash,
        "open_positions": len(portfolio.open_positions),
    })

    return portfolio
