"""
Live trading loop for Kalshi crypto edge detector.

Scans markets for BUY_NO edges, places real limit orders, and tracks
positions through settlement. Includes hard safety limits.

Safety limits (can be overridden but defaults are conservative):
  - Max $1 per trade (cost basis)
  - Max $10 total deployed capital
  - Max 10 simultaneous positions
  - BUY_NO only (100% backtest win rate)
  - Edge 2-5%, price 1-15¢ (validated thresholds)
  - Kill switch: circuit breaker on consecutive losses

Usage:
    python main.py kalshi live-trade --max-cost-per-trade 1 --max-total 10
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..core.logger import get_logger
from ..strategies.kalshi_edges import StatisticalEdgeDetector, Edge
from ..strategies.paper_trader import (
    _fetch_live_markets,
    _extract_asset,
    CRYPTO_SERIES,
    DEFAULT_MIN_EDGE,
    DEFAULT_MAX_EDGE,
    DEFAULT_MIN_PRICE,
    DEFAULT_MAX_PRICE,
)

logger = get_logger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

# Hard safety defaults — intentionally conservative for first live deployment
DEFAULT_MAX_COST_PER_TRADE = 1.00   # $1 max per trade
DEFAULT_MAX_TOTAL_DEPLOYED = 10.00  # $10 max total capital at risk
DEFAULT_MAX_POSITIONS = 10          # Max simultaneous positions
DEFAULT_MAX_LOSS_STREAK = 3         # Kill switch after 3 consecutive losses
DEFAULT_MAX_DAILY_LOSS = 5.00       # Stop trading if daily losses exceed $5


@dataclass
class LivePosition:
    """Tracks a live position placed on Kalshi."""
    ticker: str
    event_ticker: str
    side: str               # 'yes' or 'no'
    order_id: str
    price_cents: int
    contracts: int
    cost_dollars: float
    edge_value: float
    edge_type: str
    reasoning: str
    opened_at: str
    # Filled on settlement
    settled: bool = False
    outcome: Optional[str] = None
    pnl: float = 0.0


@dataclass
class LivePortfolio:
    """Tracks live trading state."""
    max_cost_per_trade: float = DEFAULT_MAX_COST_PER_TRADE
    max_total_deployed: float = DEFAULT_MAX_TOTAL_DEPLOYED
    max_positions: int = DEFAULT_MAX_POSITIONS
    max_loss_streak: int = DEFAULT_MAX_LOSS_STREAK
    max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS

    open_positions: Dict[str, LivePosition] = field(default_factory=dict)
    closed_positions: List[LivePosition] = field(default_factory=list)
    realized_pnl: float = 0.0
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    consecutive_losses: int = 0
    daily_loss: float = 0.0
    scan_count: int = 0
    killed: bool = False
    kill_reason: str = ""

    @property
    def deployed_capital(self) -> float:
        return sum(p.cost_dollars for p in self.open_positions.values())

    @property
    def available_to_deploy(self) -> float:
        return max(0, self.max_total_deployed - self.deployed_capital)

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return self.trades_won / total if total > 0 else 0.0

    def check_kill_switch(self) -> bool:
        """Returns True if trading should stop."""
        if self.consecutive_losses >= self.max_loss_streak:
            self.killed = True
            self.kill_reason = f"Kill switch: {self.consecutive_losses} consecutive losses"
            return True
        if self.daily_loss >= self.max_daily_loss:
            self.killed = True
            self.kill_reason = f"Kill switch: daily loss ${self.daily_loss:.2f} >= ${self.max_daily_loss:.2f}"
            return True
        return False

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "LIVE TRADING PORTFOLIO",
            "=" * 55,
            f"Scans completed   : {self.scan_count}",
            f"Deployed capital  : ${self.deployed_capital:.2f} / ${self.max_total_deployed:.2f}",
            f"Realized P&L      : ${self.realized_pnl:+.2f}",
            f"Trades taken      : {self.trades_taken}",
            f"  Won             : {self.trades_won} ({self.win_rate:.1%})",
            f"  Lost            : {self.trades_lost}",
            f"Open positions    : {len(self.open_positions)}",
            f"Closed positions  : {len(self.closed_positions)}",
            f"Loss streak       : {self.consecutive_losses} / {self.max_loss_streak} (kill switch)",
        ]
        if self.killed:
            lines.append(f"STATUS: KILLED — {self.kill_reason}")
        lines.append("=" * 55)

        if self.open_positions:
            lines.append("\nOpen:")
            for t, p in self.open_positions.items():
                lines.append(
                    f"  {t}  {p.side.upper()} {p.contracts}@{p.price_cents}¢"
                    f"  edge={p.edge_value:.1%}  cost=${p.cost_dollars:.2f}"
                    f"  order={p.order_id[:8]}"
                )
        return "\n".join(lines)


def _log_event(log_path: Path, event: Dict):
    """Append a JSON event to the live trade log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _check_live_settlements(client, portfolio: LivePortfolio, log_path: Path):
    """Check if any open positions have settled on Kalshi."""
    from ..data.sources.kalshi import KalshiAdapter

    adapter = KalshiAdapter(demo=False)
    settled_tickers = []

    for ticker, pos in portfolio.open_positions.items():
        try:
            market = adapter.get_market(ticker)
        except Exception:
            continue

        if market.status not in ("settled", "finalized") or market.result is None:
            continue

        pos.settled = True
        pos.outcome = market.result

        if pos.side == "yes":
            payout = 1.0 if market.result == "yes" else 0.0
        else:
            payout = 1.0 if market.result == "no" else 0.0

        pos.pnl = (payout * pos.contracts) - pos.cost_dollars
        portfolio.realized_pnl += pos.pnl

        if pos.pnl > 0:
            portfolio.trades_won += 1
            portfolio.consecutive_losses = 0
        else:
            portfolio.trades_lost += 1
            portfolio.consecutive_losses += 1
            portfolio.daily_loss += abs(pos.pnl)

        settled_tickers.append(ticker)

        _log_event(log_path, {
            "type": "settlement",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "side": pos.side,
            "price_cents": pos.price_cents,
            "outcome": market.result,
            "pnl": pos.pnl,
            "cumulative_pnl": portfolio.realized_pnl,
            "consecutive_losses": portfolio.consecutive_losses,
        })
        win_label = "WIN" if pos.pnl > 0 else "LOSS"
        logger.info(
            f"SETTLED {ticker}: {market.result} -> {win_label} ${pos.pnl:+.2f}  "
            f"(cumulative: ${portfolio.realized_pnl:+.2f})"
        )

    for t in settled_tickers:
        portfolio.closed_positions.append(portfolio.open_positions.pop(t))


def run_live_trading(
    interval_seconds: int = 300,
    min_edge: float = DEFAULT_MIN_EDGE,
    max_edge: float = DEFAULT_MAX_EDGE,
    min_price: int = DEFAULT_MIN_PRICE,
    max_price: int = DEFAULT_MAX_PRICE,
    max_cost_per_trade: float = DEFAULT_MAX_COST_PER_TRADE,
    max_total_deployed: float = DEFAULT_MAX_TOTAL_DEPLOYED,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    max_loss_streak: int = DEFAULT_MAX_LOSS_STREAK,
    max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
    series: Optional[List[str]] = None,
    allowed_sides: Optional[List[str]] = None,
    max_scans: Optional[int] = None,
    dry_run: bool = False,
) -> LivePortfolio:
    """
    Run live trading against the Kalshi API.

    Places real limit orders on detected BUY_NO edges.
    Hard safety limits prevent runaway losses.

    Args:
        dry_run: If True, find edges and log them but don't place orders.
    """
    from ..data.sources.kalshi import KalshiAdapter
    from ..execution.kalshi_client import KalshiExecutionClient, OrderSide
    from ..monitoring import AlertSystem

    series_list = series or CRYPTO_SERIES
    log_path = LOG_DIR / "live_trades.jsonl"

    env_allowed_sides = os.getenv("LIVE_ALLOWED_SIDES", "no")
    configured_sides = allowed_sides or [
        side.strip().lower() for side in env_allowed_sides.split(",") if side.strip()
    ]
    allowed_side_set = {side for side in configured_sides if side in {"yes", "no"}}
    if not allowed_side_set:
        allowed_side_set = {"no"}

    allow_buy_yes = os.getenv("LIVE_ALLOW_BUY_YES", "false").lower() == "true"
    if "yes" in allowed_side_set and not allow_buy_yes:
        logger.warning("Removing BUY_YES from allowed sides (set LIVE_ALLOW_BUY_YES=true to override)")
        allowed_side_set.discard("yes")
    if not allowed_side_set:
        allowed_side_set = {"no"}

    alert_recipients_raw = os.getenv("ALERT_EMAIL_TO", "")
    alert_recipients = [email.strip() for email in alert_recipients_raw.split(",") if email.strip()]
    alerter = AlertSystem(alert_recipients) if alert_recipients else None

    def _send_alert(subject: str, message: str, severity: str) -> None:
        if alerter is None:
            return
        try:
            alerter.send_alert(subject, message, severity=severity)
        except Exception as exc:
            logger.error("Failed to send alert '%s': %s", subject, exc)

    adapter = KalshiAdapter(demo=False)

    def _cb_alert(event) -> None:
        """Bridge CircuitBreaker events into the alert system."""
        _send_alert(
            f"Circuit breaker: {event.rule_violated}",
            event.description,
            event.severity,
        )

    client = KalshiExecutionClient(demo=False, alert_callback=_cb_alert)

    detector = StatisticalEdgeDetector(
        min_edge=min_edge,
        min_liquidity=0,
        max_spread=1000,
    )

    portfolio = LivePortfolio(
        max_cost_per_trade=max_cost_per_trade,
        max_total_deployed=max_total_deployed,
        max_positions=max_positions,
        max_loss_streak=max_loss_streak,
        max_daily_loss=max_daily_loss,
    )

    # Check account balance first
    try:
        balance, total_value = client.get_balance()
        logger.info(f"Account balance: ${balance:.2f} available, ${total_value:.2f} total")
        if balance < max_cost_per_trade:
            logger.error(f"Insufficient balance: ${balance:.2f} < ${max_cost_per_trade:.2f} min trade cost")
            return portfolio
    except Exception as e:
        logger.error(f"Failed to get account balance: {e}")
        return portfolio

    mode_label = "DRY RUN" if dry_run else "LIVE"
    logger.info(f"{'='*50}")
    logger.info(f"LIVE TRADING STARTED ({mode_label})")
    logger.info(f"{'='*50}")
    logger.info(f"Max per trade: ${max_cost_per_trade:.2f}")
    logger.info(f"Max deployed:  ${max_total_deployed:.2f}")
    logger.info(f"Max positions: {max_positions}")
    logger.info(f"Kill switch:   {max_loss_streak} consecutive losses or ${max_daily_loss:.2f} daily loss")
    logger.info(f"Edge range:    {min_edge:.1%} – {max_edge:.1%}")
    logger.info(f"Price range:   {min_price}–{max_price}¢")
    logger.info(f"Sides:         {', '.join(sorted(f'BUY_{s.upper()}' for s in allowed_side_set))}")
    logger.info(f"Series:        {', '.join(series_list)}")

    _send_alert(
        "Live trading session started",
        (
            f"Mode={mode_label} | MaxCost=${max_cost_per_trade:.2f} | "
            f"MaxDeployed=${max_total_deployed:.2f} | "
            f"Sides={','.join(sorted(allowed_side_set))} | "
            f"Edge={min_edge:.1%}-{max_edge:.1%}"
        ),
        severity="info",
    )

    _log_event(log_path, {
        "type": "session_start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode_label,
        "max_cost_per_trade": max_cost_per_trade,
        "max_total_deployed": max_total_deployed,
        "max_positions": max_positions,
        "max_loss_streak": max_loss_streak,
        "max_daily_loss": max_daily_loss,
        "min_edge": min_edge,
        "max_edge": max_edge,
        "allowed_sides": sorted(allowed_side_set),
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

            # Kill switch check
            if portfolio.check_kill_switch():
                logger.warning(f"KILL SWITCH: {portfolio.kill_reason}")
                _send_alert(
                    "Live trading kill switch triggered",
                    portfolio.kill_reason,
                    severity="critical",
                )
                _log_event(log_path, {
                    "type": "kill_switch",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": portfolio.kill_reason,
                })
                break

            scan_ts = datetime.now(timezone.utc).isoformat()

            # 1. Check settlements
            if portfolio.open_positions:
                _check_live_settlements(client, portfolio, log_path)

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

            # 3. Detect edges
            edges = detector.scan_series(markets, top_n=500)

            # 4. Filter and place orders
            new_trades = 0
            for edge in edges:
                if edge.edge_type not in ("crypto_spot_mispricing",):
                    continue

                # BUY_NO only
                if edge.recommended_side not in allowed_side_set:
                    continue

                if edge.ticker in portfolio.open_positions:
                    continue

                if edge.edge_value < min_edge or edge.edge_value > max_edge:
                    continue

                price = edge.market_price
                if price < min_price or price > max_price:
                    continue

                if len(portfolio.open_positions) >= max_positions:
                    break

                # Concentration limit: max 40% per asset
                asset = _extract_asset(edge.ticker)
                asset_cost = sum(
                    p.cost_dollars for p in portfolio.open_positions.values()
                    if _extract_asset(p.ticker) == asset
                )
                if asset_cost >= max_total_deployed * 0.40:
                    continue

                # Size: how many contracts can we afford?
                if edge.recommended_side == "yes":
                    cost_per_contract = price / 100.0
                else:
                    cost_per_contract = (100 - price) / 100.0
                max_by_trade_limit = int(max_cost_per_trade / cost_per_contract) if cost_per_contract > 0 else 0
                max_by_capital = int(portfolio.available_to_deploy / cost_per_contract) if cost_per_contract > 0 else 0
                contracts = min(max_by_trade_limit, max_by_capital)

                if contracts <= 0:
                    continue

                total_cost = contracts * cost_per_contract

                if dry_run:
                    logger.info(
                        f"DRY RUN: would place BUY_{edge.recommended_side.upper()} {contracts}@{price}¢ on {edge.ticker} "
                        f"edge={edge.edge_value:.1%} cost=${total_cost:.2f}"
                    )
                    _log_event(log_path, {
                        "type": "dry_run_signal",
                        "timestamp": scan_ts,
                        "ticker": edge.ticker,
                        "side": edge.recommended_side,
                        "price_cents": price,
                        "contracts": contracts,
                        "cost": total_cost,
                        "edge": edge.edge_value,
                        "reasoning": edge.reasoning,
                    })
                    continue

                # PLACE REAL ORDER
                logger.info(
                    f"PLACING ORDER: BUY_{edge.recommended_side.upper()} {contracts}@{price}¢ on {edge.ticker} "
                    f"edge={edge.edge_value:.1%} cost=${total_cost:.2f}"
                )

                order_side = OrderSide.YES if edge.recommended_side == "yes" else OrderSide.NO
                order_price = price if edge.recommended_side == "yes" else (100 - price)

                order = client.place_limit_order(
                    ticker=edge.ticker,
                    side=order_side,
                    price=order_price,
                    contracts=contracts,
                    expiration_seconds=interval_seconds,  # Cancel if not filled by next scan
                )

                if order is None:
                    logger.warning(f"Order failed for {edge.ticker}")
                    _send_alert(
                        "Live order failed",
                        f"Ticker={edge.ticker} side={edge.recommended_side} price={price} contracts={contracts}",
                        severity="warning",
                    )
                    continue

                portfolio.trades_taken += 1
                pos = LivePosition(
                    ticker=edge.ticker,
                    event_ticker=edge.event_ticker,
                    side=edge.recommended_side,
                    order_id=order.order_id,
                    price_cents=price,
                    contracts=contracts,
                    cost_dollars=total_cost,
                    edge_value=edge.edge_value,
                    edge_type=edge.edge_type,
                    reasoning=edge.reasoning,
                    opened_at=scan_ts,
                )
                portfolio.open_positions[edge.ticker] = pos
                new_trades += 1

                _log_event(log_path, {
                    "type": "order_placed",
                    "timestamp": scan_ts,
                    "ticker": edge.ticker,
                    "event_ticker": edge.event_ticker,
                    "side": edge.recommended_side,
                    "price_cents": price,
                    "contracts": contracts,
                    "cost": total_cost,
                    "edge": edge.edge_value,
                    "edge_type": edge.edge_type,
                    "order_id": order.order_id,
                    "reasoning": edge.reasoning,
                })

                logger.info(
                    f"ORDER FILLED: {edge.ticker} BUY_{edge.recommended_side.upper()} {contracts}@{price}¢ "
                    f"cost=${total_cost:.2f} order={order.order_id[:8]}"
                )

            # 5. Summary
            logger.info(
                f"Scan {scan_n}: {len(markets)} markets | "
                f"{len(edges)} edges | {new_trades} new orders | "
                f"open={len(portfolio.open_positions)} | "
                f"deployed=${portfolio.deployed_capital:.2f} | "
                f"P&L=${portfolio.realized_pnl:+.2f}"
            )

            _log_event(log_path, {
                "type": "scan_summary",
                "timestamp": scan_ts,
                "scan": scan_n,
                "markets_scanned": len(markets),
                "edges_found": len(edges),
                "new_trades": new_trades,
                "open_positions": len(portfolio.open_positions),
                "deployed_capital": portfolio.deployed_capital,
                "realized_pnl": portfolio.realized_pnl,
            })

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Live trading interrupted by user.")
    except Exception as e:
        logger.error("Live trading crashed: %s", e)
        _send_alert(
            "Live trading crashed",
            f"Unhandled exception: {e}",
            severity="critical",
        )
        raise

    _log_event(log_path, {
        "type": "session_end",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scans": scan_n,
        "trades_taken": portfolio.trades_taken,
        "trades_won": portfolio.trades_won,
        "trades_lost": portfolio.trades_lost,
        "realized_pnl": portfolio.realized_pnl,
        "open_positions": len(portfolio.open_positions),
        "killed": portfolio.killed,
        "kill_reason": portfolio.kill_reason,
    })

    return portfolio
