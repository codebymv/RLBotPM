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
    classify_sleeve,
    hours_to_close,
    CRYPTO_SERIES,
    LIVE_SERIES,
    FAST_SERIES,
    MACRO_SERIES_SET,
    HYBRID_FAST_HORIZON_HOURS,
    HYBRID_FAST_CAPITAL_FRAC,
    HYBRID_FAST_POSITION_FRAC,
    HYBRID_MACRO_MAX_PORTFOLIO_FRAC,
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

# ── Trade admission rules (Phase 3 philosophy reset) ─────────────
# These gates exist to prevent the three failure modes identified in
# the edge audit: model overconfidence, execution friction, and
# capital lock-up.

# Execution edge: (payout - cost) / cost must exceed this.
# A 50c cost needs to return >75c to clear a 0.50 threshold.
MIN_EXECUTION_EDGE = 0.50

# Spread gate: reject trades where bid-ask spread in cents exceeds this.
MAX_SPREAD_CENTS = 25.0

# Per-ticker contract cap to prevent concentration blowups.
MAX_CONTRACTS_PER_TICKER = 5

# Macro admission: macro_data edges are blocked unless this env var
# is set to "true".  Macro must EARN its way back via evidence.
# Set KALSHI_MACRO_ENABLED=true after 5+ positive macro settlements.
MACRO_ENABLED_DEFAULT = False

# Minimum activity (volume + OI) for a market to be tradeable.
MIN_MARKET_ACTIVITY = 5

# Don't enter a market that the exit logic will immediately flatten.
# We flatten when hours_left <= 1.0h. Add a scan-interval buffer so a new
# position can at least survive to the next scan without being force-sold.
# Effective rule: market must have >= MIN_HOURS_TO_CLOSE_ON_ENTRY of life
# left at entry time.
MIN_HOURS_TO_CLOSE_ON_ENTRY = 1.5

# H1-only mode: the most restrictive admission profile, derived from the
# edge-audit's only proven hypothesis. Activated by env H1_ONLY=true.
# Applies a hard set of filters in addition to the normal ones:
#   - edge_type must be spot_vs_strike (no macro, no weather)
#   - recommended_side must be 'no' (we sell tail probability)
#   - asset must be a crypto from H1_CRYPTO_ASSETS
#   - spot must be >= H1_MIN_SPOT_DISTANCE_PCT away from the strike
#   - hours_to_close must be in [H1_MIN_HOURS, H1_MAX_HOURS]
# When active, flatten_before_close exit is disabled so the trade rides to
# settlement, and the kill switch is widened to allow the lottery-style
# loss pattern this strategy expects.
H1_CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "DOGE", "XRP"}
H1_MIN_SPOT_DISTANCE_PCT = 0.15
H1_MIN_HOURS = 0.5
H1_MAX_HOURS = 2.0
H1_HARD_FLOOR_TOTAL_WEALTH = 25.0


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
class RestingOrder:
    """An order that was accepted by Kalshi but hasn't filled yet."""
    ticker: str
    event_ticker: str
    side: str
    order_id: str
    price_cents: int
    contracts_requested: int
    cost_per_contract: float
    edge_value: float
    edge_type: str
    reasoning: str
    placed_at: str
    filled_contracts: int = 0

    @property
    def remaining_contracts(self) -> int:
        return max(0, self.contracts_requested - self.filled_contracts)

    @property
    def reserved_cost_dollars(self) -> float:
        return self.remaining_contracts * self.cost_per_contract


@dataclass
class LivePortfolio:
    """Tracks live trading state."""
    max_cost_per_trade: float = DEFAULT_MAX_COST_PER_TRADE
    max_total_deployed: float = DEFAULT_MAX_TOTAL_DEPLOYED
    max_positions: int = DEFAULT_MAX_POSITIONS
    max_loss_streak: int = DEFAULT_MAX_LOSS_STREAK
    max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS

    open_positions: Dict[str, LivePosition] = field(default_factory=dict)
    resting_orders: Dict[str, RestingOrder] = field(default_factory=dict)
    closed_positions: List[LivePosition] = field(default_factory=list)
    realized_pnl: float = 0.0
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    consecutive_losses: int = 0
    daily_loss: float = 0.0
    _daily_loss_date: str = ""
    scan_count: int = 0
    killed: bool = False
    kill_reason: str = ""
    _phantom_count: int = 0

    def maybe_reset_daily_loss(self):
        """Reset daily_loss and un-kill the bot at the start of each new UTC day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_loss_date != today:
            if self.daily_loss > 0:
                logger.info(
                    "Daily loss reset: $%.2f (from %s) -> $0.00 (new day %s)",
                    self.daily_loss, self._daily_loss_date or "session-start", today,
                )
            self.daily_loss = 0.0
            self._daily_loss_date = today
            if self.killed and "daily loss" in self.kill_reason:
                self.killed = False
                self.kill_reason = ""

    @property
    def deployed_capital(self) -> float:
        return self.open_position_capital + self.resting_capital

    @property
    def open_position_capital(self) -> float:
        return sum(p.cost_dollars for p in self.open_positions.values())

    @property
    def resting_capital(self) -> float:
        return sum(o.reserved_cost_dollars for o in self.resting_orders.values())

    @property
    def available_to_deploy(self) -> float:
        return max(0, self.max_total_deployed - self.deployed_capital)

    @property
    def active_market_count(self) -> int:
        return len(set(self.open_positions) | set(self.resting_orders))

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
            f"  Open cost       : ${self.open_position_capital:.2f}",
            f"  Resting reserved: ${self.resting_capital:.2f}",
            f"Realized P&L      : ${self.realized_pnl:+.2f}",
            f"Trades taken      : {self.trades_taken}",
            f"  Won             : {self.trades_won} ({self.win_rate:.1%})",
            f"  Lost            : {self.trades_lost}",
            f"Resting orders    : {len(self.resting_orders)}",
            f"Open positions    : {len(self.open_positions)}",
            f"Active markets    : {self.active_market_count}",
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


def _push_event_to_api(event: Dict) -> bool:
    """Fire-and-forget POST of a trade event to the remote API."""
    import urllib.request

    api_url = (
        os.getenv("API_BASE_URL")
        or os.getenv("NEXT_PUBLIC_API_URL")
        or "http://localhost:8000"
    ).rstrip("/")
    url = f"{api_url}/api/live-trades"
    body = json.dumps(event, default=str).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception as exc:
        event_type = event.get("type", "unknown")
        if event_type != "scan_summary":
            logger.warning("Failed to push live-trade event '%s' to API: %s", event_type, exc)
        else:
            logger.debug("Failed to push live-trade event '%s' to API: %s", event_type, exc)
        return False


def _log_event(log_path: Path, event: Dict):
    """Append a JSON event to the live trade log and push to remote API."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    _push_event_to_api(event)


def _check_resting_fills(client, portfolio: LivePortfolio, log_path: Path, session_id: str):
    """Poll resting orders to see if any have filled. Promote to positions."""
    completed = []
    for ticker, resting in portfolio.resting_orders.items():
        try:
            order = client.get_order(resting.order_id, strict=True)
        except Exception as e:
            logger.warning("Failed to poll resting order %s for %s: %s", resting.order_id[:8], ticker, e)
            continue

        if order is None:
            continue

        status_str = order.status.value if order.status else "unknown"
        filled = order.filled_contracts or 0

        newly_filled = max(0, filled - resting.filled_contracts)
        if newly_filled > 0:
            actual_cost = newly_filled * resting.cost_per_contract
            pos = portfolio.open_positions.get(ticker)
            if pos is None:
                portfolio.trades_taken += 1
                pos = LivePosition(
                    ticker=ticker,
                    event_ticker=resting.event_ticker,
                    side=resting.side,
                    order_id=resting.order_id,
                    price_cents=resting.price_cents,
                    contracts=newly_filled,
                    cost_dollars=actual_cost,
                    edge_value=resting.edge_value,
                    edge_type=resting.edge_type,
                    reasoning=resting.reasoning,
                    opened_at=resting.placed_at,
                )
                portfolio.open_positions[ticker] = pos
            else:
                pos.contracts += newly_filled
                pos.cost_dollars += actual_cost

            resting.filled_contracts = filled
            _log_event(log_path, {
                "type": "order_filled",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "event_ticker": resting.event_ticker,
                "side": resting.side,
                "price_cents": resting.price_cents,
                "contracts": newly_filled,
                "filled_contracts_total": filled,
                "remaining_contracts": resting.remaining_contracts,
                "cost": actual_cost,
                "edge": resting.edge_value,
                "edge_type": resting.edge_type,
                "order_id": resting.order_id,
                "order_status": status_str,
                "reasoning": resting.reasoning,
            })
            logger.info(
                "RESTING ORDER FILL: %s %s +%s@%s¢ remaining=%s order=%s",
                ticker,
                resting.side.upper(),
                newly_filled,
                resting.price_cents,
                resting.remaining_contracts,
                resting.order_id[:8],
            )

        if status_str in ("canceled", "cancelled", "expired"):
            logger.info("Resting order %s: %s order=%s", status_str, ticker, resting.order_id[:8])
            _log_event(log_path, {
                "type": "order_closed",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "event_ticker": resting.event_ticker,
                "side": resting.side,
                "price_cents": resting.price_cents,
                "contracts": resting.contracts_requested,
                "filled_contracts_total": resting.filled_contracts,
                "remaining_contracts": resting.remaining_contracts,
                "cost": resting.reserved_cost_dollars,
                "edge": resting.edge_value,
                "edge_type": resting.edge_type,
                "order_id": resting.order_id,
                "order_status": status_str,
                "reasoning": resting.reasoning,
            })
            completed.append(ticker)
            continue

        if resting.remaining_contracts <= 0 or status_str in ("filled", "executed"):
            completed.append(ticker)

    for t in completed:
        portfolio.resting_orders.pop(t, None)


def _check_exit_opportunities(
    client,
    adapter,
    portfolio: "LivePortfolio",
    log_path: Path,
    profit_target_pct: float = 0.40,
    stop_loss_pct: float = 0.60,
    max_hold_hours: float = 168.0,
    flatten_before_close_hours: float = 1.0,
):
    """Scan open positions and sell any that meet exit criteria.

    Exit policies (checked in priority order):
      1. Profit target: mark-to-market gain >= profit_target_pct of cost
      2. Stop-loss: mark-to-market loss >= stop_loss_pct of cost
      3. Time decay flatten: market closes within flatten_before_close_hours
      4. Max hold: position has been open longer than max_hold_hours

    Uses limit sells priced at the current bid to maximize fill chance
    while avoiding market-order slippage on thin books.
    """
    if not portfolio.open_positions:
        return

    now = datetime.now(timezone.utc)
    exits = []

    for ticker, pos in list(portfolio.open_positions.items()):
        try:
            market = adapter.get_market(ticker)
        except Exception:
            continue

        if market.status in ("settled", "finalized"):
            continue

        # Mark-to-market the position
        if pos.side == "yes":
            bid = float(market.yes_bid or 0)
            mark_value = bid * pos.contracts / 100.0
        else:
            no_bid = 100 - float(market.yes_ask or 100)
            if no_bid <= 0:
                continue
            mark_value = no_bid * pos.contracts / 100.0

        cost = pos.cost_dollars
        mtm_pnl = mark_value - cost
        mtm_pct = mtm_pnl / cost if cost > 0 else 0.0

        # Time since open
        try:
            opened = datetime.fromisoformat(pos.opened_at)
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            hold_hours = (now - opened).total_seconds() / 3600.0
        except (ValueError, TypeError):
            hold_hours = 0.0

        # Hours to market close
        close_time = getattr(market, "close_time", None)
        hours_left = None
        if close_time:
            try:
                if isinstance(close_time, str):
                    close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                hours_left = (close_time - now).total_seconds() / 3600.0
            except (ValueError, TypeError):
                pass

        reason = None
        if mtm_pct >= profit_target_pct:
            reason = f"profit_target ({mtm_pct:+.0%} >= {profit_target_pct:.0%})"
        elif mtm_pct <= -stop_loss_pct:
            reason = f"stop_loss ({mtm_pct:+.0%} <= -{stop_loss_pct:.0%})"
        elif hours_left is not None and 0 < hours_left <= flatten_before_close_hours:
            reason = f"flatten_before_close ({hours_left:.1f}h left)"
        elif hold_hours >= max_hold_hours:
            reason = f"max_hold_exceeded ({hold_hours:.0f}h >= {max_hold_hours:.0f}h)"

        if reason is None:
            continue

        # Sell at the bid to maximize fill probability
        if pos.side == "yes":
            sell_price = int(bid) if bid > 0 else None
        else:
            sell_price = int(no_bid) if no_bid > 0 else None

        use_mkt = sell_price is None or sell_price < 2
        logger.info(
            "EXIT %s: %s | mtm=$%+.2f (%+.0f%%) hold=%.1fh | %s sell @%s",
            ticker, reason, mtm_pnl, mtm_pct * 100, hold_hours,
            "market" if use_mkt else "limit", sell_price if not use_mkt else "mkt",
        )

        try:
            order = client.sell_position(
                ticker=ticker,
                contracts=pos.contracts,
                use_market=use_mkt,
                limit_price=sell_price if not use_mkt else None,
            )
        except Exception as exc:
            logger.warning("Exit sell failed for %s: %s", ticker, exc)
            continue

        if order is None:
            continue

        filled = order.filled_contracts or 0
        status_str = order.status.value if order.status else "unknown"
        if filled == 0 and status_str in ("executed", "filled"):
            filled = pos.contracts

        if filled > 0:
            if pos.side == "yes":
                revenue = filled * (sell_price or order.price) / 100.0
            else:
                revenue = filled * (sell_price or order.price) / 100.0
            exit_cost = pos.cost_dollars * (filled / pos.contracts) if pos.contracts > 0 else 0
            exit_pnl = revenue - exit_cost

            portfolio.realized_pnl += exit_pnl
            if exit_pnl > 0:
                portfolio.trades_won += 1
                portfolio.consecutive_losses = 0
            elif exit_pnl < 0:
                portfolio.trades_lost += 1
                portfolio.consecutive_losses += 1
                portfolio.daily_loss += abs(exit_pnl)
            else:
                portfolio.consecutive_losses = 0

            exits.append(ticker)
            _log_event(log_path, {
                "type": "active_exit",
                "timestamp": now.isoformat(),
                "ticker": ticker,
                "reason": reason,
                "side": pos.side,
                "contracts_sold": filled,
                "sell_price": sell_price or order.price,
                "revenue": revenue,
                "cost_basis": exit_cost,
                "pnl": exit_pnl,
                "cumulative_pnl": portfolio.realized_pnl,
                "hold_hours": hold_hours,
            })
            win_label = "WIN" if exit_pnl > 0 else "LOSS"
            logger.info(
                "EXIT FILLED %s: %s $%+.2f (cumulative: $%+.2f)",
                ticker, win_label, exit_pnl, portfolio.realized_pnl,
            )

    for t in exits:
        pos = portfolio.open_positions.pop(t, None)
        if pos:
            portfolio.closed_positions.append(pos)


def _check_live_settlements(client, portfolio: LivePortfolio, log_path: Path):
    """Check if any open positions have settled on Kalshi."""
    from ..data.sources.kalshi import KalshiAdapter

    adapter = KalshiAdapter(demo=False)
    settled_tickers = []

    actual_positions = None
    exchange_pnl_map = {}
    try:
        actual_positions = {}
        for kp in client.get_positions(strict=True):
            actual_positions[kp.ticker] = kp.position
            if kp.realized_pnl and abs(kp.realized_pnl) > 1e-9:
                exchange_pnl_map[kp.ticker] = kp.realized_pnl
    except Exception as exc:
        logger.warning("Skipping live settlement reconciliation: unable to fetch Kalshi positions (%s)", exc)

    for ticker, pos in portfolio.open_positions.items():
        actual_qty = actual_positions.get(ticker, 0) if actual_positions is not None else None
        if actual_qty == 0:
            # Position gone from exchange. Try to reconcile P&L from exchange
            # data rather than silently dropping it.
            exchange_pnl = exchange_pnl_map.get(ticker)
            if exchange_pnl is not None:
                pos.pnl = exchange_pnl
                pos.settled = True
                portfolio.realized_pnl += pos.pnl
                if pos.pnl > 0:
                    portfolio.trades_won += 1
                    portfolio.consecutive_losses = 0
                elif pos.pnl < 0:
                    portfolio.trades_lost += 1
                    portfolio.consecutive_losses += 1
                    portfolio.daily_loss += abs(pos.pnl)
                else:
                    portfolio.consecutive_losses = 0
                logger.info(
                    "RECONCILED %s: exchange realized_pnl=$%+.2f (cumulative: $%+.2f)",
                    ticker, pos.pnl, portfolio.realized_pnl,
                )
                _log_event(log_path, {
                    "type": "reconciled_settlement",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ticker": ticker,
                    "side": pos.side,
                    "local_contracts": pos.contracts,
                    "exchange_realized_pnl": exchange_pnl,
                    "cumulative_pnl": portfolio.realized_pnl,
                })
            else:
                logger.warning(
                    "Position %s: Kalshi shows 0 contracts, no exchange P&L available — "
                    "removing as phantom (cost=$%.2f unreconciled)",
                    ticker, pos.cost_dollars,
                )
                portfolio._phantom_count += 1
                _log_event(log_path, {
                    "type": "phantom_removed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ticker": ticker,
                    "side": pos.side,
                    "local_contracts": pos.contracts,
                    "kalshi_contracts": 0,
                    "unreconciled_cost": pos.cost_dollars,
                })
            settled_tickers.append(ticker)
            continue
        if actual_qty is not None and abs(actual_qty) < pos.contracts:
            logger.warning(
                "Kalshi position for %s is smaller than local tracking (%s vs %s contracts); "
                "using the exchange quantity for settlement",
                ticker,
                abs(actual_qty),
                pos.contracts,
            )

        try:
            market = adapter.get_market(ticker)
        except Exception:
            continue

        if market.status not in ("settled", "finalized") or market.result is None:
            continue

        pos.settled = True
        pos.outcome = market.result

        settled_contracts = pos.contracts if actual_qty is None else min(pos.contracts, abs(actual_qty))
        settled_cost = pos.cost_dollars
        if actual_qty is not None and settled_contracts < pos.contracts and pos.contracts > 0:
            settled_cost = pos.cost_dollars * (settled_contracts / pos.contracts)

        if pos.side == "yes":
            payout = 1.0 if market.result == "yes" else 0.0
        else:
            payout = 1.0 if market.result == "no" else 0.0

        pos.pnl = (payout * settled_contracts) - settled_cost
        portfolio.realized_pnl += pos.pnl

        if pos.pnl > 0:
            portfolio.trades_won += 1
            portfolio.consecutive_losses = 0
        elif pos.pnl < 0:
            portfolio.trades_lost += 1
            portfolio.consecutive_losses += 1
            portfolio.daily_loss += abs(pos.pnl)
        else:
            portfolio.consecutive_losses = 0

        if pos.pnl != 0:
            try:
                _, total_value = client.get_balance()
                circuit_capital = total_value if total_value > 0 else (portfolio.max_total_deployed + portfolio.realized_pnl)
                client.record_trade_outcome(
                    pnl=pos.pnl,
                    capital=max(0.0, circuit_capital),
                    is_win=pos.pnl > 0,
                    timestamp=datetime.now(timezone.utc),
                )
            except Exception as exc:
                logger.warning("Failed to record live trade outcome in circuit breaker state: %s", exc)

        settled_tickers.append(ticker)

        _log_event(log_path, {
            "type": "settlement",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "side": pos.side,
            "price_cents": pos.price_cents,
            "contracts": settled_contracts,
            "cost": settled_cost,
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
    # Hybrid turnover settings
    hybrid_mode: bool = True,
    fast_capital_frac: float = HYBRID_FAST_CAPITAL_FRAC,
    fast_position_frac: float = HYBRID_FAST_POSITION_FRAC,
    fast_horizon_hours: float = HYBRID_FAST_HORIZON_HOURS,
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
    from ..monitoring.heartbeat import BotHeartbeat

    series_list = series or LIVE_SERIES
    log_path = LOG_DIR / "live_trades.jsonl"
    h1_candidates_log = LOG_DIR / "h1_candidates.jsonl"

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

    # H1-only mode widens the loss-streak / daily-loss limits because the
    # strategy is lottery-style: 80%+ of trades expected to lose, with rare
    # 5x-10x wins. The hard floor is total wealth instead of streak count.
    h1_only_run = os.getenv("H1_ONLY", "").lower() == "true"
    if h1_only_run:
        effective_loss_streak = 9999
        effective_daily_loss = 9999.0
    else:
        effective_loss_streak = max_loss_streak
        effective_daily_loss = max_daily_loss

    portfolio = LivePortfolio(
        max_cost_per_trade=max_cost_per_trade,
        max_total_deployed=max_total_deployed,
        max_positions=max_positions,
        max_loss_streak=effective_loss_streak,
        max_daily_loss=effective_daily_loss,
    )
    session_start_time = datetime.now(timezone.utc)
    session_id = session_start_time.strftime("live_%Y%m%d_%H%M%S")

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

    allow_unreconciled_start = os.getenv("KALSHI_ALLOW_UNRECONCILED_STARTUP", "false").lower() == "true"
    try:
        existing_positions = client.get_active_positions(strict=True)
        existing_orders = client.get_open_orders(strict=True)
    except Exception as exc:
        logger.error("Startup reconciliation failed: %s", exc)
        return portfolio
    if (existing_positions or existing_orders) and not allow_unreconciled_start:
        logger.error(
            "Startup blocked: Kalshi account already has %s open positions and %s open orders. "
            "Resolve them first or set KALSHI_ALLOW_UNRECONCILED_STARTUP=true to override.",
            len(existing_positions),
            len(existing_orders),
        )
        _log_event(log_path, {
            "type": "startup_reconcile_blocked",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open_positions": len(existing_positions),
            "open_orders": len(existing_orders),
        })
        return portfolio

    mode_label = "DRY RUN" if dry_run else "LIVE"
    heartbeat = BotHeartbeat(
        interval=60,
        bot_id="kalshi_live" if not dry_run else "kalshi_live_dry_run",
        metadata_fn=lambda: {
            "mode": "live" if not dry_run else "dry_run",
            "session_id": session_id,
            "scan_count": portfolio.scan_count,
            "resting_orders": len(portfolio.resting_orders),
            "open_positions": len(portfolio.open_positions),
            "active_markets": portfolio.active_market_count,
            "deployed_capital": round(portfolio.deployed_capital, 2),
            "realized_pnl": round(portfolio.realized_pnl, 2),
        },
    )
    heartbeat.start()
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
    if hybrid_mode:
        fast_max_pos = max(1, int(max_positions * fast_position_frac))
        macro_max_pos = max(1, max_positions - fast_max_pos)
        logger.info(f"Hybrid mode:   ON (fast {fast_capital_frac:.0%} / macro {1-fast_capital_frac:.0%})")
        logger.info(f"  Fast slots:  {fast_max_pos} | Macro slots: {macro_max_pos}")
        logger.info(f"  Fast budget: ${max_total_deployed * fast_capital_frac:.2f} | Macro budget: ${max_total_deployed * (1-fast_capital_frac):.2f}")
        logger.info(f"  Fast horizon: {fast_horizon_hours:.0f}h")
        logger.info(f"  Macro portfolio cap: {HYBRID_MACRO_MAX_PORTFOLIO_FRAC:.0%} of total")
        logger.info("  Active exits: profit>=40%% | stop<=-60%% | flatten<1h | max-hold<=168h")

    macro_enabled = os.getenv("KALSHI_MACRO_ENABLED", "").lower() == "true" or MACRO_ENABLED_DEFAULT
    logger.info(f"Trade admission gates:")
    logger.info(f"  Min execution edge:  {MIN_EXECUTION_EDGE:.0%}")
    logger.info(f"  Max spread:          {MAX_SPREAD_CENTS:.0f}c")
    logger.info(f"  Max contracts/ticker: {MAX_CONTRACTS_PER_TICKER}")
    logger.info(f"  Min activity (vol+OI): {MIN_MARKET_ACTIVITY}")
    logger.info(f"  Min hours to expiry: {MIN_HOURS_TO_CLOSE_ON_ENTRY:.1f}h (prevents instant flatten)")
    logger.info(f"  Macro enabled:       {'YES' if macro_enabled else 'NO (set KALSHI_MACRO_ENABLED=true)'}")
    if h1_only_run:
        logger.info("=" * 50)
        logger.info("H1-ONLY MODE ACTIVE (audit-derived lottery edge)")
        logger.info("=" * 50)
        logger.info(f"  Side:               BUY_NO only")
        logger.info(f"  Edge type:          spot_vs_strike only")
        logger.info(f"  Crypto assets:      {sorted(H1_CRYPTO_ASSETS)}")
        logger.info(f"  Min OTM distance:   {H1_MIN_SPOT_DISTANCE_PCT:.0%} from spot")
        logger.info(f"  Time window:        {H1_MIN_HOURS:.1f}h - {H1_MAX_HOURS:.1f}h to settlement")
        logger.info(f"  Flatten exit:       DISABLED (rides to settlement)")
        logger.info(f"  Stop-loss exit:     DISABLED (lottery-style)")
        logger.info(f"  Streak kill switch: DISABLED")
        logger.info(f"  Hard floor:         total wealth < ${H1_HARD_FLOOR_TOTAL_WEALTH:.2f} stops bot")

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
        "session_id": session_id,
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

            # Reset daily loss at UTC day boundary
            portfolio.maybe_reset_daily_loss()

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
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": portfolio.kill_reason,
                })
                break

            scan_ts = datetime.now(timezone.utc).isoformat()

            # 1a. Poll resting orders for fills
            if portfolio.resting_orders:
                _check_resting_fills(client, portfolio, log_path, session_id)

            # 1b. Check settlements
            if portfolio.open_positions:
                _check_live_settlements(client, portfolio, log_path)

            # 1c. Check active exit opportunities (profit-take / stop-loss / flatten)
            # H1 mode disables flatten and stop-loss because the strategy is a
            # lottery-ticket: trades are deliberately deep-OTM, expected to
            # mostly expire worthless, and the rare wins must be allowed to
            # ride to full $1 settlement. Profit-take and max-hold still apply.
            if portfolio.open_positions:
                h1_active = os.getenv("H1_ONLY", "").lower() == "true"
                _check_exit_opportunities(
                    client, adapter, portfolio, log_path,
                    profit_target_pct=0.40,
                    stop_loss_pct=10.0 if h1_active else 0.60,
                    max_hold_hours=168.0,
                    flatten_before_close_hours=0.0 if h1_active else 1.0,
                )

            # 1d. Refresh exchange-side available balance as hard cap for sizing
            try:
                exchange_cash, exchange_total = client.get_balance()
            except Exception:
                exchange_cash = 0.0
                exchange_total = 0.0

            # 1e. H1-mode hard wealth floor: stop everything if total wealth
            # (cash + positions mark-to-market) falls below the floor. This
            # replaces the kill switch as the discipline mechanism for the
            # lottery-ticket strategy.
            if h1_only_run:
                total_wealth = exchange_cash + exchange_total
                if total_wealth < H1_HARD_FLOOR_TOTAL_WEALTH and total_wealth > 0:
                    logger.warning(
                        f"H1 HARD FLOOR HIT: total wealth ${total_wealth:.2f} "
                        f"< ${H1_HARD_FLOOR_TOTAL_WEALTH:.2f}. Stopping."
                    )
                    portfolio.killed = True
                    portfolio.kill_reason = (
                        f"H1 hard floor hit: total wealth ${total_wealth:.2f}"
                    )
                    _send_alert(
                        "H1 hard floor hit",
                        f"Total wealth ${total_wealth:.2f} below ${H1_HARD_FLOOR_TOTAL_WEALTH:.2f}. Bot stopped.",
                    )
                    break

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

            # 3b. Hybrid mode: classify edges into sleeves, compute budgets
            if hybrid_mode:
                fast_max_positions = max(1, int(max_positions * fast_position_frac))
                macro_max_positions = max(1, max_positions - fast_max_positions)
                fast_max_capital = max_total_deployed * fast_capital_frac
                macro_max_capital = max_total_deployed * (1.0 - fast_capital_frac)

                fast_open = sum(
                    1 for t in portfolio.open_positions
                    if classify_sleeve(portfolio.open_positions[t].__dict__, fast_horizon_hours) == "fast"
                       or _extract_asset(t) in {a for s in FAST_SERIES for a in [_extract_asset(s)]}
                )
                macro_open = len(portfolio.open_positions) - fast_open
                fast_resting = sum(
                    1 for t, o in portfolio.resting_orders.items()
                    if classify_sleeve({"series_ticker": t.split("-")[0], "close_time": None}, fast_horizon_hours) == "fast"
                )
                macro_resting = sum(
                    1 for t, o in portfolio.resting_orders.items()
                    if classify_sleeve({"series_ticker": t.split("-")[0], "close_time": None}, fast_horizon_hours) == "macro"
                )

                fast_deployed = sum(
                    p.cost_dollars for p in portfolio.open_positions.values()
                    if _extract_asset(p.ticker) not in {"CPI", "NFP", "FED", "WEATHER"}
                ) + sum(
                    o.reserved_cost_dollars for o in portfolio.resting_orders.values()
                    if _extract_asset(o.ticker) not in {"CPI", "NFP", "FED", "WEATHER"}
                )
                macro_deployed = portfolio.deployed_capital - fast_deployed

                fast_slots_left = max(0, fast_max_positions - fast_open - fast_resting)
                macro_slots_left = max(0, macro_max_positions - macro_open - macro_resting)
                fast_capital_left = max(0.0, fast_max_capital - fast_deployed)
                macro_capital_left = max(0.0, macro_max_capital - macro_deployed)

                def _hybrid_score(e):
                    """Score an edge: base edge*confidence + time bonus for fast sleeve."""
                    base = e.edge_value * (e.confidence if hasattr(e, "confidence") and e.confidence else 1.0)
                    h = hours_to_close(e.market_data) if e.market_data else None
                    if h is not None and h <= 24:
                        base *= 1.5
                    elif h is not None and h <= fast_horizon_hours:
                        base *= 1.2
                    return base

                edges = sorted(edges, key=_hybrid_score, reverse=True)

            # 4. Filter and place orders
            new_trades = 0
            _blk = {"type": 0, "side": 0, "dup": 0, "edge_range": 0, "dead": 0, "price": 0,
                     "sleeve_full": 0}
            _sleeve_stats = {"fast_candidates": 0, "macro_candidates": 0, "other_candidates": 0,
                             "fast_placed": 0, "macro_placed": 0}
            h1_only = os.getenv("H1_ONLY", "").lower() == "true"

            for edge in edges:
                if edge.edge_type not in ("spot_vs_strike", "crypto_spot_mispricing", "macro_data", "weather"):
                    _blk["type"] += 1
                    continue

                if edge.recommended_side not in allowed_side_set:
                    _blk["side"] += 1
                    continue

                # H1-ONLY MODE: enforce the audit-derived "only proven edge"
                # profile. All filters must pass or the edge is rejected with
                # a counted-blocker reason for transparency.
                # Also logs every NO-side crypto candidate (passing or not)
                # to a JSONL so we can study the regime even when no trades
                # fire.
                if h1_only:
                    if edge.edge_type != "spot_vs_strike":
                        _blk["h1_type"] = _blk.get("h1_type", 0) + 1
                        continue
                    if edge.recommended_side != "no":
                        _blk["h1_side"] = _blk.get("h1_side", 0) + 1
                        continue
                    asset = edge.market_data.get("_h1_asset") if edge.market_data else None
                    if asset not in H1_CRYPTO_ASSETS:
                        _blk["h1_asset"] = _blk.get("h1_asset", 0) + 1
                        continue
                    md = edge.market_data or {}
                    spot = md.get("_h1_spot")
                    strike_type = md.get("_h1_strike_type")
                    floor_strike = md.get("_h1_floor_strike")
                    cap_strike = md.get("_h1_cap_strike")
                    h1_hours = hours_to_close(edge.market_data) if edge.market_data else None

                    # OTM-distance: spot must be deep enough away from the
                    # relevant strike that "buy NO" is buying obvious tail
                    # probability, not a model-derived edge.
                    #   greater [floor]: OTM YES means spot << floor
                    #   less [floor]:    OTM YES means spot >> floor
                    #   between [lo,hi]: OTM YES means spot is OUTSIDE [lo,hi];
                    #                    distance is to the nearer boundary.
                    distance_pct: Optional[float] = None
                    if not spot or spot <= 0 or not strike_type:
                        distance_pct = None
                    elif strike_type == "between" and floor_strike is not None and cap_strike is not None:
                        lo, hi = sorted((float(floor_strike), float(cap_strike)))
                        if lo <= spot <= hi:
                            distance_pct = 0.0
                        else:
                            distance_pct = min(abs(spot - lo), abs(spot - hi)) / spot
                    elif strike_type == "greater" and floor_strike is not None:
                        distance_pct = max(0.0, (float(floor_strike) - spot)) / spot
                    elif strike_type == "less" and floor_strike is not None:
                        distance_pct = max(0.0, (spot - float(floor_strike))) / spot

                    # Determine pass/fail for each H1 gate. We do this even
                    # for blocked candidates so we can study what the regime
                    # looked like at decision-time.
                    pass_distance = distance_pct is not None and distance_pct >= H1_MIN_SPOT_DISTANCE_PCT
                    pass_window = h1_hours is not None and H1_MIN_HOURS <= h1_hours <= H1_MAX_HOURS
                    h1_passes_all = (distance_pct is not None and h1_hours is not None
                                     and pass_distance and pass_window)

                    # Append candidate to the H1 ground-truth log. This file
                    # accumulates every NO-side crypto edge we evaluated so
                    # post-hoc analysis can correlate distance/time/edge to
                    # actual market settlement outcomes.
                    try:
                        _log_event(h1_candidates_log, {
                            "ts": scan_ts,
                            "scan": scan_n,
                            "ticker": edge.ticker,
                            "asset": asset,
                            "strike_type": strike_type,
                            "floor_strike": floor_strike,
                            "cap_strike": cap_strike,
                            "spot": spot,
                            "distance_pct": distance_pct,
                            "hours_to_close": h1_hours,
                            "edge_value": edge.edge_value,
                            "market_price": edge.market_price,
                            "fair_price": edge.fair_price,
                            "yes_bid": float(md.get("yes_bid", 0) or 0),
                            "yes_ask": float(md.get("yes_ask", 100) or 100),
                            "volume": float(md.get("volume", 0) or 0),
                            "open_interest": float(md.get("open_interest", 0) or 0),
                            "pass_distance": pass_distance,
                            "pass_window": pass_window,
                            "h1_passes_all": h1_passes_all,
                        })
                    except Exception:
                        pass

                    # Now actually enforce the gates.
                    if distance_pct is None:
                        _blk["h1_no_spot"] = _blk.get("h1_no_spot", 0) + 1
                        continue
                    if not pass_distance:
                        _blk["h1_near_money"] = _blk.get("h1_near_money", 0) + 1
                        continue
                    if h1_hours is None:
                        _blk["h1_no_expiry"] = _blk.get("h1_no_expiry", 0) + 1
                        continue
                    if not pass_window:
                        _blk["h1_window"] = _blk.get("h1_window", 0) + 1
                        continue

                if edge.ticker in portfolio.open_positions or edge.ticker in portfolio.resting_orders:
                    _blk["dup"] += 1
                    continue

                if edge.edge_value < min_edge or edge.edge_value > max_edge:
                    _blk["edge_range"] += 1
                    continue

                # Macro admission gate: macro_data edges are blocked by default.
                macro_enabled = os.getenv("KALSHI_MACRO_ENABLED", "").lower() == "true" or MACRO_ENABLED_DEFAULT
                if edge.edge_type == "macro_data" and not macro_enabled:
                    _blk["macro_blocked"] = _blk.get("macro_blocked", 0) + 1
                    continue

                # Skip dead markets (insufficient activity for a realistic fill)
                mkt_volume = float(edge.market_data.get("volume", 0) or 0)
                mkt_oi = float(edge.market_data.get("open_interest", 0) or 0)
                if mkt_volume + mkt_oi < MIN_MARKET_ACTIVITY:
                    _blk["dead"] += 1
                    continue

                # Determine actual order price from live bid/ask.
                yes_bid = float(edge.market_data.get("yes_bid", 0) or 0)
                yes_ask = float(edge.market_data.get("yes_ask", 100) or 100)
                spread = yes_ask - yes_bid
                edge_price = edge.market_price
                if edge.recommended_side == "no":
                    price = yes_bid if yes_bid > 0 else edge_price
                    our_cost_cents = 100 - price
                else:
                    price = yes_ask if yes_ask < 100 else edge_price
                    our_cost_cents = price

                # Spread gate: wide spreads destroy the execution edge.
                if spread > MAX_SPREAD_CENTS:
                    _blk["spread"] = _blk.get("spread", 0) + 1
                    continue

                # Execution edge: what is the actual return profile?
                # For a $1 payout contract, execution_edge = (100 - cost) / cost
                if our_cost_cents > 0:
                    execution_edge = (100.0 - our_cost_cents) / our_cost_cents
                else:
                    execution_edge = 99.0
                if execution_edge < MIN_EXECUTION_EDGE:
                    _blk["exec_edge"] = _blk.get("exec_edge", 0) + 1
                    continue

                # Price filter: applied to OUR cost (what we actually pay).
                if our_cost_cents < min_price or our_cost_cents > max_price:
                    _blk["price"] += 1
                    continue

                # Time-to-expiry guard: never open a position the exit logic
                # will immediately flatten on the next scan. We enforce a
                # minimum life-left at entry of MIN_HOURS_TO_CLOSE_ON_ENTRY
                # hours, which is flatten_threshold + a scan-interval buffer.
                # In H1-only mode this guard is bypassed because its narrow
                # 0.5–2h window is intentional and the flatten exit is also
                # disabled so trades ride to settlement.
                if not h1_only:
                    mkt_hours_left = hours_to_close(edge.market_data) if edge.market_data else None
                    if mkt_hours_left is not None and mkt_hours_left < MIN_HOURS_TO_CLOSE_ON_ENTRY:
                        _blk["expiring_soon"] = _blk.get("expiring_soon", 0) + 1
                        continue

                # Hybrid sleeve gating
                sleeve = classify_sleeve(edge.market_data, fast_horizon_hours) if hybrid_mode else "any"
                if hybrid_mode:
                    _sleeve_stats[f"{sleeve}_candidates"] = _sleeve_stats.get(f"{sleeve}_candidates", 0) + 1

                    if sleeve == "fast":
                        if fast_slots_left <= 0 or fast_capital_left <= 0:
                            _blk["sleeve_full"] += 1
                            continue
                    elif sleeve == "macro":
                        if macro_slots_left <= 0 or macro_capital_left <= 0:
                            _blk["sleeve_full"] += 1
                            continue
                        # Absolute macro cap: block new macro entries if macro
                        # exposure already exceeds HYBRID_MACRO_MAX_PORTFOLIO_FRAC
                        # of total portfolio value.
                        if exchange_total > 0 and macro_deployed / exchange_total >= HYBRID_MACRO_MAX_PORTFOLIO_FRAC:
                            _blk["sleeve_full"] += 1
                            continue
                    else:
                        # 'other' markets: only allow if both sleeves have room
                        if (fast_slots_left <= 0 and macro_slots_left <= 0):
                            _blk["sleeve_full"] += 1
                            continue

                if portfolio.active_market_count >= max_positions:
                    break

                MAX_NEW_TRADES_PER_SCAN = 6
                if new_trades >= MAX_NEW_TRADES_PER_SCAN:
                    break

                # Concentration limit: max 30% per asset, max 50% per correlated
                # cluster (e.g. CPI+NFP are both macro-employment linked).
                MACRO_CLUSTER = {"CPI", "NFP", "FED"}
                asset = _extract_asset(edge.ticker)
                asset_cost = sum(
                    p.cost_dollars for p in portfolio.open_positions.values()
                    if _extract_asset(p.ticker) == asset
                )
                asset_cost += sum(
                    o.reserved_cost_dollars for o in portfolio.resting_orders.values()
                    if _extract_asset(o.ticker) == asset
                )
                if asset_cost >= max_total_deployed * 0.30:
                    continue

                if asset in MACRO_CLUSTER:
                    cluster_cost = sum(
                        p.cost_dollars for p in portfolio.open_positions.values()
                        if _extract_asset(p.ticker) in MACRO_CLUSTER
                    )
                    cluster_cost += sum(
                        o.reserved_cost_dollars for o in portfolio.resting_orders.values()
                        if _extract_asset(o.ticker) in MACRO_CLUSTER
                    )
                    if cluster_cost >= max_total_deployed * 0.50:
                        continue

                # Size: how many contracts can we afford?
                # Use the lesser of local accounting, exchange cash, and per-ticker cap.
                if edge.recommended_side == "yes":
                    cost_per_contract = price / 100.0
                else:
                    cost_per_contract = (100 - price) / 100.0
                max_by_trade_limit = int(max_cost_per_trade / cost_per_contract) if cost_per_contract > 0 else 0
                max_by_capital = int(portfolio.available_to_deploy / cost_per_contract) if cost_per_contract > 0 else 0
                max_by_exchange = int(exchange_cash / cost_per_contract) if cost_per_contract > 0 else 0

                # Per-ticker cap: count existing contracts on this exact ticker
                existing_on_ticker = 0
                if edge.ticker in portfolio.open_positions:
                    existing_on_ticker += portfolio.open_positions[edge.ticker].contracts
                if edge.ticker in portfolio.resting_orders:
                    existing_on_ticker += portfolio.resting_orders[edge.ticker].contracts_requested
                max_by_ticker = max(0, MAX_CONTRACTS_PER_TICKER - existing_on_ticker)

                contracts = min(max_by_trade_limit, max_by_capital, max_by_exchange, max_by_ticker)

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
                        "session_id": session_id,
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

                # Pre-check: skip markets closing within one scan interval
                close_time_utc = detector._parse_ts(edge.market_data.get("close_time"))
                if close_time_utc is not None:
                    secs_to_close = (close_time_utc - datetime.now(timezone.utc)).total_seconds()
                    if 0 < secs_to_close <= interval_seconds:
                        logger.debug(
                            "Skipping %s: closes in %ds (< scan interval %ds)",
                            edge.ticker, int(secs_to_close), interval_seconds,
                        )
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
                    close_time_utc=close_time_utc,
                )

                if order is None:
                    # Order rejected by exchange client (circuit breaker, no-trade
                    # window, or exchange-side position limit). Do NOT break the
                    # whole scan -- skip this edge and try the next one.
                    logger.warning("Order blocked for %s (client returned None)", edge.ticker)
                    continue

                filled = order.filled_contracts or 0
                status_str = order.status.value if order.status else "unknown"

                # Kalshi v2 returns 'executed' for immediately-filled orders.
                # If status says executed but fill_count is missing, assume fully filled.
                if filled == 0 and status_str in ("executed", "filled"):
                    filled = contracts

                if filled == 0:
                    logger.info(
                        f"ORDER RESTING (not filled): {edge.ticker} status={status_str} "
                        f"order={order.order_id[:8]} — will poll for fills"
                    )
                    portfolio.resting_orders[edge.ticker] = RestingOrder(
                        ticker=edge.ticker,
                        event_ticker=edge.event_ticker,
                        side=edge.recommended_side,
                        order_id=order.order_id,
                        price_cents=price,
                        contracts_requested=contracts,
                        cost_per_contract=cost_per_contract,
                        edge_value=edge.edge_value,
                        edge_type=edge.edge_type,
                        reasoning=edge.reasoning,
                        placed_at=scan_ts,
                    )
                    exchange_cash -= total_cost
                    if hybrid_mode:
                        if sleeve == "fast":
                            fast_slots_left -= 1
                            fast_capital_left -= total_cost
                        elif sleeve == "macro":
                            macro_slots_left -= 1
                            macro_capital_left -= total_cost
                    _log_event(log_path, {
                        "type": "order_resting",
                        "session_id": session_id,
                        "timestamp": scan_ts,
                        "ticker": edge.ticker,
                        "side": edge.recommended_side,
                        "price_cents": price,
                        "contracts": contracts,
                        "remaining_contracts": contracts,
                        "cost": total_cost,
                        "edge": edge.edge_value,
                        "edge_type": edge.edge_type,
                        "order_id": order.order_id,
                        "order_status": status_str,
                        "reasoning": edge.reasoning,
                    })
                    continue

                actual_cost = filled * cost_per_contract
                portfolio.trades_taken += 1
                pos = LivePosition(
                    ticker=edge.ticker,
                    event_ticker=edge.event_ticker,
                    side=edge.recommended_side,
                    order_id=order.order_id,
                    price_cents=price,
                    contracts=filled,
                    cost_dollars=actual_cost,
                    edge_value=edge.edge_value,
                    edge_type=edge.edge_type,
                    reasoning=edge.reasoning,
                    opened_at=scan_ts,
                )
                portfolio.open_positions[edge.ticker] = pos
                new_trades += 1
                exchange_cash -= actual_cost

                # Update hybrid sleeve counters
                if hybrid_mode:
                    if sleeve == "fast":
                        fast_slots_left -= 1
                        fast_capital_left -= actual_cost
                        _sleeve_stats["fast_placed"] += 1
                    elif sleeve == "macro":
                        macro_slots_left -= 1
                        macro_capital_left -= actual_cost
                        _sleeve_stats["macro_placed"] += 1

                _log_event(log_path, {
                    "type": "order_placed",
                    "session_id": session_id,
                    "timestamp": scan_ts,
                    "ticker": edge.ticker,
                    "event_ticker": edge.event_ticker,
                    "side": edge.recommended_side,
                    "price_cents": price,
                    "contracts": filled,
                    "cost": actual_cost,
                    "edge": edge.edge_value,
                    "execution_edge": round(execution_edge, 3),
                    "spread": spread,
                    "edge_type": edge.edge_type,
                    "order_id": order.order_id,
                    "order_status": status_str,
                    "reasoning": edge.reasoning,
                })

                logger.info(
                    f"ORDER FILLED: {edge.ticker} BUY_{edge.recommended_side.upper()} "
                    f"{filled}@{price}¢ cost=${actual_cost:.2f} order={order.order_id[:8]}"
                )

            # 5. Summary
            logger.info(
                f"Scan {scan_n}: {len(markets)} markets | "
                f"{len(edges)} edges | {new_trades} new orders | "
                f"open={len(portfolio.open_positions)} | "
                f"resting={len(portfolio.resting_orders)} | "
                f"deployed=${portfolio.deployed_capital:.2f} | "
                f"P&L=${portfolio.realized_pnl:+.2f}"
            )
            if new_trades == 0 and edges:
                logger.info(
                    f"  Filter breakdown: type={_blk['type']} side={_blk['side']} "
                    f"dup={_blk['dup']} edge_range={_blk['edge_range']} "
                    f"dead={_blk['dead']} spread={_blk.get('spread', 0)} "
                    f"exec_edge={_blk.get('exec_edge', 0)} price={_blk['price']} "
                    f"macro_blocked={_blk.get('macro_blocked', 0)} "
                    f"expiring_soon={_blk.get('expiring_soon', 0)} "
                    f"sleeve_full={_blk.get('sleeve_full', 0)} "
                    f"other={len(edges) - sum(_blk.values())}"
                )
                if os.getenv("H1_ONLY", "").lower() == "true":
                    logger.info(
                        f"  H1 filter:        h1_type={_blk.get('h1_type', 0)} "
                        f"h1_side={_blk.get('h1_side', 0)} "
                        f"h1_asset={_blk.get('h1_asset', 0)} "
                        f"h1_no_spot={_blk.get('h1_no_spot', 0)} "
                        f"h1_near_money={_blk.get('h1_near_money', 0)} "
                        f"h1_no_expiry={_blk.get('h1_no_expiry', 0)} "
                        f"h1_window={_blk.get('h1_window', 0)}"
                    )
            if hybrid_mode:
                settling_24h = sum(
                    1 for p in portfolio.open_positions.values()
                    if hours_to_close(p.__dict__) is not None and 0 < hours_to_close(p.__dict__) <= 24
                )
                settling_72h = sum(
                    1 for p in portfolio.open_positions.values()
                    if hours_to_close(p.__dict__) is not None and 0 < hours_to_close(p.__dict__) <= 72
                )
                logger.info(
                    f"  Sleeves: fast_cand={_sleeve_stats.get('fast_candidates', 0)} "
                    f"macro_cand={_sleeve_stats.get('macro_candidates', 0)} "
                    f"fast_placed={_sleeve_stats.get('fast_placed', 0)} "
                    f"macro_placed={_sleeve_stats.get('macro_placed', 0)} | "
                    f"settling <24h={settling_24h} <72h={settling_72h}"
                )

            _log_event(log_path, {
                "type": "scan_summary",
                "session_id": session_id,
                "timestamp": scan_ts,
                "scan": scan_n,
                "markets_scanned": len(markets),
                "edges_found": len(edges),
                "new_trades": new_trades,
                "resting_orders": len(portfolio.resting_orders),
                "open_positions": len(portfolio.open_positions),
                "deployed_capital": portfolio.deployed_capital,
                "realized_pnl": portfolio.realized_pnl,
                "exchange_cash": exchange_cash,
                "exchange_total": exchange_total,
                "closed_positions": len(portfolio.closed_positions),
                "blocked_by": {k: v for k, v in _blk.items() if v > 0},
            })

            # Hourly status report (every ~12 scans at 300s interval)
            if scan_n % 12 == 1 or scan_n == 1:
                try:
                    avail_bal, total_bal = client.get_balance()
                except Exception:
                    avail_bal = total_bal = 0.0
                logger.info(
                    "HOURLY STATUS | cash=$%.2f portfolio=$%.2f | open=%d resting=%d "
                    "deployed=$%.2f | realized=$%+.2f daily_loss=$%.2f | W/L=%d/%d streak=%d",
                    avail_bal, total_bal,
                    len(portfolio.open_positions), len(portfolio.resting_orders),
                    portfolio.deployed_capital, portfolio.realized_pnl,
                    portfolio.daily_loss, portfolio.trades_won, portfolio.trades_lost,
                    portfolio.consecutive_losses,
                )

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
    finally:
        # Compute turnover metrics for session report
        session_end = datetime.now(timezone.utc)
        session_hours = (session_end - session_start_time).total_seconds() / 3600.0
        total_closed = len(portfolio.closed_positions)
        avg_hold_hours = 0.0
        if total_closed > 0:
            hold_sum = 0.0
            for cp in portfolio.closed_positions:
                try:
                    opened = datetime.fromisoformat(cp.opened_at)
                    if opened.tzinfo is None:
                        opened = opened.replace(tzinfo=timezone.utc)
                    hold_sum += (session_end - opened).total_seconds() / 3600.0
                except (ValueError, TypeError):
                    pass
            avg_hold_hours = hold_sum / total_closed
        daily_pnl = portfolio.realized_pnl / max(session_hours / 24.0, 0.01)
        win_rate = portfolio.trades_won / max(portfolio.trades_won + portfolio.trades_lost, 1)

        # Sleeve-level analytics
        sleeve_stats = {"fast": {"pnl": 0.0, "wins": 0, "losses": 0, "cost": 0.0, "open": 0},
                        "macro": {"pnl": 0.0, "wins": 0, "losses": 0, "cost": 0.0, "open": 0},
                        "other": {"pnl": 0.0, "wins": 0, "losses": 0, "cost": 0.0, "open": 0}}

        def _ticker_sleeve(ticker: str) -> str:
            series = ticker.split("-")[0] if ticker else ""
            if series in FAST_SERIES:
                return "fast"
            if series in MACRO_SERIES_SET:
                return "macro"
            return "other"

        for cp in portfolio.closed_positions:
            sl = _ticker_sleeve(cp.ticker)
            sleeve_stats[sl]["pnl"] += cp.pnl
            sleeve_stats[sl]["cost"] += cp.cost_dollars
            if cp.pnl > 0:
                sleeve_stats[sl]["wins"] += 1
            elif cp.pnl < 0:
                sleeve_stats[sl]["losses"] += 1

        for pos in portfolio.open_positions.values():
            sl = _ticker_sleeve(pos.ticker)
            sleeve_stats[sl]["open"] += 1
            sleeve_stats[sl]["cost"] += pos.cost_dollars

        open_capital = sum(p.cost_dollars for p in portfolio.open_positions.values())
        resting_capital = sum(r.reserved_cost_dollars for r in portfolio.resting_orders.values())
        phantom_count = getattr(portfolio, "_phantom_count", 0)

        _log_event(log_path, {
            "type": "session_end",
            "session_id": session_id,
            "timestamp": session_end.isoformat(),
            "scans": scan_n,
            "session_hours": round(session_hours, 2),
            "trades_taken": portfolio.trades_taken,
            "trades_won": portfolio.trades_won,
            "trades_lost": portfolio.trades_lost,
            "win_rate": round(win_rate, 3),
            "realized_pnl": portfolio.realized_pnl,
            "daily_pnl_rate": round(daily_pnl, 4),
            "positions_closed": total_closed,
            "positions_open": len(portfolio.open_positions),
            "avg_hold_hours": round(avg_hold_hours, 1),
            "resting_orders": len(portfolio.resting_orders),
            "killed": portfolio.killed,
            "kill_reason": portfolio.kill_reason,
            "open_capital": round(open_capital, 2),
            "resting_capital": round(resting_capital, 2),
            "phantom_count": phantom_count,
            "sleeves": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                            for kk, vv in v.items()}
                        for k, v in sleeve_stats.items()},
        })

        logger.info("=" * 50)
        logger.info("SESSION TURNOVER REPORT")
        logger.info("=" * 50)
        logger.info("Duration:       %.1f hours", session_hours)
        logger.info("Trades taken:   %d (W:%d L:%d, %.0f%% win)",
                     portfolio.trades_taken, portfolio.trades_won,
                     portfolio.trades_lost, win_rate * 100)
        logger.info("Realized P&L:   $%+.2f", portfolio.realized_pnl)
        logger.info("Daily P&L rate: $%+.2f/day", daily_pnl)
        logger.info("Positions:      %d closed, %d still open",
                     total_closed, len(portfolio.open_positions))
        logger.info("Avg hold time:  %.1f hours", avg_hold_hours)
        for sl_name in ("fast", "macro", "other"):
            sl = sleeve_stats[sl_name]
            if sl["wins"] or sl["losses"] or sl["open"]:
                logger.info("  %s: %dW/%dL pnl=$%+.2f open=%d cost=$%.2f",
                            sl_name, sl["wins"], sl["losses"],
                            sl["pnl"], sl["open"], sl["cost"])
        if phantom_count:
            logger.info("Phantoms:       %d removed this session", phantom_count)
        logger.info("=" * 50)

        heartbeat.stop()

    return portfolio
