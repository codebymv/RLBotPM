import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BOT_DIR = REPO_ROOT / "bot"
for path in (str(REPO_ROOT), str(BOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.strategies.live_trader import LivePortfolio, LivePosition, RestingOrder
from src.execution.kalshi_client import KalshiExecutionClient, KalshiPosition

try:
    from api.main import _read_live_trade_log_metrics
except ModuleNotFoundError:
    _read_live_trade_log_metrics = None


def test_live_portfolio_counts_resting_exposure():
    portfolio = LivePortfolio(max_total_deployed=10.0)
    portfolio.open_positions["BTC"] = LivePosition(
        ticker="BTCUSD-TEST",
        event_ticker="BTCUSD-TEST",
        side="no",
        order_id="ord-open",
        price_cents=5,
        contracts=2,
        cost_dollars=0.40,
        edge_value=0.02,
        edge_type="crypto_spot_mispricing",
        reasoning="filled",
        opened_at="2026-04-12T00:00:00+00:00",
    )
    portfolio.resting_orders["ETHUSD-TEST"] = RestingOrder(
        ticker="ETHUSD-TEST",
        event_ticker="ETHUSD-TEST",
        side="no",
        order_id="ord-rest",
        price_cents=4,
        contracts_requested=5,
        cost_per_contract=0.20,
        edge_value=0.03,
        edge_type="crypto_spot_mispricing",
        reasoning="resting",
        placed_at="2026-04-12T00:00:00+00:00",
        filled_contracts=2,
    )

    assert portfolio.open_position_capital == 0.40
    assert portfolio.resting_capital == pytest.approx(0.60)
    assert portfolio.deployed_capital == pytest.approx(1.00)
    assert portfolio.available_to_deploy == pytest.approx(9.00)
    assert portfolio.active_market_count == 2


def test_active_position_filter_ignores_zero_rows():
    zero_row = KalshiPosition(
        ticker="BTCUSD-ZERO",
        position=0,
        market_exposure=0.0,
        realized_pnl=0.0,
        total_cost=0.0,
    )
    active_row = KalshiPosition(
        ticker="BTCUSD-LIVE",
        position=-2,
        market_exposure=1.5,
        realized_pnl=0.0,
        total_cost=1.5,
    )

    assert not KalshiExecutionClient.is_active_position(zero_row)
    assert KalshiExecutionClient.is_active_position(active_row)


def test_live_log_replay_handles_resting_fill_and_close(tmp_path):
    if _read_live_trade_log_metrics is None:
        pytest.skip("fastapi dependencies are not installed in this test environment")
    log_path = tmp_path / "live_trades.jsonl"
    events = [
        {
            "type": "session_start",
            "session_id": "live_1",
            "timestamp": "2026-04-12T00:00:00+00:00",
            "allowed_sides": ["no"],
            "series": ["BTCUSD"],
        },
        {
            "type": "order_resting",
            "session_id": "live_1",
            "timestamp": "2026-04-12T00:01:00+00:00",
            "ticker": "BTCUSD-TEST",
            "event_ticker": "BTCUSD-TEST",
            "side": "no",
            "price_cents": 5,
            "contracts": 5,
            "remaining_contracts": 5,
            "cost": 0.50,
            "edge": 0.02,
            "edge_type": "crypto_spot_mispricing",
            "order_id": "ord-1",
            "reasoning": "resting",
        },
        {
            "type": "order_filled",
            "session_id": "live_1",
            "timestamp": "2026-04-12T00:02:00+00:00",
            "ticker": "BTCUSD-TEST",
            "event_ticker": "BTCUSD-TEST",
            "side": "no",
            "price_cents": 5,
            "contracts": 2,
            "filled_contracts_total": 2,
            "remaining_contracts": 3,
            "cost": 0.20,
            "edge": 0.02,
            "edge_type": "crypto_spot_mispricing",
            "order_id": "ord-1",
            "reasoning": "partial fill",
        },
        {
            "type": "order_closed",
            "session_id": "live_1",
            "timestamp": "2026-04-12T00:03:00+00:00",
            "ticker": "BTCUSD-TEST",
            "event_ticker": "BTCUSD-TEST",
            "side": "no",
            "price_cents": 5,
            "contracts": 5,
            "filled_contracts_total": 2,
            "remaining_contracts": 0,
            "cost": 0.0,
            "edge": 0.02,
            "edge_type": "crypto_spot_mispricing",
            "order_id": "ord-1",
            "order_status": "expired",
            "reasoning": "expired remainder",
        },
        {
            "type": "settlement",
            "session_id": "live_1",
            "timestamp": "2026-04-12T00:10:00+00:00",
            "ticker": "BTCUSD-TEST",
            "side": "no",
            "price_cents": 5,
            "contracts": 2,
            "cost": 0.20,
            "outcome": "no",
            "pnl": 1.80,
        },
    ]
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    replay = _read_live_trade_log_metrics(log_path)

    assert replay["metrics"]["total_trades"] == 1
    assert replay["metrics"]["settled_trades"] == 1
    assert replay["metrics"]["realized_pnl"] == 1.80
    assert replay["metrics"]["open_positions"] == 0
    assert replay["metrics"]["resting_orders"] == 0
    assert replay["current_session"]["settled_trades"] == 1
    assert replay["current_session"]["resting_orders"] == 0


def test_live_log_replay_removes_phantom_positions(tmp_path):
    if _read_live_trade_log_metrics is None:
        pytest.skip("fastapi dependencies are not installed in this test environment")
    log_path = tmp_path / "live_trades.jsonl"
    events = [
        {
            "type": "session_start",
            "session_id": "live_2",
            "timestamp": "2026-04-12T00:00:00+00:00",
        },
        {
            "type": "order_placed",
            "session_id": "live_2",
            "timestamp": "2026-04-12T00:01:00+00:00",
            "ticker": "ETHUSD-TEST",
            "event_ticker": "ETHUSD-TEST",
            "side": "no",
            "price_cents": 10,
            "contracts": 1,
            "cost": 0.10,
            "edge": 0.03,
            "edge_type": "crypto_spot_mispricing",
            "order_id": "ord-phantom",
            "reasoning": "filled",
        },
        {
            "type": "phantom_removed",
            "session_id": "live_2",
            "timestamp": "2026-04-12T00:02:00+00:00",
            "ticker": "ETHUSD-TEST",
            "side": "no",
            "local_contracts": 1,
            "kalshi_contracts": 0,
        },
    ]
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    replay = _read_live_trade_log_metrics(log_path)

    assert replay["metrics"]["open_positions"] == 0
    assert replay["metrics"]["open_cost"] == 0.0
    assert replay["current_session"]["open_positions"] == 0
