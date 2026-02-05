"""
Walk-forward training and evaluation.

Trains on a rolling window and evaluates on the next window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
from sqlalchemy import func

from ..agents import PPOAgent
from ..environment import CryptoTradingEnv
from ..data import CryptoCandle, CryptoSymbol, get_db_session
from ..data.collectors import CryptoDataLoader
from ..data.sources.base import DataUnavailableError
from ..core.config import get_settings
from ..core.logger import get_logger


logger = get_logger(__name__)


@dataclass
class WalkForwardResult:
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_pnl: float


def run_walk_forward(
    folds: int,
    train_days: int,
    test_days: int,
    train_episodes: int,
    eval_episodes: int,
) -> List[WalkForwardResult]:
    settings = get_settings()
    source = settings.DATA_SOURCE
    interval = settings.DATA_INTERVAL

    min_window_days = _min_window_days(interval)
    if train_days < min_window_days or test_days < min_window_days:
        raise DataUnavailableError(
            "Window too short for feature prep. "
            f"Use at least {min_window_days} day(s) for train/test windows."
        )

    symbols = _get_symbols(source)
    if not symbols:
        raise DataUnavailableError("No symbols found in database.")

    min_ts, max_ts = _get_data_range(source, interval, symbols)
    if min_ts is None or max_ts is None:
        raise DataUnavailableError("No candles found for the selected symbols.")

    total_days = (max_ts - min_ts).total_seconds() / 86400.0
    max_folds = int((total_days - train_days) // test_days) if total_days > train_days else 0
    if max_folds <= 0:
        raise DataUnavailableError(
            "Insufficient history for walk-forward. "
            f"Need > {train_days + test_days} days, have {total_days:.1f} days."
        )

    if folds > max_folds:
        logger.warning(
            "Requested folds exceed available history; "
            f"reducing folds from {folds} to {max_folds}."
        )
        folds = max_folds

    loader = CryptoDataLoader(source=source)
    results: List[WalkForwardResult] = []

    for fold in range(1, folds + 1):
        train_start = min_ts + timedelta(days=(fold - 1) * test_days)
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)

        train_data = loader.load_dataset(
            symbols=symbols,
            interval=interval,
            start=train_start,
            end=train_end,
        )
        test_data = loader.load_dataset(
            symbols=symbols,
            interval=interval,
            start=test_start,
            end=test_end,
        )

        if train_data is None or train_data.empty:
            raise DataUnavailableError("Training dataset is empty for a fold.")
        if test_data is None or test_data.empty:
            raise DataUnavailableError("Test dataset is empty for a fold.")

        train_env = CryptoTradingEnv(dataset=train_data, interval=interval)
        agent = PPOAgent(env=train_env, use_gpu=False)
        agent.train(total_timesteps=train_episodes, log_interval=10)

        metrics = _evaluate_agent(agent, test_data, interval, eval_episodes)
        results.append(
            WalkForwardResult(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                total_return=metrics["total_return"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                win_rate=metrics["win_rate"],
                avg_trade_pnl=metrics["avg_trade_pnl"],
            )
        )

    return results


def _evaluate_agent(
    agent: PPOAgent,
    dataset,
    interval: str,
    num_episodes: int,
) -> Dict[str, float]:
    env = CryptoTradingEnv(dataset=dataset, interval=interval)

    episode_returns: List[float] = []
    episode_drawdowns: List[float] = []
    episode_win_rates: List[float] = []
    trade_pnls: List[float] = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        max_drawdown = 0.0

        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            max_drawdown = max(max_drawdown, float(info.get("drawdown", 0.0)))

            trade_result = info.get("trade_result")
            if trade_result and trade_result.get("executed"):
                action_name = trade_result.get("action_name", "")
                if action_name.startswith("SELL") or action_name.startswith("CLOSE"):
                    trade_pnls.append(float(trade_result.get("pnl", 0.0)))

        portfolio_value = float(info.get("portfolio_value", get_settings().INITIAL_CAPITAL))
        total_return = (portfolio_value - get_settings().INITIAL_CAPITAL) / get_settings().INITIAL_CAPITAL
        episode_returns.append(total_return)
        episode_drawdowns.append(max_drawdown)
        episode_win_rates.append(float(info.get("win_rate", 0.0)))

    sharpe_ratio = _compute_sharpe_ratio(episode_returns)

    return {
        "total_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(np.max(episode_drawdowns)) if episode_drawdowns else 0.0,
        "win_rate": float(np.mean(episode_win_rates)) if episode_win_rates else 0.0,
        "avg_trade_pnl": float(np.mean(trade_pnls)) if trade_pnls else 0.0,
    }


def _compute_sharpe_ratio(returns: List[float]) -> float:
    if not returns:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    return (mean_return / std_return) * np.sqrt(len(returns))


def _get_symbols(source: str) -> List[str]:
    settings = get_settings()
    if settings.DATA_SYMBOLS:
        return [s.strip() for s in settings.DATA_SYMBOLS.split(",") if s.strip()]

    session = get_db_session()
    try:
        return [
            row.symbol
            for row in session.query(CryptoSymbol)
            .filter_by(source=source, status="active")
            .all()
        ]
    finally:
        session.close()


def _get_data_range(source: str, interval: str, symbols: List[str]):
    session = get_db_session()
    try:
        min_ts, max_ts = (
            session.query(func.min(CryptoCandle.timestamp), func.max(CryptoCandle.timestamp))
            .filter(CryptoCandle.source == source)
            .filter(CryptoCandle.interval == interval)
            .filter(CryptoCandle.symbol.in_(symbols))
            .one()
        )
        return min_ts, max_ts
    finally:
        session.close()


def _min_window_days(interval: str) -> int:
    minutes = _interval_to_minutes(interval)
    candles_per_day = max(1, int(1440 / minutes))
    feature_window_candles = int((48 * 60) / minutes)
    min_steps = feature_window_candles + 2
    return int(np.ceil(min_steps / candles_per_day))


def _interval_to_minutes(interval: str) -> int:
    mapping = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    if interval not in mapping:
        raise DataUnavailableError(f"Unsupported interval: {interval}")
    return mapping[interval]
