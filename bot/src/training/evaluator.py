"""
Model evaluation for RL agents.

Runs the trained policy against the real-data environment and
computes high-level performance metrics.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence
import numpy as np

from ..agents import PPOAgent
from ..environment import CryptoTradingEnv, SequenceStackWrapper
from ..data import CryptoSymbol, get_db_session
from ..data.collectors import CryptoDataLoader
from ..data.sources.base import DataUnavailableError
from ..core.config import get_settings
from ..core.logger import get_logger


logger = get_logger(__name__)


class Evaluator:
    """
    Evaluate a trained PPO model on real OHLCV data.
    """

    def __init__(self, model_path: str, policy_type: str = "MlpPolicy", sequence_length: int = 1):
        self.settings = get_settings()
        self.model_path = model_path
        self.policy_type = policy_type
        self.sequence_length = int(sequence_length)

        dataset = self._load_dataset()
        base_env = CryptoTradingEnv(
            dataset=dataset,
            interval=self.settings.DATA_INTERVAL,
            sequence_length=1,
        )
        if self.policy_type == "MlpPolicy" and self.sequence_length > 1:
            self.env = SequenceStackWrapper(base_env, sequence_length=self.sequence_length, flatten=True)
        else:
            self.env = base_env

        self.agent = PPOAgent(env=self.env, use_gpu=False, policy_type=self.policy_type)
        self.agent.load(model_path)

    def evaluate(
        self,
        num_episodes: int = 100,
        deterministic: bool = True,
        seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation episodes and compute metrics.
        """
        episode_returns: List[float] = []
        episode_drawdowns: List[float] = []
        episode_win_rates: List[float] = []
        trade_pnls: List[float] = []
        trade_durations: List[int] = []
        action_counts: Dict[int, int] = {}
        executed_counts: Dict[str, int] = {
            "BUY": 0,
            "SELL": 0,
            "CLOSE": 0,
            "NO_ACTION": 0,
        }
        block_reasons: Dict[str, int] = {}
        auto_exits: Dict[str, int] = {}
        total_steps = 0
        total_flat_steps = 0
        total_position_steps = 0
        hold_streaks: List[int] = []
        total_trade_value = 0.0
        total_fees = 0.0
        profit_wins = 0.0
        profit_losses = 0.0

        if seeds is None:
            seeds = list(range(num_episodes))

        for idx in range(num_episodes):
            obs, info = self.env.reset(seed=seeds[idx])
            done = False
            truncated = False
            max_drawdown = 0.0
            position_streak = 0
            lstm_state = None
            episode_start = np.ones((1,), dtype=bool)

            while not (done or truncated):
                action, lstm_state = self.agent.predict(
                    obs,
                    deterministic=deterministic,
                    state=lstm_state,
                    episode_start=episode_start,
                )
                action_counts[action] = action_counts.get(action, 0) + 1
                obs, reward, done, truncated, info = self.env.step(action)
                episode_start = np.array([done or truncated], dtype=bool)

                total_steps += 1
                if info.get("num_positions", 0) > 0:
                    total_position_steps += 1
                    position_streak += 1
                else:
                    total_flat_steps += 1
                    if position_streak:
                        hold_streaks.append(position_streak)
                        position_streak = 0

                max_drawdown = max(max_drawdown, float(info.get("drawdown", 0.0)))

                trade_result = info.get("trade_result")
                if trade_result and trade_result.get("executed"):
                    action_name = trade_result.get("action_name", "")
                    if action_name.startswith("BUY"):
                        executed_counts["BUY"] += 1
                    elif action_name.startswith("SELL"):
                        executed_counts["SELL"] += 1
                    elif action_name.startswith("CLOSE"):
                        executed_counts["CLOSE"] += 1
                    else:
                        executed_counts["NO_ACTION"] += 1

                    size = float(trade_result.get("size", 0.0))
                    total_trade_value += size
                    total_fees += float(trade_result.get("cost", 0.0))

                    if action_name.startswith("SELL") or action_name.startswith("CLOSE"):
                        pnl = float(trade_result.get("pnl", 0.0))
                        trade_pnls.append(pnl)
                        if pnl > 0:
                            profit_wins += pnl
                        elif pnl < 0:
                            profit_losses += abs(pnl)
                        trade_durations.append(int(trade_result.get("hold_steps", 0)))
                    if trade_result.get("reason") in {"AUTO_STOP_LOSS", "AUTO_TAKE_PROFIT", "AUTO_MAX_HOLD"}:
                        reason = trade_result.get("reason")
                        auto_exits[reason] = auto_exits.get(reason, 0) + 1
                elif trade_result and not trade_result.get("executed"):
                    reason = trade_result.get("reason") or "Unknown"
                    block_reasons[reason] = block_reasons.get(reason, 0) + 1

            if position_streak:
                hold_streaks.append(position_streak)

            portfolio_value = float(info.get("portfolio_value", self.settings.INITIAL_CAPITAL))
            total_return = (portfolio_value - self.settings.INITIAL_CAPITAL) / self.settings.INITIAL_CAPITAL
            episode_returns.append(total_return)
            episode_drawdowns.append(max_drawdown)
            episode_win_rates.append(float(info.get("win_rate", 0.0)))

        sharpe_ratio = self._compute_sharpe_ratio(episode_returns)
        sortino_ratio = self._compute_sortino_ratio(episode_returns)
        cvar_95 = self._compute_cvar(episode_returns, alpha=0.05)
        avg_hold_steps = float(np.mean(hold_streaks)) if hold_streaks else 0.0
        avg_trade_duration = float(np.mean(trade_durations)) if trade_durations else 0.0
        flat_ratio = float(total_flat_steps / total_steps) if total_steps else 0.0
        in_position_ratio = float(total_position_steps / total_steps) if total_steps else 0.0
        turnover = float(total_trade_value / (self.settings.INITIAL_CAPITAL * num_episodes)) if num_episodes else 0.0
        profit_factor = (profit_wins / profit_losses) if profit_losses > 0 else 0.0

        return {
            "total_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(np.max(episode_drawdowns)) if episode_drawdowns else 0.0,
            "win_rate": float(np.mean(episode_win_rates)) if episode_win_rates else 0.0,
            "avg_trade_pnl": float(np.mean(trade_pnls)) if trade_pnls else 0.0,
            "avg_trade_duration": avg_trade_duration,
            "turnover": turnover,
            "total_fees": float(total_fees),
            "profit_factor": float(profit_factor),
            "flat_ratio": flat_ratio,
            "in_position_ratio": in_position_ratio,
            "avg_hold_steps": avg_hold_steps,
            "auto_exits": auto_exits,
            "action_counts": action_counts,
            "executed_counts": executed_counts,
            "block_reasons": block_reasons,
        }

    def _compute_sharpe_ratio(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return (mean_return / std_return) * np.sqrt(len(returns))

    def _compute_sortino_ratio(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        mean_return = np.mean(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 0.0
        std_down = np.std(downside)
        if std_down == 0:
            return 0.0
        return (mean_return / std_down) * np.sqrt(len(returns))

    def _compute_cvar(self, returns: List[float], alpha: float = 0.05) -> float:
        if not returns:
            return 0.0
        percentile = np.percentile(returns, alpha * 100)
        tail = [r for r in returns if r <= percentile]
        if not tail:
            return 0.0
        return float(np.mean(tail))

    def _load_dataset(self):
        source = self.settings.DATA_SOURCE
        interval = self.settings.DATA_INTERVAL
        days = self.settings.REQUIRE_HISTORICAL_DAYS

        symbols: List[str] = []
        if self.settings.DATA_SYMBOLS:
            symbols = [s.strip() for s in self.settings.DATA_SYMBOLS.split(",") if s.strip()]
        else:
            session = get_db_session()
            try:
                symbols = [
                    row.symbol
                    for row in session.query(CryptoSymbol)
                    .filter_by(source=source, status="active")
                    .all()
                ]
            finally:
                session.close()

        if not symbols:
            raise DataUnavailableError(
                "No symbols found in database. Run CryptoDataLoader to sync symbols and candles."
            )

        loader = CryptoDataLoader(source=source)
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        dataset = loader.load_dataset(
            symbols=symbols,
            interval=interval,
            start=start,
            end=end,
        )

        if dataset is None or dataset.empty:
            raise DataUnavailableError("Dataset is empty after loading. Check data pipeline.")

        min_rows = 500 + (self.sequence_length if self.policy_type == "MlpPolicy" else 1) + 1
        counts = dataset.groupby("symbol").size()
        valid_symbols = counts[counts >= min_rows].index.tolist()
        dataset = dataset[dataset["symbol"].isin(valid_symbols)].reset_index(drop=True)
        if dataset.empty:
            raise DataUnavailableError("No symbols meet minimum row requirement for evaluation.")

        return dataset


def compare_models(
    model_a_path: str,
    model_b_path: str,
    policy_a: str = "MlpPolicy",
    policy_b: str = "MlpLstmPolicy",
    seq_a: int = 1,
    seq_b: int = 1,
    episodes: int = 100,
    deterministic: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two models on identical episode seeds.
    """
    seeds = list(range(episodes))
    eval_a = Evaluator(model_path=model_a_path, policy_type=policy_a, sequence_length=seq_a)
    eval_b = Evaluator(model_path=model_b_path, policy_type=policy_b, sequence_length=seq_b)

    results_a = eval_a.evaluate(num_episodes=episodes, deterministic=deterministic, seeds=seeds)
    results_b = eval_b.evaluate(num_episodes=episodes, deterministic=deterministic, seeds=seeds)

    return {"model_a": results_a, "model_b": results_b}
