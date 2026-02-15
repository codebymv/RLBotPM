"""
Model evaluation for RL agents.

Runs the trained policy against the real-data environment and
computes high-level performance metrics.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any
import numpy as np
import yaml

from src.agents.ppo_agent import PPOAgent  # type: ignore
from src.agents.specialist_manager import SpecialistManager  # type: ignore
from src.environment.gym_env import CryptoTradingEnv  # type: ignore
from src.environment.sequence_wrapper import SequenceStackWrapper  # type: ignore
from src.data.database import CryptoSymbol, get_db_session  # type: ignore
from src.data.collectors import CryptoDataLoader, MultiSourceLoader  # type: ignore
from src.data.sources.base import DataUnavailableError  # type: ignore
from src.core.config import get_settings  # type: ignore
from src.core.logger import get_logger  # type: ignore


logger = get_logger(__name__)


class Evaluator:
    """
    Evaluate a trained PPO model on real OHLCV data.
    """

    def __init__(
        self,
        model_path: str,
        policy_type: str = "MlpPolicy",
        sequence_length: int = 1,
        arbitrage_enabled: bool = False,
        use_specialist_router: bool = False,
    ):
        self.settings = get_settings()
        self.model_path = model_path
        self.policy_type = policy_type
        self.sequence_length = int(sequence_length)
        self.arbitrage_enabled = arbitrage_enabled

        dataset = self._load_dataset()
        # Calculate sequence length for env (same logic as Trainer)
        use_sequence_stack = self.policy_type == "MlpPolicy" and self.sequence_length > 1
        env_sequence_length = self.sequence_length if use_sequence_stack else 1

        base_env = CryptoTradingEnv(
            dataset=dataset,
            interval=self.settings.DATA_INTERVAL,
            sequence_length=env_sequence_length,
            arbitrage_enabled=self.arbitrage_enabled,
        )
        # print(f"DEBUG: Evaluator init policy={self.policy_type} seq_len={self.sequence_length}")
        
        if self.policy_type == "MlpPolicy" and self.sequence_length > 1:
            self.env = SequenceStackWrapper(base_env, sequence_length=self.sequence_length, flatten=True)
            # print(f"DEBUG: Wrapped with SequenceStackWrapper. Obs space: {self.env.observation_space}")
        else:
            self.env = base_env
            # print(f"DEBUG: Not wrapped. Obs space: {self.env.observation_space}")

        self.specialist_manager: Optional[SpecialistManager] = None
        self.agent: Optional[PPOAgent] = None
        self.use_specialist_router = bool(use_specialist_router)

        if self.use_specialist_router:
            specialist_cfg = self._load_specialist_config()
            enabled = bool(specialist_cfg.get("enabled", False))
            if enabled:
                try:
                    self.specialist_manager = SpecialistManager.from_config(
                        env=self.env,
                        config=specialist_cfg,
                        policy_type=self.policy_type,
                        use_gpu=False,
                    )
                    logger.info("Specialist router enabled for evaluation.")
                except Exception as exc:
                    logger.warning("Failed to initialize specialist router: %s", exc)

        if self.specialist_manager is None:
            self.agent = PPOAgent(env=self.env, use_gpu=False, policy_type=self.policy_type)
            self.agent.load(model_path)

    def _load_specialist_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "model_config.yaml"
        if not config_path.exists():
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:
            return {}
        return data.get("specialist_router", {}) or {}

    def _get_action_masks(self) -> Optional[np.ndarray]:
        """Get action masks from wrapped eval env when available."""
        env = self.env
        if hasattr(env, "action_masks"):
            try:
                return np.asarray(env.action_masks(), dtype=np.float32)
            except Exception:
                pass
        if hasattr(env, "get_wrapper_attr"):
            try:
                return np.asarray(env.get_wrapper_attr("action_masks")(), dtype=np.float32)
            except Exception:
                pass
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "action_masks"):
            try:
                return np.asarray(env.unwrapped.action_masks(), dtype=np.float32)
            except Exception:
                pass
        return None

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
        total_steps: int = 0
        total_flat_steps: int = 0
        total_position_steps: int = 0
        hold_streaks: List[int] = []
        total_trade_value: float = 0.0
        total_fees: float = 0.0
        total_buy_fees: float = 0.0
        total_sell_fees: float = 0.0
        profit_wins: float = 0.0
        profit_losses: float = 0.0
        episode_end_closes_count: int = 0
        unmatched_positions: int = 0
        regime_counts: Dict[str, int] = {}

        # Ensure we have a valid sequence of seeds
        if seeds is None:
            episode_seeds = list(range(num_episodes))
        else:
            episode_seeds = seeds

        for idx in range(num_episodes):
            # Use episode_seeds to avoid type checker confusion
            current_seed = episode_seeds[idx] if idx < len(episode_seeds) else None
            obs, info = self.env.reset(seed=current_seed)
            done = False
            truncated = False
            max_drawdown = 0.0
            position_streak = 0
            lstm_state = None
            episode_start = np.ones((1,), dtype=bool)

            while not (done or truncated):
                action_masks = self._get_action_masks()
                if self.specialist_manager is not None:
                    action_int, lstm_state, regime = self.specialist_manager.predict(
                        obs,
                        deterministic=deterministic,
                        state=lstm_state,
                        episode_start=episode_start,
                        action_masks=action_masks,
                    )
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                else:
                    if self.agent is None:
                        raise RuntimeError("Evaluator agent is not initialized.")
                    action, lstm_state = self.agent.predict(
                        obs,
                        deterministic=deterministic,
                        state=lstm_state,
                        episode_start=episode_start,
                        action_masks=action_masks,
                    )
                    action_int = int(action) if isinstance(action, (np.integer, int, np.ndarray)) else int(action)
                # Safe increment for action counts
                count = action_counts.get(action_int, 0)
                action_counts[action_int] = count + 1

                obs, reward, done, truncated, info = self.env.step(action_int)
                episode_start = np.array([done or truncated], dtype=bool)

                total_steps += 1  # type: ignore
                if info.get("num_positions", 0) > 0:
                    total_position_steps += 1  # type: ignore
                    position_streak += 1  # type: ignore
                else:
                    total_flat_steps += 1  # type: ignore
                    if position_streak:
                        hold_streaks.append(position_streak)
                        position_streak = 0

                max_drawdown = max(max_drawdown, float(info.get("drawdown", 0.0)))

                trade_result: Optional[Dict[str, Any]] = info.get("trade_result")
                if trade_result and trade_result.get("executed"):
                    action_name = str(trade_result.get("action_name", ""))
                    if action_name.startswith("BUY"):
                        executed_counts["BUY"] += 1  # type: ignore
                    elif action_name.startswith("SELL"):
                        executed_counts["SELL"] += 1  # type: ignore
                    elif action_name.startswith("CLOSE"):
                        executed_counts["CLOSE"] += 1  # type: ignore
                    else:
                        executed_counts["NO_ACTION"] += 1  # type: ignore

                    size = float(trade_result.get("size", 0.0))
                    total_trade_value += size  # type: ignore
                    cost = float(trade_result.get("cost", 0.0))
                    total_fees += cost  # type: ignore

                    if action_name.startswith("BUY"):
                        total_buy_fees += cost  # type: ignore
                    elif action_name.startswith(("SELL", "CLOSE")):
                        total_sell_fees += cost  # type: ignore
                        pnl = float(trade_result.get("pnl", 0.0))
                        trade_pnls.append(pnl)
                        if pnl > 0:
                            profit_wins += pnl  # type: ignore
                        elif pnl < 0:
                            profit_losses += abs(pnl)  # type: ignore
                        trade_durations.append(int(trade_result.get("hold_steps", 0)))
                    
                    reason = str(trade_result.get("reason", ""))
                    if reason in {"AUTO_STOP_LOSS", "AUTO_TAKE_PROFIT", "AUTO_MAX_HOLD"}:
                        auto_exits[reason] = auto_exits.get(reason, 0) + 1
                        
                elif trade_result and not trade_result.get("executed"):
                    reason = str(trade_result.get("reason") or "Unknown")
                    block_reasons[reason] = block_reasons.get(reason, 0) + 1

            if position_streak:
                hold_streaks.append(position_streak)

            # Track episode-end force-closes
            ep_end_closes: List[Dict[str, Any]] = info.get("episode_end_closes", [])
            for ec in ep_end_closes:
                if ec.get("executed"):
                    episode_end_closes_count += 1  # type: ignore
                    pnl = float(ec.get("pnl", 0.0))
                    cost = float(ec.get("cost", 0.0))
                    trade_pnls.append(pnl)
                    total_fees += cost  # type: ignore
                    total_sell_fees += cost  # type: ignore
                    total_trade_value += float(ec.get("size", 0.0))  # type: ignore
                    if pnl > 0:
                        profit_wins += pnl  # type: ignore
                    elif pnl < 0:
                        profit_losses += abs(pnl)  # type: ignore
                    auto_exits["EPISODE_END_CLOSE"] = auto_exits.get("EPISODE_END_CLOSE", 0) + 1

            # Count unmatched positions (buys that were force-closed at end)
            unmatched_positions += len(ep_end_closes)  # type: ignore

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
        flat_ratio = float(total_flat_steps / total_steps) if total_steps else 0.0  # type: ignore
        in_position_ratio = float(total_position_steps / total_steps) if total_steps else 0.0  # type: ignore
        turnover = float(total_trade_value / (self.settings.INITIAL_CAPITAL * num_episodes)) if num_episodes else 0.0
        profit_factor = (profit_wins / profit_losses) if profit_losses > 0 else 0.0
        max_drawdown = float(np.max(episode_drawdowns)) if episode_drawdowns else 0.0
        drawdown_guard = float(
            getattr(self.env.unwrapped, "reward_config", {}).get("drawdown_threshold", 0.2)
        )
        if max_drawdown > drawdown_guard:
            profit_factor = 0.0
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [abs(pnl) for pnl in trade_pnls if pnl < 0]
        avg_win_size = float(np.mean(winning_trades)) if winning_trades else 0.0
        avg_loss_size = float(np.mean(losing_trades)) if losing_trades else 0.0
        win_loss_ratio = (avg_win_size / avg_loss_size) if avg_loss_size > 0 else 0.0
        fees_pct_of_gross_pnl = float(total_fees / profit_wins) if profit_wins > 0 else 0.0  # type: ignore
        trades_per_episode = float(len(trade_pnls) / num_episodes) if num_episodes else 0.0

        return {
            "total_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "cvar_95": float(cvar_95),
            "max_drawdown": max_drawdown,
            "win_rate": float(np.mean(episode_win_rates)) if episode_win_rates else 0.0,
            "avg_trade_pnl": float(np.mean(trade_pnls)) if trade_pnls else 0.0,
            "avg_win_size": avg_win_size,
            "avg_loss_size": avg_loss_size,
            "win_loss_ratio": float(win_loss_ratio),
            "avg_trade_duration": avg_trade_duration,
            "turnover": turnover,
            "total_fees": float(total_fees),
            "profit_factor": float(profit_factor),
            "drawdown_guard": float(drawdown_guard),
            "fees_pct_of_gross_pnl": fees_pct_of_gross_pnl,
            "trades_per_episode": trades_per_episode,
            "flat_ratio": flat_ratio,
            "in_position_ratio": in_position_ratio,
            "avg_hold_steps": avg_hold_steps,
            "auto_exits": auto_exits,  # type: ignore
            "action_counts": action_counts,  # type: ignore
            "executed_counts": executed_counts,  # type: ignore
            "block_reasons": block_reasons,  # type: ignore
            "total_buy_fees": float(total_buy_fees),
            "total_sell_fees": float(total_sell_fees),
            "episode_end_closes": float(episode_end_closes_count),
            "unmatched_positions_per_ep": float(unmatched_positions / num_episodes) if num_episodes else 0.0,
            "regime_counts": regime_counts,  # type: ignore
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

        end = datetime.utcnow()
        start = end - timedelta(days=days)

        # Use multi-source loader if arbitrage is enabled
        if self.arbitrage_enabled and self.settings.DATA_SOURCES:
            sources = [s.strip() for s in self.settings.DATA_SOURCES.split(",") if s.strip()]
            if len(sources) >= 2:
                logger.info(f"Evaluator loading multi-exchange data from: {sources}")
                multi_loader = MultiSourceLoader(sources=sources)
                dataset = multi_loader.load_aligned_dataset(
                    symbols=symbols,
                    interval=interval,
                    start=start,
                    end=end
                )
            else:
                loader = CryptoDataLoader(source=source)
                dataset = loader.load_dataset(
                    symbols=symbols,
                    interval=interval,
                    start=start,
                    end=end,
                )
        else:
            loader = CryptoDataLoader(source=source)
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
