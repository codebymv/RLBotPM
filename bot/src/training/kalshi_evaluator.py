"""
Evaluator for Kalshi binary market RL models.

Runs trained policy on held-out Kalshi events and computes
win rate, ROI, and other metrics for binary outcome trading.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from ..agents import PPOAgent
from ..environment.kalshi_env import KalshiEventEnv, load_kalshi_settled_markets
from ..data import get_db_session
from ..core.logger import get_logger


logger = get_logger(__name__)


class KalshiEvaluator:
    """
    Evaluate a trained Kalshi PPO model on held-out events.
    """

    def __init__(
        self,
        model_path: str,
        policy_type: str = "MlpPolicy",
        series_tickers: Optional[List[str]] = None,
        train_ratio: float = 0.8,
    ):
        """
        Args:
            model_path: Path to saved model checkpoint
            policy_type: MlpPolicy or MlpLstmPolicy
            series_tickers: Series to evaluate on (None = all)
            train_ratio: Fraction of events used for training (rest for eval)
        """
        self.model_path = model_path
        self.policy_type = policy_type
        self.train_ratio = train_ratio

        # Load held-out test events
        session = get_db_session()
        try:
            df = load_kalshi_settled_markets(session, series_tickers=series_tickers)
            if df is None or df.empty:
                raise ValueError("No Kalshi settled markets found in DB")
            
            # Split by event: train on first 80%, eval on last 20%
            unique_events = df['event_ticker'].unique()
            n_train = int(len(unique_events) * train_ratio)
            test_events = set(unique_events[n_train:])
            df_test = df[df['event_ticker'].isin(test_events)].reset_index(drop=True)
            
            logger.info(
                f"KalshiEvaluator: {len(test_events)} test events, "
                f"{len(df_test)} contracts (train_ratio={train_ratio:.2f})"
            )
            
            self.env = KalshiEventEnv(
                settled_markets=df_test,
                initial_capital=25.0,
                max_positions_per_event=3,
                contracts_per_trade=1,
            )
        finally:
            session.close()

        self.agent = PPOAgent(env=self.env, use_gpu=False, policy_type=self.policy_type)
        self.agent.load(model_path)

    def evaluate(
        self,
        num_episodes: int = 100,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Run evaluation episodes and compute metrics.
        
        Returns:
            Dict with metrics: win_rate, roi, sharpe_ratio, avg_return, etc.
        """
        if num_episodes > len(self.env.events):
            num_episodes = len(self.env.events)
            logger.warning(f"Capping episodes to {num_episodes} (total test events)")

        episode_returns: List[float] = []
        episode_win_rates: List[float] = []
        total_trades = 0
        total_wins = 0
        all_trade_pnls: List[float] = []

        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            
            episode_returns.append(info['episode_return'])
            episode_win_rates.append(info['win_rate'])
            total_trades += info['trade_count']
            total_wins += info['win_count']
            
            # Collect individual trade PnLs
            if hasattr(self.env, 'episode_trades'):
                all_trade_pnls.extend(self.env.episode_trades)

        # Compute aggregate metrics
        mean_return = float(np.mean(episode_returns))
        std_return = float(np.std(episode_returns))
        sharpe = mean_return / std_return if std_return > 0 else 0.0
        win_rate = total_wins / max(1, total_trades)
        
        max_dd = 0.0
        cumulative = np.cumsum(episode_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        if len(drawdowns) > 0:
            max_dd = float(np.max(drawdowns))

        metrics = {
            "mean_return": mean_return,
            "std_return": std_return,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "max_drawdown": max_dd,
            "num_episodes": num_episodes,
            "min_return": float(np.min(episode_returns)),
            "max_return": float(np.max(episode_returns)),
        }

        logger.info(
            f"Eval: {num_episodes} eps | Return={mean_return:.4f}±{std_return:.4f} | "
            f"Sharpe={sharpe:.3f} | WinRate={win_rate:.1%} ({total_wins}/{total_trades})"
        )

        return metrics
