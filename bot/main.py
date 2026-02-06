"""
RLTrade Bot - Main entry point for training and evaluation

This is the command-line interface for the RL trading bot.
Use this to train models, evaluate performance, and manage the bot.
"""

import asyncio
import click
import numpy as np
import json
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

console = Console()


@click.group()
def cli():
    """RLTrade - Reinforcement Learning Crypto Trading Bot"""
    console.print(Panel.fit(
        "[bold cyan]RLTrade Bot[/bold cyan]\n"
        "[dim]Reinforcement Learning Trading System[/dim]",
        border_style="cyan"
    ))


@cli.command()
@click.option('--episodes', default=10000, help='Number of training episodes')
@click.option('--checkpoint', default=None, help='Resume from checkpoint path')
@click.option('--config', default='shared/config/model_config.yaml', help='Config file path')
@click.option('--policy', default=None, help='Policy type (MlpPolicy or MlpLstmPolicy)')
@click.option('--sequence-length', default=None, type=int, help='Sequence length for frame stacking')
@click.option('--checkpoint-frequency', default=10000, type=int, help='Save checkpoints every N episodes')
@click.option('--eval-frequency', default=10000, type=int, help='Evaluate every N episodes')
def train(episodes, checkpoint, config, policy, sequence_length, checkpoint_frequency, eval_frequency):
    """Train the RL agent on historical data"""
    console.print(f"\n[bold green]Starting training:[/bold green] {episodes} episodes")
    
    if checkpoint:
        console.print(f"[yellow]Resuming from checkpoint:[/yellow] {checkpoint}")
    
    # Import here to avoid loading heavy dependencies on CLI help
    from src.training.trainer import Trainer
    
    try:
        overrides = {}
        if policy:
            overrides["ppo"] = {"policy_type": policy}
        if sequence_length:
            overrides["recurrent"] = {"sequence_length": sequence_length}
        trainer = Trainer(config_path=config, overrides=overrides or None)
        
        if checkpoint:
            trainer.load_checkpoint(checkpoint)
        
        trainer.train(
            total_episodes=episodes,
            checkpoint_frequency=checkpoint_frequency,
            eval_frequency=eval_frequency,
        )
        
        console.print("\n[bold green]OK[/bold green] Training completed!")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise


@cli.command()
@click.option('--model', required=True, help='Path to trained model')
@click.option('--episodes', default=100, help='Number of evaluation episodes')
@click.option('--stochastic', is_flag=True, help='Use stochastic actions for evaluation')
@click.option('--policy', default="MlpPolicy", help='Policy type (MlpPolicy or MlpLstmPolicy)')
@click.option('--sequence-length', default=1, type=int, help='Sequence length for frame stacking')
def evaluate(model, episodes, stochastic, policy, sequence_length):
    """Evaluate a trained model on test data"""
    console.print(f"\n[bold cyan]Evaluating model:[/bold cyan] {model}")
    
    from src.training.evaluator import Evaluator
    
    try:
        evaluator = Evaluator(model_path=model, policy_type=policy, sequence_length=sequence_length)
        results = evaluator.evaluate(num_episodes=episodes, deterministic=not stochastic)
        
        # Display results
        console.print("\n[bold green]Evaluation Results:[/bold green]")
        console.print(f"  Total Return: {results['total_return']:.2%}")
        console.print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        if "sortino_ratio" in results:
            console.print(f"  Sortino Ratio: {results['sortino_ratio']:.3f}")
        if "cvar_95" in results:
            console.print(f"  CVaR (95%): {results['cvar_95']:.3f}")
        console.print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        console.print(f"  Win Rate: {results['win_rate']:.2%}")
        console.print(f"  Avg Trade P&L: ${results['avg_trade_pnl']:.2f}")
        if "avg_win_size" in results:
            console.print(f"  Avg Win Size: ${results['avg_win_size']:.2f}")
        if "avg_loss_size" in results:
            console.print(f"  Avg Loss Size: ${results['avg_loss_size']:.2f}")
        if "win_loss_ratio" in results:
            console.print(f"  Win/Loss Ratio: {results['win_loss_ratio']:.2f}")
        if "profit_factor" in results:
            console.print(f"  Profit Factor: {results['profit_factor']:.2f}")
        if "fees_pct_of_gross_pnl" in results:
            console.print(f"  Fees % of Gross PnL: {results['fees_pct_of_gross_pnl']:.2%}")
        if "trades_per_episode" in results:
            console.print(f"  Trades / Episode: {results['trades_per_episode']:.2f}")
        if "avg_trade_duration" in results:
            console.print(f"  Avg Trade Duration: {results['avg_trade_duration']:.1f}")
        if "turnover" in results:
            console.print(f"  Turnover: {results['turnover']:.2f}x")
        if "total_fees" in results:
            console.print(f"  Total Fees: ${results['total_fees']:.2f}")
        if "flat_ratio" in results:
            console.print(f"  Flat Ratio: {results['flat_ratio']:.2%}")
        if "in_position_ratio" in results:
            console.print(f"  In-Position Ratio: {results['in_position_ratio']:.2%}")
        if "avg_hold_steps" in results:
            console.print(f"  Avg Hold Steps: {results['avg_hold_steps']:.1f}")

        action_counts = results.get("action_counts", {})
        executed_counts = results.get("executed_counts", {})
        block_reasons = results.get("block_reasons", {})

        if action_counts:
            console.print("\n[bold]Action counts:[/bold]")
            for action, count in sorted(action_counts.items()):
                console.print(f"  {action}: {count}")

        if executed_counts:
            console.print("\n[bold]Executed actions:[/bold]")
            for action, count in executed_counts.items():
                console.print(f"  {action}: {count}")

        if block_reasons:
            console.print("\n[bold]Blocked reasons:[/bold]")
            for reason, count in block_reasons.items():
                console.print(f"  {reason}: {count}")

        auto_exits = results.get("auto_exits", {})
        if auto_exits:
            console.print("\n[bold]Auto exits:[/bold]")
            for reason, count in auto_exits.items():
                console.print(f"  {reason}: {count}")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise


@cli.command("compare")
@click.option('--model-a', required=True, help='Path to model A')
@click.option('--model-b', required=True, help='Path to model B')
@click.option('--policy-a', default="MlpPolicy", help='Policy type for model A')
@click.option('--policy-b', default="MlpLstmPolicy", help='Policy type for model B')
@click.option('--seq-a', default=1, type=int, help='Sequence length for model A')
@click.option('--seq-b', default=1, type=int, help='Sequence length for model B')
@click.option('--episodes', default=100, type=int, help='Number of evaluation episodes')
@click.option('--stochastic', is_flag=True, help='Use stochastic actions for evaluation')
def compare_models_cmd(model_a, model_b, policy_a, policy_b, seq_a, seq_b, episodes, stochastic):
    """Compare two models (A/B) on identical episodes"""
    console.print("\n[bold cyan]Comparing models[/bold cyan]")

    from src.training.evaluator import compare_models

    results = compare_models(
        model_a_path=model_a,
        model_b_path=model_b,
        policy_a=policy_a,
        policy_b=policy_b,
        seq_a=seq_a,
        seq_b=seq_b,
        episodes=episodes,
        deterministic=not stochastic,
    )

    console.print("\n[bold]Model A:[/bold]")
    for key, value in results["model_a"].items():
        console.print(f"  {key}: {value}")

    console.print("\n[bold]Model B:[/bold]")
    for key, value in results["model_b"].items():
        console.print(f"  {key}: {value}")


@cli.command("data-qa")
@click.option('--days', default=None, type=int, help='Override days to inspect')
def data_qa(days):
    """Run data quality checks on OHLCV candles"""
    console.print("\n[bold cyan]Data Quality Report[/bold cyan]")

    from src.data.qa_report import build_data_quality_report
    from src.data.sources.base import DataUnavailableError

    try:
        report = build_data_quality_report(days=days)
        console.print(
            f"Source={report['source']} | Interval={report['interval']} | Days={report['days']} | "
            f"Symbols={report['symbols']} | Rows={report['total_rows']}"
        )
        console.print(
            f"Total missing={report['total_missing']} | Total duplicates={report['total_duplicates']}"
        )

        worst = sorted(
            report["symbols_report"].items(),
            key=lambda item: item[1]["missing_pct"],
            reverse=True,
        )[:5]
        if worst:
            console.print("\n[bold]Worst symbols (by missing %):[/bold]")
            for symbol, stats in worst:
                console.print(
                    f"  {symbol}: missing={stats['missing']} "
                    f"({stats['missing_pct']:.2%}), duplicates={stats['duplicates']}, "
                    f"max_gap_hours={stats['max_gap_hours']:.1f}"
                )

    except DataUnavailableError as e:
        console.print(f"\n[bold red]Data Error:[/bold red] {str(e)}")
        raise
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise


@cli.command("paper-trade")
@click.option('--model', required=True, help='Path to trained model')
@click.option('--duration', default="24h", help='Duration to run (e.g., 24h, 7d)')
@click.option('--capital', default=1000.0, help='Starting capital')
@click.option('--interval', default="5m", help='Trading interval (1m,5m,15m,1h)')
@click.option('--policy', default="MlpPolicy", help='Policy type (MlpPolicy or MlpLstmPolicy)')
@click.option('--sequence-length', default=1, type=int, help='Sequence length for frame stacking')
def paper_trade(model, duration, capital, interval, policy, sequence_length):
    """Run paper trading with historical data replay."""
    console.print("\n[bold cyan]Starting paper trading (replay mode)[/bold cyan]")
    console.print(f"  Model: {model}")
    console.print(f"  Duration: {duration}")
    console.print(f"  Capital: ${capital:.2f}")
    console.print(f"  Interval: {interval}")

    from src.core.config import get_settings
    from src.data.collectors.crypto_loader import CryptoDataLoader
    from src.data.sources.base import DataUnavailableError
    from src.environment import CryptoTradingEnv, SequenceStackWrapper
    from src.execution.paper_trader import PaperTradingEngine
    from src.agents import PPOAgent

    settings = get_settings()
    symbols = settings.DATA_SYMBOLS
    if not symbols:
        console.print("\n[bold red]Error:[/bold red] DATA_SYMBOLS is not set")
        return

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

    interval_seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
    }
    if interval not in interval_seconds:
        console.print(f"\n[bold red]Error:[/bold red] Unsupported interval: {interval}")
        return

    if duration.endswith("h"):
        total_seconds = int(duration[:-1]) * 3600
    elif duration.endswith("d"):
        total_seconds = int(duration[:-1]) * 86400
    else:
        console.print("\n[bold red]Error:[/bold red] Duration must end with h or d")
        return

    total_steps = max(1, total_seconds // interval_seconds[interval])

    try:
        loader = CryptoDataLoader(source=settings.DATA_SOURCE)
        days_back = max(1, total_seconds // 86400)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        dataset = loader.load_dataset(
            symbols=symbol_list,
            interval=interval,
            start=start_time,
            end=end_time,
        )

        env = CryptoTradingEnv(
            dataset=dataset,
            interval=interval,
            initial_capital=capital,
            max_steps=total_steps,
            transaction_cost=settings.TRANSACTION_COST_PCT,
            sequence_length=1,
        )

        if policy == "MlpPolicy" and sequence_length > 1:
            env = SequenceStackWrapper(env, sequence_length=sequence_length, flatten=True)

        agent = PPOAgent(env=env, policy_type=policy)
        agent.load(model)

        paper_engine = PaperTradingEngine(initial_capital=capital, transaction_cost_pct=settings.TRANSACTION_COST_PCT)

        obs, _ = env.reset()
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

        for _ in range(total_steps):
            action, lstm_state = agent.predict(
                obs,
                deterministic=True,
                state=lstm_state,
                episode_start=episode_start,
            )
            obs, _, terminated, truncated, info = env.step(action)
            episode_start = np.array([terminated or truncated], dtype=bool)
            trade_result = info.get("trade_result", {})
            if trade_result:
                trade_result["symbol"] = info.get("current_symbol")
                paper_engine.record_trade(trade_result)

            if terminated or truncated:
                obs, _ = env.reset()

        metrics = paper_engine.get_performance_metrics()
        metrics["timestamp"] = datetime.utcnow().isoformat()

        metrics_path = os.getenv("PAPER_TRADING_METRICS_PATH", "./logs/paper_trading/metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        console.print("\n[bold green]Paper Trading Results:[/bold green]")
        console.print(f"  Final Capital: ${metrics['capital']:.2f}")
        console.print(f"  Total Return: {metrics['total_return_pct']:.2%}")
        console.print(f"  Win Rate: {metrics['win_rate']:.2%}")
        console.print(f"  Total Trades: {metrics['num_trades']}")

    except DataUnavailableError as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise


@cli.command("data-dedupe")
@click.option('--dry-run', is_flag=True, help='Only report duplicate count')
def data_dedupe(dry_run):
    """Remove duplicate candles from the database"""
    console.print("\n[bold cyan]Data Deduplication[/bold cyan]")

    from src.data.cleanup import count_duplicate_candles, dedupe_crypto_candles

    duplicates = count_duplicate_candles()
    console.print(f"Duplicate candles found: {duplicates}")

    if dry_run or duplicates == 0:
        return

    removed = dedupe_crypto_candles()
    console.print(f"Removed {removed} duplicate candles")


@cli.command("walk-forward")
@click.option('--folds', default=3, type=int, help='Number of folds')
@click.option('--train-days', default=8, type=int, help='Training window in days')
@click.option('--test-days', default=3, type=int, help='Test window in days')
@click.option('--train-episodes', default=1000, type=int, help='Training episodes per fold')
@click.option('--eval-episodes', default=20, type=int, help='Evaluation episodes per fold')
def walk_forward(folds, train_days, test_days, train_episodes, eval_episodes):
    """Run walk-forward training and evaluation"""
    console.print("\n[bold cyan]Walk-Forward Evaluation[/bold cyan]")

    from src.training.walk_forward import run_walk_forward
    from src.data.sources.base import DataUnavailableError

    try:
        results = run_walk_forward(
            folds=folds,
            train_days=train_days,
            test_days=test_days,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
        )

        for result in results:
            console.print(
                f"Fold {result.fold}: "
                f"return={result.total_return:.2%}, "
                f"sharpe={result.sharpe_ratio:.3f}, "
                f"drawdown={result.max_drawdown:.2%}, "
                f"win_rate={result.win_rate:.2%}, "
                f"avg_trade_pnl=${result.avg_trade_pnl:.2f}"
            )

    except DataUnavailableError as e:
        console.print(f"\n[bold red]Data Error:[/bold red] {str(e)}")
        raise
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise


@cli.command()
@click.option('--source', default="coinbase", help='Data source adapter (coinbase, kraken)')
@click.option('--symbols', required=True, help='Comma-separated symbols (e.g., BTC-USD,ETH-USD)')
@click.option('--interval', default="1h", help='Candle interval (e.g., 1m,5m,1h,1d)')
@click.option('--days', default=30, help='Number of days of history to collect')
def collect_data(source, symbols, interval, days):
    """Collect real OHLCV data from crypto exchange"""
    console.print(
        f"\n[bold cyan]Collecting data:[/bold cyan] source={source}, interval={interval}, days={days}"
    )

    from src.data.collectors.crypto_loader import CryptoDataLoader
    from src.data.sources.base import DataUnavailableError

    try:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        loader = CryptoDataLoader(source=source)
        loader.sync_symbols(symbol_list)

        end = datetime.utcnow()
        start = end - timedelta(days=days)

        loader.collect_ohlcv(
            symbols=symbol_list,
            interval=interval,
            start=start,
            end=end
        )

        console.print("\n[bold green]OK[/bold green] Data collection completed!")

    except DataUnavailableError as e:
        console.print(f"\n[bold red]Data Error:[/bold red] {str(e)}")
        raise
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise


@cli.command("paper-trade-live")
@click.option('--symbols', required=True, help='Comma-separated symbols (e.g., BTC-USD,ETH-USD)')
@click.option('--duration-hours', default=24, type=int, help='Run duration in hours')
@click.option('--capital', default=1000.0, type=float, help='Starting capital')
@click.option('--trade-size-pct', default=0.05, type=float, help='Trade size as fraction of capital')
@click.option('--entry-threshold', default=0.002, type=float, help='Entry threshold (price change pct)')
@click.option('--exit-threshold', default=0.002, type=float, help='Exit threshold (price change pct)')
def paper_trade_live(symbols, duration_hours, capital, trade_size_pct, entry_threshold, exit_threshold):
    """Run paper trading with live data feed."""
    console.print("\n[bold cyan]Starting paper trading (live mode)[/bold cyan]")
    console.print(f"  Symbols: {symbols}")
    console.print(f"  Duration: {duration_hours}h")
    console.print(f"  Capital: ${capital:.2f}")

    from src.execution.paper_trader import PaperTradingEngine, LivePaperTradingSession

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        console.print("\n[bold red]Error:[/bold red] No symbols provided")
        return

    engine = PaperTradingEngine(initial_capital=capital)

    def _on_trade(trade: dict) -> None:
        console.print(
            f"[dim]{trade['side'].upper()} {trade['symbol']} "
            f"@ ${trade['price']:.4f} size=${trade['size']:.2f}[/dim]"
        )

    session = LivePaperTradingSession(
        symbols=symbol_list,
        engine=engine,
        trade_size_pct=trade_size_pct,
        entry_threshold_pct=entry_threshold,
        exit_threshold_pct=exit_threshold,
        on_trade=_on_trade,
    )

    try:
        asyncio.run(session.run(duration_seconds=duration_hours * 3600))
    except KeyboardInterrupt:
        console.print("\n[yellow]Live paper trading interrupted by user[/yellow]")
    finally:
        metrics = engine.get_performance_metrics()
        console.print("\n[bold green]Live Paper Trading Results:[/bold green]")
        console.print(f"  Final Capital: ${metrics['capital']:.2f}")
        console.print(f"  Total Return: {metrics['total_return_pct']:.2%}")
        console.print(f"  Win Rate: {metrics['win_rate']:.2%}")
        console.print(f"  Total Trades: {metrics['num_trades']}")


@cli.command("rl-paper-trade")
@click.option('--model', required=True, help='Path to trained model (e.g., models/best_model_run_73_step_955500)')
@click.option('--symbol', default='BTC-USD', help='Trading symbol (e.g., BTC-USD, ETH-USD)')
@click.option('--capital', default=1000.0, type=float, help='Starting capital in USD')
@click.option('--interval', default='1h', help='Trading interval (1m, 5m, 15m, 1h)')
@click.option('--duration', default=24, type=int, help='Duration in hours (0 = run indefinitely)')
@click.option('--log-dir', default='./logs/paper_trading', help='Directory for trade logs')
def rl_paper_trade(model, symbol, capital, interval, duration, log_dir):
    """Run the trained RL model on live Coinbase data (paper trading).

    This connects to Coinbase's public API, fetches real-time candles,
    and uses the trained model to make buy/sell decisions with simulated
    capital. All decisions are logged for later analysis.

    \b
    Example:
      python main.py rl-paper-trade --model models/best_model_run_73_step_955500 --symbol BTC-USD --duration 48
    """
    console.print("\n[bold cyan]Starting RL Paper Trading (Live Mode)[/bold cyan]")
    console.print(f"  Model: {model}")
    console.print(f"  Symbol: {symbol}")
    console.print(f"  Capital: ${capital:.2f}")
    console.print(f"  Interval: {interval}")
    console.print(f"  Duration: {duration}h {'(indefinite)' if duration == 0 else ''}")

    from src.execution.live_rl_trader import LiveRLPaperTrader

    try:
        trader = LiveRLPaperTrader(
            model_path=model,
            symbol=symbol,
            initial_capital=capital,
            interval=interval,
            log_dir=log_dir,
        )

        metrics = trader.run(duration_hours=duration, verbose=True)

        # Print final metrics via Rich
        console.print("\n[bold green]Final Metrics:[/bold green]")
        console.print(f"  Total Return: {metrics['total_return_pct']:+.2f}%")
        console.print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        console.print(f"  Profit Factor: {metrics['profit_factor']}")
        console.print(f"  Total Trades: {metrics['total_trades']}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Paper trading stopped by user[/yellow]")
    except FileNotFoundError as e:
        console.print(f"\n[bold red]Model not found:[/bold red] {str(e)}")
        console.print("[dim]Make sure the model path is correct (e.g., models/best_model_run_73_step_955500)[/dim]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise


@cli.command()
def info():
    """Display system information and configuration"""
    from src.core.config import get_settings
    
    settings = get_settings()
    
    console.print("\n[bold]System Configuration:[/bold]")
    console.print(f"  Environment: {settings.ENVIRONMENT}")
    console.print(f"  Database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'Not configured'}")
    console.print(f"  Model Save Path: {settings.MODEL_SAVE_PATH}")
    console.print(f"  Log Level: {settings.LOG_LEVEL}")
    console.print(f"  Data Source: {settings.DATA_SOURCE}")
    console.print(f"  Data Interval: {settings.DATA_INTERVAL}")
    console.print(f"  Require History (days): {settings.REQUIRE_HISTORICAL_DAYS}")
    
    console.print("\n[bold]Risk Limits:[/bold]")
    console.print(f"  Max Daily Loss: ${settings.MAX_DAILY_LOSS_USD}")
    console.print(f"  Max Position Size: {settings.MAX_POSITION_SIZE_PCT * 100}%")
    console.print(f"  Max Open Positions: {settings.MAX_OPEN_POSITIONS}")


@cli.command()
def test_env():
    """Test the Gym environment setup"""
    console.print("\n[bold cyan]Testing Gym Environment...[/bold cyan]")

    from src.environment.gym_env import CryptoTradingEnv
    from src.data.collectors.crypto_loader import CryptoDataLoader
    from src.data.sources.base import DataUnavailableError
    from src.core.config import get_settings

    try:
        settings = get_settings()
        loader = CryptoDataLoader(source=settings.DATA_SOURCE)

        # Load dataset from DB
        symbols = []
        if settings.DATA_SYMBOLS:
            symbols = [s.strip() for s in settings.DATA_SYMBOLS.split(",") if s.strip()]
        if not symbols:
            raise DataUnavailableError("DATA_SYMBOLS not set. Provide symbols to test environment.")

        end = datetime.utcnow()
        start = end - timedelta(days=settings.REQUIRE_HISTORICAL_DAYS)
        dataset = loader.load_dataset(
            symbols=symbols,
            interval=settings.DATA_INTERVAL,
            start=start,
            end=end
        )

        env = CryptoTradingEnv(dataset=dataset, interval=settings.DATA_INTERVAL)

        console.print(f"  State space: {env.observation_space}")
        console.print(f"  Action space: {env.action_space}")

        obs, info = env.reset()
        console.print(f"\n  Initial observation shape: {obs.shape}")

        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            console.print(f"  Step {i+1}: action={action}, reward={reward:.4f}")

            if terminated or truncated:
                break

        console.print("\n[bold green]OK[/bold green] Environment test passed!")

    except DataUnavailableError as e:
        console.print(f"\n[bold red]Data Error:[/bold red] {str(e)}")
        raise
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise


if __name__ == '__main__':
    cli()
