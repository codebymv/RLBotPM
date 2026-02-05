"""
RLTrade Bot - Main entry point for training and evaluation

This is the command-line interface for the RL trading bot.
Use this to train models, evaluate performance, and manage the bot.
"""

import click
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
def train(episodes, checkpoint, config):
    """Train the RL agent on historical data"""
    console.print(f"\n[bold green]Starting training:[/bold green] {episodes} episodes")
    
    if checkpoint:
        console.print(f"[yellow]Resuming from checkpoint:[/yellow] {checkpoint}")
    
    # Import here to avoid loading heavy dependencies on CLI help
    from src.training.trainer import Trainer
    
    try:
        trainer = Trainer(config_path=config)
        
        if checkpoint:
            trainer.load_checkpoint(checkpoint)
        
        trainer.train(total_episodes=episodes)
        
        console.print("\n[bold green]✓[/bold green] Training completed!")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        raise


@cli.command()
@click.option('--model', required=True, help='Path to trained model')
@click.option('--episodes', default=100, help='Number of evaluation episodes')
@click.option('--stochastic', is_flag=True, help='Use stochastic actions for evaluation')
def evaluate(model, episodes, stochastic):
    """Evaluate a trained model on test data"""
    console.print(f"\n[bold cyan]Evaluating model:[/bold cyan] {model}")
    
    from src.training.evaluator import Evaluator
    
    try:
        evaluator = Evaluator(model_path=model)
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
        if "profit_factor" in results:
            console.print(f"  Profit Factor: {results['profit_factor']:.2f}")
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
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        raise


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
        console.print(f"\n[bold red]✗ Data Error:[/bold red] {str(e)}")
        raise
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
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
        console.print(f"\n[bold red]✗ Data Error:[/bold red] {str(e)}")
        raise
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
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
        loader = CryptoDataLoader(source=source)
        loader.sync_symbols()

        end = datetime.utcnow()
        start = end - timedelta(days=days)
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

        loader.collect_ohlcv(
            symbols=symbol_list,
            interval=interval,
            start=start,
            end=end
        )

        console.print("\n[bold green]✓[/bold green] Data collection completed!")

    except DataUnavailableError as e:
        console.print(f"\n[bold red]✗ Data Error:[/bold red] {str(e)}")
        raise
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        raise


@cli.command()
def paper_trade():
    """Run paper trading with live data (Phase 2)"""
    console.print("\n[bold yellow]Paper Trading Mode[/bold yellow]")
    console.print("[dim]Phase 2 feature - Coming soon![/dim]")


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

        console.print("\n[bold green]✓[/bold green] Environment test passed!")

    except DataUnavailableError as e:
        console.print(f"\n[bold red]✗ Data Error:[/bold red] {str(e)}")
        raise
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        raise


if __name__ == '__main__':
    cli()
