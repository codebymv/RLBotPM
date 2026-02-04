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
def evaluate(model, episodes):
    """Evaluate a trained model on test data"""
    console.print(f"\n[bold cyan]Evaluating model:[/bold cyan] {model}")
    
    from src.training.evaluator import Evaluator
    
    try:
        evaluator = Evaluator(model_path=model)
        results = evaluator.evaluate(num_episodes=episodes)
        
        # Display results
        console.print("\n[bold green]Evaluation Results:[/bold green]")
        console.print(f"  Total Return: {results['total_return']:.2%}")
        console.print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        console.print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        console.print(f"  Win Rate: {results['win_rate']:.2%}")
        console.print(f"  Avg Trade P&L: ${results['avg_trade_pnl']:.2f}")
        
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
