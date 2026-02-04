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

# Load environment variables
load_dotenv()

console = Console()


@click.group()
def cli():
    """RLTrade - Reinforcement Learning Trading Bot for Polymarket"""
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
@click.option('--months', default=6, help='Months of historical data to collect')
@click.option('--min-volume', default=10000, help='Minimum market volume in USD')
def collect_data(months, min_volume):
    """Collect historical data from Polymarket"""
    console.print(f"\n[bold cyan]Collecting data:[/bold cyan] {months} months, min volume ${min_volume}")
    
    from src.data.collectors.historical_loader import HistoricalDataLoader
    
    try:
        loader = HistoricalDataLoader()
        loader.collect_historical_data(
            months=months,
            min_volume=min_volume
        )
        
        console.print("\n[bold green]✓[/bold green] Data collection completed!")
        
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
    
    console.print("\n[bold]Risk Limits:[/bold]")
    console.print(f"  Max Daily Loss: ${settings.MAX_DAILY_LOSS_USD}")
    console.print(f"  Max Position Size: {settings.MAX_POSITION_SIZE_PCT * 100}%")
    console.print(f"  Max Open Positions: {settings.MAX_OPEN_POSITIONS}")


@cli.command()
def test_env():
    """Test the Gym environment setup"""
    console.print("\n[bold cyan]Testing Gym Environment...[/bold cyan]")
    
    from src.environment.gym_env import PolymarketTradingEnv
    
    try:
        env = PolymarketTradingEnv()
        
        console.print(f"  State space: {env.observation_space}")
        console.print(f"  Action space: {env.action_space}")
        
        # Run a few random steps
        obs, info = env.reset()
        console.print(f"\n  Initial observation shape: {obs.shape}")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            console.print(f"  Step {i+1}: action={action}, reward={reward:.4f}")
            
            if terminated or truncated:
                break
        
        console.print("\n[bold green]✓[/bold green] Environment test passed!")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
        raise


if __name__ == '__main__':
    cli()
