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
@click.option('--arbitrage', is_flag=True, help='Enable arbitrage mode (33-dim obs)')
def evaluate(model, episodes, stochastic, policy, sequence_length, arbitrage):
    """Evaluate a trained model on test data"""
    console.print(f"\n[bold cyan]Evaluating model:[/bold cyan] {model}")
    
    from src.training.evaluator import Evaluator
    from src.core.config import get_settings
    
    # Auto-detect arbitrage mode from settings if not explicitly set
    settings = get_settings()
    use_arbitrage = arbitrage or settings.ARBITRAGE_ENABLED
    
    try:
        evaluator = Evaluator(
            model_path=model, 
            policy_type=policy, 
            sequence_length=sequence_length,
            arbitrage_enabled=use_arbitrage
        )
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


@cli.command("collect-multi")
@click.option('--sources', default="coinbase,kraken", help='Comma-separated exchanges (e.g., coinbase,kraken)')
@click.option('--symbols', required=True, help='Comma-separated symbols (e.g., BTC-USD,ETH-USD)')
@click.option('--interval', default="1h", help='Candle interval (e.g., 1m,5m,1h,1d)')
@click.option('--days', default=30, help='Number of days of history to collect')
def collect_multi(sources, symbols, interval, days):
    """Collect OHLCV data from multiple exchanges for arbitrage analysis"""
    source_list = [s.strip() for s in sources.split(",") if s.strip()]
    console.print(
        f"\n[bold cyan]Collecting multi-exchange data:[/bold cyan]"
        f"\n  Sources: {source_list}"
        f"\n  Interval: {interval}"
        f"\n  Days: {days}"
    )

    from src.data.collectors.crypto_loader import CryptoDataLoader, MultiSourceLoader
    from src.data.sources.base import DataUnavailableError

    try:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        
        # Collect from each source
        for source in source_list:
            console.print(f"\n[cyan]Collecting from {source}...[/cyan]")
            loader = CryptoDataLoader(source=source)
            loader.sync_symbols(symbol_list)

            end = datetime.utcnow()
            start = end - timedelta(days=days)

            count = loader.collect_ohlcv(
                symbols=symbol_list,
                interval=interval,
                start=start,
                end=end
            )
            console.print(f"  Stored {count} candles from {source}")

        # Verify alignment
        if len(source_list) >= 2:
            console.print("\n[cyan]Verifying cross-exchange alignment...[/cyan]")
            multi_loader = MultiSourceLoader(sources=source_list)
            end = datetime.utcnow()
            start = end - timedelta(days=days)
            
            aligned = multi_loader.load_aligned_dataset(
                symbols=symbol_list,
                interval=interval,
                start=start,
                end=end
            )
            console.print(f"  Aligned rows: {len(aligned)}")
            console.print(f"  Spread columns: {[c for c in aligned.columns if 'spread' in c or 'diff' in c]}")

        console.print("\n[bold green]OK[/bold green] Multi-exchange data collection completed!")
        console.print("\n[dim]To enable arbitrage training, set in .env:[/dim]")
        console.print("[dim]  ARBITRAGE_ENABLED=true[/dim]")
        console.print(f"[dim]  DATA_SOURCES={sources}[/dim]")

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
@click.option('--model', required=True, help='Path to trained model (e.g., models/best_model_run_75_step_674000)')
@click.option('--symbols', default=None, help='Comma-separated symbols (default: all from DATA_SYMBOLS in .env)')
@click.option('--capital', default=1000.0, type=float, help='Starting capital in USD')
@click.option('--interval', default='1h', help='Trading interval (1m, 5m, 15m, 1h)')
@click.option('--duration', default=24, type=int, help='Duration in hours (0 = run indefinitely)')
@click.option('--log-dir', default='./logs/paper_trading', help='Directory for trade logs')
def rl_paper_trade(model, symbols, capital, interval, duration, log_dir):
    """Run the trained RL model on live Coinbase data (paper trading).

    Scans ALL configured symbols each hour for trading opportunities.
    Holds one position at a time, just like during training.
    Logs every decision to JSONL files for later analysis.

    \b
    Examples:
      python main.py rl-paper-trade --model models/best_model_run_75_step_674000 --duration 0
      python main.py rl-paper-trade --model models/best_model_run_75_step_674000 --symbols BTC-USD,ETH-USD
    """
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

    console.print("\n[bold cyan]Starting RL Paper Trading (Live Mode)[/bold cyan]")
    console.print(f"  Model: {model}")
    console.print(f"  Symbols: {len(symbol_list) if symbol_list else 'all from .env'}")
    console.print(f"  Capital: ${capital:.2f}")
    console.print(f"  Interval: {interval}")
    console.print(f"  Duration: {duration}h {'(indefinite)' if duration == 0 else ''}")

    from src.execution.live_rl_trader import LiveRLPaperTrader

    try:
        trader = LiveRLPaperTrader(
            model_path=model,
            symbols=symbol_list,
            initial_capital=capital,
            interval=interval,
            log_dir=log_dir,
        )

        metrics = trader.run(duration_hours=duration, verbose=True)

        console.print("\n[bold green]Final Metrics:[/bold green]")
        console.print(f"  Total Return: {metrics['total_return_pct']:+.2f}%")
        console.print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        console.print(f"  Profit Factor: {metrics['profit_factor']}")
        console.print(f"  Total Trades: {metrics['total_trades']}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Paper trading stopped by user[/yellow]")
    except FileNotFoundError as e:
        console.print(f"\n[bold red]Model not found:[/bold red] {str(e)}")
        console.print("[dim]Make sure the model path is correct (e.g., models/best_model_run_75_step_674000)[/dim]")
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


# =============================================================================
# Kalshi Prediction Markets Commands
# =============================================================================

@cli.group()
def kalshi():
    """Kalshi prediction markets trading commands"""
    pass


@kalshi.command()
@click.option('--live', is_flag=True, help='Check production API instead of demo')
def status(live):
    """Check Kalshi API connection and account status"""
    mode = "PRODUCTION" if live else "DEMO"
    console.print(f"\n[bold cyan]Checking Kalshi API Status ({mode})...[/bold cyan]")
    
    from src.data.sources.kalshi import KalshiAdapter
    from src.execution.kalshi_client import KalshiExecutionClient
    
    # Check if credentials are set
    api_key = os.getenv("KALSHI_API_KEY")
    api_secret = os.getenv("KALSHI_API_SECRET")
    
    if not api_key or not api_secret:
        console.print("\n[yellow]Warning:[/yellow] KALSHI_API_KEY and KALSHI_API_SECRET not set")
        console.print("  Set these environment variables to enable trading")
        return
    
    use_demo = not live
    
    # Test adapter (read-only operations)
    try:
        adapter = KalshiAdapter(demo=use_demo)
        health = adapter.healthcheck()
        
        console.print(f"\n[bold]Adapter Health:[/bold]")
        console.print(f"  Status: {'[green]OK[/green]' if health['ok'] else '[red]ERROR[/red]'}")
        console.print(f"  Demo Mode: {health.get('demo', True)}")
        console.print(f"  Markets Available: {health.get('markets', 0)}")
        console.print(f"  RSA Key Loaded: {health.get('authenticated', False)}")
        
    except Exception as e:
        console.print(f"\n[red]Adapter Error:[/red] {e}")
    
    # Test execution client
    try:
        client = KalshiExecutionClient(demo=use_demo)
        health = client.healthcheck()
        
        console.print(f"\n[bold]Execution Client:[/bold]")
        console.print(f"  Status: {'[green]OK[/green]' if health['ok'] else '[red]ERROR[/red]'}")
        console.print(f"  RSA Key Loaded: {health.get('authenticated', False)}")
        console.print(f"  Available Balance: ${health.get('balance_available', 0):.2f}")
        console.print(f"  Portfolio Value: ${health.get('balance_total', 0):.2f}")
        if health.get('error'):
            console.print(f"  [yellow]Note:[/yellow] {health.get('error', '')[:100]}")
        
    except Exception as e:
        console.print(f"\n[red]Client Error:[/red] {e}")


@kalshi.command()
@click.option('--asset', default='BTC', help='Crypto asset to filter (BTC, ETH)')
@click.option('--limit', default=20, help='Max markets to show')
@click.option('--live/--demo', default=True, help='Use production or demo API')
def markets(asset, limit, live):
    """List available Kalshi crypto markets"""
    console.print(f"\n[bold cyan]Fetching Kalshi {asset} Markets...[/bold cyan]")
    
    from src.data.sources.kalshi import KalshiAdapter
    
    try:
        adapter = KalshiAdapter(demo=not live)
        crypto_markets = adapter.get_crypto_markets(asset=asset)
        
        if not crypto_markets:
            console.print(f"\n[yellow]No {asset} markets found[/yellow]")
            return
        
        console.print(f"\n[bold]Found {len(crypto_markets)} {asset} Markets:[/bold]\n")
        
        for i, market in enumerate(crypto_markets[:limit]):
            yes_pct = market.yes_price
            spread = market.yes_ask - market.yes_bid
            
            console.print(f"[bold]{market.ticker}[/bold]")
            console.print(f"  {market.title}")
            console.print(f"  YES: {yes_pct:.0f}c | Spread: {spread:.1f}c | Volume: {market.volume}")
            console.print(f"  Status: {market.status} | Expires: {market.expiration_time}")
            console.print()
            
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")


@kalshi.command()
def positions():
    """Show current Kalshi positions"""
    console.print("\n[bold cyan]Fetching Kalshi Positions...[/bold cyan]")
    
    from src.execution.kalshi_client import KalshiExecutionClient
    
    try:
        client = KalshiExecutionClient(demo=True)
        positions = client.get_positions()
        
        if not positions:
            console.print("\n[dim]No open positions[/dim]")
            return
        
        console.print(f"\n[bold]Open Positions ({len(positions)}):[/bold]\n")
        
        total_exposure = 0
        total_pnl = 0
        
        for pos in positions:
            side = "YES" if pos.position > 0 else "NO"
            contracts = abs(pos.position)
            
            console.print(f"[bold]{pos.ticker}[/bold]")
            console.print(f"  {contracts} {side} contracts")
            console.print(f"  Exposure: ${pos.market_exposure:.2f}")
            console.print(f"  Realized P&L: ${pos.realized_pnl:.2f}")
            console.print()
            
            total_exposure += pos.market_exposure
            total_pnl += pos.realized_pnl
        
        console.print(f"[bold]Totals:[/bold]")
        console.print(f"  Total Exposure: ${total_exposure:.2f}")
        console.print(f"  Total Realized P&L: ${total_pnl:.2f}")
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")


@kalshi.command()
@click.argument('ticker')
@click.option('--side', type=click.Choice(['yes', 'no']), required=True, help='YES or NO')
@click.option('--contracts', default=10, help='Number of contracts')
@click.option('--price', default=None, type=int, help='Limit price in cents (1-99). Omit for market order')
@click.option('--demo/--live', default=True, help='Use demo or live API')
def buy(ticker, side, contracts, price, demo):
    """Place a buy order on Kalshi"""
    mode = "DEMO" if demo else "LIVE"
    console.print(f"\n[bold cyan]Placing {mode} Order...[/bold cyan]")
    
    if not demo:
        if not click.confirm("\n[bold red]WARNING: This is a LIVE order with real money. Continue?[/bold red]"):
            console.print("Order cancelled")
            return
    
    from src.execution.kalshi_client import KalshiExecutionClient, OrderSide
    
    try:
        client = KalshiExecutionClient(demo=demo)
        order_side = OrderSide.YES if side == 'yes' else OrderSide.NO
        
        if price:
            order = client.place_limit_order(
                ticker=ticker.upper(),
                side=order_side,
                price=price,
                contracts=contracts,
            )
            order_type = f"limit @ {price}c"
        else:
            order = client.place_market_order(
                ticker=ticker.upper(),
                side=order_side,
                contracts=contracts,
            )
            order_type = "market"
        
        if order:
            console.print(f"\n[bold green]Order Placed![/bold green]")
            console.print(f"  Order ID: {order.order_id}")
            console.print(f"  Market: {order.ticker}")
            console.print(f"  Side: {order.side.value.upper()}")
            console.print(f"  Type: {order_type}")
            console.print(f"  Contracts: {order.contracts}")
            console.print(f"  Status: {order.status.value}")
        else:
            console.print("\n[red]Order failed[/red]")
            
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")


@kalshi.command()
@click.argument('ticker')
@click.option('--contracts', default=None, type=int, help='Contracts to sell (all if omitted)')
@click.option('--demo/--live', default=True, help='Use demo or live API')
def sell(ticker, contracts, demo):
    """Sell (close) a Kalshi position"""
    mode = "DEMO" if demo else "LIVE"
    console.print(f"\n[bold cyan]Closing {mode} Position...[/bold cyan]")
    
    if not demo:
        if not click.confirm("\n[bold red]WARNING: This is a LIVE trade. Continue?[/bold red]"):
            console.print("Cancelled")
            return
    
    from src.execution.kalshi_client import KalshiExecutionClient
    
    try:
        client = KalshiExecutionClient(demo=demo)
        order = client.sell_position(ticker=ticker.upper(), contracts=contracts, use_market=True)
        
        if order:
            console.print(f"\n[bold green]Position Closed![/bold green]")
            console.print(f"  Order ID: {order.order_id}")
            console.print(f"  Contracts: {order.contracts}")
            console.print(f"  Fill Price: {order.price}c")
        else:
            console.print("\n[yellow]No position to close[/yellow]")
            
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")


def _get_kalshi_ppo_obs():
    """Build current crypto observation for PPO signal (optional). Returns None if unavailable."""
    try:
        from src.core.config import get_settings
        from src.data.collectors.crypto_loader import CryptoDataLoader
        settings = get_settings()
        loader = CryptoDataLoader(source=settings.DATA_SOURCE)
        symbols = [s.strip() for s in (settings.DATA_SYMBOLS or "BTC-USD").split(",") if s.strip()]
        if not symbols:
            return None
        from datetime import datetime, timedelta
        end = datetime.utcnow()
        start = end - timedelta(days=min(7, getattr(settings, "REQUIRE_HISTORICAL_DAYS", 7)))
        dataset = loader.load_dataset(symbols=symbols, interval=settings.DATA_INTERVAL, start=start, end=end)
        if dataset is None or dataset.empty:
            return None
        from src.environment.gym_env import CryptoTradingEnv
        env = CryptoTradingEnv(dataset=dataset, interval=settings.DATA_INTERVAL)
        obs, _ = env.reset()
        return obs
    except Exception:
        return None


@kalshi.command()
@click.option('--min-edge', default=0.10, help='Minimum edge to show (0.10 = 10%)')
@click.option('--limit', default=20, help='Max opportunities to show')
@click.option('--category', default=None, help='Filter by category (elections, economics, fed)')
def scan(min_edge, limit, category):
    """Scan all markets for trading opportunities"""
    console.print(f"\n[bold cyan]Scanning Kalshi Markets for Opportunities...[/bold cyan]")
    console.print(f"[dim]Min edge: {min_edge:.0%} | Limit: {limit}[/dim]\n")
    
    from pathlib import Path
    import yaml
    from src.data.sources.kalshi import KalshiAdapter
    from src.strategies.kalshi_signals import KalshiSignalAggregator, MarketCategory
    
    kalshi_cfg = {}
    cfg_path = Path(__file__).parent.parent / "shared" / "config" / "kalshi_config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            kalshi_cfg = yaml.safe_load(f) or {}
    crypto_cfg = kalshi_cfg.get("strategy", {}).get("crypto_signals", {})
    use_ppo = bool(crypto_cfg.get("use_ppo_signal", False))
    ppo_path = crypto_cfg.get("ppo_model_path") or None
    
    try:
        adapter = KalshiAdapter(demo=False)  # Use production for real data
        aggregator = KalshiSignalAggregator(
            use_ppo_signal=use_ppo,
            ppo_model_path=ppo_path,
            get_ppo_obs=_get_kalshi_ppo_obs if use_ppo else None,
        )
        
        # Fetch all open markets
        console.print("[dim]Fetching markets...[/dim]")
        all_markets = adapter.get_markets(status="open", limit=500)
        
        if not all_markets:
            console.print("[yellow]No markets found[/yellow]")
            return
        
        console.print(f"[dim]Analyzing {len(all_markets)} markets...[/dim]\n")
        
        # Convert to dict format for analyzer
        market_dicts = [
            {
                "ticker": m.ticker,
                "title": m.title,
                "category": m.category,
                "yes_price": m.yes_price,
                "volume": m.volume,
            }
            for m in all_markets
        ]
        
        # Filter by category if specified
        if category:
            cat_enum = MarketCategory(category.lower())
            market_dicts = [
                m for m in market_dicts
                if aggregator.categorize_market(m) == cat_enum
            ]
            console.print(f"[dim]Filtered to {len(market_dicts)} {category} markets[/dim]\n")
        
        # Find opportunities
        opportunities = aggregator.find_opportunities(market_dicts, min_edge=min_edge)
        
        if not opportunities:
            console.print("[yellow]No opportunities found with sufficient edge[/yellow]")
            console.print(f"[dim]Try lowering --min-edge below {min_edge:.0%}[/dim]")
            return
        
        console.print(f"[bold green]Found {len(opportunities)} opportunities:[/bold green]\n")
        
        for i, opp in enumerate(opportunities[:limit]):
            edge_color = "green" if opp.edge > 0 else "red"
            action_color = "green" if opp.recommendation == "BUY_YES" else "yellow" if opp.recommendation == "BUY_NO" else "dim"
            
            console.print(f"[bold]{i+1}. {opp.ticker}[/bold]")
            console.print(f"   {opp.title[:80]}...")
            console.print(f"   Market: {opp.market_price:.0%} | Model: {opp.our_probability:.0%} | [{edge_color}]Edge: {opp.edge:+.1%}[/{edge_color}]")
            console.print(f"   [{action_color}]{opp.recommendation}[/{action_color}] | Confidence: {opp.confidence:.0%} | Size: {opp.position_size_pct:.1%}")
            if opp.signals:
                signals_str = ", ".join(f"{s.source}:{s.probability:.0%}" for s in opp.signals[:3])
                console.print(f"   [dim]Signals: {signals_str}[/dim]")
            console.print()
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()


@kalshi.command()
@click.argument('ticker')
def analyze(ticker):
    """Deep analysis of a specific market"""
    console.print(f"\n[bold cyan]Analyzing {ticker}...[/bold cyan]\n")
    
    from src.data.sources.kalshi import KalshiAdapter
    from src.strategies.kalshi_signals import KalshiSignalAggregator
    
    try:
        adapter = KalshiAdapter(demo=False)
        aggregator = KalshiSignalAggregator()
        
        # Get market details
        market = adapter.get_market(ticker.upper())
        
        console.print(f"[bold]{market.title}[/bold]")
        console.print(f"Ticker: {market.ticker}")
        console.print(f"Status: {market.status}")
        console.print(f"Expires: {market.expiration_time}")
        console.print(f"Volume: {market.volume} | Open Interest: {market.open_interest}")
        console.print()
        
        console.print("[bold]Current Prices:[/bold]")
        console.print(f"  YES: {market.yes_price:.0f}c (bid: {market.yes_bid:.0f}, ask: {market.yes_ask:.0f})")
        console.print(f"  NO:  {market.no_price:.0f}c")
        console.print(f"  Spread: {market.yes_ask - market.yes_bid:.1f}c")
        console.print()
        
        # Run analysis
        market_dict = {
            "ticker": market.ticker,
            "title": market.title,
            "category": market.category,
            "yes_price": market.yes_price,
            "volume": market.volume,
        }
        
        analysis = aggregator.analyze_market(market_dict)
        
        console.print("[bold]Model Analysis:[/bold]")
        console.print(f"  Category: {aggregator.categorize_market(market_dict).value}")
        console.print(f"  Market Probability: {analysis.market_price:.1%}")
        console.print(f"  Model Probability:  {analysis.our_probability:.1%}")
        console.print(f"  Edge: {analysis.edge:+.1%}")
        console.print(f"  Confidence: {analysis.confidence:.1%}")
        console.print()
        
        console.print("[bold]Signals:[/bold]")
        if analysis.signals:
            for signal in analysis.signals:
                console.print(f"  {signal.source}: {signal.probability:.1%} (conf: {signal.confidence:.1%})")
        else:
            console.print("  [dim]No external signals available[/dim]")
        console.print()
        
        rec_color = "green" if "YES" in analysis.recommendation else "yellow" if "NO" in analysis.recommendation else "dim"
        console.print(f"[bold]Recommendation:[/bold] [{rec_color}]{analysis.recommendation}[/{rec_color}]")
        console.print(f"[bold]Position Size:[/bold] {analysis.position_size_pct:.1%} of portfolio")
        console.print(f"\n[dim]{analysis.reasoning}[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")


@kalshi.command()
@click.option('--tickers', default=None, help='Comma-separated tickers to backfill (default: fetch open crypto markets)')
@click.option('--limit', default=500, help='Max history points per market')
@click.option('--demo/--live', default=True, help='Use demo or live API')
@click.option('--dry-run', is_flag=True, help='Fetch and report only, do not write to DB')
def backfill(tickers, limit, demo, dry_run):
    """Backfill Kalshi market history to DB for RL training"""
    from src.data.sources.kalshi import KalshiAdapter, backfill_kalshi_market_to_db
    from src.data.database import init_db, DatabaseSession

    console.print("\n[bold cyan]Kalshi history backfill[/bold cyan]")
    if dry_run:
        console.print("[yellow]Dry run: no DB writes[/yellow]\n")

    init_db()
    adapter = KalshiAdapter(demo=demo)

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        try:
            markets = adapter.get_crypto_markets("BTC") + adapter.get_crypto_markets("ETH")
            ticker_list = list({m.ticker for m in markets})[:30]
        except Exception as e:
            console.print(f"[red]Failed to list markets: {e}[/red]")
            return
        if not ticker_list:
            console.print("[yellow]No markets to backfill. Use --tickers TICKER1,TICKER2[/yellow]")
            return

    console.print(f"Markets: {len(ticker_list)}")
    total = 0
    for ticker in ticker_list:
        if dry_run:
            rows = adapter.fetch_and_normalize_history(ticker, limit=limit)
            n = len(rows)
            console.print(f"  {ticker}: {n} rows (dry run)")
            total += n
            continue
        try:
            with DatabaseSession() as session:
                n = backfill_kalshi_market_to_db(adapter, session, ticker, limit=limit)
            if n > 0:
                console.print(f"  {ticker}: +{n} rows")
            total += n
        except Exception as e:
            console.print(f"  [red]{ticker}: {e}[/red]")
    console.print(f"\n[bold]Total: {total} rows[/bold]")


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
