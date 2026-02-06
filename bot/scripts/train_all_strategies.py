"""Train all specialized strategies in parallel."""

from __future__ import annotations

import multiprocessing as mp
import subprocess
from typing import Dict


STRATEGIES = [
    {
        "name": "momentum",
        "config": "shared/config/strategy_momentum.yaml",
        "episodes": 200000,
    },
    {
        "name": "mean_reversion",
        "config": "shared/config/strategy_mean_reversion.yaml",
        "episodes": 200000,
    },
    {
        "name": "breakout",
        "config": "shared/config/strategy_breakout.yaml",
        "episodes": 200000,
    },
]


def train_strategy(strategy: Dict[str, str]) -> None:
    """Train a single strategy."""
    name = strategy["name"]
    print(f"Training {name} strategy...")

    cmd = [
        "python",
        "main.py",
        "train",
        "--episodes",
        str(strategy["episodes"]),
        "--config",
        strategy["config"],
    ]

    result = subprocess.run(cmd, cwd="../bot", capture_output=True, text=True, check=False)
    if result.returncode == 0:
        print(f"✓ {name} training complete!")
    else:
        print(f"✗ {name} training failed!")
        print(result.stderr)


if __name__ == "__main__":
    with mp.Pool(processes=3) as pool:
        pool.map(train_strategy, STRATEGIES)

    print("\nAll strategies trained!")
