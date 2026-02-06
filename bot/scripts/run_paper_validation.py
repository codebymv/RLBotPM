"""Run 30-day paper trading validation."""

from __future__ import annotations

import subprocess


def main() -> None:
    cmd = [
        "python",
        "main.py",
        "paper-trade",
        "--duration",
        "30d",
        "--interval",
        "5m",
        "--model",
        "models/final_run_49",
    ]
    subprocess.run(cmd, cwd="../bot", check=False)


if __name__ == "__main__":
    main()
