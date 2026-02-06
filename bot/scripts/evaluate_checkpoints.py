"""
Evaluate multiple checkpoints for a training run.

Usage:
  python scripts/evaluate_checkpoints.py --run-id 53 --policy MlpLstmPolicy --episodes 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.evaluator import Evaluator


def _parse_step(path: Path) -> int:
    match = re.search(r"_step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _collect_models(models_dir: Path, run_id: int) -> List[Path]:
    patterns = [
        f"best_model_run_{run_id}_step_*.zip",
        f"checkpoint_run_{run_id}_step_*.zip",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(models_dir.glob(pattern))
    final_model = models_dir / f"final_run_{run_id}.zip"
    if final_model.exists():
        candidates.append(final_model)
    return candidates


def _sort_models(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=_parse_step)


def _evaluate_model(path: Path, policy: str, sequence_length: int, episodes: int) -> dict:
    evaluator = Evaluator(
        model_path=str(path.with_suffix("")),
        policy_type=policy,
        sequence_length=sequence_length,
    )
    results = evaluator.evaluate(num_episodes=episodes, deterministic=True)
    results["model_path"] = str(path)
    results["step"] = _parse_step(path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints for a run.")
    parser.add_argument("--run-id", type=int, required=True, help="Training run id")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--policy", type=str, default="MlpLstmPolicy", help="Policy type")
    parser.add_argument("--sequence-length", type=int, default=1, help="Sequence length for MLP stacking")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes per checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models = _sort_models(_collect_models(models_dir, args.run_id))
    if not models:
        raise SystemExit(f"No models found in {models_dir} for run_id={args.run_id}")

    results: List[dict] = []
    for model_path in models:
        result = _evaluate_model(
            model_path,
            policy=args.policy,
            sequence_length=args.sequence_length,
            episodes=args.episodes,
        )
        results.append(result)
        print(
            f"{model_path.name}: return={result['total_return']:.2%}, "
            f"win_rate={result['win_rate']:.2%}, drawdown={result['max_drawdown']:.2%}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
