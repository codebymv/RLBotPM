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
from typing import List, Dict, Tuple

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


def _golden_score(metrics: Dict[str, float]) -> float:
    """Composite score aligned with durable profitability."""
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    total_return = float(metrics.get("total_return", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    fees = float(metrics.get("fees_pct_of_gross_pnl", 1.0))
    drawdown = float(metrics.get("max_drawdown", 1.0))
    in_position = float(metrics.get("in_position_ratio", 0.0))
    return (
        0.35 * sharpe
        + 45.0 * total_return
        + 0.60 * profit_factor
        - 10.0 * fees
        - 6.0 * drawdown
        + 1.5 * in_position
    )


def _passes_golden_gate(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Hard gate for shortlist promotion."""
    failures: List[str] = []
    if float(metrics.get("total_return", -1.0)) < 0.0:
        failures.append("total_return<0")
    if float(metrics.get("profit_factor", 0.0)) < 1.2:
        failures.append("profit_factor<1.2")
    if float(metrics.get("sharpe_ratio", -999.0)) < 0.5:
        failures.append("sharpe_ratio<0.5")
    if float(metrics.get("max_drawdown", 1.0)) > 0.20:
        failures.append("max_drawdown>0.20")
    if float(metrics.get("fees_pct_of_gross_pnl", 1.0)) > 0.30:
        failures.append("fees_pct_of_gross_pnl>0.30")
    if float(metrics.get("in_position_ratio", 0.0)) < 0.35:
        failures.append("in_position_ratio<0.35")
    return len(failures) == 0, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints for a run.")
    parser.add_argument("--run-id", type=int, required=True, help="Training run id")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--policy", type=str, default="MlpLstmPolicy", help="Policy type")
    parser.add_argument("--sequence-length", type=int, default=1, help="Sequence length for MLP stacking")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes per checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top models to print")
    parser.add_argument("--golden-gate", action="store_true", help="Apply golden ticket gate to shortlist")
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
        result["golden_score"] = _golden_score(result)
        passed, failures = _passes_golden_gate(result)
        result["golden_gate_pass"] = passed
        result["golden_gate_failures"] = failures
        results.append(result)
        print(
            f"{model_path.name}: return={result['total_return']:.2%}, "
            f"win_rate={result['win_rate']:.2%}, drawdown={result['max_drawdown']:.2%}, "
            f"score={result['golden_score']:.3f}, gate={'PASS' if passed else 'FAIL'}"
        )

    ranked = sorted(results, key=lambda r: float(r.get("golden_score", -1e9)), reverse=True)
    shortlist = [r for r in ranked if r.get("golden_gate_pass")] if args.golden_gate else ranked
    top_k = shortlist[: max(1, int(args.top_k))]

    print("\nTop candidates:")
    for i, r in enumerate(top_k, start=1):
        print(
            f"{i}. {Path(r['model_path']).name} | score={r['golden_score']:.3f} | "
            f"ret={r['total_return']:.2%} pf={r['profit_factor']:.2f} "
            f"sharpe={r['sharpe_ratio']:.2f} fees={r['fees_pct_of_gross_pnl']:.2%} "
            f"dd={r['max_drawdown']:.2%} inpos={r['in_position_ratio']:.2%}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
