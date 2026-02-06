import json
from pathlib import Path
import sys
import yaml

BOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_ROOT))

from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _policy_settings(config: dict) -> tuple[str, int]:
    policy_type = config.get("ppo", {}).get("policy_type", "MlpPolicy")
    sequence_length = int(config.get("recurrent", {}).get("sequence_length", 1))
    if policy_type != "MlpPolicy":
        sequence_length = 1
    return policy_type, sequence_length


def run_ab_test(episodes: int, eval_episodes: int) -> None:
    reward_config_path = BOT_ROOT.parent / "shared" / "config" / "reward_config.yaml"
    original_reward_config = _load_yaml(reward_config_path)

    if not original_reward_config:
        raise FileNotFoundError(f"Reward config not found at {reward_config_path}")

    old_reward_config = json.loads(json.dumps(original_reward_config))
    old_reward_config["version"] = "v1"
    reward_weights = old_reward_config.setdefault("reward_weights", {})
    reward_weights["sell_profit_bonus"] = 1.0
    reward_weights["manual_exit_bonus"] = 1.2
    reward_weights["loss_aversion_scale"] = 0.0

    results = {}
    try:
        for label, config in (("old_reward_v1", old_reward_config), ("new_reward_v2", original_reward_config)):
            _write_yaml(reward_config_path, config)
            trainer = Trainer(config_path="shared/config/model_config.yaml")
            trainer.train(total_episodes=episodes, checkpoint_frequency=episodes, eval_frequency=episodes)

            model_path = f"models/final_run_{trainer.training_run.id}"
            policy_type, sequence_length = _policy_settings(trainer.config)
            evaluator = Evaluator(
                model_path=model_path,
                policy_type=policy_type,
                sequence_length=sequence_length,
            )
            results[label] = {
                "model_path": model_path,
                "policy_type": policy_type,
                "sequence_length": sequence_length,
                "metrics": evaluator.evaluate(num_episodes=eval_episodes, deterministic=True),
            }
    finally:
        _write_yaml(reward_config_path, original_reward_config)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_ab_test(episodes=5000, eval_episodes=50)
