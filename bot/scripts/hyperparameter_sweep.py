"""
Run a lightweight hyperparameter sweep for LSTM configs.

Usage:
  python scripts/hyperparameter_sweep.py --episodes 5000
"""

from __future__ import annotations

import argparse
from itertools import product

from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM hyperparameter sweep")
    parser.add_argument("--config", default="shared/config/model_config.yaml", help="Config path")
    parser.add_argument("--episodes", type=int, default=5000, help="Episodes per config")
    parser.add_argument("--hidden-sizes", default="128,256,512", help="Comma-separated hidden sizes")
    parser.add_argument("--layers", default="1,2", help="Comma-separated LSTM layer counts")
    parser.add_argument("--shared-lstm", default="false,true", help="Comma-separated shared_lstm flags")
    parser.add_argument("--enable-critic-lstm", default="true", help="Enable critic LSTM")
    args = parser.parse_args()

    hidden_sizes = [int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip()]
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    shared_flags = [x.strip().lower() == "true" for x in args.shared_lstm.split(",") if x.strip()]
    enable_critic_lstm = args.enable_critic_lstm.strip().lower() == "true"

    for hidden_size, layer_count, shared_lstm in product(hidden_sizes, layers, shared_flags):
        overrides = {
            "ppo": {"policy_type": "MlpLstmPolicy"},
            "recurrent": {
                "lstm_hidden_size": hidden_size,
                "lstm_layers": layer_count,
                "shared_lstm": shared_lstm,
                "enable_critic_lstm": enable_critic_lstm,
            },
        }

        print(
            f"Training config: hidden={hidden_size}, layers={layer_count}, "
            f"shared_lstm={shared_lstm}, critic_lstm={enable_critic_lstm}"
        )

        trainer = Trainer(config_path=args.config, overrides=overrides)
        trainer.train(
            total_episodes=args.episodes,
            checkpoint_frequency=args.episodes,
            eval_frequency=args.episodes,
        )


if __name__ == "__main__":
    main()
