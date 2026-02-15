"""RL agents and baseline strategies"""

from .ppo_agent import PPOAgent, create_parallel_env
from .ensemble_agent import EnsembleAgent
from .specialist_manager import SpecialistManager
from .baseline_agents import (
    RandomAgent,
    BuyAndHoldAgent,
    MeanReversionAgent,
    MomentumAgent,
    ConservativeAgent,
    get_baseline_agents,
    compare_agents
)

__all__ = [
    "PPOAgent",
    "create_parallel_env",
    "RandomAgent",
    "BuyAndHoldAgent",
    "MeanReversionAgent",
    "MomentumAgent",
    "ConservativeAgent",
    "get_baseline_agents",
    "compare_agents",
    "EnsembleAgent",
    "SpecialistManager",
]
