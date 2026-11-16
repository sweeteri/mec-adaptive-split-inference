"""
Model Manager for managing RL model checkpoints, loading pre-trained weights, and status tracking.
"""

import os
import json
from typing import Dict, Any, Optional
from algorithms.base_policy import BasePolicy
from algorithms.dqn import DQNPolicy
from algorithms.ppo import PPOPolicy
from algorithms.sac import SACPolicy


class ModelManager:
    """Manages persistence of trained RL checkpoints in the results/models/ directory."""

    def __init__(self, storage_dir: str = "results/models"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def get_checkpoint_path(self, algorithm_name: str) -> str:
        clean_name = algorithm_name.lower().replace(" ", "_")
        return os.path.join(self.storage_dir, f"{clean_name}_checkpoint.pth")

    def save_model(self, agent: BasePolicy) -> str:
        path = self.get_checkpoint_path(agent.name)
        if hasattr(agent, "save_checkpoint"):
            agent.save_checkpoint(path)
        return path

    def load_model(self, agent: BasePolicy) -> bool:
        path = self.get_checkpoint_path(agent.name)
        if hasattr(agent, "load_checkpoint") and os.path.exists(path):
            return agent.load_checkpoint(path)
        return False

    def check_status(self, agent: BasePolicy) -> Dict[str, Any]:
        path = self.get_checkpoint_path(agent.name)
        exists = os.path.exists(path)
        return {
            "name": agent.name,
            "is_trainable": agent.is_trainable,
            "is_trained": getattr(agent, "is_trained", False),
            "checkpoint_available": exists,
            "checkpoint_path": path if exists else None
        }
