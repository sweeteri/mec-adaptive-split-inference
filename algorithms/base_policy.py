"""
Base Policy Abstract Interface for MEC Split Inference decision algorithms.
Ensures identical interaction protocols across Baseline algorithms and RL policies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


class BasePolicy(ABC):
    """
    Abstract base class for all decision-making policies in MEC Adaptive Split Inference.
    
    Action representation: (split_point, bitwidth_idx, freq_idx)
    - split_point: k in [0, total_layers]
    - bitwidth_idx: 0 -> 4-bit, 1 -> 8-bit, 2 -> 16-bit, 3 -> 32-bit (FP32)
    - freq_idx: index in device frequency levels [0, num_freqs - 1]
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.is_trainable: bool = False
        self.is_trained: bool = True  # Non-learning baselines are ready immediately

    @abstractmethod
    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int, int]:
        """
        Selects an action tuple (split_point k, bitwidth_idx b_idx, freq_idx f_idx)
        given the current environment observation and optional auxiliary info.
        """
        pass

    def reset(self) -> None:
        """Resets policy internal state if applicable between episodes."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Returns policy hyperparameter configuration dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "is_trainable": self.is_trainable,
            "is_trained": self.is_trained
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
