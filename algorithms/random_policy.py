"""
Algorithm 5: Random Policy Baseline.
Samples split point, bitwidth, and CPU frequency uniformly from valid discrete action spaces.
Serves as an empirical stochastic control baseline.
"""

from typing import Dict, Any, Tuple, Optional
import random
from algorithms.base_policy import BasePolicy
from simulation.environment import VALID_BITWIDTHS


class RandomPolicy(BasePolicy):
    """
    Random Policy:
    Uniformly samples k in [0, total_layers], bitwidth index in [0, 3], and freq index in [0, num_freqs - 1].
    """

    def __init__(
        self,
        total_layers: int = 6,
        num_bitwidths: int = 4,
        num_freq_levels: int = 5,
        seed: Optional[int] = 42
    ):
        super().__init__(
            name="Random Policy",
            description="Uniform random selection across valid action spaces (control baseline)"
        )
        self.total_layers = total_layers
        self.num_bitwidths = num_bitwidths
        self.num_freq_levels = num_freq_levels
        self.rng = random.Random(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int, int]:
        k = self.rng.randint(0, self.total_layers)
        b_idx = self.rng.randint(0, self.num_bitwidths - 1)
        f_idx = self.rng.randint(0, self.num_freq_levels - 1)
        return (k, b_idx, f_idx)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            "total_layers": self.total_layers,
            "num_bitwidths": self.num_bitwidths,
            "num_freq_levels": self.num_freq_levels
        })
        return cfg
