"""
Algorithm 3: Fixed Split Inference Baseline Policy.
Partitions the model at a user-configured static split point k and constant bitwidth b.
"""

from typing import Dict, Any, Tuple, Optional
from algorithms.base_policy import BasePolicy


class FixedSplitPolicy(BasePolicy):
    """
    Fixed Split Policy:
    - User selects static split point k in [1, total_layers - 1]
    - User selects static bitwidth (e.g., INT8 -> index 1)
    - User selects static CPU frequency
    """

    def __init__(
        self,
        split_point: int = 3,
        bitwidth_idx: int = 1,  # Default INT8 (index 1)
        freq_idx: int = 2       # Default mid frequency (e.g. 1.8 GHz)
    ):
        super().__init__(
            name=f"Fixed Split (k={split_point})",
            description=f"Static split point at layer {split_point} with fixed parameters"
        )
        self.split_point = split_point
        self.bitwidth_idx = bitwidth_idx
        self.freq_idx = freq_idx

    def set_parameters(self, split_point: int, bitwidth_idx: int, freq_idx: int) -> None:
        self.split_point = split_point
        self.bitwidth_idx = bitwidth_idx
        self.freq_idx = freq_idx
        self.name = f"Fixed Split (k={split_point})"

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int, int]:
        return (self.split_point, self.bitwidth_idx, self.freq_idx)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            "split_point": self.split_point,
            "bitwidth_idx": self.bitwidth_idx,
            "freq_idx": self.freq_idx
        })
        return cfg
