"""
Algorithm 2: Full Offloading Baseline Policy.
Transfers raw input tokens to the MEC server (k = 0), which computes the entire model.
Mobile device CPU operates at minimum frequency to minimize standby energy drain.
"""

from typing import Dict, Any, Tuple, Optional
from algorithms.base_policy import BasePolicy


class FullOffloadingPolicy(BasePolicy):
    """
    Full Offloading Policy (k = 0):
    - Split point: k = 0
    - Bitwidth: 32 (FP32 raw tokens) -> index 3
    - CPU Frequency: Minimum frequency (index 0) to conserve mobile battery while waiting
    """

    def __init__(self, freq_idx: int = 0):
        super().__init__(
            name="Full Offloading",
            description="Entire Transformer executed on MEC server (k=0, raw tokens transmitted)"
        )
        self.split_point = 0
        self.bitwidth_idx = 3  # FP32
        self.freq_idx = freq_idx

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
