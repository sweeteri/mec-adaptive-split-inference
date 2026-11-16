"""
Algorithm 1: Local Inference Baseline Policy.
Executes the entire Transformer model locally on the Mobile Device (k = L).
Zero network transmission overhead and zero server queuing delay.
"""

from typing import Dict, Any, Tuple, Optional
from algorithms.base_policy import BasePolicy


class LocalInferencePolicy(BasePolicy):
    """
    Local Inference Policy (k = L):
    - Split point: k = total_layers
    - Bitwidth: 32 (FP32, no intermediate quantization needed) -> index 3
    - CPU Frequency: High/Max frequency level to minimize device latency
    """

    def __init__(
        self,
        total_layers: int = 6,
        freq_idx: Optional[int] = None,
        num_freq_levels: int = 5
    ):
        super().__init__(
            name="Local Inference",
            description="Entire Transformer executed locally on device (k=L, no offloading)"
        )
        self.total_layers = total_layers
        # Default to max CPU frequency (last index) for local execution performance
        self.freq_idx = (num_freq_levels - 1) if freq_idx is None else freq_idx
        self.bitwidth_idx = 3  # FP32

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int, int]:
        return (self.total_layers, self.bitwidth_idx, self.freq_idx)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            "split_point": self.total_layers,
            "bitwidth_idx": self.bitwidth_idx,
            "freq_idx": self.freq_idx
        })
        return cfg
