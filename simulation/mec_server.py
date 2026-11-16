"""
MEC (Mobile Edge Computing) Server Simulation Model.
Simulates high-performance edge server with queuing, workload variations, and accelerated layer processing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import random


@dataclass
class MECServerConfig:
    """Hardware and queuing parameters for the edge server."""
    # Speedup factor compared to mobile reference CPU (e.g. GPU/NPU acceleration)
    acceleration_factor: float = 8.5  # 8.5x faster than mobile CPU at reference frequency
    
    # Base server latency per layer for seq_len=128
    base_server_layer_latency_s: float = 0.0015  # ~1.5 ms per Transformer layer
    classifier_head_latency_s: float = 0.0008     # ~0.8 ms for task head / output projection
    
    # Queuing model
    service_rate_per_task_s: float = 0.018       # Mean task processing time for queuing delay
    max_queue_capacity: int = 50                 # Maximum tasks in waiting queue
    
    # Dynamic workload variation bounds
    workload_min: float = 0.1                    # Low load (10%)
    workload_max: float = 0.9                    # High load (90%)
    default_workload: float = 0.35


class MECServer:
    """
    Simulates the MEC server execution of remaining layers (L - k), background workload, and queuing delay.
    """

    def __init__(self, config: Optional[MECServerConfig] = None, seed: Optional[int] = None):
        self.config = config or MECServerConfig()
        self.rng = random.Random(seed)
        self.current_workload = self.config.default_workload
        self.current_queue_length = max(0, int(self.current_workload * 8))

    def set_seed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    def update_workload(self, target_workload: Optional[float] = None) -> float:
        """
        Updates server background load. If target not provided, performs stochastic random walk.
        """
        if target_workload is not None:
            self.current_workload = max(0.0, min(1.0, target_workload))
        else:
            delta = self.rng.gauss(0.0, 0.05)
            self.current_workload = max(
                self.config.workload_min,
                min(self.config.workload_max, self.current_workload + delta)
            )

        # Queue length correlates with workload (M/M/1 tendency)
        lambda_factor = self.current_workload / (1.0 - min(0.95, self.current_workload))
        mean_q = lambda_factor * 1.5
        self.current_queue_length = min(
            self.config.max_queue_capacity,
            max(0, int(self.rng.expovariate(1.0 / max(0.5, mean_q))))
        )
        return self.current_workload

    def calculate_server_latency(
        self,
        split_point: int,
        total_layers: int,
        sequence_length: int,
        workload: Optional[float] = None
    ) -> float:
        """
        Computes MEC server execution time for the remaining (total_layers - split_point) layers.
        If split_point == total_layers (fully local), server computation is 0.
        """
        remaining_layers = max(0, total_layers - split_point)
        if remaining_layers == 0:
            return 0.0

        load = workload if workload is not None else self.current_workload
        # Server slowdown under high multi-tenant load
        load_penalty = 1.0 + 0.65 * (load ** 1.5)
        
        seq_scaling = sequence_length / 128.0
        t_layers = (
            remaining_layers
            * self.config.base_server_layer_latency_s
            * load_penalty
            * seq_scaling
        )
        t_head = self.config.classifier_head_latency_s * load_penalty * seq_scaling
        return t_layers + t_head

    def calculate_queue_latency(self, queue_length: Optional[int] = None) -> float:
        """
        Computes queue waiting time before the task is scheduled on server compute resources.
        """
        q = queue_length if queue_length is not None else self.current_queue_length
        if q <= 0:
            return 0.0
        # Average wait is q * mean_service_time with slight variance
        return float(q) * self.config.service_rate_per_task_s * 0.4
