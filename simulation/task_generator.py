"""
Task Generator for Transformer inference workloads in MEC environment.
Generates inference requests with dynamic sequence lengths, target SLAs, and batch sizes.
"""

from dataclasses import dataclass
from typing import List, Optional
import random


@dataclass
class InferenceTask:
    """Represents a single NLP inference request arriving at the mobile edge device."""
    task_id: str
    sequence_length: int
    batch_size: int = 1
    sla_target_ms: float = 120.0  # Latency deadline SLA (e.g. 120 ms for real-time NLP)
    arrival_step: int = 0


class TaskGenerator:
    """
    Generates reproducible stream of NLP inference tasks.
    Supports fixed sequence lengths or variable token distributions (e.g. user prompts).
    """

    def __init__(
        self,
        default_seq_len: int = 128,
        min_seq_len: int = 32,
        max_seq_len: int = 256,
        default_sla_ms: float = 120.0,
        seed: Optional[int] = None
    ):
        self.default_seq_len = default_seq_len
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.default_sla_ms = default_sla_ms
        self.rng = random.Random(seed)
        self.task_counter: int = 0

    def set_seed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    def generate_task(
        self,
        step: int = 0,
        fixed_seq_len: Optional[int] = None,
        sla_target_ms: Optional[float] = None
    ) -> InferenceTask:
        """Creates a single task. If fixed_seq_len is None, samples from realistic NLP prompt distribution."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter:05d}"
        
        if fixed_seq_len is not None:
            seq_len = fixed_seq_len
        else:
            # Common NLP prompt lengths cluster around 64, 128, 192, 256
            choices = [32, 64, 96, 128, 160, 192, 256]
            weights = [0.1, 0.25, 0.2, 0.25, 0.1, 0.05, 0.05]
            seq_len = self.rng.choices(choices, weights=weights, k=1)[0]

        sla = sla_target_ms if sla_target_ms is not None else self.default_sla_ms

        return InferenceTask(
            task_id=task_id,
            sequence_length=seq_len,
            batch_size=1,
            sla_target_ms=sla,
            arrival_step=step
        )
