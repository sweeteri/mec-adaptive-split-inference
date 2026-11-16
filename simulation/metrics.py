"""
Metrics collection and dataclasses for MEC Transformer Split Inference simulation.
Tracks detailed latency, energy, transmission, quality, and reward measurements.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import math


@dataclass
class StepMetrics:
    """Detailed performance and resource metrics for a single inference step."""
    step_id: int
    task_id: str
    sequence_length: int
    batch_size: int
    
    # Decisions (Actions)
    split_point: int          # k in [0, total_layers]
    bitwidth: int             # b in [4, 8, 16, 32]
    cpu_frequency_ghz: float  # f in GHz
    
    # Latency Breakdown (seconds)
    t_device: float           # Mobile local computation time
    t_tx_data: float          # Raw transmission duration over network
    t_propagation: float      # Network propagation/round-trip delay
    t_transmission: float     # Total network time = t_tx_data + t_propagation
    t_mec: float              # MEC server computation time
    t_queue: float            # MEC queue waiting time
    t_total: float            # Total end-to-end inference latency
    
    # Energy Breakdown (Joules)
    e_computation: float      # Mobile computation energy
    e_transmission: float     # Mobile RF transmission energy
    e_mobile: float           # Total mobile device energy consumption
    
    # Network & Data Metrics
    transmitted_bytes: int    # Payload size in bytes
    bandwidth_mbps: float     # Channel bandwidth at decision time
    channel_latency_ms: float # Propagation delay in ms
    
    # Quality & Accuracy
    quality_score: float      # Model output quality [0.0 - 1.0] (1.0 = FP32 baseline)
    quality_degradation: float# D = Q_FP32 - Q_quantized
    sla_target_ms: float      # Target latency SLA in ms
    sla_satisfied: bool       # True if t_total * 1000 <= sla_target_ms
    quality_constraint_met: bool # True if quality_score >= min_acceptable_quality
    
    # Optimization Objective
    objective_cost: float     # J = alpha*(T/T_ref) + beta*(E/E_ref) + lambda*D
    reward: float             # R = -J - penalty
    
    # MEC Server State at decision
    mec_workload: float       # Background load in [0.0, 1.0]
    mec_queue_length: int     # Queue length at execution
    
    # Method tag
    is_simulated: bool = True # Differentiates analytical simulation vs empirical profiling

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationEpisodeMetrics:
    """Aggregated summary of an entire evaluation or simulation episode."""
    episode_id: int
    algorithm_name: str
    scenario_name: str
    total_steps: int
    
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    
    avg_energy_joules: float
    total_energy_joules: float
    
    total_transmitted_kb: float
    avg_bandwidth_mbps: float
    
    avg_quality: float
    avg_quality_degradation: float
    quality_constraint_success_rate: float
    sla_success_rate: float
    
    avg_reward: float
    total_reward: float
    
    steps: List[StepMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("steps", None)
        return d


class MetricsCollector:
    """Stores step metrics and computes episode summaries."""

    def __init__(self):
        self.step_history: List[StepMetrics] = []

    def record_step(self, metrics: StepMetrics) -> None:
        self.step_history.append(metrics)

    def clear(self) -> None:
        self.step_history.clear()

    def get_summary(
        self,
        algorithm_name: str = "Unknown",
        scenario_name: str = "Default",
        episode_id: int = 0
    ) -> SimulationEpisodeMetrics:
        steps = list(self.step_history)
        n = len(steps)
        if n == 0:
            return SimulationEpisodeMetrics(
                episode_id=episode_id,
                algorithm_name=algorithm_name,
                scenario_name=scenario_name,
                total_steps=0,
                avg_latency_ms=0.0,
                median_latency_ms=0.0,
                p95_latency_ms=0.0,
                avg_energy_joules=0.0,
                total_energy_joules=0.0,
                total_transmitted_kb=0.0,
                avg_bandwidth_mbps=0.0,
                avg_quality=0.0,
                avg_quality_degradation=0.0,
                quality_constraint_success_rate=0.0,
                sla_success_rate=0.0,
                avg_reward=0.0,
                total_reward=0.0,
                steps=[],
            )

        latencies_ms = sorted([s.t_total * 1000.0 for s in steps])
        energies = [s.e_mobile for s in steps]
        tx_bytes = [s.transmitted_bytes for s in steps]
        bws = [s.bandwidth_mbps for s in steps]
        qualities = [s.quality_score for s in steps]
        degradations = [s.quality_degradation for s in steps]
        rewards = [s.reward for s in steps]
        q_met = [1.0 if s.quality_constraint_met else 0.0 for s in steps]
        sla_met = [1.0 if s.sla_satisfied else 0.0 for s in steps]

        avg_lat = sum(latencies_ms) / n
        median_lat = latencies_ms[n // 2] if n % 2 == 1 else (latencies_ms[n // 2 - 1] + latencies_ms[n // 2]) / 2.0
        p95_idx = min(n - 1, int(math.ceil(0.95 * n)) - 1)
        p95_lat = latencies_ms[p95_idx]

        total_e = sum(energies)
        avg_e = total_e / n

        return SimulationEpisodeMetrics(
            episode_id=episode_id,
            algorithm_name=algorithm_name,
            scenario_name=scenario_name,
            total_steps=n,
            avg_latency_ms=round(avg_lat, 3),
            median_latency_ms=round(median_lat, 3),
            p95_latency_ms=round(p95_lat, 3),
            avg_energy_joules=round(avg_e, 4),
            total_energy_joules=round(total_e, 4),
            total_transmitted_kb=round(sum(tx_bytes) / 1024.0, 2),
            avg_bandwidth_mbps=round(sum(bws) / n, 2),
            avg_quality=round(sum(qualities) / n, 4),
            avg_quality_degradation=round(sum(degradations) / n, 4),
            quality_constraint_success_rate=round(sum(q_met) / n, 4),
            sla_success_rate=round(sum(sla_met) / n, 4),
            avg_reward=round(sum(rewards) / n, 4),
            total_reward=round(sum(rewards), 4),
            steps=steps,
        )
