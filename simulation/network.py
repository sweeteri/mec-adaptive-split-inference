"""
Network Channel Simulation Model.
Simulates wireless channel conditions (bandwidth, propagation latency, packet overhead)
across 5 scenarios: Stable, Fluctuating, Congested, Poor, and Random.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import math
import random


class NetworkScenario(str, Enum):
    STABLE = "stable"
    FLUCTUATING = "fluctuating"
    CONGESTED = "congested"
    POOR = "poor"
    RANDOM = "random"


@dataclass
class NetworkChannelConfig:
    """Parameters governing wireless communication between Mobile Device and MEC."""
    scenario: NetworkScenario = NetworkScenario.FLUCTUATING
    packet_overhead_ratio: float = 0.04  # 4% header and protocol overhead
    
    # Stable scenario bounds
    stable_bandwidth_mbps: float = 65.0
    stable_latency_ms: float = 8.0
    
    # Fluctuating scenario bounds
    fluctuating_min_mbps: float = 15.0
    fluctuating_max_mbps: float = 75.0
    fluctuating_period_steps: int = 40
    fluctuating_latency_min_ms: float = 10.0
    fluctuating_latency_max_ms: float = 35.0
    
    # Congested scenario bounds
    congested_min_mbps: float = 5.0
    congested_max_mbps: float = 18.0
    congested_latency_min_ms: float = 40.0
    congested_latency_max_ms: float = 95.0
    
    # Poor scenario bounds
    poor_min_mbps: float = 1.2
    poor_max_mbps: float = 4.5
    poor_latency_min_ms: float = 80.0
    poor_latency_max_ms: float = 160.0


class NetworkChannel:
    """
    Simulates dynamic wireless channel conditions and activation transmission latency.
    """

    def __init__(self, config: Optional[NetworkChannelConfig] = None, seed: Optional[int] = None):
        self.config = config or NetworkChannelConfig()
        self.rng = random.Random(seed)
        self.step_counter: int = 0
        
        # State variables
        self.current_bandwidth_mbps: float = self.config.stable_bandwidth_mbps
        self.current_latency_ms: float = self.config.stable_latency_ms
        self.update_channel()

    def set_seed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    def set_scenario(self, scenario: NetworkScenario) -> None:
        self.config.scenario = scenario
        self.update_channel()

    def update_channel(self) -> Tuple[float, float]:
        """
        Advances network time step and updates channel conditions based on current scenario.
        Returns: (bandwidth_mbps, latency_ms)
        """
        self.step_counter += 1
        scenario = self.config.scenario

        if scenario == NetworkScenario.STABLE:
            bw_noise = self.rng.gauss(0, 1.5)
            lat_noise = self.rng.gauss(0, 0.4)
            self.current_bandwidth_mbps = max(30.0, self.config.stable_bandwidth_mbps + bw_noise)
            self.current_latency_ms = max(4.0, self.config.stable_latency_ms + lat_noise)

        elif scenario == NetworkScenario.FLUCTUATING:
            # Multi-frequency sinusoids + noise simulating shadow fading and mobility
            t = self.step_counter
            base = 0.5 * (1.0 + math.sin(2.0 * math.pi * t / self.config.fluctuating_period_steps))
            harmonic = 0.25 * math.sin(4.0 * math.pi * t / self.config.fluctuating_period_steps)
            factor = max(0.0, min(1.0, base + harmonic))
            
            bw_range = self.config.fluctuating_max_mbps - self.config.fluctuating_min_mbps
            bw = self.config.fluctuating_min_mbps + factor * bw_range + self.rng.gauss(0, 2.0)
            
            lat_range = self.config.fluctuating_latency_max_ms - self.config.fluctuating_latency_min_ms
            # Lower bandwidth typically correlates with higher buffering/queuing latency
            lat = self.config.fluctuating_latency_min_ms + (1.0 - factor) * lat_range + self.rng.gauss(0, 1.5)
            
            self.current_bandwidth_mbps = max(self.config.fluctuating_min_mbps * 0.7, bw)
            self.current_latency_ms = max(5.0, lat)

        elif scenario == NetworkScenario.CONGESTED:
            bw = self.rng.uniform(self.config.congested_min_mbps, self.config.congested_max_mbps)
            lat = self.rng.uniform(self.config.congested_latency_min_ms, self.config.congested_latency_max_ms)
            self.current_bandwidth_mbps = max(1.0, bw)
            self.current_latency_ms = max(20.0, lat)

        elif scenario == NetworkScenario.POOR:
            bw = self.rng.uniform(self.config.poor_min_mbps, self.config.poor_max_mbps)
            lat = self.rng.uniform(self.config.poor_latency_min_ms, self.config.poor_latency_max_ms)
            self.current_bandwidth_mbps = max(0.5, bw)
            self.current_latency_ms = max(40.0, lat)

        elif scenario == NetworkScenario.RANDOM:
            # Transitions randomly between distinct channel regimes
            regime = self.rng.random()
            if regime < 0.35:
                bw = self.rng.uniform(40.0, 80.0)
                lat = self.rng.uniform(6.0, 15.0)
            elif regime < 0.75:
                bw = self.rng.uniform(12.0, 35.0)
                lat = self.rng.uniform(20.0, 50.0)
            else:
                bw = self.rng.uniform(1.5, 9.0)
                lat = self.rng.uniform(60.0, 130.0)
            self.current_bandwidth_mbps = bw
            self.current_latency_ms = lat

        return self.current_bandwidth_mbps, self.current_latency_ms

    def calculate_activation_size_bytes(
        self,
        split_point: int,
        total_layers: int,
        sequence_length: int,
        hidden_dim: int,
        bitwidth: int,
        batch_size: int = 1
    ) -> int:
        """
        Computes transmitted payload size:
        - k = total_layers (fully local): 0 bytes.
        - k = 0 (full offloading): input tokens (raw int32 token IDs: batch * seq_len * 4 bytes).
        - 0 < k < total_layers (split inference):
            S(k, b, N) = N_batch * N_tokens * hidden_dim * (bitwidth / 8) bytes.
        """
        if split_point >= total_layers:
            return 0
        if split_point == 0:
            # Input tokens sent to server: 4 bytes per token ID (int32)
            return batch_size * sequence_length * 4
        
        # Split point activation tensor: shape (batch_size, sequence_length, hidden_dim)
        total_elements = batch_size * sequence_length * hidden_dim
        # bitwidth / 8 gives bytes per element
        size_bytes = math.ceil(total_elements * (bitwidth / 8.0))
        return size_bytes

    def calculate_transmission_time(
        self,
        size_bytes: int,
        bandwidth_mbps: Optional[float] = None,
        latency_ms: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculates network latency strictly distinguishing:
        1. t_tx_data = raw transmission time over the link (depends on size & bandwidth)
        2. t_propagation = channel propagation latency
        3. t_total_tx = t_tx_data + t_propagation
        Returns: (t_tx_data, t_propagation, t_total_tx) in seconds.
        """
        if size_bytes <= 0:
            return 0.0, 0.0, 0.0

        bw_mbps = bandwidth_mbps if bandwidth_mbps is not None else self.current_bandwidth_mbps
        lat_ms = latency_ms if latency_ms is not None else self.current_latency_ms

        # Bandwidth in bytes per second: 1 Mbps = 10^6 bits/sec = 125,000 bytes/sec
        bytes_per_sec = max(1000.0, bw_mbps * 125000.0)
        
        effective_bytes = size_bytes * (1.0 + self.config.packet_overhead_ratio)
        t_tx_data = effective_bytes / bytes_per_sec
        t_propagation = (lat_ms / 1000.0)
        t_total_tx = t_tx_data + t_propagation

        return t_tx_data, t_propagation, t_total_tx
