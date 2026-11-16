"""
Mobile Device Simulation Model.
Simulates edge CPU frequency scaling (DVFS), execution time, and energy consumption.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MobileDeviceConfig:
    """Hardware and power parameters for the mobile edge device."""
    # Available CPU frequencies in GHz
    frequency_levels_ghz: List[float] = field(
        default_factory=lambda: [1.0, 1.4, 1.8, 2.2, 2.5]
    )
    default_frequency_ghz: float = 1.8
    reference_frequency_ghz: float = 2.0  # Base frequency for profiling benchmarks
    
    # Power Model (DVFS: P = P_idle + kappa * f^3)
    p_idle_watts: float = 0.35        # Standby idle power of device SoC
    kappa_dvfs: float = 0.28           # Dynamic power scaling coefficient (Watts / (GHz)^3)
    p_tx_watts: float = 1.25          # Radio RF transmission power (Wi-Fi/4G/5G uplink)
    
    # Memory
    total_memory_mb: float = 4096.0   # Device RAM
    available_memory_mb: float = 2800.0
    
    # Baseline layer execution benchmark at reference frequency for seq_len=128
    base_layer_latency_s: float = 0.0125  # ~12.5 ms per Transformer layer on mobile CPU at 2.0 GHz
    embedding_latency_s: float = 0.002    # ~2 ms for token embedding lookup


class MobileDevice:
    """
    Mobile Device execution and energy simulator with Dynamic Voltage and Frequency Scaling (DVFS).
    """

    def __init__(self, config: Optional[MobileDeviceConfig] = None):
        self.config = config or MobileDeviceConfig()
        self.current_frequency_ghz = self.config.default_frequency_ghz
        self.available_memory_mb = self.config.available_memory_mb

    def set_frequency(self, frequency_ghz: float) -> float:
        """Sets CPU frequency, snapping to closest valid level if specified discretely."""
        valid_freqs = self.config.frequency_levels_ghz
        closest = min(valid_freqs, key=lambda x: abs(x - frequency_ghz))
        self.current_frequency_ghz = closest
        return self.current_frequency_ghz

    def set_frequency_by_index(self, index: int) -> float:
        """Sets CPU frequency by discrete index [0, len(frequency_levels)-1]."""
        clamped_idx = max(0, min(index, len(self.config.frequency_levels_ghz) - 1))
        self.current_frequency_ghz = self.config.frequency_levels_ghz[clamped_idx]
        return self.current_frequency_ghz

    def get_power_consumption_watts(self, frequency_ghz: Optional[float] = None) -> float:
        """
        Calculates instantaneous CPU computation power using CMOS DVFS model:
        P_comp(f) = P_idle + kappa * f^3
        """
        f = frequency_ghz if frequency_ghz is not None else self.current_frequency_ghz
        return self.config.p_idle_watts + self.config.kappa_dvfs * (f ** 3)

    def calculate_device_latency(
        self,
        split_point: int,
        sequence_length: int,
        frequency_ghz: Optional[float] = None
    ) -> float:
        """
        Computes local device inference time for the first k layers.
        If split_point == 0 (full offloading), local computation time is 0.
        Otherwise includes embedding lookup + k transformer layers scaled by CPU frequency and seq_len.
        """
        if split_point <= 0:
            return 0.0

        f = frequency_ghz if frequency_ghz is not None else self.current_frequency_ghz
        freq_scaling = self.config.reference_frequency_ghz / max(0.2, f)
        
        # Sequence length scaling factor (combination of linear FFN and self-attention)
        seq_scaling = sequence_length / 128.0

        t_layers = split_point * self.config.base_layer_latency_s * freq_scaling * seq_scaling
        t_embed = self.config.embedding_latency_s * freq_scaling * seq_scaling
        return t_embed + t_layers

    def calculate_computation_energy(
        self,
        device_latency_s: float,
        frequency_ghz: Optional[float] = None
    ) -> float:
        """E_comp = P_comp(f) * T_device"""
        if device_latency_s <= 0.0:
            return 0.0
        p_comp = self.get_power_consumption_watts(frequency_ghz)
        return p_comp * device_latency_s

    def calculate_transmission_energy(self, tx_duration_s: float) -> float:
        """E_tx = P_tx * T_tx_raw"""
        if tx_duration_s <= 0.0:
            return 0.0
        return self.config.p_tx_watts * tx_duration_s

    def calculate_total_energy(self, device_latency_s: float, tx_duration_s: float) -> float:
        """E_mobile = E_computation + E_transmission"""
        return (
            self.calculate_computation_energy(device_latency_s)
            + self.calculate_transmission_energy(tx_duration_s)
        )
