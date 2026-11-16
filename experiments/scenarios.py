"""
Simulation Scenario Definitions for MEC Research Experiments.
"""

from typing import Dict, Any
from simulation.network import NetworkScenario, NetworkChannelConfig


def get_scenario_presets() -> Dict[str, Dict[str, Any]]:
    """Returns standard network scenarios for comparative experiments."""
    return {
        "Stable Network": {
            "scenario": NetworkScenario.STABLE,
            "description": "High throughput (65 Mbps), low propagation latency (8 ms), minimal jitter",
            "recommended_steps": 25
        },
        "Fluctuating Network": {
            "scenario": NetworkScenario.FLUCTUATING,
            "description": "Sinusoidal shadow fading and mobility (15 - 75 Mbps, 10 - 35 ms latency)",
            "recommended_steps": 40
        },
        "Congested Network": {
            "scenario": NetworkScenario.CONGESTED,
            "description": "Heavy cellular traffic (5 - 18 Mbps, 40 - 95 ms latency)",
            "recommended_steps": 25
        },
        "Poor Network": {
            "scenario": NetworkScenario.POOR,
            "description": "Degraded edge channel (1.2 - 4.5 Mbps, 80 - 160 ms latency)",
            "recommended_steps": 20
        },
        "Random Network": {
            "scenario": NetworkScenario.RANDOM,
            "description": "Stochastic transitions between channel regimes",
            "recommended_steps": 30
        }
    }
