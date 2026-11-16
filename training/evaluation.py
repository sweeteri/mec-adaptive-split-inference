"""
Evaluation Engine for evaluating policies under rigorously identical conditions.
"""

from typing import List, Dict, Any, Optional
import sys
import os

from simulation.environment import MECSimulationEnv
from simulation.metrics import SimulationEpisodeMetrics
from algorithms.base_policy import BasePolicy


def evaluate_policy(
    policy: BasePolicy,
    env: MECSimulationEnv,
    num_episodes: int = 5,
    seed: int = 42
) -> List[SimulationEpisodeMetrics]:
    """
    Evaluates policy over num_episodes, resetting environment with deterministic seed sequence.
    Returns list of episode metrics summaries.
    """
    summaries: List[SimulationEpisodeMetrics] = []

    for ep in range(num_episodes):
        ep_seed = seed + ep * 100
        obs, info = env.reset(seed=ep_seed)
        policy.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

        summary = env.metrics_collector.get_summary(
            algorithm_name=policy.name,
            scenario_name=env.network.config.scenario.value,
            episode_id=ep + 1
        )
        summaries.append(summary)

    return summaries
