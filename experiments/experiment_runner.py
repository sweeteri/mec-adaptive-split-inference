"""
Experiment Runner for Comparative Evaluation of Split Inference Algorithms.
Executes controlled benchmarking across multiple algorithms under strictly identical conditions.
Supports full configuration of 13+ research parameters, collecting all 10 scientific evaluation metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
import time
import math
import statistics
import os

from simulation.environment import (
    MECSimulationEnv,
    NetworkScenario,
    ModelProfile,
    RewardWeights,
    MobileDeviceConfig,
    MECServerConfig,
    NetworkChannelConfig
)
from simulation.metrics import SimulationEpisodeMetrics
from algorithms.base_policy import BasePolicy
from algorithms.local_inference import LocalInferencePolicy
from algorithms.full_offloading import FullOffloadingPolicy
from algorithms.fixed_split import FixedSplitPolicy
from algorithms.greedy import GreedyAdaptivePolicy
from algorithms.random_policy import RandomPolicy
from algorithms.dqn import DQNPolicy
from algorithms.ppo import PPOPolicy
from algorithms.sac import SACPolicy
from experiments.result_storage import ResultStorage


@dataclass
class ExperimentConfig:
    """Complete experimental setup encompassing all research configuration parameters."""
    experiment_name: str = "Comparative_Benchmark"
    
    # 1. Algorithms for comparison
    selected_algorithms: List[str] = field(
        default_factory=lambda: [
            "Local Inference",
            "Full Offloading",
            "Fixed Split (k=3)",
            "Greedy Adaptive",
            "Random Policy",
            "PPO",
            "DQN"
        ]
    )
    
    # 2. Experiment scale
    num_episodes: int = 3
    tasks_per_episode: int = 25
    seed: int = 42
    
    # 3. Network scenario & channel parameters
    scenario: NetworkScenario = NetworkScenario.FLUCTUATING
    bandwidth_override_mbps: Optional[float] = None
    latency_override_ms: Optional[float] = None
    
    # 4. Model profile parameters
    total_layers: int = 6           # L: number of Transformer encoder layers
    hidden_dim: int = 768           # d: hidden state dimension
    sequence_length: Optional[int] = 128 # Fixed or None for dynamic task prompts
    
    # 5. Device & MEC hardware parameters
    device_freq_ghz: float = 1.8
    mec_acceleration_factor: float = 8.5
    mec_target_workload: Optional[float] = None
    
    # 6. Multi-objective reward weights & constraints
    alpha: float = 0.50             # Latency weight
    beta: float = 0.35              # Energy weight
    lambd: float = 0.15             # Quality degradation weight
    min_acceptable_quality: float = 0.90
    sla_target_ms: float = 100.0


class ExperimentRunner:
    """
    Orchestrates comparative experiments between baseline algorithms and RL agents.
    Guarantees reproducibility, identical evaluation seeds, and collects all 10 target metrics:
      1. Average inference latency
      2. Median inference latency
      3. 95th percentile latency
      4. Mobile energy consumption
      5. Total transmitted data
      6. Average bandwidth utilization
      7. Average inference quality
      8. Quality degradation
      9. Average reward
     10. Quality constraint success rate (plus SLA compliance rate)
    """

    def __init__(self, result_storage: Optional[ResultStorage] = None):
        self.storage = result_storage or ResultStorage()

    def build_environment_from_config(self, config: ExperimentConfig) -> MECSimulationEnv:
        """Constructs and initializes MECSimulationEnv according to ExperimentConfig."""
        model_profile = ModelProfile(
            model_name=f"Transformer-L{config.total_layers}-d{config.hidden_dim}",
            total_layers=config.total_layers,
            hidden_dim=config.hidden_dim,
            reference_seq_len=config.sequence_length or 128
        )

        device_config = MobileDeviceConfig(default_frequency_ghz=config.device_freq_ghz)
        mec_config = MECServerConfig(
            acceleration_factor=config.mec_acceleration_factor,
            default_workload=config.mec_target_workload or 0.35
        )
        network_config = NetworkChannelConfig(scenario=config.scenario)
        if config.bandwidth_override_mbps is not None:
            network_config.stable_bandwidth_mbps = config.bandwidth_override_mbps
        if config.latency_override_ms is not None:
            network_config.stable_latency_ms = config.latency_override_ms

        reward_weights = RewardWeights(
            alpha=config.alpha,
            beta=config.beta,
            lambd=config.lambd,
            min_acceptable_quality=config.min_acceptable_quality
        )

        env = MECSimulationEnv(
            model_profile=model_profile,
            device_config=device_config,
            mec_config=mec_config,
            network_config=network_config,
            reward_weights=reward_weights,
            max_steps_per_episode=config.tasks_per_episode,
            fixed_seq_len=config.sequence_length,
            seed=config.seed
        )
        return env

    def instantiate_policies(
        self,
        algorithm_names: List[str],
        env: MECSimulationEnv
    ) -> List[BasePolicy]:
        """Instantiates requested algorithm policy objects."""
        L = env.model.total_layers
        d = env.model.hidden_dim
        w = env.reward_weights

        policy_map = {
            "Local Inference": lambda: LocalInferencePolicy(total_layers=L),
            "Full Offloading": lambda: FullOffloadingPolicy(),
            "Fixed Split (k=3)": lambda: FixedSplitPolicy(split_point=min(3, L), bitwidth_idx=1),
            "Greedy Adaptive": lambda: GreedyAdaptivePolicy(total_layers=L, hidden_dim=d, reward_weights=w),
            "Random Policy": lambda: RandomPolicy(total_layers=L, seed=42),
            "PPO": lambda: PPOPolicy(total_layers=L),
            "DQN": lambda: DQNPolicy(total_layers=L),
            "SAC": lambda: SACPolicy(total_layers=L)
        }

        policies: List[BasePolicy] = []
        for name in algorithm_names:
            if name in policy_map:
                policies.append(policy_map[name]())
            else:
                # Custom fixed split format e.g. "Fixed Split (k=2)"
                if name.startswith("Fixed Split"):
                    try:
                        k_val = int(name.split("k=")[1].replace(")", "").strip())
                        policies.append(FixedSplitPolicy(split_point=k_val, bitwidth_idx=1))
                    except Exception:
                        policies.append(FixedSplitPolicy(split_point=L // 2, bitwidth_idx=1))

        return policies

    def run_benchmark(
        self,
        policies: List[BasePolicy],
        env: MECSimulationEnv,
        num_episodes: int = 3,
        seed: int = 42,
        experiment_name: str = "Benchmark",
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Runs benchmark across all policies under identical seeds per episode.
        Guarantees that every algorithm faces identical task arrivals, sequence lengths, and channel fading.
        """
        all_episode_metrics: Dict[str, List[SimulationEpisodeMetrics]] = {}
        total_runs = len(policies) * num_episodes
        current_run = 0

        for policy in policies:
            policy_episodes: List[SimulationEpisodeMetrics] = []

            for ep in range(num_episodes):
                current_run += 1
                if progress_callback:
                    progress_callback(policy.name, ep + 1, total_runs)

                # Exactly identical pseudo-random seed per episode across all compared algorithms
                ep_seed = seed + ep * 79
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
                policy_episodes.append(summary)

            all_episode_metrics[policy.name] = policy_episodes
            self.storage.save_episode_metrics_to_csv(policy_episodes, experiment_name=experiment_name)

        # Aggregate across episodes for each algorithm
        aggregated_results: List[Dict[str, Any]] = []
        for policy_name, ep_list in all_episode_metrics.items():
            n = len(ep_list)
            latencies = [e.avg_latency_ms for e in ep_list]
            medians = [e.median_latency_ms for e in ep_list]
            p95s = [e.p95_latency_ms for e in ep_list]
            energies = [e.avg_energy_joules for e in ep_list]
            tot_energies = [e.total_energy_joules for e in ep_list]
            tx_data = [e.total_transmitted_kb for e in ep_list]
            bandwidths = [e.avg_bandwidth_mbps for e in ep_list]
            qualities = [e.avg_quality for e in ep_list]
            degradations = [e.avg_quality_degradation for e in ep_list]
            rewards = [e.avg_reward for e in ep_list]
            q_success = [e.quality_constraint_success_rate for e in ep_list]
            sla_success = [e.sla_success_rate for e in ep_list]

            # Compute means and standard deviations
            avg_lat = statistics.mean(latencies)
            std_lat = statistics.stdev(latencies) if n > 1 else 0.0

            avg_e = statistics.mean(energies)
            std_e = statistics.stdev(energies) if n > 1 else 0.0

            avg_r = statistics.mean(rewards)
            std_r = statistics.stdev(rewards) if n > 1 else 0.0

            aggregated_results.append({
                "algorithm": policy_name,
                "avg_latency_ms": round(avg_lat, 2),
                "std_latency_ms": round(std_lat, 2),
                "median_latency_ms": round(statistics.mean(medians), 2),
                "p95_latency_ms": round(statistics.mean(p95s), 2),
                "avg_energy_joules": round(avg_e, 4),
                "std_energy_joules": round(std_e, 4),
                "total_energy_joules": round(statistics.mean(tot_energies), 4),
                "total_transmitted_kb": round(statistics.mean(tx_data), 2),
                "avg_bandwidth_mbps": round(statistics.mean(bandwidths), 2),
                "avg_quality": round(statistics.mean(qualities), 4),
                "quality_degradation": round(statistics.mean(degradations), 4),
                "avg_reward": round(avg_r, 4),
                "std_reward": round(std_r, 4),
                "quality_constraint_success_rate": round(statistics.mean(q_success), 4),
                "sla_success_rate": round(statistics.mean(sla_success), 4)
            })

        return {
            "experiment_name": experiment_name,
            "scenario": env.network.config.scenario.value,
            "episodes_per_policy": num_episodes,
            "results": aggregated_results,
            "csv_path": self.storage.csv_path,
            "raw_episodes": all_episode_metrics
        }

    def run_configured_experiment(
        self,
        config: ExperimentConfig,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """Runs complete benchmark directly from an ExperimentConfig object."""
        env = self.build_environment_from_config(config)
        policies = self.instantiate_policies(config.selected_algorithms, env)
        return self.run_benchmark(
            policies=policies,
            env=env,
            num_episodes=config.num_episodes,
            seed=config.seed,
            experiment_name=config.experiment_name,
            progress_callback=progress_callback
        )
