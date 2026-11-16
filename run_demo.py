"""
Command-line Demo Runner for MEC Adaptive Split Inference System.
Executes a quick 5-algorithm comparative benchmark and prints results to console.

Usage:
    python run_demo.py
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.environment import MECSimulationEnv, NetworkScenario
from algorithms.local_inference import LocalInferencePolicy
from algorithms.full_offloading import FullOffloadingPolicy
from algorithms.fixed_split import FixedSplitPolicy
from algorithms.greedy import GreedyAdaptivePolicy
from algorithms.random_policy import RandomPolicy
from algorithms.ppo import PPOPolicy
from algorithms.dqn import DQNPolicy
from experiments.experiment_runner import ExperimentRunner
from experiments.result_storage import ResultStorage


def main():
    print("=" * 75)
    print("  MEC Adaptive Split Inference of Transformer Models (RL Prototype)")
    print("  Demonstration Console Runner")
    print("=" * 75)

    # 1. Initialize simulation environment
    print("\n[1/4] Initializing MEC Simulation Environment...")
    env = MECSimulationEnv(seed=42, max_steps_per_episode=20)
    env.set_scenario(NetworkScenario.FLUCTUATING)
    print(f"  Model Profile: {env.model.model_name} (Layers: {env.model.total_layers}, Hidden Dim: {env.model.hidden_dim})")
    print(f"  Active Scenario: {env.network.config.scenario.value.capitalize()}")
    print(f"  Mobile CPU: {env.device.current_frequency_ghz:.1f} GHz | MEC Server Workload: {env.mec.current_workload*100:.1f}%")

    # 2. Setup candidate policies
    print("\n[2/4] Setting up Evaluation Algorithms...")
    policies = [
        LocalInferencePolicy(total_layers=env.model.total_layers),
        FullOffloadingPolicy(),
        FixedSplitPolicy(split_point=3, bitwidth_idx=1),
        GreedyAdaptivePolicy(total_layers=env.model.total_layers),
        RandomPolicy(total_layers=env.model.total_layers, seed=42),
        PPOPolicy(total_layers=env.model.total_layers),
        DQNPolicy(total_layers=env.model.total_layers)
    ]
    for p in policies:
        print(f"  - {p.name}: {p.description}")

    # 3. Run comparative benchmark
    print("\n[3/4] Running Comparative Benchmark under Identical Channel Conditions...")
    storage = ResultStorage()
    runner = ExperimentRunner(result_storage=storage)

    def progress(algo_name, ep, total):
        print(f"    Evaluating {algo_name:<20} | Episode {ep} ...")

    benchmark = runner.run_benchmark(
        policies=policies,
        env=env,
        num_episodes=3,
        seed=42,
        experiment_name="Demo_Fluctuating_Channel",
        progress_callback=progress
    )

    # 4. Print results table
    print("\n[4/4] Benchmark Results Summary:")
    print("-" * 105)
    header = f"{'Algorithm':<22} | {'Latency (ms)':<12} | {'Energy (J)':<10} | {'Tx Data (KB)':<12} | {'Quality':<8} | {'Reward':<8} | {'SLA Met'}"
    print(header)
    print("-" * 105)

    for r in benchmark["results"]:
        row = (
            f"{r['algorithm']:<22} | "
            f"{r['avg_latency_ms']:<12.1f} | "
            f"{r['avg_energy_joules']:<10.4f} | "
            f"{r['total_transmitted_kb']:<12.1f} | "
            f"{r['avg_quality']*100:<7.1f}% | "
            f"{r['avg_reward']:<8.3f} | "
            f"{r['sla_success_rate']*100:.0f}%"
        )
        print(row)
    print("-" * 105)

    print(f"\nResults successfully saved to: {benchmark['csv_path']}")
    print("\nTo launch the interactive Web Dashboard with visual charts, run:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()
