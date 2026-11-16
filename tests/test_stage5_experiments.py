"""
Unit tests and verification script for Stage 5: Experiment Runner & Result Storage.
Validates all 10 evaluation metrics, ExperimentConfig parameterization, and CSV persistence.
"""

import sys
import os
import shutil

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import MECSimulationEnv, NetworkScenario
from algorithms.local_inference import LocalInferencePolicy
from algorithms.greedy import GreedyAdaptivePolicy
from experiments.scenarios import get_scenario_presets
from experiments.result_storage import ResultStorage
from experiments.experiment_runner import ExperimentRunner, ExperimentConfig


def test_experiment_config_execution():
    print("Testing Stage 5: ExperimentConfig execution and metric validation...")
    storage = ResultStorage(results_dir="results/test_stage5_metrics")
    runner = ExperimentRunner(result_storage=storage)

    config = ExperimentConfig(
        experiment_name="Stage5_Validation_Test",
        selected_algorithms=["Local Inference", "Full Offloading", "Fixed Split (k=3)", "Greedy Adaptive"],
        num_episodes=2,
        tasks_per_episode=15,
        seed=101,
        scenario=NetworkScenario.FLUCTUATING,
        total_layers=6,
        hidden_dim=768,
        sequence_length=128,
        device_freq_ghz=1.8,
        mec_acceleration_factor=8.5,
        alpha=0.50,
        beta=0.35,
        lambd=0.15,
        min_acceptable_quality=0.90,
        sla_target_ms=100.0
    )

    res = runner.run_configured_experiment(config)

    assert len(res["results"]) == 4, f"Expected 4 policy results, got {len(res['results'])}"
    assert os.path.exists(res["csv_path"])

    # Verify all 10 required metrics are present and populated with valid numbers
    required_metrics = [
        "avg_latency_ms",                   # 1. Average inference latency
        "median_latency_ms",                # 2. Median inference latency
        "p95_latency_ms",                   # 3. 95th percentile latency
        "avg_energy_joules",                # 4. Mobile energy consumption
        "total_transmitted_kb",             # 5. Total transmitted data
        "avg_bandwidth_mbps",               # 6. Average bandwidth utilization
        "avg_quality",                      # 7. Average inference quality
        "quality_degradation",              # 8. Quality degradation
        "avg_reward",                       # 9. Average reward
        "quality_constraint_success_rate",  # 10. Success rate for quality constraint
        "sla_success_rate"                  # SLA success rate
    ]

    for item in res["results"]:
        for metric_name in required_metrics:
            assert metric_name in item, f"Metric {metric_name} missing from results"
            assert isinstance(item[metric_name], (int, float)), f"Metric {metric_name} must be numeric"

    print("  All 10 required evaluation metrics verified:")
    for m in res["results"]:
        print(f"    - {m['algorithm']:<20}: Latency={m['avg_latency_ms']:.1f}ms (p95={m['p95_latency_ms']:.1f}ms), Energy={m['avg_energy_joules']:.4f}J, Q={m['avg_quality']*100:.1f}%, Reward={m['avg_reward']:.3f}")

    # Verify CSV storage integrity
    records = storage.load_all_results_as_dicts()
    assert len(records) >= 8  # 4 policies * 2 episodes = 8 episode records
    print(f"  CSV persistence verified: {len(records)} records loaded from {res['csv_path']}")

    # Clean up test output directory
    if os.path.exists("results/test_stage5_metrics"):
        shutil.rmtree("results/test_stage5_metrics")


if __name__ == "__main__":
    test_experiment_config_execution()
    print("\nALL STAGE 5 EXPERIMENT RUNNER & METRICS TESTS PASSED SUCCESSFULLY!")
