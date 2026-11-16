"""
Unit tests and verification script for Stage 3: Baseline Algorithms.
Evaluates all 5 baselines under identical initial conditions and channel scenarios.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import MECSimulationEnv, NetworkScenario
from algorithms.local_inference import LocalInferencePolicy
from algorithms.full_offloading import FullOffloadingPolicy
from algorithms.fixed_split import FixedSplitPolicy
from algorithms.greedy import GreedyAdaptivePolicy
from algorithms.random_policy import RandomPolicy


def run_policy_evaluation(policy, env, steps=20):
    obs, info = env.reset(seed=42)
    policy.reset()
    
    total_reward = 0.0
    for step in range(steps):
        action = policy.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
            
    summary = env.metrics_collector.get_summary(
        algorithm_name=policy.name,
        scenario_name=env.network.config.scenario.value
    )
    return summary


def test_baseline_algorithms():
    print("Testing Stage 3: Baseline Algorithms on MECSimulationEnv...")
    env = MECSimulationEnv(seed=42, max_steps_per_episode=20)
    
    # 1. Local Inference
    local_pol = LocalInferencePolicy(total_layers=6)
    s_local = run_policy_evaluation(local_pol, env)
    assert s_local.total_steps == 20
    assert s_local.total_transmitted_kb == 0.0, "Local inference must transmit 0 KB"
    assert s_local.avg_quality == 1.0, "Local inference must have quality 1.0"
    print(f"  [1] Local Inference: Latency={s_local.avg_latency_ms:.2f}ms, Energy={s_local.total_energy_joules:.4f}J, Reward={s_local.avg_reward:.4f}")

    # 2. Full Offloading
    offload_pol = FullOffloadingPolicy()
    s_offload = run_policy_evaluation(offload_pol, env)
    assert s_offload.total_steps == 20
    assert s_offload.total_transmitted_kb > 0.0
    assert s_offload.avg_quality == 1.0
    print(f"  [2] Full Offloading: Latency={s_offload.avg_latency_ms:.2f}ms, Energy={s_offload.total_energy_joules:.4f}J, Reward={s_offload.avg_reward:.4f}")

    # 3. Fixed Split (k = 3, INT8)
    fixed_pol = FixedSplitPolicy(split_point=3, bitwidth_idx=1)
    s_fixed = run_policy_evaluation(fixed_pol, env)
    assert s_fixed.total_steps == 20
    assert s_fixed.total_transmitted_kb > 0.0
    print(f"  [3] Fixed Split (k=3): Latency={s_fixed.avg_latency_ms:.2f}ms, Energy={s_fixed.total_energy_joules:.4f}J, Reward={s_fixed.avg_reward:.4f}")

    # 4. Greedy Adaptive
    greedy_pol = GreedyAdaptivePolicy(total_layers=6)
    s_greedy = run_policy_evaluation(greedy_pol, env)
    assert s_greedy.total_steps == 20
    assert s_greedy.quality_constraint_success_rate == 1.0, "Greedy must satisfy quality constraint"
    print(f"  [4] Greedy Adaptive: Latency={s_greedy.avg_latency_ms:.2f}ms, Energy={s_greedy.total_energy_joules:.4f}J, Reward={s_greedy.avg_reward:.4f}")

    # 5. Random Policy
    random_pol = RandomPolicy(total_layers=6, seed=42)
    s_random = run_policy_evaluation(random_pol, env)
    assert s_random.total_steps == 20
    print(f"  [5] Random Policy: Latency={s_random.avg_latency_ms:.2f}ms, Energy={s_random.total_energy_joules:.4f}J, Reward={s_random.avg_reward:.4f}")

    # Sanity comparison: Greedy adaptive should outperform random policy
    assert s_greedy.avg_reward > s_random.avg_reward, f"Greedy reward ({s_greedy.avg_reward}) should exceed Random ({s_random.avg_reward})"
    print(f"  Greedy superiority verified: Reward {s_greedy.avg_reward} vs Random {s_random.avg_reward}")


def test_poor_network_behavior():
    print("Testing algorithm behavior under POOR network conditions...")
    env = MECSimulationEnv(seed=100, max_steps_per_episode=15)
    env.set_scenario(NetworkScenario.POOR)
    
    greedy_pol = GreedyAdaptivePolicy(total_layers=6)
    s_greedy = run_policy_evaluation(greedy_pol, env, steps=15)
    
    offload_pol = FullOffloadingPolicy()
    s_offload = run_policy_evaluation(offload_pol, env, steps=15)
    
    print(f"  Poor Network -> Greedy Latency: {s_greedy.avg_latency_ms:.2f}ms vs Offload Latency: {s_offload.avg_latency_ms:.2f}ms")
    # Under poor network (1-4 Mbps), greedy should adaptively shift computation to mobile or compress aggressively
    assert s_greedy.avg_latency_ms < s_offload.avg_latency_ms, "Greedy should achieve lower latency than pure offloading in poor network"
    print("  Adaptive behavior in poor network verified.")


if __name__ == "__main__":
    test_baseline_algorithms()
    test_poor_network_behavior()
    print("\nALL STAGE 3 BASELINE ALGORITHMS VERIFIED SUCCESSFULLY!")
