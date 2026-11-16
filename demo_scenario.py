"""
Scientific Demonstration Scenario for Adaptive Split Inference in MEC.
Illustrates dynamic network degradation, adaptive split point shifting,
4-bit tensor quantization, and comparative performance of all 8 algorithms.

Usage:
    python demo_scenario.py
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.environment import MECSimulationEnv, NetworkScenario
from simulation.device import MobileDevice
from simulation.mec_server import MECServer
from models.quantization import ActivationQuantizer
from models.transformer_model import ModularTransformer, TransformerConfig
from models.split_inference import SplitInferenceExecutor
from algorithms.local_inference import LocalInferencePolicy
from algorithms.full_offloading import FullOffloadingPolicy
from algorithms.fixed_split import FixedSplitPolicy
from algorithms.greedy import GreedyAdaptivePolicy
from algorithms.random_policy import RandomPolicy
from algorithms.dqn import DQNPolicy
from algorithms.ppo import PPOPolicy
from algorithms.sac import SACPolicy
from experiments.experiment_runner import ExperimentRunner
from experiments.result_storage import ResultStorage


def print_banner(text: str):
    print("\n" + "=" * 90)
    print(f"  {text}")
    print("=" * 90)


def run_demonstration():
    print_banner("SCIENTIFIC RESEARCH DEMONSTRATION:\n  Adaptive Split Inference of Transformer NLP Models in Mobile Edge Computing")
    
    # ---------------------------------------------------------
    # PART 1: PHYSICAL ENVIRONMENT & HARDWARE MODELING
    # ---------------------------------------------------------
    print("\n>>> [PHASE 1] HARDWARE & DVFS POWER MODELING")
    env = MECSimulationEnv(seed=42)
    device = env.device
    print(f"  Mobile Device SoC: P_idle = {device.config.p_idle_watts:.2f} W, kappa = {device.config.kappa_dvfs}")
    print("  CPU DVFS Power Scaling (P_comp = P_idle + kappa * f^3):")
    for f in device.config.frequency_levels_ghz:
        p_comp = device.get_power_consumption_watts(f)
        print(f"    - Frequency {f:.1f} GHz -> Power: {p_comp:.4f} W")

    # ---------------------------------------------------------
    # PART 2: REAL TENSOR QUANTIZATION WITH BITWISE PACKING
    # ---------------------------------------------------------
    print("\n>>> [PHASE 2] REAL ACTIVATION QUANTIZATION (FP32 -> INT16 -> INT8 -> INT4)")
    quantizer = ActivationQuantizer()
    batch_size, seq_len, hidden_dim = 1, 128, 768
    simulated_activations = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"  Reference Activation Tensor Shape: {list(simulated_activations.shape)}")
    print(f"  Original FP32 Size: {batch_size * seq_len * hidden_dim * 4:,} bytes")
    print("  " + "-" * 75)
    print(f"  {'Precision':<10} | {'Payload (B)':<12} | {'Ratio':<8} | {'SNR (dB)':<10} | {'Cosine Sim':<10}")
    print("  " + "-" * 75)

    for bw in [32, 16, 8, 4]:
        q_res = quantizer.quantize(simulated_activations, bitwidth=bw)
        deq_tensor = quantizer.dequantize(q_res, as_torch_tensor=True)
        quality = quantizer.evaluate_quality(simulated_activations, deq_tensor)
        print(f"  {bw:>2}-bit {'(FP32)' if bw==32 else 'INT'+str(bw):<6} | {q_res.compressed_size_bytes:<12,} | {q_res.compression_ratio:<8.2f}x | {quality['snr_db']:<10.2f} | {quality['cosine_similarity']:<10.4f}")
    print("  " + "-" * 75)
    print("  Note: INT4 utilizes physical bitwise packing (2 nibbles per uint8 byte) achieving ~8x compression.")

    # ---------------------------------------------------------
    # PART 3: DYNAMIC ADAPTATION TO CHANNEL DEGRADATION
    # ---------------------------------------------------------
    print("\n>>> [PHASE 3] DYNAMIC REACTION TO CHANNEL DEGRADATION (5G -> Cellular Deadzone)")
    print("  Simulating a mobile user transitioning from high-speed 5G (80 Mbps) into a weak cellular cell (1.5 Mbps)...")
    
    # Scenario A: Fast Channel
    env.set_scenario(NetworkScenario.STABLE)
    env.network.current_bandwidth_mbps = 80.0
    env.network.current_latency_ms = 8.0
    obs, info = env.reset()

    greedy = GreedyAdaptivePolicy(total_layers=6)
    action_fast = greedy.select_action(obs, info)
    k_fast, b_idx_fast, f_idx_fast = action_fast
    bitwidths = [32, 16, 8, 4]
    print(f"\n  [High-Speed 5G: 80 Mbps, 8 ms delay]")
    print(f"  -> Adaptive Policy Decision: Split Point k={k_fast}/6, Quantization={bitwidths[b_idx_fast]}-bit, CPU Freq={env.device.config.frequency_levels_ghz[f_idx_fast]} GHz")
    _, _, _, _, step_info_fast = env.step(action_fast)
    m_fast = step_info_fast["metrics"]
    print(f"     Latency: {m_fast['t_total']*1000:.1f} ms | Mobile Energy: {m_fast['e_mobile']*1000:.2f} mJ | Transmitted: {m_fast['transmitted_bytes']:,} B | SLA Met: {m_fast['sla_satisfied']}")

    # Scenario B: Poor Channel
    env.set_scenario(NetworkScenario.POOR)
    env.network.current_bandwidth_mbps = 1.5
    env.network.current_latency_ms = 95.0
    obs, info = env.reset()

    # Compare static Offload vs Adaptive Greedy
    offload = FullOffloadingPolicy()
    action_offload = offload.select_action(obs)
    _, _, _, _, step_info_offload = env.step(action_offload)
    m_offload = step_info_offload["metrics"]

    # Reset to identical state for greedy
    obs, info = env.reset()
    action_poor = greedy.select_action(obs, info)
    k_poor, b_idx_poor, f_idx_poor = action_poor
    print(f"\n  [Degraded Channel: 1.5 Mbps, 95 ms delay]")
    print(f"  -> Static Full Offload: Latency = {m_offload['t_total']*1000:.1f} ms (SLA VIOLATION: {not m_offload['sla_satisfied']})")
    _, _, _, _, step_info_poor = env.step(action_poor)
    m_poor = step_info_poor["metrics"]
    print(f"  -> Adaptive Policy Decision: Split Point k={k_poor}/6, Quantization={bitwidths[b_idx_poor]}-bit, CPU Freq={env.device.config.frequency_levels_ghz[f_idx_poor]} GHz")
    print(f"     Latency: {m_poor['t_total']*1000:.1f} ms (SLA Met: {m_poor['sla_satisfied']}) | Mobile Energy: {m_poor['e_mobile']*1000:.2f} mJ")
    speedup = ((m_offload['t_total'] - m_poor['t_total']) / m_offload['t_total']) * 100.0
    print(f"     >>> Adaptive policy achieves {speedup:.1f}% latency reduction under degraded network conditions!")

    # ---------------------------------------------------------
    # PART 4: COMPREHENSIVE 8-ALGORITHM COMPARATIVE BENCHMARK
    # ---------------------------------------------------------
    print("\n>>> [PHASE 4] COMPREHENSIVE BENCHMARK ACROSS ALL 8 POLICIES")
    print("  Evaluating all baselines and RL algorithms under Fluctuating Network Conditions (Seed=42)...")
    
    policies = [
        LocalInferencePolicy(total_layers=6),
        FullOffloadingPolicy(),
        FixedSplitPolicy(split_point=3, bitwidth_idx=2),
        GreedyAdaptivePolicy(total_layers=6),
        RandomPolicy(total_layers=6, seed=42),
        DQNPolicy(total_layers=6),
        PPOPolicy(total_layers=6),
        SACPolicy(total_layers=6)
    ]

    storage = ResultStorage()
    runner = ExperimentRunner(result_storage=storage)
    bench_res = runner.run_benchmark(
        policies=policies,
        env=env,
        num_episodes=3,
        seed=42,
        experiment_name="Scientific_Demonstration"
    )

    print("\n" + "=" * 108)
    print(f"{'Algorithm':<22} | {'Avg Lat (ms)':<12} | {'P95 Lat (ms)':<12} | {'Energy (J)':<10} | {'Tx (KB)':<10} | {'Quality':<8} | {'SLA Met':<8} | {'Reward':<8}")
    print("=" * 108)
    for r in bench_res["results"]:
        print(
            f"{r['algorithm']:<22} | "
            f"{r['avg_latency_ms']:<12.1f} | "
            f"{r['p95_latency_ms']:<12.1f} | "
            f"{r['avg_energy_joules']:<10.4f} | "
            f"{r['total_transmitted_kb']:<10.1f} | "
            f"{r['avg_quality']*100:<7.1f}% | "
            f"{r['sla_success_rate']*100:<7.0f}% | "
            f"{r['avg_reward']:<8.3f}"
        )
    print("=" * 108)

    print_banner("DEMONSTRATION COMPLETED SUCCESSFULLY!\n"
                 "To launch the full interactive Streamlit dashboard:\n"
                 "    streamlit run app.py\n\n"
                 "Or access the live React web simulator in your browser preview.")


if __name__ == "__main__":
    run_demonstration()
