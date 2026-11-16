"""
Unit tests and verification script for Stage 1: MEC Simulation Environment.
Runs standalone without external test runners or with pytest.
"""

import sys
import os
import math

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.device import MobileDevice, MobileDeviceConfig
from simulation.mec_server import MECServer, MECServerConfig
from simulation.network import NetworkChannel, NetworkChannelConfig, NetworkScenario
from simulation.task_generator import TaskGenerator
from simulation.environment import MECSimulationEnv, ModelProfile, RewardWeights, VALID_BITWIDTHS
from simulation.metrics import MetricsCollector


def test_mobile_device():
    print("Testing MobileDevice...")
    dev = MobileDevice()
    # Test frequency scaling
    f = dev.set_frequency_by_index(2)
    assert f == 1.8, f"Expected 1.8 GHz, got {f}"
    
    # Power CMOS cubic scaling check
    p = dev.get_power_consumption_watts(2.0)
    expected_p = dev.config.p_idle_watts + dev.config.kappa_dvfs * (2.0 ** 3)
    assert abs(p - expected_p) < 1e-5, f"Power calculation mismatch: {p} vs {expected_p}"
    
    # Latency for k=0 (full offloading) must be 0
    t0 = dev.calculate_device_latency(split_point=0, sequence_length=128)
    assert t0 == 0.0, f"Local latency for k=0 must be 0, got {t0}"
    
    # Latency for k=3
    t3 = dev.calculate_device_latency(split_point=3, sequence_length=128)
    assert t3 > 0.0, "Local latency for k=3 must be > 0"
    
    # Energy computation
    e_comp = dev.calculate_computation_energy(t3)
    assert e_comp > 0.0
    print("  MobileDevice passed.")


def test_mec_server():
    print("Testing MECServer...")
    server = MECServer(seed=42)
    # Server latency for k=total_layers (fully local) must be 0
    t_server_local = server.calculate_server_latency(split_point=6, total_layers=6, sequence_length=128)
    assert t_server_local == 0.0, f"MEC server latency for k=L must be 0, got {t_server_local}"
    
    # Server latency for k=2
    t_server_split = server.calculate_server_latency(split_point=2, total_layers=6, sequence_length=128)
    assert t_server_split > 0.0
    
    # Workload update
    w = server.update_workload(0.65)
    assert abs(w - 0.65) < 1e-4
    q_lat = server.calculate_queue_latency()
    assert q_lat >= 0.0
    print("  MECServer passed.")


def test_network_channel():
    print("Testing NetworkChannel...")
    net = NetworkChannel(seed=42)
    
    # Check activation sizes:
    # 1. k = L (fully local) -> 0 bytes
    s_local = net.calculate_activation_size_bytes(split_point=6, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=32)
    assert s_local == 0, f"Expected 0 bytes for k=L, got {s_local}"
    
    # 2. k = 0 (full offloading) -> raw tokens: 128 * 4 = 512 bytes
    s_offload = net.calculate_activation_size_bytes(split_point=0, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=32)
    assert s_offload == 128 * 4, f"Expected 512 bytes for k=0, got {s_offload}"
    
    # 3. 0 < k < L -> N_tokens * hidden_dim * (bitwidth / 8)
    # For seq_len=128, hidden_dim=768, bitwidth=32 (FP32) -> 128 * 768 * 4 = 393,216 bytes (~384 KB)
    s_fp32 = net.calculate_activation_size_bytes(split_point=3, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=32)
    assert s_fp32 == 128 * 768 * 4, f"Expected 393216 bytes, got {s_fp32}"
    
    # For INT8 -> 128 * 768 * 1 = 98,304 bytes (~96 KB, 4x compression)
    s_int8 = net.calculate_activation_size_bytes(split_point=3, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=8)
    assert s_int8 == 128 * 768 * 1, f"Expected 98304 bytes, got {s_int8}"
    assert s_int8 == s_fp32 // 4, "INT8 must be 1/4 the size of FP32"
    
    # For INT4 -> 128 * 768 * 0.5 = 49,152 bytes (8x compression)
    s_int4 = net.calculate_activation_size_bytes(split_point=3, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=4)
    assert s_int4 == 128 * 768 // 2, f"Expected 49152 bytes, got {s_int4}"
    
    # Transmission time calculation
    t_tx_data, t_prop, t_total_tx = net.calculate_transmission_time(s_int8, bandwidth_mbps=50.0, latency_ms=10.0)
    assert t_tx_data > 0
    assert abs(t_prop - 0.010) < 1e-5
    assert abs(t_total_tx - (t_tx_data + t_prop)) < 1e-5, "Transmission latency must sum raw transfer and propagation delay"
    
    # Test all 5 scenarios
    for sc in NetworkScenario:
        net.set_scenario(sc)
        bw, lat = net.update_channel()
        assert bw > 0, f"Bandwidth must be positive in scenario {sc}"
        assert lat > 0, f"Latency must be positive in scenario {sc}"
    print("  NetworkChannel passed.")


def test_task_generator():
    print("Testing TaskGenerator...")
    gen = TaskGenerator(seed=123)
    task1 = gen.generate_task(step=1, fixed_seq_len=128)
    assert task1.sequence_length == 128
    assert task1.task_id == "task_00001"
    
    task2 = gen.generate_task(step=2)
    assert task2.sequence_length in [32, 64, 96, 128, 160, 192, 256]
    print("  TaskGenerator passed.")


def test_environment_lifecycle():
    print("Testing MECSimulationEnv lifecycle and actions...")
    env = MECSimulationEnv(seed=42, max_steps_per_episode=10)
    obs, info = env.reset(seed=42)
    assert len(obs) == 8, f"Observation vector length must be 8, got {len(obs)}"
    
    # Test local inference action: k = 6, bitwidth = 32 (idx 3), freq = 2.5 GHz (idx 4)
    obs, reward, terminated, truncated, info = env.step([6, 3, 4])
    m = info["metrics"]
    assert m["split_point"] == 6
    assert m["t_transmission"] == 0.0, "Transmission time must be 0 for local inference"
    assert m["t_mec"] == 0.0, "MEC time must be 0 for local inference"
    assert m["transmitted_bytes"] == 0
    assert m["quality_score"] == 1.0
    assert m["quality_degradation"] == 0.0
    
    # Test full offloading action: k = 0, bitwidth = 32 (idx 3), freq = 1.0 GHz (idx 0)
    obs, reward, terminated, truncated, info = env.step([0, 3, 0])
    m = info["metrics"]
    assert m["split_point"] == 0
    assert m["t_device"] == 0.0, "Device computation time must be 0 for full offloading"
    assert m["t_mec"] > 0.0
    assert m["quality_score"] == 1.0
    
    # Test split inference with INT8 quantization: k = 3, bitwidth = 8 (idx 1), freq = 1.8 GHz (idx 2)
    obs, reward, terminated, truncated, info = env.step([3, 1, 2])
    m = info["metrics"]
    assert m["split_point"] == 3
    assert m["bitwidth"] == 8
    assert m["t_device"] > 0.0
    assert m["t_transmission"] > 0.0
    assert m["t_mec"] > 0.0
    assert m["quality_score"] < 1.0  # Slight degradation from INT8
    assert m["quality_degradation"] > 0.0
    
    # Run episode to termination
    step_count = 3
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step([2, 1, 1])
        step_count += 1
    
    summary = env.metrics_collector.get_summary(algorithm_name="TestPolicy", scenario_name="Fluctuating")
    assert summary.total_steps == 10
    assert summary.avg_latency_ms > 0
    assert summary.total_energy_joules > 0
    print(f"  MECSimulationEnv passed. Episode completed: {summary.total_steps} steps, avg latency = {summary.avg_latency_ms} ms, avg reward = {summary.avg_reward}")


if __name__ == "__main__":
    test_mobile_device()
    test_mec_server()
    test_network_channel()
    test_task_generator()
    test_environment_lifecycle()
    print("\nALL STAGE 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
