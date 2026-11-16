"""
Unit tests for Stage 1b: SimPy Discrete-Event Simulation (DES) Engine.
Verifies simpy.Environment, simpy.Resource contention, Poisson background traffic,
and integration inside MECSimulationEnv.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.simpy_des import SimPyDESEngine, HAS_SIMPY
from simulation.environment import MECSimulationEnv


def test_simpy_engine_lifecycle():
    print("Testing SimPyDESEngine lifecycle and resources...")
    assert HAS_SIMPY, "SimPy package must be installed"

    engine = SimPyDESEngine(mec_workers=1, service_rate_s=0.018, seed=42)
    engine.reset(seed=42, initial_workload=0.40, initial_channel_util=0.20, warmup_s=0.30)

    assert engine.sim_env is not None
    assert engine.sim_env.now >= 0.30, "Warmup should advance SimPy simulation clock"
    assert engine.uplink_resource is not None
    assert engine.mec_resource is not None

    # Local inference step (k = L) must bypass wireless and MEC queues
    res_local = engine.execute_split_inference_step(
        split_point=6,
        total_layers=6,
        t_device_s=0.050,
        t_tx_data_s=0.0,
        t_propagation_s=0.0,
        t_mec_compute_s=0.0
    )
    assert res_local.t_transmission_s == 0.0
    assert res_local.t_mec_compute_s == 0.0
    assert res_local.t_queue_total_s == 0.0
    assert abs(res_local.t_total_s - 0.050) < 1e-6

    # Split inference step (k = 2) passes through SimPy uplink & MEC resources
    res_split = engine.execute_split_inference_step(
        split_point=2,
        total_layers=6,
        t_device_s=0.025,
        t_tx_data_s=0.015,
        t_propagation_s=0.012,
        t_mec_compute_s=0.008
    )
    assert res_split.backend == "simpy_des"
    assert res_split.sim_time_end_s > res_split.sim_time_start_s
    assert res_split.t_total_s >= (0.025 + 0.015 + 0.012 + 0.008 - 1e-6)
    assert res_split.events_processed > 0
    print(f"  SimPyDESEngine passed: t_total={res_split.t_total_s*1000:.2f}ms, queue_wait={res_split.t_queue_total_s*1000:.2f}ms, events={res_split.events_processed}")


def test_simpy_load_impact_on_queue():
    print("Testing SimPyDESEngine queue growth under heavy Poisson load...")
    low_load = SimPyDESEngine(mec_workers=1, service_rate_s=0.020, seed=101)
    low_load.reset(seed=101, initial_workload=0.10, initial_channel_util=0.05, warmup_s=0.50)
    q_wait_low = sum(
        low_load.execute_split_inference_step(2, 6, 0.02, 0.01, 0.01, 0.01).t_queue_total_s
        for _ in range(20)
    ) / 20.0

    high_load = SimPyDESEngine(mec_workers=1, service_rate_s=0.020, seed=101)
    high_load.reset(seed=101, initial_workload=0.85, initial_channel_util=0.55, warmup_s=0.50)
    q_wait_high = sum(
        high_load.execute_split_inference_step(2, 6, 0.02, 0.01, 0.01, 0.01).t_queue_total_s
        for _ in range(20)
    ) / 20.0

    assert q_wait_high > q_wait_low, f"Expected higher queue delay under 85% load ({q_wait_high:.4f}s) vs 10% load ({q_wait_low:.4f}s)"
    print(f"  Queue scaling verified: Low load avg wait={q_wait_low*1000:.2f}ms vs High load avg wait={q_wait_high*1000:.2f}ms")


def test_env_uses_simpy_backend():
    print("Testing MECSimulationEnv with SimPy DES backend enabled...")
    env = MECSimulationEnv(seed=42, max_steps_per_episode=10, use_simpy_des=True)
    assert env.use_simpy_des is True
    obs, info = env.reset(seed=42)
    assert info["sim_backend"] == "simpy_des"

    obs, reward, term, trunc, info = env.step([2, 1, 2])
    assert info["sim_backend"] == "simpy_des"
    assert info["simpy_events"] > 0
    assert info["simpy_time_s"] > 0.0
    print(f"  MECSimulationEnv SimPy integration verified: backend={info['sim_backend']}, events={info['simpy_events']}")


if __name__ == "__main__":
    test_simpy_engine_lifecycle()
    test_simpy_load_impact_on_queue()
    test_env_uses_simpy_backend()
    print("\nALL STAGE 1B SIMPY DES TESTS PASSED SUCCESSFULLY!")
