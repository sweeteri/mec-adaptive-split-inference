"""
Stage 7: Comprehensive End-to-End Integration and Verification Test Suite.
Validates the entire pipeline from environment to models, all 8 algorithms,
experiment execution, result storage, and scientific metrics computation.
"""

import unittest
import sys
import os
import shutil

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import MECSimulationEnv, NetworkScenario
from simulation.device import MobileDevice, MobileDeviceConfig
from simulation.mec_server import MECServer, MECServerConfig
from simulation.network import NetworkChannel, NetworkChannelConfig
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
from experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from experiments.result_storage import ResultStorage


class TestStage7Integration(unittest.TestCase):
    """End-to-end integration test suite verifying the complete system."""

    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "stage7_test_output")
        os.makedirs(self.test_dir, exist_ok=True)
        self.env = MECSimulationEnv(seed=123, max_steps_per_episode=10)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_environment_and_physics_models(self):
        """Verify device DVFS, MEC queuing, and network transmission separation."""
        # 1. Device DVFS cubic scaling: P_comp(f) = P_idle + kappa * f^3
        f_low = 1.0
        f_high = 2.5
        p_low = self.env.device.get_power_consumption_watts(f_low)
        p_high = self.env.device.get_power_consumption_watts(f_high)
        self.assertGreater(p_high, p_low)
        
        # 2. Network transmission: verify S(k, b, N) calculation
        # k=L (Local): size must be 0
        s_local = self.env.network.calculate_activation_size_bytes(
            split_point=6, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=8, batch_size=1
        )
        self.assertEqual(s_local, 0)

        # k=0 (Offload): raw token indices (4 bytes per token)
        s_offload = self.env.network.calculate_activation_size_bytes(
            split_point=0, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=8, batch_size=1
        )
        self.assertEqual(s_offload, 128 * 4)

        # 0 < k < L (INT8 split): 128 * 768 * 1 byte = 98304 bytes
        s_split8 = self.env.network.calculate_activation_size_bytes(
            split_point=3, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=8, batch_size=1
        )
        self.assertEqual(s_split8, 128 * 768 * 1)

        # 0 < k < L (INT4 split): half the size of INT8
        s_split4 = self.env.network.calculate_activation_size_bytes(
            split_point=3, total_layers=6, sequence_length=128, hidden_dim=768, bitwidth=4, batch_size=1
        )
        self.assertEqual(s_split4, 128 * 768 * 4 // 8)

        # 3. Channel separation of raw transfer and propagation delay
        t_tx, t_prop, t_total = self.env.network.calculate_transmission_time(s_split8)
        self.assertGreater(t_tx, 0.0)
        self.assertGreater(t_prop, 0.0)
        self.assertAlmostEqual(t_total, t_tx + t_prop, places=6)

    def test_tensor_quantization_and_split_executor(self):
        """Verify PyTorch tensor quantization across FP32, INT16, INT8, and INT4."""
        import torch
        quantizer = ActivationQuantizer()
        x = torch.randn(2, 64, 256)

        for bw in [32, 16, 8, 4]:
            res = quantizer.quantize(x, bitwidth=bw)
            x_rec = quantizer.dequantize(res)
            self.assertEqual(x_rec.shape, tuple(x.shape))
            metrics = quantizer.evaluate_quality(x, x_rec)
            self.assertGreaterEqual(metrics["cosine_similarity"], 0.85)

        # Test SplitInferenceExecutor
        cfg = TransformerConfig.create_tiny_test_config(num_layers=4, hidden_dim=128)
        model = ModularTransformer(cfg)
        executor = SplitInferenceExecutor(model=model)

        tokens = torch.randint(0, 1000, (1, 32))
        split_res = executor.execute(tokens, split_point=2, bitwidth=8)
        self.assertGreater(split_res.t_device_measured_s + split_res.t_mec_measured_s, 0.0)
        self.assertEqual(split_res.split_point, 2)
        self.assertEqual(split_res.bitwidth, 8)
        self.assertIn("cosine_similarity", split_res.quality_metrics)

    def test_all_eight_algorithms_execution(self):
        """Verify that all 8 candidate algorithms execute valid actions and observe rewards."""
        total_layers = self.env.model.total_layers
        policies = {
            "Local": LocalInferencePolicy(total_layers=total_layers),
            "Offload": FullOffloadingPolicy(),
            "FixedSplit": FixedSplitPolicy(split_point=2, bitwidth_idx=2),
            "Greedy": GreedyAdaptivePolicy(total_layers=total_layers),
            "Random": RandomPolicy(total_layers=total_layers, seed=42),
            "DQN": DQNPolicy(total_layers=total_layers),
            "PPO": PPOPolicy(total_layers=total_layers),
            "SAC": SACPolicy(total_layers=total_layers)
        }

        obs, _ = self.env.reset()
        for name, policy in policies.items():
            action = policy.select_action(obs)
            self.assertEqual(len(action), 3, f"Policy {name} must return 3-tuple action (k, b, f)")
            k, b, f = action
            self.assertTrue(0 <= k <= total_layers, f"{name}: k={k} out of bounds")
            self.assertTrue(0 <= b < 4, f"{name}: b={b} out of bounds")
            self.assertTrue(0 <= f < len(self.env.device.config.frequency_levels_ghz), f"{name}: f={f} out of bounds")

            # Execute step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertIsInstance(reward, float)
            self.assertIn("metrics", info)
            step_m = info["metrics"]
            self.assertIn("t_total", step_m)
            self.assertIn("e_mobile", step_m)

    def test_full_experiment_runner_and_csv_storage(self):
        """Verify experiment runner, multi-scenario evaluation, and CSV persistence."""
        storage = ResultStorage(results_dir=self.test_dir)
        runner = ExperimentRunner(result_storage=storage)

        policies = [
            LocalInferencePolicy(total_layers=self.env.model.total_layers),
            FullOffloadingPolicy(),
            GreedyAdaptivePolicy(total_layers=self.env.model.total_layers)
        ]

        benchmark_res = runner.run_benchmark(
            policies=policies,
            env=self.env,
            num_episodes=2,
            seed=42,
            experiment_name="Stage7_E2E_Test"
        )
        self.assertIn("results", benchmark_res)
        self.assertGreater(len(benchmark_res["results"]), 0)

        # Check CSV file creation and readability
        csv_path = benchmark_res["csv_path"]
        self.assertTrue(os.path.exists(csv_path))
        loaded_df = storage.load_results()
        self.assertGreater(len(loaded_df), 0)
        self.assertIn("avg_latency_ms", loaded_df.columns)
        self.assertIn("sla_success_rate", loaded_df.columns)

    def test_simpy_des_engine(self):
        """Verify genuine SimPy discrete-event engine execution and resource contention."""
        from simulation.simpy_des import HAS_SIMPY, SimPyDESEngine
        self.assertTrue(HAS_SIMPY, "SimPy must be installed for DES verification")
        self.assertTrue(self.env.use_simpy_des, "MECSimulationEnv must default to SimPy DES")

        # Execute split step and verify SimPy virtual clock & event telemetry
        obs, _ = self.env.reset(seed=77)
        obs, reward, term, trunc, info = self.env.step([2, 1, 2])
        self.assertEqual(info["sim_backend"], "simpy_des")
        self.assertGreater(info["simpy_events"], 0)
        self.assertGreater(info["simpy_time_s"], 0.0)

        # Verify that higher background load increases DES queueing delay statistically
        engine_low = SimPyDESEngine(mec_workers=1, service_rate_s=0.02, seed=42)
        engine_low.reset(seed=42, initial_workload=0.10, initial_channel_util=0.05, warmup_s=0.5)
        q_low = sum(
            engine_low.execute_split_inference_step(2, 6, 0.02, 0.01, 0.01, 0.01).t_queue_total_s
            for _ in range(15)
        )

        engine_high = SimPyDESEngine(mec_workers=1, service_rate_s=0.02, seed=42)
        engine_high.reset(seed=42, initial_workload=0.85, initial_channel_util=0.55, warmup_s=0.5)
        q_high = sum(
            engine_high.execute_split_inference_step(2, 6, 0.02, 0.01, 0.01, 0.01).t_queue_total_s
            for _ in range(15)
        )
        self.assertGreater(q_high, q_low, "High background Poisson load must cause greater SimPy queue delay")


def suite():
    s = unittest.TestSuite()
    s.addTest(unittest.makeSuite(TestStage7Integration))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    if result.wasSuccessful():
        print("\nALL STAGE 7 INTEGRATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        sys.exit(1)
