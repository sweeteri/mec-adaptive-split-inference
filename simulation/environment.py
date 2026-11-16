"""
Mobile Edge Computing Simulation Environment for Adaptive Split Inference.
Compliant with Gymnasium API, supporting multi-criteria optimization of Latency, Energy, and Quality.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List, Union
import math
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    # Standalone lightweight fallback if gymnasium is not yet installed
    HAS_GYMNASIUM = False
    class gym:  # type: ignore
        class Env:
            pass
    class spaces:  # type: ignore
        class Box:
            def __init__(self, low, high, shape=None, dtype=None):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
            def sample(self):
                if HAS_NUMPY:
                    return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)
                import random
                return [random.uniform(self.low, self.high) for _ in range(self.shape[0] if self.shape else 1)]
        class MultiDiscrete:
            def __init__(self, nvec):
                self.nvec = list(nvec)
            def sample(self):
                import random
                return [random.randint(0, n - 1) for n in self.nvec]

from simulation.device import MobileDevice, MobileDeviceConfig
from simulation.mec_server import MECServer, MECServerConfig
from simulation.network import NetworkChannel, NetworkChannelConfig, NetworkScenario
from simulation.task_generator import TaskGenerator, InferenceTask
from simulation.metrics import StepMetrics, MetricsCollector, SimulationEpisodeMetrics
from simulation.simpy_des import SimPyDESEngine, HAS_SIMPY


# Supported quantization bitwidths
VALID_BITWIDTHS = [4, 8, 16, 32]
BITWIDTH_TO_INDEX = {4: 0, 8: 1, 16: 2, 32: 3}
INDEX_TO_BITWIDTH = {0: 4, 1: 8, 2: 16, 3: 32}


@dataclass
class ModelProfile:
    """Transformer Model profile parameters."""
    model_name: str = "DistilBERT-base"
    total_layers: int = 6           # L: number of transformer encoder layers (6 for DistilBERT, 12 for BERT)
    hidden_dim: int = 768           # d: hidden representation dimension
    num_attention_heads: int = 12
    intermediate_dim: int = 3072
    reference_seq_len: int = 128


@dataclass
class RewardWeights:
    """Weights and constraints for the multi-objective reward function."""
    alpha: float = 0.50             # Latency weight
    beta: float = 0.35              # Energy weight
    lambd: float = 0.15             # Quality degradation weight
    
    # Reference normalization baselines (based on full local inference at mid frequency)
    t_ref_seconds: float = 0.080    # 80 ms reference latency
    e_ref_joules: float = 0.150     # 0.15 J reference energy
    
    # Quality & SLA constraints
    min_acceptable_quality: float = 0.90   # Minimum quality threshold Q_min
    quality_penalty_factor: float = 5.0    # Penalty applied when Q < Q_min
    sla_penalty: float = 1.0               # Penalty if latency SLA is violated


class MECSimulationEnv(gym.Env):
    """
    Simulation Environment for Adaptive Split Inference in MEC.
    
    State Space: 8-dimensional continuous vector:
    [
        0: Bandwidth (Mbps) normalized [0, 100],
        1: Network Latency (ms) normalized [0, 200],
        2: Mobile CPU Frequency (GHz) normalized [0.5, 3.0],
        3: MEC Workload normalized [0, 1],
        4: MEC Queue Length normalized [0, 50],
        5: Task Sequence Length normalized [16, 512],
        6: Previous Total Latency (s) normalized [0, 0.5],
        7: Previous Mobile Energy (J) normalized [0, 0.5]
    ]
    
    Action Space: MultiDiscrete([total_layers + 1, 4, num_frequency_levels])
    - Action 0: Split point k in [0, total_layers]
    - Action 1: Bitwidth index b_idx in [0, 3] -> [4, 8, 16, 32] bits
    - Action 2: CPU Frequency index f_idx in [0, len(frequency_levels) - 1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        model_profile: Optional[ModelProfile] = None,
        device_config: Optional[MobileDeviceConfig] = None,
        mec_config: Optional[MECServerConfig] = None,
        network_config: Optional[NetworkChannelConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
        max_steps_per_episode: int = 50,
        fixed_seq_len: Optional[int] = None,
        seed: Optional[int] = 42,
        use_simpy_des: bool = True
    ):
        super().__init__()

        self.model = model_profile or ModelProfile()
        self.reward_weights = reward_weights or RewardWeights()
        self.max_steps = max_steps_per_episode
        self.fixed_seq_len = fixed_seq_len
        self.seed_val = seed
        self.use_simpy_des = bool(use_simpy_des and HAS_SIMPY)

        # Core subsystems
        self.device = MobileDevice(device_config)
        self.mec = MECServer(mec_config, seed=seed)
        self.network = NetworkChannel(network_config, seed=seed)
        self.task_generator = TaskGenerator(
            default_seq_len=self.model.reference_seq_len,
            seed=seed
        )
        self.des_engine = SimPyDESEngine(
            mec_workers=1,
            service_rate_s=self.mec.config.service_rate_per_task_s,
            seed=seed
        )
        self.metrics_collector = MetricsCollector()

        # Spaces definition
        self.num_layers = self.model.total_layers
        self.num_bitwidths = len(VALID_BITWIDTHS)
        self.num_freqs = len(self.device.config.frequency_levels_ghz)

        # MultiDiscrete: [split_point, bitwidth_idx, freq_idx]
        self.action_space = spaces.MultiDiscrete([
            self.num_layers + 1,
            self.num_bitwidths,
            self.num_freqs
        ])

        # 8 normalized continuous observations
        dtype_spec = np.float32 if HAS_NUMPY else float
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),
            dtype=dtype_spec
        )

        # Step state tracking
        self.current_step = 0
        self.current_task: Optional[InferenceTask] = None
        self.prev_latency_s = self.reward_weights.t_ref_seconds
        self.prev_energy_j = self.reward_weights.e_ref_joules

        if seed is not None:
            self.reset(seed=seed)

    def set_seed(self, seed: Optional[int]) -> None:
        self.seed_val = seed
        self.device = MobileDevice(self.device.config)
        self.mec.set_seed(seed)
        self.network.set_seed(seed)
        self.task_generator.set_seed(seed)
        if self.use_simpy_des:
            self.des_engine.reset(seed=seed, initial_workload=self.mec.current_workload)

    def set_scenario(self, scenario: Union[str, NetworkScenario]) -> None:
        if isinstance(scenario, str):
            scenario = NetworkScenario(scenario)
        self.network.set_scenario(scenario)

    def _get_obs(self) -> Any:
        seq_len = self.current_task.sequence_length if self.current_task else self.model.reference_seq_len

        # Normalized feature list
        raw_features = [
            min(1.0, max(0.0, self.network.current_bandwidth_mbps / 100.0)),
            min(1.0, max(0.0, self.network.current_latency_ms / 200.0)),
            min(1.0, max(0.0, (self.device.current_frequency_ghz - 0.5) / 2.5)),
            min(1.0, max(0.0, self.mec.current_workload)),
            min(1.0, max(0.0, self.mec.current_queue_length / 50.0)),
            min(1.0, max(0.0, (seq_len - 16) / 496.0)),
            min(1.0, max(0.0, self.prev_latency_s / 0.5)),
            min(1.0, max(0.0, self.prev_energy_j / 0.5))
        ]
        if HAS_NUMPY:
            return np.array(raw_features, dtype=np.float32)
        return raw_features

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            self.set_seed(seed)

        self.current_step = 0
        self.prev_latency_s = self.reward_weights.t_ref_seconds
        self.prev_energy_j = self.reward_weights.e_ref_joules
        self.metrics_collector.clear()

        # Update dynamic states
        self.network.update_channel()
        self.mec.update_workload()
        self.device.set_frequency_by_index(len(self.device.config.frequency_levels_ghz) // 2)

        if self.use_simpy_des:
            self.des_engine.reset(
                seed=self.seed_val,
                initial_workload=self.mec.current_workload
            )
            self.des_engine.update_background_intensity(
                mec_workload=self.mec.current_workload,
                scenario_name=self.network.config.scenario.value
            )

        self.current_task = self.task_generator.generate_task(
            step=0,
            fixed_seq_len=self.fixed_seq_len
        )

        obs = self._get_obs()
        info = {
            "task_id": self.current_task.task_id,
            "seq_len": self.current_task.sequence_length,
            "bandwidth_mbps": self.network.current_bandwidth_mbps,
            "latency_ms": self.network.current_latency_ms,
            "mec_workload": self.mec.current_workload,
            "sim_backend": "simpy_des" if self.use_simpy_des else "analytical",
        }
        return obs, info

    def _estimate_quality(self, split_point: int, bitwidth: int) -> Tuple[float, float]:
        """
        Calculates output quality Q and degradation D relative to FP32.
        - If split_point == 0 (raw tokens sent) or split_point == total_layers (fully local),
          no activation quantization occurs -> Q = 1.0, D = 0.0.
        - For 0 < k < total_layers, activation quantization introduces controlled degradation:
          FP32 (32): D = 0.000, Q = 1.000
          INT16 (16): D = 0.002, Q = 0.998
          INT8   (8): D = 0.018, Q = 0.982
          INT4   (4): D = 0.085, Q = 0.915
        """
        if split_point == 0 or split_point >= self.model.total_layers:
            return 1.0, 0.0

        degradation_table = {
            32: 0.000,
            16: 0.0025,
            8: 0.018,
            4: 0.085
        }
        deg = degradation_table.get(bitwidth, 0.02)
        quality = max(0.0, 1.0 - deg)
        return quality, deg

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Executes one split inference decision step.
        action: tuple/list of (split_point, bitwidth_idx, freq_idx)
        """
        self.current_step += 1

        # Unpack action components
        if isinstance(action, (list, tuple)) or hasattr(action, "__len__"):
            k = int(action[0])
            b_idx = int(action[1])
            f_idx = int(action[2])
        else:
            # Fallback single integer decoder if discrete joint space is used
            total_b = self.num_bitwidths
            total_f = self.num_freqs
            k = int(action) // (total_b * total_f)
            remainder = int(action) % (total_b * total_f)
            b_idx = remainder // total_f
            f_idx = remainder % total_f

        # Clamp actions to valid bounds
        k = max(0, min(self.model.total_layers, k))
        b_idx = max(0, min(self.num_bitwidths - 1, b_idx))
        f_idx = max(0, min(self.num_freqs - 1, f_idx))
        
        bitwidth = INDEX_TO_BITWIDTH[b_idx]
        freq_ghz = self.device.set_frequency_by_index(f_idx)

        task = self.current_task or self.task_generator.generate_task(
            step=self.current_step,
            fixed_seq_len=self.fixed_seq_len
        )

        # 1. Device Computation Time and Energy
        t_device = self.device.calculate_device_latency(
            split_point=k,
            sequence_length=task.sequence_length,
            frequency_ghz=freq_ghz
        )
        e_computation = self.device.calculate_computation_energy(
            device_latency_s=t_device,
            frequency_ghz=freq_ghz
        )

        # 2. Network Transmission Time
        size_bytes = self.network.calculate_activation_size_bytes(
            split_point=k,
            total_layers=self.model.total_layers,
            sequence_length=task.sequence_length,
            hidden_dim=self.model.hidden_dim,
            bitwidth=bitwidth,
            batch_size=task.batch_size
        )
        t_tx_data, t_propagation, t_transmission = self.network.calculate_transmission_time(size_bytes)
        
        # Transmission Energy on Mobile Device
        # Mobile RF radio operates during raw uplink transmission
        e_transmission = self.device.calculate_transmission_energy(t_tx_data)
        e_mobile = e_computation + e_transmission

        # 3. MEC Server Computation & Queue Latency (SimPy DES or Analytical)
        t_mec = self.mec.calculate_server_latency(
            split_point=k,
            total_layers=self.model.total_layers,
            sequence_length=task.sequence_length
        )

        if self.use_simpy_des:
            self.des_engine.update_background_intensity(
                mec_workload=self.mec.current_workload,
                scenario_name=self.network.config.scenario.value
            )
            des_res = self.des_engine.execute_split_inference_step(
                split_point=k,
                total_layers=self.model.total_layers,
                t_device_s=t_device,
                t_tx_data_s=t_tx_data,
                t_propagation_s=t_propagation,
                t_mec_compute_s=t_mec
            )
            t_queue = des_res.t_queue_total_s
            t_total = des_res.t_total_s
            simpy_events = des_res.events_processed
            simpy_time_s = des_res.sim_time_end_s
            if k < self.model.total_layers:
                self.mec.current_queue_length = des_res.mec_queue_length_at_arrival
        else:
            t_queue = 0.0 if k >= self.model.total_layers else self.mec.calculate_queue_latency()
            t_total = t_device + t_transmission + t_mec + t_queue
            simpy_events = 0
            simpy_time_s = 0.0

        # 4. Quality & Degradation
        quality_score, quality_deg = self._estimate_quality(k, bitwidth)
        quality_constraint_met = quality_score >= self.reward_weights.min_acceptable_quality
        sla_satisfied = (t_total * 1000.0) <= task.sla_target_ms

        # 5. Objective Cost and Reward:
        # J = alpha * (T / T_ref) + beta * (E / E_ref) + lambda * D
        w = self.reward_weights
        cost_latency = w.alpha * (t_total / max(1e-4, w.t_ref_seconds))
        cost_energy = w.beta * (e_mobile / max(1e-4, w.e_ref_joules))
        cost_quality = w.lambd * quality_deg
        objective_cost = cost_latency + cost_energy + cost_quality

        # Penalty terms for constraint violations
        penalty = 0.0
        if not quality_constraint_met:
            penalty += w.quality_penalty_factor * (w.min_acceptable_quality - quality_score)
        if not sla_satisfied:
            penalty += w.sla_penalty * ((t_total * 1000.0 - task.sla_target_ms) / task.sla_target_ms)

        reward = -objective_cost - penalty

        # Save previous step references
        self.prev_latency_s = t_total
        self.prev_energy_j = e_mobile

        # Record metrics
        step_metrics = StepMetrics(
            step_id=self.current_step,
            task_id=task.task_id,
            sequence_length=task.sequence_length,
            batch_size=task.batch_size,
            split_point=k,
            bitwidth=bitwidth,
            cpu_frequency_ghz=freq_ghz,
            t_device=round(t_device, 6),
            t_tx_data=round(t_tx_data, 6),
            t_propagation=round(t_propagation, 6),
            t_transmission=round(t_transmission, 6),
            t_mec=round(t_mec, 6),
            t_queue=round(t_queue, 6),
            t_total=round(t_total, 6),
            e_computation=round(e_computation, 6),
            e_transmission=round(e_transmission, 6),
            e_mobile=round(e_mobile, 6),
            transmitted_bytes=size_bytes,
            bandwidth_mbps=round(self.network.current_bandwidth_mbps, 3),
            channel_latency_ms=round(self.network.current_latency_ms, 3),
            quality_score=round(quality_score, 4),
            quality_degradation=round(quality_deg, 4),
            sla_target_ms=task.sla_target_ms,
            sla_satisfied=sla_satisfied,
            quality_constraint_met=quality_constraint_met,
            objective_cost=round(objective_cost, 4),
            reward=round(reward, 4),
            mec_workload=round(self.mec.current_workload, 3),
            mec_queue_length=self.mec.current_queue_length,
            is_simulated=True
        )
        self.metrics_collector.record_step(step_metrics)

        # Advance environment state for next step
        self.network.update_channel()
        self.mec.update_workload()
        self.current_task = self.task_generator.generate_task(
            step=self.current_step,
            fixed_seq_len=self.fixed_seq_len
        )

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()
        info = {
            "metrics": step_metrics.to_dict(),
            "k": k,
            "bitwidth": bitwidth,
            "freq_ghz": freq_ghz,
            "t_total_ms": round(t_total * 1000.0, 2),
            "e_mobile_mj": round(e_mobile * 1000.0, 2),
            "transmitted_kb": round(size_bytes / 1024.0, 2),
            "quality": quality_score,
            "sim_backend": "simpy_des" if self.use_simpy_des else "analytical",
            "simpy_events": simpy_events,
            "simpy_time_s": round(simpy_time_s, 4)
        }

        return obs, reward, terminated, truncated, info
