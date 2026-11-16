"""
Algorithm 4: Greedy Adaptive Heuristic Policy.
Analytically estimates execution latency and energy across all candidate (k, b, f) actions,
selecting the optimal configuration minimizing cost J subject to quality and SLA constraints.
"""

from typing import Dict, Any, Tuple, Optional, List
import math
from algorithms.base_policy import BasePolicy
from simulation.environment import VALID_BITWIDTHS, BITWIDTH_TO_INDEX, INDEX_TO_BITWIDTH, RewardWeights


class GreedyAdaptivePolicy(BasePolicy):
    """
    Greedy Adaptive Policy:
    Evaluates instantaneous channel conditions (bandwidth, latency), device capacity,
    and MEC server workload to pick action (k, b, f) minimizing J = alpha*(T/T_ref) + beta*(E/E_ref) + lambda*D.
    """

    def __init__(
        self,
        total_layers: int = 6,
        hidden_dim: int = 768,
        frequency_levels_ghz: Optional[List[float]] = None,
        reward_weights: Optional[RewardWeights] = None,
        min_acceptable_quality: float = 0.90
    ):
        super().__init__(
            name="Greedy Adaptive",
            description="Optimal heuristic search over analytical latency, energy, and quality models"
        )
        self.total_layers = total_layers
        self.hidden_dim = hidden_dim
        self.freq_levels = frequency_levels_ghz or [1.0, 1.4, 1.8, 2.2, 2.5]
        self.weights = reward_weights or RewardWeights()
        self.min_quality = min_acceptable_quality

        # Device & MEC benchmark profiling constants
        self.base_layer_latency_s = 0.0125
        self.embedding_latency_s = 0.002
        self.base_server_layer_latency_s = 0.0015
        self.classifier_head_latency_s = 0.0008
        self.service_rate_per_task_s = 0.018
        self.p_idle_watts = 0.35
        self.kappa_dvfs = 0.28
        self.p_tx_watts = 1.25

    def _estimate_quality(self, k: int, bitwidth: int) -> Tuple[float, float]:
        if k == 0 or k >= self.total_layers:
            return 1.0, 0.0
        deg_map = {32: 0.0, 16: 0.0025, 8: 0.018, 4: 0.085}
        deg = deg_map.get(bitwidth, 0.02)
        return 1.0 - deg, deg

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int, int]:
        # Extract features from observation vector or info dictionary
        if info and "bandwidth_mbps" in info:
            bw_mbps = float(info["bandwidth_mbps"])
            lat_ms = float(info["latency_ms"])
            workload = float(info.get("mec_workload", 0.35))
            seq_len = int(info.get("seq_len", 128))
            queue_len = int(info.get("queue_len", int(workload * 8)))
        else:
            # Observation vector unpacking:
            # [bw_norm, lat_norm, freq_norm, load_norm, q_norm, seq_norm, prev_lat, prev_e]
            obs_list = list(observation)
            bw_mbps = max(0.5, obs_list[0] * 100.0)
            lat_ms = max(2.0, obs_list[1] * 200.0)
            workload = max(0.0, min(1.0, obs_list[3]))
            queue_len = int(obs_list[4] * 50.0)
            seq_len = int(16 + obs_list[5] * 496.0)

        bytes_per_sec = bw_mbps * 125000.0
        prop_latency_s = lat_ms / 1000.0

        best_cost = float("inf")
        best_action = (0, 3, 0)  # Default fallback: k=0, FP32, lowest freq

        # Search over all candidate combinations
        for k in range(self.total_layers + 1):
            for b_idx, bitwidth in enumerate(VALID_BITWIDTHS):
                # If local or offload, bitwidth is effectively FP32
                if (k == 0 or k == self.total_layers) and b_idx != 3:
                    continue

                q_score, q_deg = self._estimate_quality(k, bitwidth)
                if q_score < self.min_quality:
                    continue  # Hard constraint: discard actions violating minimum quality

                # Network transmission calculation
                if k == self.total_layers:
                    s_bytes = 0
                    t_tx_data = 0.0
                    t_tx_total = 0.0
                elif k == 0:
                    s_bytes = seq_len * 4  # raw tokens
                    t_tx_data = (s_bytes * 1.04) / bytes_per_sec
                    t_tx_total = t_tx_data + prop_latency_s
                else:
                    s_bytes = math.ceil(seq_len * self.hidden_dim * (bitwidth / 8.0))
                    t_tx_data = (s_bytes * 1.04) / bytes_per_sec
                    t_tx_total = t_tx_data + prop_latency_s

                # Server computation & queue delay
                if k == self.total_layers:
                    t_server = 0.0
                    t_queue = 0.0
                else:
                    rem_layers = self.total_layers - k
                    load_penalty = 1.0 + 0.65 * (workload ** 1.5)
                    seq_scale = seq_len / 128.0
                    t_server = (
                        rem_layers * self.base_server_layer_latency_s * load_penalty * seq_scale
                        + self.classifier_head_latency_s * load_penalty * seq_scale
                    )
                    t_queue = queue_len * self.service_rate_per_task_s * 0.4

                # Device CPU frequency evaluation
                for f_idx, f_ghz in enumerate(self.freq_levels):
                    if k == 0 and f_idx != 0:
                        continue  # In offload, mobile CPU stays at lowest power

                    if k == 0:
                        t_device = 0.0
                        p_comp = self.p_idle_watts + self.kappa_dvfs * (f_ghz ** 3)
                        e_comp = 0.0
                    else:
                        freq_scale = 2.0 / max(0.2, f_ghz)
                        seq_scale = seq_len / 128.0
                        t_device = (
                            k * self.base_layer_latency_s * freq_scale * seq_scale
                            + self.embedding_latency_s * freq_scale * seq_scale
                        )
                        p_comp = self.p_idle_watts + self.kappa_dvfs * (f_ghz ** 3)
                        e_comp = p_comp * t_device

                    e_tx = self.p_tx_watts * t_tx_data
                    e_mobile = e_comp + e_tx
                    t_total = t_device + t_tx_total + t_server + t_queue

                    # Multi-objective normalized cost: J = alpha*(T/T_ref) + beta*(E/E_ref) + lambda*D
                    w = self.weights
                    cost = (
                        w.alpha * (t_total / max(1e-4, w.t_ref_seconds))
                        + w.beta * (e_mobile / max(1e-4, w.e_ref_joules))
                        + w.lambd * q_deg
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_action = (k, b_idx, f_idx)

        return best_action

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            "total_layers": self.total_layers,
            "hidden_dim": self.hidden_dim,
            "freq_levels": self.freq_levels,
            "min_quality": self.min_quality
        })
        return cfg
