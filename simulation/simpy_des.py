"""
Discrete-Event Simulation (DES) Engine based on SimPy.
Models real event-driven execution timelines, shared wireless uplink contention,
stochastic multi-tenant background traffic, and MEC server compute queues using simpy.Resource.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import random

try:
    import simpy
    HAS_SIMPY = True
except ImportError:
    HAS_SIMPY = False
    simpy = None  # type: ignore


@dataclass
class DESExecutionResult:
    """Measured timings and queue telemetry from a SimPy discrete-event execution."""
    t_device_s: float
    t_tx_queue_s: float
    t_tx_data_s: float
    t_propagation_s: float
    t_transmission_s: float
    t_mec_queue_s: float
    t_mec_compute_s: float
    t_queue_total_s: float
    t_total_s: float
    mec_queue_length_at_arrival: int
    uplink_queue_length_at_arrival: int
    sim_time_start_s: float
    sim_time_end_s: float
    events_processed: int
    backend: str = "simpy_des"


class SimPyDESEngine:
    """
    SimPy Discrete-Event Simulator for Mobile Edge Computing Split Inference.
    Maintains:
      - simpy.Environment virtual clock
      - simpy.Resource for wireless uplink channel access (capacity=1)
      - simpy.Resource for MEC GPU/CPU compute queue (capacity=mec_workers)
      - Continuous stochastic Poisson background traffic competing for both resources
    """

    def __init__(
        self,
        mec_workers: int = 1,
        service_rate_s: float = 0.018,
        seed: Optional[int] = 42
    ):
        self.mec_workers = max(1, mec_workers)
        self.service_rate_s = service_rate_s
        self.rng = random.Random(seed)

        self.target_mec_workload: float = 0.35
        self.target_channel_utilization: float = 0.15

        self.sim_env: Optional[Any] = None
        self.uplink_resource: Optional[Any] = None
        self.mec_resource: Optional[Any] = None

        self.events_processed: int = 0
        self.tasks_completed: int = 0
        self.background_tasks_served: int = 0

        if HAS_SIMPY:
            self.reset(seed=seed)

    def reset(
        self,
        seed: Optional[int] = None,
        initial_workload: float = 0.35,
        initial_channel_util: float = 0.15,
        warmup_s: float = 0.20
    ) -> None:
        """Initializes a fresh SimPy discrete-event environment and warms up queues."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.target_mec_workload = max(0.05, min(0.88, initial_workload))
        self.target_channel_utilization = max(0.02, min(0.75, initial_channel_util))
        self.events_processed = 0
        self.tasks_completed = 0
        self.background_tasks_served = 0

        if not HAS_SIMPY:
            return

        self.sim_env = simpy.Environment()
        self.uplink_resource = simpy.Resource(self.sim_env, capacity=1)
        self.mec_resource = simpy.Resource(self.sim_env, capacity=self.mec_workers)

        # Spawn continuous background traffic generators
        self.sim_env.process(self._mec_background_generator())
        self.sim_env.process(self._uplink_background_generator())

        if warmup_s > 0:
            self.sim_env.run(until=warmup_s)

    def update_background_intensity(self, mec_workload: float, scenario_name: str) -> None:
        """Updates Poisson background arrival intensity based on server load and wireless scenario."""
        self.target_mec_workload = max(0.05, min(0.88, mec_workload))
        scenario_util_map = {
            "stable": 0.05,
            "fluctuating": 0.15,
            "congested": 0.48,
            "poor": 0.35,
            "random": 0.22
        }
        self.target_channel_utilization = scenario_util_map.get(scenario_name.lower(), 0.15)

    def _background_mec_job(self, service_time_s: float):
        with self.mec_resource.request() as req:
            yield req
            yield self.sim_env.timeout(service_time_s)
        self.background_tasks_served += 1
        self.events_processed += 1

    def _mec_background_generator(self):
        """Poisson arrival process generating competing multi-tenant tasks on the MEC server."""
        while True:
            mean_service = max(0.002, self.service_rate_s)
            # Arrival rate lambda = rho * workers / E[S]
            arrival_rate = (self.target_mec_workload * self.mec_workers) / mean_service
            interarrival = self.rng.expovariate(max(1e-3, arrival_rate))
            yield self.sim_env.timeout(interarrival)
            self.events_processed += 1

            service_time = self.rng.expovariate(1.0 / mean_service)
            # Cap extreme exponential tail to keep discrete steps physically bounded
            service_time = min(service_time, mean_service * 4.0)
            self.sim_env.process(self._background_mec_job(service_time))

    def _background_uplink_job(self, air_time_s: float):
        with self.uplink_resource.request() as req:
            yield req
            yield self.sim_env.timeout(air_time_s)
        self.events_processed += 1

    def _uplink_background_generator(self):
        """Poisson background traffic competing on the shared wireless uplink."""
        while True:
            mean_air_time = 0.006
            arrival_rate = self.target_channel_utilization / mean_air_time
            interarrival = self.rng.expovariate(max(1e-3, arrival_rate))
            yield self.sim_env.timeout(interarrival)
            self.events_processed += 1

            air_time = min(mean_air_time * 3.5, self.rng.expovariate(1.0 / mean_air_time))
            self.sim_env.process(self._background_uplink_job(air_time))

    def execute_split_inference_step(
        self,
        split_point: int,
        total_layers: int,
        t_device_s: float,
        t_tx_data_s: float,
        t_propagation_s: float,
        t_mec_compute_s: float
    ) -> DESExecutionResult:
        """
        Runs one split inference request through the real SimPy discrete-event timeline.
        """
        if not HAS_SIMPY or self.sim_env is None:
            # Fallback analytical timeline if simpy is absent
            t_tx_total = t_tx_data_s + t_propagation_s if split_point < total_layers else 0.0
            t_total = t_device_s + t_tx_total + t_mec_compute_s
            return DESExecutionResult(
                t_device_s=t_device_s,
                t_tx_queue_s=0.0,
                t_tx_data_s=t_tx_data_s,
                t_propagation_s=t_propagation_s,
                t_transmission_s=t_tx_total,
                t_mec_queue_s=0.0,
                t_mec_compute_s=t_mec_compute_s,
                t_queue_total_s=0.0,
                t_total_s=t_total,
                mec_queue_length_at_arrival=0,
                uplink_queue_length_at_arrival=0,
                sim_time_start_s=0.0,
                sim_time_end_s=t_total,
                events_processed=0,
                backend="analytical_fallback"
            )

        t_start = float(self.sim_env.now)
        metrics_out: Dict[str, Any] = {
            "t_tx_queue": 0.0,
            "t_mec_queue": 0.0,
            "uplink_q_len": 0,
            "mec_q_len": 0
        }

        def inference_process():
            # Phase 1: Mobile Device local execution (layers 0..k-1)
            if t_device_s > 0.0:
                yield self.sim_env.timeout(t_device_s)
                self.events_processed += 1

            # If local inference (k == L), no wireless or MEC steps occur
            if split_point >= total_layers:
                return

            # Phase 2: Shared Wireless Uplink Transmission
            if t_tx_data_s > 0.0 or t_propagation_s > 0.0:
                metrics_out["uplink_q_len"] = len(self.uplink_resource.queue)
                tx_req_time = self.sim_env.now
                with self.uplink_resource.request() as req:
                    yield req
                    metrics_out["t_tx_queue"] = float(self.sim_env.now - tx_req_time)
                    if t_tx_data_s > 0.0:
                        yield self.sim_env.timeout(t_tx_data_s)
                if t_propagation_s > 0.0:
                    yield self.sim_env.timeout(t_propagation_s)
                self.events_processed += 2

            # Phase 3: MEC Server Resource Contention & Execution (layers k..L-1)
            metrics_out["mec_q_len"] = len(self.mec_resource.queue)
            mec_req_time = self.sim_env.now
            with self.mec_resource.request() as req:
                yield req
                metrics_out["t_mec_queue"] = float(self.sim_env.now - mec_req_time)
                if t_mec_compute_s > 0.0:
                    yield self.sim_env.timeout(t_mec_compute_s)
            self.events_processed += 2

        proc = self.sim_env.process(inference_process())
        self.sim_env.run(until=proc)
        t_end = float(self.sim_env.now)
        self.tasks_completed += 1

        t_tx_queue = metrics_out["t_tx_queue"]
        t_mec_queue = metrics_out["t_mec_queue"]
        t_queue_total = t_tx_queue + t_mec_queue

        if split_point >= total_layers:
            t_transmission = 0.0
            t_mec_compute = 0.0
            t_queue_total = 0.0
            t_total = t_device_s
        else:
            t_transmission = t_tx_data_s + t_propagation_s
            t_mec_compute = t_mec_compute_s
            t_total = t_end - t_start

        return DESExecutionResult(
            t_device_s=t_device_s,
            t_tx_queue_s=t_tx_queue,
            t_tx_data_s=t_tx_data_s,
            t_propagation_s=t_propagation_s,
            t_transmission_s=t_transmission,
            t_mec_queue_s=t_mec_queue,
            t_mec_compute_s=t_mec_compute,
            t_queue_total_s=t_queue_total,
            t_total_s=t_total,
            mec_queue_length_at_arrival=metrics_out["mec_q_len"],
            uplink_queue_length_at_arrival=metrics_out["uplink_q_len"],
            sim_time_start_s=t_start,
            sim_time_end_s=t_end,
            events_processed=self.events_processed,
            backend="simpy_des"
        )

    def get_current_mec_queue_length(self) -> int:
        if HAS_SIMPY and self.mec_resource is not None:
            return len(self.mec_resource.queue)
        return 0

    def get_current_uplink_queue_length(self) -> int:
        if HAS_SIMPY and self.uplink_resource is not None:
            return len(self.uplink_resource.queue)
        return 0
