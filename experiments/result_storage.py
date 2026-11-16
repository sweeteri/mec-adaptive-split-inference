"""
Result Storage and CSV Export Module for Comparative Research Experiments.
"""

import os
import csv
import json
from typing import List, Dict, Any, Optional
import datetime

from simulation.metrics import SimulationEpisodeMetrics


class ResultStorage:
    """Manages writing and reading experiment evaluation metrics to CSV and JSON."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.csv_path = os.path.join(self.results_dir, "experiment_results.csv")

    def save_episode_metrics_to_csv(
        self,
        metrics_list: List[SimulationEpisodeMetrics],
        experiment_name: str = "Experiment"
    ) -> str:
        """Appends episode metrics to the master CSV file."""
        file_exists = os.path.exists(self.csv_path)
        timestamp = datetime.datetime.now().isoformat()

        fieldnames = [
            "timestamp",
            "experiment_name",
            "episode_id",
            "algorithm_name",
            "scenario_name",
            "total_steps",
            "avg_latency_ms",
            "median_latency_ms",
            "p95_latency_ms",
            "avg_energy_joules",
            "total_energy_joules",
            "total_transmitted_kb",
            "avg_bandwidth_mbps",
            "avg_quality",
            "avg_quality_degradation",
            "quality_constraint_success_rate",
            "sla_success_rate",
            "avg_reward",
            "total_reward"
        ]

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for m in metrics_list:
                row = m.to_dict()
                row["timestamp"] = timestamp
                row["experiment_name"] = experiment_name
                writer.writerow(row)

        return self.csv_path

    def load_all_results_as_dicts(self) -> List[Dict[str, Any]]:
        """Reads all historical results from CSV."""
        if not os.path.exists(self.csv_path):
            return []

        rows: List[Dict[str, Any]] = []
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Convert numeric fields
                for k in [
                    "avg_latency_ms", "median_latency_ms", "p95_latency_ms",
                    "avg_energy_joules", "total_energy_joules", "total_transmitted_kb",
                    "avg_bandwidth_mbps", "avg_quality", "avg_quality_degradation",
                    "quality_constraint_success_rate", "sla_success_rate",
                    "avg_reward", "total_reward"
                ]:
                    if k in r and r[k] != "":
                        r[k] = float(r[k])
                for k in ["episode_id", "total_steps"]:
                    if k in r and r[k] != "":
                        r[k] = int(r[k])
                rows.append(r)
        return rows

    def load_results(self) -> Any:
        """Loads results into a Pandas DataFrame if available, otherwise returns list of dicts."""
        rows = self.load_all_results_as_dicts()
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows
