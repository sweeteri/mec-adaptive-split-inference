"""
Page 5: Comparative Evaluation of Baselines and RL Algorithms.
Renders latency, energy, quality, data transmission, and reward comparison charts and data tables.
"""

from typing import Dict, Any, List
import pandas as pd
from simulation.environment import NetworkScenario
from algorithms.local_inference import LocalInferencePolicy
from algorithms.full_offloading import FullOffloadingPolicy
from algorithms.fixed_split import FixedSplitPolicy
from algorithms.greedy import GreedyAdaptivePolicy
from algorithms.random_policy import RandomPolicy
from algorithms.dqn import DQNPolicy
from algorithms.ppo import PPOPolicy
from algorithms.sac import SACPolicy
from experiments.experiment_runner import ExperimentRunner


def render_comparison_page(st: Any, env: Any, storage: Any) -> None:
    """Renders multi-algorithm benchmark comparisons with charts and export."""
    st.title("⚖️ Algorithm Comparison & Benchmarks")
    st.caption("Benchmark baseline policies against Reinforcement Learning agents under identical conditions")

    # Available algorithm pool
    available_policies = {
        "Local Inference": LocalInferencePolicy(total_layers=env.model.total_layers),
        "Full Offloading": FullOffloadingPolicy(),
        "Fixed Split (k=3)": FixedSplitPolicy(split_point=3, bitwidth_idx=1),
        "Greedy Adaptive": GreedyAdaptivePolicy(total_layers=env.model.total_layers),
        "Random Policy": RandomPolicy(total_layers=env.model.total_layers),
        "PPO": PPOPolicy(total_layers=env.model.total_layers),
        "DQN": DQNPolicy(total_layers=env.model.total_layers),
        "SAC": SACPolicy(total_layers=env.model.total_layers)
    }

    # Sidebar / configuration controls
    st.subheader("1. Benchmark Setup")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_algo_names = st.multiselect(
            "Select Algorithms to Compare",
            options=list(available_policies.keys()),
            default=["Local Inference", "Full Offloading", "Fixed Split (k=3)", "Greedy Adaptive", "PPO"]
        )

    with col2:
        scenario = st.selectbox(
            "Evaluation Scenario",
            [s.value.capitalize() for s in NetworkScenario],
            index=1
        )
        env.set_scenario(scenario.lower())

    with col3:
        num_episodes = st.slider("Evaluation Episodes", 1, 10, 3)
        seed = st.number_input("Random Seed", value=42, step=1)

    btn_benchmark = st.button("🚀 Run Comparative Benchmark", type="primary", use_container_width=True)

    runner = ExperimentRunner(result_storage=storage)

    if btn_benchmark:
        if not selected_algo_names:
            st.error("Please select at least one algorithm.")
            return

        policies = [available_policies[name] for name in selected_algo_names]
        prog_bar = st.progress(0.0)
        status_txt = st.empty()

        def prog_cb(algo_name, ep, total):
            status_txt.text(f"Running {algo_name} | Episode {ep}/{num_episodes}")

        with st.spinner("Evaluating algorithms under identical conditions..."):
            res = runner.run_benchmark(
                policies=policies,
                env=env,
                num_episodes=num_episodes,
                seed=seed,
                experiment_name=f"Benchmark_{scenario}",
                progress_callback=prog_cb
            )

        st.session_state["last_benchmark"] = res
        st.success("Benchmark completed! Results recorded to CSV.")

    res = st.session_state.get("last_benchmark")
    if res:
        st.divider()
        st.subheader("2. Comparative Evaluation Metrics")

        df = pd.DataFrame(res["results"])
        st.dataframe(df, use_container_width=True)

        # Download CSV button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Benchmark Results (CSV)",
            data=csv_data,
            file_name=f"mec_split_inference_{res['scenario']}_results.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("3. Visual Comparative Charts")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Average Latency (ms) [Lower is better]")
            lat_chart_df = df.set_index("algorithm")[["avg_latency_ms", "median_latency_ms", "p95_latency_ms"]]
            st.bar_chart(lat_chart_df)

        with c2:
            st.markdown("##### Energy Consumption (Joules) [Lower is better]")
            e_chart_df = df.set_index("algorithm")["avg_energy_joules"]
            st.bar_chart(e_chart_df)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("##### Transmitted Payload (KB) [Lower is better]")
            tx_chart_df = df.set_index("algorithm")["total_transmitted_kb"]
            st.bar_chart(tx_chart_df)

        with c4:
            st.markdown("##### Average Reward [Higher is better]")
            r_chart_df = df.set_index("algorithm")["avg_reward"]
            st.bar_chart(r_chart_df)
