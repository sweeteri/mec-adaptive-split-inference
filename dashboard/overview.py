"""
Page 1: System Overview and Architecture Dashboard.
"""

from typing import Dict, Any


def render_overview_page(st: Any, env: Any, storage: Any) -> None:
    """Renders the architecture overview and real-time state metrics."""
    st.title("🔬 MEC Adaptive Split Inference of Transformers")
    st.caption("Research Demonstration System using Reinforcement Learning in Mobile Edge Computing")

    st.markdown("""
    This research platform investigates **dynamic partitioning (split inference)** of Transformer-based NLP models 
    between battery-constrained mobile edge devices and remote MEC edge servers.
    """)

    # 3 High-Level System Architecture Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📱 Mobile Edge Device")
        freq = env.device.current_frequency_ghz
        p_comp = env.device.get_power_consumption_watts(freq)
        st.metric(label="CPU Frequency", value=f"{freq:.1f} GHz")
        st.metric(label="Dynamic Power (DVFS)", value=f"{p_comp:.2f} W")
        st.markdown(f"""
        - **CPU Model**: CMOS DVFS ($P = P_{{idle}} + \\kappa f^3$)
        - **Available RAM**: {env.device.config.available_memory_mb:.0f} MB
        - **Transmission RF Power**: {env.device.config.p_tx_watts:.2f} W
        - **Supported Frequencies**: {env.device.config.frequency_levels_ghz} GHz
        """)

    with col2:
        st.subheader("📡 Wireless Channel")
        bw = env.network.current_bandwidth_mbps
        lat = env.network.current_latency_ms
        sc = env.network.config.scenario.value.capitalize()
        st.metric(label="Active Scenario", value=sc)
        st.metric(label="Current Bandwidth", value=f"{bw:.1f} Mbps")
        st.metric(label="Propagation Latency", value=f"{lat:.1f} ms")
        st.markdown("""
        - **Protocol Overhead**: 4%
        - **Variations**: Fading, congestion, mobility
        - **Channel Models**: 5 dynamic regimes
        """)

    with col3:
        st.subheader("☁️ MEC Edge Server")
        wl = env.mec.current_workload
        q = env.mec.current_queue_length
        st.metric(label="Server Workload", value=f"{wl * 100:.1f} %")
        st.metric(label="Queue Length", value=f"{q} tasks")
        st.markdown(f"""
        - **Acceleration Factor**: {env.mec.config.acceleration_factor:.1f}x vs mobile reference
        - **Layer Processing Rate**: ~{env.mec.config.base_server_layer_latency_s * 1000:.1f} ms/layer
        - **Queuing Engine**: {'SimPy DES (`simpy.Resource`)' if getattr(env, 'use_simpy_des', False) else 'Analytical M/M/1'}
        """)

    st.divider()

    # Mathematical Framework Section
    st.subheader("📐 Mathematical Formulation")
    mcol1, mcol2 = st.columns(2)

    with mcol1:
        st.markdown("##### 1. Transmitted Activation Size")
        st.latex(r"S(k, b, N) = N_{\text{batch}} \cdot N_{\text{tokens}} \cdot d \cdot \frac{b}{8} \text{ bytes}")
        st.markdown("##### 2. Total Split Inference Latency")
        st.latex(r"T_{\text{total}} = T_{\text{device}}(k, f) + T_{\text{transmission}}(k, b, B) + T_{\text{MEC}}(k) + T_{\text{queue}}")

    with mcol2:
        st.markdown("##### 3. Mobile Device Energy Consumption")
        st.latex(r"E_{\text{mobile}} = P_{\text{comp}}(f) \cdot T_{\text{device}} + P_{\text{tx}} \cdot T_{\text{tx,data}}")
        st.markdown("##### 4. Multi-Objective Cost & Reward")
        st.latex(r"J = \alpha \frac{T_{\text{total}}}{T_{\text{ref}}} + \beta \frac{E_{\text{mobile}}}{E_{\text{ref}}} + \lambda (Q_{\text{FP32}} - Q_{\text{quant}})")

    st.divider()

    # Recent Experiment History
    st.subheader("📊 Recent Experiment Runs")
    history = storage.load_all_results_as_dicts()
    if history:
        st.dataframe(history[-10:], use_container_width=True)
    else:
        st.info("No prior experiment runs recorded. Use the 'Algorithm Comparison' page to launch benchmark evaluations.")
