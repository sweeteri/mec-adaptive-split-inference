"""
Page 2: Simulation System Configuration.
Allows researchers to configure hardware, model, network, and optimization parameters.
"""

from typing import Any
from simulation.environment import (
    ModelProfile,
    MobileDeviceConfig,
    MECServerConfig,
    NetworkChannelConfig,
    RewardWeights,
    NetworkScenario
)


def render_config_page(st: Any, env: Any) -> None:
    """Renders comprehensive parameter tuning controls for the MEC simulation platform."""
    st.title("⚙️ Simulation Configuration")
    st.caption("Fine-tune model architecture, device SoC power models, MEC server specs, and reward weights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Transformer Model Specifications")
        model_name = st.selectbox("Base Architecture", ["DistilBERT-base", "BERT-base", "Custom Transformer"])
        total_layers = st.slider("Total Encoder Layers (L)", min_value=2, max_value=12, value=env.model.total_layers)
        hidden_dim = st.selectbox("Hidden State Dimension (d)", [256, 384, 768, 1024], index=2)
        seq_len = st.slider("Reference Sequence Length (Tokens)", min_value=32, max_value=512, value=env.model.reference_seq_len, step=32)

        st.divider()

        st.subheader("2. Mobile Device SoC & Power Model")
        p_idle = st.number_input("Idle Standby Power (Watts)", value=env.device.config.p_idle_watts, step=0.05)
        kappa = st.number_input("CMOS DVFS Dynamic Coefficient kappa (W / GHz^3)", value=env.device.config.kappa_dvfs, step=0.02)
        p_tx = st.number_input("Wireless RF Transmission Power (Watts)", value=env.device.config.p_tx_watts, step=0.1)
        freq_list = st.multiselect(
            "Available CPU Frequency Levels (GHz)",
            [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5],
            default=[1.0, 1.4, 1.8, 2.2, 2.5]
        )

    with col2:
        st.subheader("3. MEC Server Specifications & DES Backend")
        use_simpy = st.checkbox(
            "Use SimPy Discrete-Event Simulation (DES) Engine",
            value=getattr(env, "use_simpy_des", True),
            help="Simulates real resource contention & Poisson background arrivals via simpy.Resource"
        )
        accel = st.slider("MEC Acceleration Factor (vs Reference Mobile CPU)", min_value=2.0, max_value=20.0, value=env.mec.config.acceleration_factor, step=0.5)
        service_rate = st.number_input("Mean Task Service Time (seconds)", value=env.mec.config.service_rate_per_task_s, step=0.005, format="%.4f")
        base_workload = st.slider("Default Background Server Workload", min_value=0.0, max_value=1.0, value=env.mec.config.default_workload, step=0.05)

        st.divider()

        st.subheader("4. Multi-Objective Cost & SLA Parameters")
        st.markdown(r"Optimization Cost: $J = \alpha \frac{T}{T_{\text{ref}}} + \beta \frac{E}{E_{\text{ref}}} + \lambda D$")
        alpha = st.slider("Latency Weight (alpha)", 0.0, 1.0, env.reward_weights.alpha, step=0.05)
        beta = st.slider("Energy Weight (beta)", 0.0, 1.0, env.reward_weights.beta, step=0.05)
        lambd = st.slider("Quality Degradation Weight (lambda)", 0.0, 1.0, env.reward_weights.lambd, step=0.05)
        
        min_quality = st.slider("Minimum Acceptable Quality Q_min", 0.70, 0.99, env.reward_weights.min_acceptable_quality, step=0.01)
        sla_ms = st.number_input("Target Latency SLA Deadline (ms)", value=100.0, step=10.0)

    btn_apply = st.button("💾 Apply Configuration to Environment", type="primary", use_container_width=True)

    if btn_apply:
        env.model = ModelProfile(
            model_name=model_name,
            total_layers=total_layers,
            hidden_dim=hidden_dim,
            reference_seq_len=seq_len
        )
        env.device.config.p_idle_watts = p_idle
        env.device.config.kappa_dvfs = kappa
        env.device.config.p_tx_watts = p_tx
        if freq_list:
            env.device.config.frequency_levels_ghz = sorted(freq_list)

        env.mec.config.acceleration_factor = accel
        env.mec.config.service_rate_per_task_s = service_rate
        env.mec.config.default_workload = base_workload

        if hasattr(env, "use_simpy_des"):
            from simulation.simpy_des import HAS_SIMPY
            env.use_simpy_des = bool(use_simpy and HAS_SIMPY)
            if hasattr(env, "des_engine"):
                env.des_engine.service_rate_s = service_rate

        env.reward_weights.alpha = alpha
        env.reward_weights.beta = beta
        env.reward_weights.lambd = lambd
        env.reward_weights.min_acceptable_quality = min_quality

        env.fixed_seq_len = seq_len
        env.reset()
        st.success("New configuration applied successfully to the active simulation environment!")
