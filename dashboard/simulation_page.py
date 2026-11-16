"""
Page 2 & 3: Interactive Simulation Configuration & Live Execution Visualizer.
"""

from typing import Dict, Any, List
from simulation.environment import VALID_BITWIDTHS, BITWIDTH_TO_INDEX, INDEX_TO_BITWIDTH, NetworkScenario


def render_simulation_page(st: Any, env: Any) -> None:
    """Renders interactive simulation controls and live layer partition diagram."""
    st.title("⚡ Live Split Inference Simulation")
    st.caption("Interact with model partitioning, activation quantization, and wireless dynamics")

    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.subheader("⚙️ Control Panel")
        
        scenario_choice = st.selectbox(
            "Network Scenario",
            [s.value.capitalize() for s in NetworkScenario],
            index=1
        )
        env.set_scenario(scenario_choice.lower())

        total_layers = env.model.total_layers
        split_point = st.slider(
            "Split Point k (Mobile Layers)",
            min_value=0,
            max_value=total_layers,
            value=st.session_state.get("sim_k", 2),
            help="k=0: Full Offloading, k=L: Full Local, 0<k<L: Split Inference"
        )
        st.session_state["sim_k"] = split_point

        bitwidth = st.select_slider(
            "Quantization Bitwidth b",
            options=[4, 8, 16, 32],
            value=st.session_state.get("sim_b", 8),
            help="4-bit (INT4), 8-bit (INT8), 16-bit (INT16), 32-bit (FP32)"
        )
        st.session_state["sim_b"] = bitwidth
        b_idx = BITWIDTH_TO_INDEX[bitwidth]

        freq_options = env.device.config.frequency_levels_ghz
        freq_ghz = st.select_slider(
            "Mobile CPU Frequency f (GHz)",
            options=freq_options,
            value=freq_options[2]
        )
        f_idx = freq_options.index(freq_ghz)

        seq_len = st.slider("Input Sequence Length (Tokens)", 32, 256, 128, step=32)
        env.fixed_seq_len = seq_len

        btn_step = st.button("▶️ Execute Single Inference Step", type="primary", use_container_width=True)

    # Execute step on button press or maintain last metrics
    if btn_step:
        obs, reward, term, trunc, info = env.step([split_point, b_idx, f_idx])
        st.session_state["last_step_info"] = info
    
    info = st.session_state.get("last_step_info")
    if not info:
        # Initial step
        obs, info_init = env.reset()
        obs, reward, term, trunc, info = env.step([split_point, b_idx, f_idx])
        st.session_state["last_step_info"] = info

    m = info["metrics"]

    with col_viz:
        st.subheader("🧩 Transformer Architecture Partitioning")
        
        # Visual representation of layers distribution
        st.markdown(f"**Total Layers:** {total_layers} | **Device:** {split_point} layers | **MEC Server:** {total_layers - split_point} layers")
        
        layer_cols = st.columns(total_layers)
        for i in range(total_layers):
            with layer_cols[i]:
                if i < split_point:
                    st.success(f"L{i+1}\n📱")
                else:
                    st.info(f"L{i+1}\n☁️")

        if 0 < split_point < total_layers:
            st.warning(
                f"📡 **Network Offloading at Layer {split_point}**: "
                f"Transmitting {m['transmitted_bytes']:,} bytes ({m['transmitted_bytes']/1024:.1f} KB) "
                f"using **INT{bitwidth}** quantization."
            )
        elif split_point == 0:
            st.info(f"📡 **Full Offloading (k=0)**: Transmitting raw input prompt ({m['transmitted_bytes']} bytes).")
        else:
            st.success("📱 **Local Inference (k=L)**: Zero data transmitted across network.")

        st.divider()

        # Key Output Metrics Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Latency", f"{m['t_total']*1000:.1f} ms", delta=f"{m['sla_target_ms'] - m['t_total']*1000:.1f} ms to SLA")
        mcol2.metric("Energy", f"{m['e_mobile']*1000:.1f} mJ", delta=f"{m['e_computation']*1000:.1f} mJ comp")
        mcol3.metric("Data Size", f"{m['transmitted_bytes']/1024:.1f} KB", delta=f"b={bitwidth} bits")
        mcol4.metric("Quality", f"{m['quality_score']*100:.1f} %", delta=f"-{m['quality_degradation']*100:.2f}% deg")

        # Detailed Latency Breakdown
        st.markdown("##### ⏱️ Latency Breakdown")
        lat_cols = st.columns(4)
        lat_cols[0].write(f"**Device Comp:** {m['t_device']*1000:.2f} ms")
        lat_cols[1].write(f"**Network Tx:** {m['t_transmission']*1000:.2f} ms")
        lat_cols[2].write(f"**Server Comp:** {m['t_mec']*1000:.2f} ms")
        lat_cols[3].write(f"**Server Queue:** {m['t_queue']*1000:.2f} ms")

        # Channel State
        st.markdown(f"**Current Channel:** Bandwidth = **{m['bandwidth_mbps']:.1f} Mbps** | Latency = **{m['channel_latency_ms']:.1f} ms** | Server Load = **{m['mec_workload']*100:.1f}%**")
