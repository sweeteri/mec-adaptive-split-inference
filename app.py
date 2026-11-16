"""
Main Application Entry Point for Streamlit Research Dashboard.
Run locally via: streamlit run app.py
"""

import sys
import os

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from simulation.environment import MECSimulationEnv
from experiments.result_storage import ResultStorage
from dashboard.overview import render_overview_page
from dashboard.config_page import render_config_page
from dashboard.simulation_page import render_simulation_page
from dashboard.training_page import render_training_page
from dashboard.comparison_page import render_comparison_page
from dashboard.analysis_page import render_analysis_page
from dashboard.spam_page import render_spam_page


def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed in the standard environment. Install via pip install streamlit.")
        return

    st.set_page_config(
        page_title="MEC Adaptive Split Inference",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize persistent session state
    if "env" not in st.session_state:
        st.session_state["env"] = MECSimulationEnv(seed=42)
    if "storage" not in st.session_state:
        st.session_state["storage"] = ResultStorage()

    env = st.session_state["env"]
    storage = st.session_state["storage"]

    # Sidebar Navigation
    st.sidebar.title("🔬 MEC RL Simulator")
    st.sidebar.caption("Adaptive Transformer Split Inference")
    st.sidebar.divider()

    pages = [
        "1. Overview & Architecture",
        "2. Simulation Configuration",
        "3. Live Simulation & Partitioning",
        "4. RL Agent Training",
        "5. Algorithm Comparison",
        "6. Results & Scientific Analysis",
        "7. Real Spam Detection (NLP + MEC)"
    ]

    selected_page = st.sidebar.radio("Navigation", pages)
    st.sidebar.divider()
    st.sidebar.markdown(f"""
    **Current Scenario**: `{env.network.config.scenario.value.capitalize()}`  
    **Bandwidth**: `{env.network.current_bandwidth_mbps:.1f} Mbps`  
    **Mobile CPU**: `{env.device.current_frequency_ghz:.1f} GHz`  
    **Server Load**: `{env.mec.current_workload * 100:.1f}%`
    """)

    if selected_page == "1. Overview & Architecture":
        render_overview_page(st, env, storage)
    elif selected_page == "2. Simulation Configuration":
        render_config_page(st, env)
    elif selected_page == "3. Live Simulation & Partitioning":
        render_simulation_page(st, env)
    elif selected_page == "4. RL Agent Training":
        render_training_page(st, env)
    elif selected_page == "5. Algorithm Comparison":
        render_comparison_page(st, env, storage)
    elif selected_page == "6. Results & Scientific Analysis":
        render_analysis_page(st, storage)
    elif selected_page == "7. Real Spam Detection (NLP + MEC)":
        render_spam_page(st, env)


if __name__ == "__main__":
    main()
