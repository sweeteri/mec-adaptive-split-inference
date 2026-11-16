"""
Page 4: Reinforcement Learning Training and Model Management.
"""

from typing import Dict, Any
from algorithms.dqn import DQNPolicy
from algorithms.ppo import PPOPolicy
from algorithms.sac import SACPolicy
from training.train import train_rl_agent
from training.model_manager import ModelManager


def render_training_page(st: Any, env: Any) -> None:
    """Renders RL training controls, loss curves, and model persistence."""
    st.title("🧠 Reinforcement Learning Agent Training")
    st.caption("Train and evaluate DQN, PPO, or SAC policies for dynamic split inference")

    manager = ModelManager()

    col_setup, col_metrics = st.columns([1, 2])

    with col_setup:
        st.subheader("🛠️ Training Setup")
        algo_choice = st.selectbox("Select RL Algorithm", ["PPO", "DQN", "SAC"])
        num_episodes = st.slider("Number of Episodes", 10, 200, 40, step=10)
        lr = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=3e-4)

        if "rl_agents" not in st.session_state:
            st.session_state["rl_agents"] = {
                "PPO": PPOPolicy(lr=lr),
                "DQN": DQNPolicy(lr=lr),
                "SAC": SACPolicy(lr=lr)
            }

        agent = st.session_state["rl_agents"][algo_choice]
        status = manager.check_status(agent)

        st.info(f"**Agent Status**: {'✅ Trained' if agent.is_trained else '⏳ Not trained'}")
        if status["checkpoint_available"]:
            st.caption(f"Checkpoint found: `{status['checkpoint_path']}`")

        c1, c2 = st.columns(2)
        btn_train = c1.button("🚀 Train Agent", type="primary", use_container_width=True)
        btn_load = c2.button("📥 Load Model", use_container_width=True)

        if btn_load:
            success = manager.load_model(agent)
            if success:
                st.success("Checkpoint successfully loaded!")
            else:
                st.warning("No saved checkpoint available.")

    with col_metrics:
        st.subheader("📈 Training Performance")

        if btn_train:
            prog_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(ep, r, loss, pct):
                prog_bar.progress(pct)
                status_text.text(f"Episode {ep}/{num_episodes} | Reward: {r:.3f} | Loss: {loss:.4f}")

            save_path = manager.get_checkpoint_path(agent.name)
            res = train_rl_agent(
                agent=agent,
                env=env,
                num_episodes=num_episodes,
                progress_callback=update_progress,
                save_path=save_path
            )
            st.success(f"Training completed in {res['training_duration_seconds']:.1f} s! Model saved.")

        if agent.episode_rewards:
            st.markdown("##### Episode Rewards History")
            st.line_chart(agent.episode_rewards)

            if agent.loss_history:
                st.markdown("##### Loss History")
                st.line_chart(agent.loss_history)
        else:
            st.write("Click **Train Agent** to start training or evaluate with current weights.")
