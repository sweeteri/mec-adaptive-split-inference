"""
Page 7: Real-World Spam Detection (NLP + Adaptive MEC Split Inference).
Combines real SMS/text classification (PyTorch ModularTransformer or HuggingFace DistilBERT)
with real SimPy DES MEC simulation and activation quantization.
"""

from typing import Any, Dict, List
import pandas as pd

from simulation.environment import (
    NetworkScenario,
    BITWIDTH_TO_INDEX,
    INDEX_TO_BITWIDTH,
)
from algorithms.local_inference import LocalInferencePolicy
from algorithms.full_offloading import FullOffloadingPolicy
from algorithms.fixed_split import FixedSplitPolicy
from algorithms.greedy import GreedyAdaptivePolicy
from algorithms.dqn import DQNPolicy
from algorithms.ppo import PPOPolicy
from algorithms.sac import SACPolicy
from training.model_manager import ModelManager
from models.spam_classifier import (
    HAS_TRANSFORMERS,
    SpamDatasetLoader,
    ModularSpamClassifier,
    HuggingFaceSplitSpamClassifier,
    evaluate_spam_dataset_with_policy,
)


def _get_or_init_policies(env: Any) -> Dict[str, Any]:
    """Builds policy dictionary, reusing trained RL agents from session_state or disk checkpoints."""
    total_layers = env.model.total_layers
    manager = ModelManager()

    rl_agents = None
    try:
        import streamlit as st_mod
        rl_agents = st_mod.session_state.get("rl_agents")
    except Exception:
        rl_agents = None

    if rl_agents:
        ppo_pol = rl_agents["PPO"]
        dqn_pol = rl_agents["DQN"]
        sac_pol = rl_agents["SAC"]
    else:
        ppo_pol = PPOPolicy(total_layers=total_layers)
        dqn_pol = DQNPolicy(total_layers=total_layers)
        sac_pol = SACPolicy(total_layers=total_layers)
        manager.load_model(ppo_pol)
        manager.load_model(dqn_pol)
        manager.load_model(sac_pol)

    return {
        "Local Inference": LocalInferencePolicy(total_layers=total_layers),
        "Full Offloading": FullOffloadingPolicy(),
        "Fixed Split (k=2, INT8)": FixedSplitPolicy(split_point=min(2, total_layers), bitwidth_idx=BITWIDTH_TO_INDEX[8]),
        "Fixed Split (k=2, INT4)": FixedSplitPolicy(split_point=min(2, total_layers), bitwidth_idx=BITWIDTH_TO_INDEX[4]),
        "Fixed Split (k=2, FP32)": FixedSplitPolicy(split_point=min(2, total_layers), bitwidth_idx=BITWIDTH_TO_INDEX[32]),
        "Greedy Adaptive": GreedyAdaptivePolicy(total_layers=total_layers),
        "PPO": ppo_pol,
        "DQN": dqn_pol,
        "SAC": sac_pol,
    }


def render_spam_page(st: Any, env: Any) -> None:
    """Renders real-world Spam Detection + Adaptive Split Inference workflow."""
    st.title("📩 Real-World Spam Detection (NLP + MEC Split Inference)")
    st.caption(
        "Test real SMS/text Spam Classification end-to-end: Tokenization → RL/Adaptive Split Decision (k, b, f) → "
        "PyTorch Activation Quantization → SimPy DES Network/Server Queueing → Accuracy, F1, Latency & Energy"
    )

    # ------------------------------------------------------------------
    # 1. Dataset & Model Configuration
    # ------------------------------------------------------------------
    st.subheader("1. Dataset & Transformer Classifier Setup")
    col_ds, col_mdl = st.columns(2)

    with col_ds:
        st.markdown("##### 📂 Dataset Source")
        ds_source = st.radio(
            "Choose Dataset",
            [
                "Built-in EN + RU SMS Benchmark (160 messages)",
                "Download UCI SMS Spam Collection (5,574 real SMS)",
                "Upload Custom CSV / TSV File",
            ],
            index=0,
        )

        if "spam_dataset" not in st.session_state:
            tr_x, tr_y, te_x, te_y = SpamDatasetLoader.load_builtin_dataset(test_ratio=0.25, seed=42)
            st.session_state["spam_dataset"] = (tr_x, tr_y, te_x, te_y)
            st.session_state["spam_ds_name"] = "Built-in EN+RU Benchmark"

        if ds_source.startswith("Built-in"):
            if st.session_state.get("spam_ds_name") != "Built-in EN+RU Benchmark":
                if st.button("🔄 Load Built-in Benchmark", use_container_width=True):
                    st.session_state["spam_dataset"] = SpamDatasetLoader.load_builtin_dataset(test_ratio=0.25, seed=42)
                    st.session_state["spam_ds_name"] = "Built-in EN+RU Benchmark"
                    st.rerun()

        elif ds_source.startswith("Download UCI"):
            if st.button("🌐 Download & Load UCI SMSSpamCollection", type="primary", use_container_width=True):
                try:
                    with st.spinner("Downloading UCI SMS Spam Collection..."):
                        tr_x, tr_y, te_x, te_y = SpamDatasetLoader.download_uci_sms_spam(test_ratio=0.20, seed=42)
                    st.session_state["spam_dataset"] = (tr_x, tr_y, te_x, te_y)
                    st.session_state["spam_ds_name"] = "UCI SMS Spam Collection"
                    st.success(f"Loaded {len(tr_x) + len(te_x)} real SMS messages!")
                except Exception as exc:
                    st.error(f"Download failed (offline or blocked): {exc}")

        else:
            uploaded = st.file_uploader(
                "Upload CSV (columns: label, text or v1, v2) or UCI TSV",
                type=["csv", "tsv", "txt"],
            )
            if uploaded is not None:
                try:
                    tr_x, tr_y, te_x, te_y = SpamDatasetLoader.parse_csv_or_tsv_bytes(uploaded.read(), test_ratio=0.20)
                    st.session_state["spam_dataset"] = (tr_x, tr_y, te_x, te_y)
                    st.session_state["spam_ds_name"] = f"Custom File ({uploaded.name})"
                    st.success(f"Parsed {len(tr_x) + len(te_x)} labeled messages!")
                except Exception as exc:
                    st.error(f"Could not parse file: {exc}")

        tr_x, tr_y, te_x, te_y = st.session_state["spam_dataset"]
        st.info(
            f"**Active Dataset:** `{st.session_state.get('spam_ds_name')}` | "
            f"**Train:** {len(tr_x)} | **Test:** {len(te_x)} "
            f"(Spam ratio: {sum(te_y) / max(1, len(te_y)) * 100:.1f}%)"
        )

    with col_mdl:
        st.markdown("##### 🤖 Split-Capable NLP Model")
        backend_options = ["ModularTransformer (Offline PyTorch, 6 Layers)"]
        if HAS_TRANSFORMERS:
            backend_options.append("HuggingFace DistilBERT (mrm8488/distilbert-finetuned-sms-spam-detection)")
        else:
            st.caption("ℹ️ Install `transformers` (`pip install transformers`) to also unlock pretrained HuggingFace DistilBERT.")

        backend_choice = st.selectbox("Classifier Architecture", backend_options)

        if backend_choice.startswith("ModularTransformer"):
            if "spam_classifier" not in st.session_state or not isinstance(
                st.session_state["spam_classifier"], ModularSpamClassifier
            ):
                clf = ModularSpamClassifier(
                    num_layers=env.model.total_layers,
                    hidden_dim=128,
                    num_heads=4,
                    max_seq_len=96,
                )
                clf.load()  # Load saved checkpoint automatically if present
                st.session_state["spam_classifier"] = clf

            clf = st.session_state["spam_classifier"]
            epochs = st.slider("Training Epochs", min_value=4, max_value=30, value=14, step=2)

            b1, b2 = st.columns(2)
            btn_train = b1.button("🚀 Train / Retrain Classifier", type="primary", use_container_width=True)
            btn_load = b2.button("📥 Load Saved Checkpoint", use_container_width=True)

            if btn_load:
                if clf.load():
                    st.success("Loaded checkpoint from `saved_models/spam_transformer/`.")
                else:
                    st.warning("No checkpoint found yet. Click Train Classifier first.")

            if btn_train:
                prog = st.progress(0.0)
                status = st.empty()

                def on_epoch(ep: int, tot: int, loss: float, acc: float) -> None:
                    prog.progress(ep / tot)
                    status.text(f"Epoch {ep}/{tot} | Loss: {loss:.4f} | Train Acc: {acc * 100:.1f}%")

                clf.train_on_dataset(tr_x, tr_y, epochs=epochs, batch_size=16, lr=8e-4, progress_callback=on_epoch)
                saved_dir = clf.save()
                st.success(f"Model trained ({clf.training_history[-1]['train_acc'] * 100:.1f}% train acc) & saved to `{saved_dir}`!")

            if clf.is_trained:
                st.success("✅ Classifier Status: Trained & Ready for Split Inference")
                if clf.training_history:
                    hist_df = pd.DataFrame(clf.training_history).set_index("epoch")
                    with st.expander("📉 Training Loss & Accuracy Curve"):
                        st.line_chart(hist_df[["loss", "train_acc"]])
            else:
                st.warning("⏳ Classifier has random weights. Click **Train / Retrain Classifier** before evaluating.")

        else:
            if st.button("📥 Load Pretrained HuggingFace DistilBERT", type="primary", use_container_width=True):
                with st.spinner("Loading mrm8488/distilbert-finetuned-sms-spam-detection..."):
                    st.session_state["spam_classifier"] = HuggingFaceSplitSpamClassifier()
                st.success("Loaded Pretrained DistilBERT Spam Classifier!")
            clf = st.session_state.get("spam_classifier")

    if clf is None:
        return

    # Ensure environment layer count aligns with classifier layers
    max_layers = clf.num_layers
    policies = _get_or_init_policies(env)

    st.divider()

    # ------------------------------------------------------------------
    # 2. Live Single Message Split Inference Inspector
    # ------------------------------------------------------------------
    st.subheader("2. Live Real Message Inspector (Try Any SMS / Text)")

    sample_presets = {
        "Custom text below...": "",
        "EN Spam (Phishing Bank)": "URGENT! Your bank account suspended. Verify card now: http://secure-verify.info",
        "EN Spam (Prize Win)": "CONGRATULATIONS! You won $1000 cash prize! Claim at http://bit.ly/claim-win today!",
        "EN Ham (Normal Chat)": "Hey, are we still meeting for coffee at 3pm near the library?",
        "RU Spam (Выигрыш)": "СРОЧНО! Вы выиграли 500 000 рублей! Заберите приз по ссылке http://priz-bonus.ru",
        "RU Ham (Учёба)": "Привет! Отправь, пожалуйста, презентацию к диплому перед созвоном в 17:00.",
    }

    c_pre, c_net, c_pol = st.columns(3)
    preset_choice = c_pre.selectbox("Quick Sample Preset", list(sample_presets.keys()), index=1)
    live_scenario = c_net.selectbox("Network Scenario", [s.value.capitalize() for s in NetworkScenario], index=1, key="spam_live_scen")
    policy_mode = c_pol.selectbox(
        "Partitioning Mode",
        ["Manual (k, b, f)"] + list(policies.keys()),
        index=0,
    )

    default_msg = sample_presets[preset_choice] if sample_presets[preset_choice] else st.session_state.get(
        "spam_custom_msg", "URGENT! You won a £900 cash reward! Click http://promo-gift.xyz"
    )
    user_msg = st.text_area("Message Text to Classify", value=default_msg, height=80)
    st.session_state["spam_custom_msg"] = user_msg

    if policy_mode == "Manual (k, b, f)":
        mc1, mc2, mc3 = st.columns(3)
        manual_k = mc1.slider("Split Layer k (Device Layers)", 0, max_layers, min(2, max_layers))
        manual_b = mc2.select_slider("Activation Bitwidth b", options=[4, 8, 16, 32], value=8)
        freq_opts = env.device.config.frequency_levels_ghz
        manual_f = mc3.select_slider("Mobile CPU Freq f (GHz)", options=freq_opts, value=freq_opts[min(2, len(freq_opts) - 1)])

    if st.button("⚡ Classify Message & Simulate MEC Split", type="primary", use_container_width=True):
        if not user_msg.strip():
            st.warning("Please enter a non-empty message.")
        else:
            env.set_scenario(live_scenario.lower())
            obs, info = env.reset(seed=42)

            # Set real sequence length from tokenized message
            tok_ids, _ = clf.tokenizer.encode(user_msg) if hasattr(clf.tokenizer, "encode") else ([], None)
            seq_len = max(8, int(tok_ids.shape[1]) if hasattr(tok_ids, "shape") else len(user_msg.split()))
            if env.current_task is not None:
                env.current_task.sequence_length = seq_len
                obs = env._get_obs()
                info["seq_len"] = seq_len

            if policy_mode == "Manual (k, b, f)":
                b_idx = BITWIDTH_TO_INDEX[manual_b]
                f_idx = env.device.config.frequency_levels_ghz.index(manual_f)
                action = [manual_k, b_idx, f_idx]
            else:
                sel_pol = policies[policy_mode]
                sel_pol.reset()
                action = sel_pol.select_action(obs, info)

            chosen_k = max(0, min(max_layers, int(action[0])))
            chosen_b = INDEX_TO_BITWIDTH[max(0, min(3, int(action[1])))]
            chosen_f = env.device.config.frequency_levels_ghz[int(action[2])]

            # 1. Real PyTorch Split Inference
            nlp_res = clf.predict_split(user_msg, split_point=chosen_k, bitwidth=chosen_b)
            # 2. Real SimPy DES Environment step
            _, reward, _, _, step_info = env.step([chosen_k, BITWIDTH_TO_INDEX[chosen_b], int(action[2])])
            mec_m = step_info["metrics"]

            if nlp_res.predicted_class == 1:
                st.error(f"🚨 **Verdict: SPAM** (Confidence: {nlp_res.spam_probability * 100:.1f}%)")
            else:
                st.success(f"✅ **Verdict: HAM (Legitimate)** (Confidence: {nlp_res.ham_probability * 100:.1f}%)")

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Action (k, b, f)", f"k={chosen_k}, INT{chosen_b}" if chosen_b < 32 else f"k={chosen_k}, FP32", delta=f"{chosen_f:.1f} GHz CPU")
            r2.metric("SimPy E2E Latency", f"{mec_m['t_total'] * 1000:.1f} ms", delta=f"Queue {mec_m['t_queue'] * 1000:.1f} ms")
            r3.metric("Mobile Energy", f"{mec_m['e_mobile'] * 1000:.2f} mJ", delta=f"SLA {'OK' if mec_m['sla_satisfied'] else 'MISS'}")
            r4.metric("Network Payload", f"{nlp_res.transmitted_bytes / 1024:.2f} KB", delta=f"{nlp_res.compression_ratio:.1f}x comp")
            r5.metric("Quant Quality", f"{nlp_res.cosine_similarity:.4f} cos", delta="Matches FP32" if nlp_res.top1_match_vs_fp32 else "Flipped!")

            st.caption(
                f"Tokens: **{nlp_res.num_tokens}** | PyTorch CPU time: Device={nlp_res.t_device_measured_ms:.2f}ms, "
                f"MEC={nlp_res.t_mec_measured_ms:.2f}ms | SimPy Channel: {mec_m['bandwidth_mbps']:.1f} Mbps, "
                f"Server Load: {mec_m['mec_workload'] * 100:.1f}%"
            )

    st.divider()

    # ------------------------------------------------------------------
    # 3. Full Test Set Benchmark: Accuracy vs Latency vs Quantization
    # ------------------------------------------------------------------
    st.subheader("3. Real Dataset Evaluation: Accuracy (F1) vs MEC Latency & Energy")
    st.caption(
        "Evaluates the held-out test set message by message through both the PyTorch Split Classifier and the SimPy DES MEC Simulator."
    )

    ev1, ev2, ev3 = st.columns(3)
    eval_scenario = ev1.selectbox("Benchmark Scenario", [s.value.capitalize() for s in NetworkScenario], index=2, key="spam_bench_scen")
    max_eval_msgs = ev2.slider("Max Test Messages", 10, min(500, max(10, len(te_x))), min(100, len(te_x)), step=10)
    selected_bench_policies = ev3.multiselect(
        "Policies to Compare",
        options=list(policies.keys()),
        default=[
            "Local Inference",
            "Full Offloading",
            "Fixed Split (k=2, FP32)",
            "Fixed Split (k=2, INT8)",
            "Fixed Split (k=2, INT4)",
            "Greedy Adaptive",
            "PPO",
        ],
    )

    if st.button("📊 Run Real Spam + MEC Benchmark", type="primary", use_container_width=True):
        if not selected_bench_policies:
            st.warning("Select at least one policy.")
        else:
            eval_texts = te_x[:max_eval_msgs]
            eval_labels = te_y[:max_eval_msgs]
            rows: List[Dict[str, Any]] = []
            logs_by_algo: Dict[str, Any] = {}

            prog = st.progress(0.0)
            status = st.empty()
            env.set_scenario(eval_scenario.lower())

            for idx_p, pname in enumerate(selected_bench_policies):
                status.text(f"Evaluating {pname} on {len(eval_texts)} real test messages...")
                pol = policies[pname]
                res = evaluate_spam_dataset_with_policy(
                    classifier=clf,
                    policy=pol,
                    env=env,
                    test_texts=eval_texts,
                    test_labels=eval_labels,
                    seed=42,
                )
                rows.append({
                    "Algorithm": pname,
                    "Accuracy (%)": res["accuracy_pct"],
                    "Precision (%)": res["precision_pct"],
                    "Recall (%)": res["recall_pct"],
                    "F1 (%)": res["f1_pct"],
                    "FP32 Match (%)": res["fp32_agreement_pct"],
                    "Avg Lat (ms)": res["avg_latency_ms"],
                    "P95 Lat (ms)": res["p95_latency_ms"],
                    "Avg Energy (mJ)": res["avg_energy_mj"],
                    "Total Tx (KB)": res["total_tx_kb"],
                    "SLA (%)": res["sla_success_pct"],
                    "Avg k": res["avg_split_k"],
                    "Avg Bits": res["avg_bitwidth"],
                })
                logs_by_algo[pname] = res
                prog.progress((idx_p + 1) / len(selected_bench_policies))

            st.session_state["spam_bench_df"] = pd.DataFrame(rows)
            st.session_state["spam_bench_logs"] = logs_by_algo
            status.empty()
            st.success(f"Evaluated {len(selected_bench_policies)} policies on {len(eval_texts)} real messages under `{eval_scenario}`!")

    if "spam_bench_df" in st.session_state:
        df = st.session_state["spam_bench_df"]
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Spam + MEC Benchmark CSV",
            data=csv_bytes,
            file_name=f"spam_mec_benchmark_{eval_scenario.lower()}.csv",
            mime="text/csv",
        )

        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            st.markdown("##### 🎯 Classification Quality (Accuracy & F1 %)")
            st.bar_chart(df.set_index("Algorithm")[["Accuracy (%)", "F1 (%)"]])
        with ch2:
            st.markdown("##### ⏱️ End-to-End Latency (ms)")
            st.bar_chart(df.set_index("Algorithm")[["Avg Lat (ms)", "P95 Lat (ms)"]])
        with ch3:
            st.markdown("##### 🔋 Energy (mJ) & SLA (%)")
            st.bar_chart(df.set_index("Algorithm")[["Avg Energy (mJ)", "SLA (%)"]])

        logs_dict = st.session_state.get("spam_bench_logs", {})
        if logs_dict:
            inspect_algo = st.selectbox("Inspect Per-Message Predictions & Confusion Matrix", list(logs_dict.keys()))
            chosen_res = logs_dict[inspect_algo]
            cm = chosen_res["confusion_matrix"]
            st.markdown(
                f"**Confusion Matrix (`{inspect_algo}`):** "
                f"TP (Spam→Spam) = **{cm['tp']}** | FP (Ham→Spam) = **{cm['fp']}** | "
                f"TN (Ham→Ham) = **{cm['tn']}** | FN (Spam→Ham) = **{cm['fn']}**"
            )
            st.dataframe(pd.DataFrame(chosen_res["sample_logs"]), use_container_width=True)
