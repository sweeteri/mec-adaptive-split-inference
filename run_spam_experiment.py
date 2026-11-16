"""
CLI Experiment Runner for Real-World Spam Detection with Adaptive MEC Split Inference.

Usage:
    python run_spam_experiment.py
    python run_spam_experiment.py --dataset uci --epochs 10 --scenario congested --max-test 200
    python run_spam_experiment.py --dataset csv --csv-path my_messages.csv --train-ppo
"""

import argparse
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from simulation.environment import MECSimulationEnv, BITWIDTH_TO_INDEX
from algorithms.local_inference import LocalInferencePolicy
from algorithms.full_offloading import FullOffloadingPolicy
from algorithms.fixed_split import FixedSplitPolicy
from algorithms.greedy import GreedyAdaptivePolicy
from algorithms.ppo import PPOPolicy
from training.train import train_rl_agent
from models.spam_classifier import (
    SpamDatasetLoader,
    ModularSpamClassifier,
    evaluate_spam_dataset_with_policy,
)


def main():
    parser = argparse.ArgumentParser(description="Real Spam Classification + SimPy MEC Split Inference Experiment")
    parser.add_argument("--dataset", choices=["builtin", "uci", "csv"], default="builtin", help="Dataset source")
    parser.add_argument("--csv-path", type=str, default="", help="Path to CSV/TSV file when --dataset csv")
    parser.add_argument("--epochs", type=int, default=14, help="Training epochs for ModularSpamClassifier")
    parser.add_argument("--scenario", choices=["ideal", "urban", "congested", "edge"], default="congested")
    parser.add_argument("--max-test", type=int, default=100, help="Maximum test messages to evaluate")
    parser.add_argument("--train-ppo", action="store_true", help="Train PPO agent briefly before evaluation")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if checkpoint exists")
    args = parser.parse_args()

    print("=" * 88)
    print("  REAL-WORLD SPAM DETECTION + ADAPTIVE SPLIT INFERENCE (PyTorch + SimPy DES MEC)")
    print("=" * 88)

    # 1. Load dataset
    if args.dataset == "builtin":
        tr_x, tr_y, te_x, te_y = SpamDatasetLoader.load_builtin_dataset(test_ratio=0.25, seed=42)
        ds_label = "Built-in EN+RU Benchmark"
    elif args.dataset == "uci":
        tr_x, tr_y, te_x, te_y = SpamDatasetLoader.download_uci_sms_spam(test_ratio=0.20, seed=42)
        ds_label = "UCI SMS Spam Collection"
    else:
        if not args.csv_path or not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
        with open(args.csv_path, "rb") as f:
            tr_x, tr_y, te_x, te_y = SpamDatasetLoader.parse_csv_or_tsv_bytes(f.read(), test_ratio=0.20, seed=42)
        ds_label = os.path.basename(args.csv_path)

    eval_texts = te_x[: args.max_test]
    eval_labels = te_y[: args.max_test]
    print(f"[Dataset] {ds_label} -> Train={len(tr_x)}, Test={len(eval_texts)} (evaluating)")

    # 2. Initialize MEC Environment (SimPy DES enabled by default)
    env = MECSimulationEnv(seed=42, use_simpy_des=True)
    env.set_scenario(args.scenario)
    print(f"[MEC Env] Scenario={args.scenario.upper()}, SimPy DES={env.use_simpy_des}, Layers={env.model.total_layers}")

    # 3. Load or Train ModularSpamClassifier
    clf = ModularSpamClassifier(num_layers=env.model.total_layers, hidden_dim=128, num_heads=4, max_seq_len=96)
    loaded = False if args.retrain else clf.load()
    if not loaded or args.dataset != "builtin":
        print(f"[Training] Training ModularTransformer on {len(tr_x)} real messages ({args.epochs} epochs)...")
        hist = clf.train_on_dataset(tr_x, tr_y, epochs=args.epochs, batch_size=16, lr=8e-4)
        print(f"[Training] Final Epoch Loss={hist[-1]['loss']:.4f}, Train Acc={hist[-1]['train_acc'] * 100:.1f}%")
        if args.dataset == "builtin":
            ckpt_dir = clf.save()
            print(f"[Checkpoint] Saved to {ckpt_dir}")
    else:
        print(f"[Checkpoint] Loaded pre-trained ModularSpamClassifier from {clf.checkpoint_dir}")

    # 4. Prepare Policies
    ppo = PPOPolicy(total_layers=env.model.total_layers)
    if args.train_ppo:
        print("[RL] Training PPO agent for 30 episodes on SimPy MEC environment...")
        train_rl_agent(ppo, env, num_episodes=30)

    policies = [
        LocalInferencePolicy(total_layers=env.model.total_layers),
        FullOffloadingPolicy(),
        FixedSplitPolicy(split_point=2, bitwidth_idx=BITWIDTH_TO_INDEX[32]),
        FixedSplitPolicy(split_point=2, bitwidth_idx=BITWIDTH_TO_INDEX[8]),
        FixedSplitPolicy(split_point=2, bitwidth_idx=BITWIDTH_TO_INDEX[4]),
        GreedyAdaptivePolicy(total_layers=env.model.total_layers),
        ppo,
    ]
    # Label fixed splits distinctly
    policies[2].name = "FixedSplit_k2_FP32"
    policies[3].name = "FixedSplit_k2_INT8"
    policies[4].name = "FixedSplit_k2_INT4"

    # 5. Run Evaluation across Policies
    results = []
    print("\n[Evaluation] Running real messages through Split Inference + SimPy DES...")
    for pol in policies:
        res = evaluate_spam_dataset_with_policy(
            classifier=clf,
            policy=pol,
            env=env,
            test_texts=eval_texts,
            test_labels=eval_labels,
            seed=42,
        )
        results.append({
            "Algorithm": res["algorithm"],
            "Acc%": res["accuracy_pct"],
            "F1%": res["f1_pct"],
            "FP32_Match%": res["fp32_agreement_pct"],
            "AvgLat_ms": res["avg_latency_ms"],
            "P95Lat_ms": res["p95_latency_ms"],
            "Energy_mJ": res["avg_energy_mj"],
            "TotalTx_KB": res["total_tx_kb"],
            "SLA%": res["sla_success_pct"],
            "Avg_k": res["avg_split_k"],
            "Avg_b": res["avg_bitwidth"],
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", f"spam_mec_{args.dataset}_{args.scenario}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[Saved] Full summary exported to {out_csv}")

    # 6. Show sample real predictions using Greedy Adaptive policy
    sample_res = evaluate_spam_dataset_with_policy(
        classifier=clf,
        policy=GreedyAdaptivePolicy(total_layers=env.model.total_layers),
        env=env,
        test_texts=eval_texts[:5],
        test_labels=eval_labels[:5],
        seed=42,
    )
    print("\n[Sample Real Predictions (Greedy Adaptive)]")
    for row in sample_res["sample_logs"]:
        print(
            f"  True={row['True']:4s} | Pred={row['Predicted']:4s} (p={row['Prob(Spam)']:.2f}) | "
            f"k={row['Split_k']} {row['Bits']:4s} | Tx={row['Tx_KB']:5.2f}KB | Lat={row['Latency_ms']:6.1f}ms | {row['Message']}"
        )


if __name__ == "__main__":
    main()
