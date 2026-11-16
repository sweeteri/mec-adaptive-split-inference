"""
Master Test Runner for the MEC Adaptive Split Inference System.
Executes all unit and integration test suites across Stages 1 to 7.

Usage:
    python run_tests.py
"""

import sys
import os
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

TEST_SCRIPTS = [
    ("Stage 1: MEC Environment & DVFS", "tests/test_simulation.py"),
    ("Stage 1b: SimPy Discrete-Event Simulation", "tests/test_simpy_des.py"),
    ("Stage 2: Transformer & Tensor Quantization", "tests/test_stage2_transformer_quantization.py"),
    ("Stage 3: Baseline Algorithms (1-5)", "tests/test_stage3_baseline_algorithms.py"),
    ("Stage 4: RL Algorithms (DQN, PPO, SAC)", "tests/test_stage4_rl_algorithms.py"),
    ("Stage 5: Experiment Runner & Metrics", "tests/test_stage5_experiments.py"),
    ("Stage 7: End-to-End System Integration", "tests/test_stage7_integration.py"),
    ("Stage 8: Real-World Spam Classification & Split Inference", "tests/test_spam_classifier.py"),
]


def run_all_tests():
    print("=" * 85)
    print("  MEC RL SPLIT INFERENCE SYSTEM - SCIENTIFIC VERIFICATION SUITE")
    print("  Testing Stages 1 through 8")
    print("=" * 85)

    start_time = time.time()
    passed_stages = 0
    failed_stages = 0

    for title, script_rel_path in TEST_SCRIPTS:
        script_full_path = os.path.join(PROJECT_ROOT, script_rel_path)
        print(f"\n[EXEC] {title} ({script_rel_path})...")
        print("-" * 85)

        t0 = time.time()
        res = subprocess.run(
            [sys.executable, script_full_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        t_elapsed = time.time() - t0

        if res.returncode == 0:
            print(res.stdout.strip())
            print(f"---> [PASS] {title} in {t_elapsed:.2f}s")
            passed_stages += 1
        else:
            print(res.stdout)
            print(res.stderr)
            print(f"---> [FAIL] {title} (Exit code {res.returncode})")
            failed_stages += 1

    total_time = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"  EXECUTION SUMMARY: {passed_stages} passed, {failed_stages} failed in {total_time:.2f}s")
    print("=" * 85)

    if failed_stages == 0:
        print("\nSUCCESS: All scientific models, algorithms, and integration tests verified!")
        sys.exit(0)
    else:
        print(f"\nFAILURE: {failed_stages} test suites failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
