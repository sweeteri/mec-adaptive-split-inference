"""
Unit and Integration Tests for Real-World Spam Classification + Adaptive MEC Split Inference.
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simulation.environment import MECSimulationEnv, BITWIDTH_TO_INDEX
from algorithms.fixed_split import FixedSplitPolicy
from algorithms.greedy import GreedyAdaptivePolicy
from models.spam_classifier import (
    SpamDatasetLoader,
    SpamTextTokenizer,
    ModularSpamClassifier,
    compute_classification_metrics,
    evaluate_spam_dataset_with_policy,
)


class TestSpamClassifierSplitInference(unittest.TestCase):
    def test_dataset_loader_and_csv_parser(self):
        tr_x, tr_y, te_x, te_y = SpamDatasetLoader.load_builtin_dataset(test_ratio=0.25, seed=42)
        self.assertGreaterEqual(len(tr_x), 100)
        self.assertGreaterEqual(len(te_x), 30)
        self.assertEqual(len(tr_x), len(tr_y))
        self.assertEqual(len(te_x), len(te_y))
        self.assertEqual(set(tr_y), {0, 1})

        # CSV parser
        csv_sample = (
            "label,text\n"
            + "\n".join([f"spam,Win FREE cash prize #{i} at http://win.biz!" for i in range(8)])
            + "\n"
            + "\n".join([f"ham,Are we meeting at campus library at {i}pm?" for i in range(8)])
        ).encode("utf-8")
        c_tr_x, c_tr_y, c_te_x, c_te_y = SpamDatasetLoader.parse_csv_or_tsv_bytes(csv_sample, test_ratio=0.25)
        self.assertEqual(len(c_tr_x) + len(c_te_x), 16)
        self.assertIn(1, c_te_y)
        self.assertIn(0, c_te_y)

        # UCI TSV parser
        tsv_sample = (
            "\n".join([f"spam\tURGENT claim prize {i} http://promo.ru" for i in range(6)])
            + "\n"
            + "\n".join([f"ham\tHey see you tomorrow for lunch {i}" for i in range(6)])
        ).encode("utf-8")
        t_tr_x, t_tr_y, t_te_x, t_te_y = SpamDatasetLoader.parse_csv_or_tsv_bytes(tsv_sample, test_ratio=0.25)
        self.assertEqual(len(t_tr_x) + len(t_te_x), 12)

    def test_tokenizer_encode_and_save_load(self):
        tok = SpamTextTokenizer(max_vocab_size=500, max_seq_len=64)
        tok.fit(["URGENT! Win $500 at http://spam.com!", "Привет, пошли пить кофе завтра."])
        ids, mask = tok.encode("Win $500 at http://spam.com!", pad_to_max=False)
        self.assertEqual(ids.shape[0], 1)
        self.assertGreater(ids.shape[1], 3)
        self.assertEqual(mask.shape, (1, 1, 1, ids.shape[1]))
        self.assertEqual(float(mask[0, 0, 0, 0].item()), 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "tok.json")
            tok.save(p)
            loaded = SpamTextTokenizer.load(p)
            ids2, _ = loaded.encode("Win $500 at http://spam.com!", pad_to_max=False)
            self.assertEqual(ids.tolist(), ids2.tolist())

    def test_train_save_load_and_split_quantization(self):
        tr_x, tr_y, te_x, te_y = SpamDatasetLoader.load_builtin_dataset(test_ratio=0.25, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            clf = ModularSpamClassifier(
                num_layers=6,
                hidden_dim=64,
                num_heads=4,
                max_seq_len=64,
                checkpoint_dir=tmpdir,
            )
            hist = clf.train_on_dataset(tr_x, tr_y, epochs=12, batch_size=16, lr=1e-3, seed=42)
            self.assertGreater(hist[-1]["train_acc"], 0.90)
            clf.save()

            clf_loaded = ModularSpamClassifier(checkpoint_dir=tmpdir)
            self.assertTrue(clf_loaded.load())
            self.assertTrue(clf_loaded.is_trained)

            # Evaluate FP32 vs INT8 vs INT4 split inference on a real spam message
            spam_msg = "CONGRATULATIONS! Claim $1000 cash prize now: http://promo-gift.xyz"
            res_fp32 = clf_loaded.predict_split(spam_msg, split_point=2, bitwidth=32)
            res_int8 = clf_loaded.predict_split(spam_msg, split_point=2, bitwidth=8)
            res_int4 = clf_loaded.predict_split(spam_msg, split_point=2, bitwidth=4)
            res_local = clf_loaded.predict_split(spam_msg, split_point=6, bitwidth=32)

            self.assertEqual(res_fp32.predicted_label, "SPAM")
            self.assertEqual(res_int8.predicted_label, "SPAM")
            self.assertEqual(res_int4.predicted_label, "SPAM")
            self.assertEqual(res_local.transmitted_bytes, 0)
            self.assertGreater(res_int4.compression_ratio, 6.0)
            self.assertLess(res_int4.transmitted_bytes, res_int8.transmitted_bytes)
            self.assertLess(res_int8.transmitted_bytes, res_fp32.transmitted_bytes)

            # Evaluate test accuracy cleanly > 80%
            preds = [clf_loaded.predict_split(t, split_point=2, bitwidth=8).predicted_class for t in te_x]
            m = compute_classification_metrics(te_y, preds)
            self.assertGreaterEqual(m["accuracy"], 0.80)
            self.assertGreaterEqual(m["f1_score"], 0.80)

    def test_end_to_end_simpy_mec_evaluation(self):
        tr_x, tr_y, te_x, te_y = SpamDatasetLoader.load_builtin_dataset(test_ratio=0.25, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            clf = ModularSpamClassifier(num_layers=6, hidden_dim=64, num_heads=4, max_seq_len=64, checkpoint_dir=tmpdir)
            clf.train_on_dataset(tr_x, tr_y, epochs=10, batch_size=16, lr=1e-3, seed=42)

            env = MECSimulationEnv(seed=42, use_simpy_des=True)
            env.set_scenario("congested")
            pol_fixed_fp32 = FixedSplitPolicy(split_point=2, bitwidth_idx=BITWIDTH_TO_INDEX[32])
            pol_greedy = GreedyAdaptivePolicy(total_layers=6)

            res_fp32 = evaluate_spam_dataset_with_policy(clf, pol_fixed_fp32, env, te_x[:20], te_y[:20], seed=42)
            res_greedy = evaluate_spam_dataset_with_policy(clf, pol_greedy, env, te_x[:20], te_y[:20], seed=42)

            self.assertGreaterEqual(res_fp32["accuracy_pct"], 75.0)
            self.assertGreaterEqual(res_greedy["accuracy_pct"], 75.0)
            self.assertGreater(res_greedy["avg_latency_ms"], 0.0)
            self.assertGreater(res_greedy["avg_energy_mj"], 0.0)
            self.assertLess(res_greedy["total_tx_kb"], res_fp32["total_tx_kb"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
