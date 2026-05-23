import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "validity_gated_exp" / "compare_results.py"

spec = importlib.util.spec_from_file_location("compare_results", MODULE_PATH)
compare_results = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(compare_results)


class CompareResultsTest(unittest.TestCase):
    def test_tradeoff_claim_is_suggested_when_naive_has_higher_pairacc(self):
        results = {
            "Baseline": {
                "f1": [0.793, 0.798, 0.786],
                "flip_rate": [0.050, 0.059, 0.042],
                "strict_flip_rate": [0.054, 0.052, 0.055],
                "prob_gap": [0.040, 0.037, 0.042],
                "strict_prob_gap": [0.042, 0.039, 0.044],
            },
            "Naive Swap": {
                "f1": [0.792, 0.796, 0.788],
                "pair_accuracy": [0.823, 0.827, 0.820],
                "strict_pair_accuracy": [0.828, 0.832, 0.824],
                "flip_rate": [0.018, 0.019, 0.017],
                "strict_flip_rate": [0.018, 0.021, 0.016],
                "prob_gap": [0.020, 0.023, 0.019],
                "strict_prob_gap": [0.021, 0.023, 0.018],
                "train_valid_cf_ratio": [0.045, 0.045, 0.045],
                "cons_batch_ratio": [0.95, 0.96, 0.94],
                "avg_valid_cf_per_batch": [2.9, 2.8, 2.9],
            },
            "Strict-Gated": {
                "f1": [0.794, 0.797, 0.790],
                "pair_accuracy": [0.818, 0.821, 0.815],
                "strict_pair_accuracy": [0.822, 0.825, 0.819],
                "flip_rate": [0.021, 0.028, 0.015],
                "strict_flip_rate": [0.024, 0.031, 0.016],
                "prob_gap": [0.019, 0.018, 0.019],
                "strict_prob_gap": [0.018, 0.016, 0.020],
                "train_valid_cf_ratio": [0.035, 0.035, 0.035],
                "cons_batch_ratio": [0.89, 0.90, 0.88],
                "avg_valid_cf_per_batch": [2.2, 2.3, 2.2],
            },
            "Strict-Matched": {
                "f1": [0.794, 0.797, 0.790],
                "pair_accuracy": [0.824, 0.826, 0.821],
                "strict_pair_accuracy": [0.829, 0.831, 0.826],
                "flip_rate": [0.019, 0.020, 0.018],
                "strict_flip_rate": [0.021, 0.022, 0.020],
                "prob_gap": [0.018, 0.018, 0.019],
                "strict_prob_gap": [0.017, 0.017, 0.018],
                "train_valid_cf_ratio": [0.035, 0.035, 0.035],
                "cons_batch_ratio": [0.89, 0.90, 0.88],
                "avg_valid_cf_per_batch": [2.2, 2.3, 2.2],
                "lambda": [0.129, 0.129, 0.129],
            },
        }

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_interpretation_notes(results)
        text = out.getvalue()

        self.assertIn("invariance-validity tradeoff", text)
        self.assertIn("Use the tradeoff claim", text)
        self.assertIn("Naive=4.50% vs Strict=3.50%", text)
        self.assertIn("Regularized batches: Naive=95.00% vs Strict=89.00%", text)
        self.assertIn("Valid CF per batch: Naive=2.87 vs Strict=2.23", text)
        self.assertIn("Coverage-matched diagnostic", text)
        self.assertIn("Strict-Matched improves Strict PairAcc", text)

    def test_loading_and_markdown_table_from_json(self):
        payload = {
            "_meta": {"git_commit": "abc123", "gate_version": "v1", "model": "klue/roberta-base"},
            "Baseline": {"f1": [0.79], "flip_rate": [0.05]},
            "metadata": {"ignored": True},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            results, metadata = compare_results.load_results_with_metadata([path])

        self.assertEqual(list(results), ["Baseline"])
        self.assertEqual(metadata[0]["git_commit"], "abc123")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_markdown_table(results)
        self.assertIn("| Baseline | 0.7900 |", out.getvalue())

    def test_metadata_warning_for_mixed_commits(self):
        metadata = [
            {"path": "a.json", "git_commit": "aaa", "gate_version": "v1", "model": "m"},
            {"path": "b.json", "git_commit": "bbb", "gate_version": "v1", "model": "m"},
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_metadata_warnings(metadata)
        self.assertIn("mix different git_commit", out.getvalue())

    def test_experiment_config_warning_for_mixed_commits_and_dirty_state(self):
        results = {
            "Naive Swap": {
                "f1": [0.79],
                "config": {
                    "mode": "swap",
                    "lambda": 0.1,
                    "git_commit": "aaa",
                    "git_dirty": False,
                    "gate_version": "v1",
                    "model": "m",
                },
            },
            "Strict-Gated": {
                "f1": [0.79],
                "config": {
                    "mode": "strict",
                    "lambda": 0.2,
                    "git_commit": "bbb",
                    "git_dirty": True,
                    "gate_version": "v1",
                    "model": "m",
                },
            },
        }
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_experiment_config_warnings(results)
        text = out.getvalue()
        self.assertIn("experiments mix different git_commit", text)
        self.assertIn("experiments were run from dirty git state", text)


if __name__ == "__main__":
    unittest.main()
