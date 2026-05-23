import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "validity_gated_exp" / "compare_results.py"
sys.path.insert(0, str(MODULE_PATH.parent))

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
                "fpr_min_group_n": [7, 7, 7],
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
                "fpr_min_group_n": [7, 7, 7],
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
                "fpr_min_group_n": [7, 7, 7],
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
                "fpr_min_group_n": [7, 7, 7],
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
        self.assertIn("FPR Gap has small normal-group support", text)
        self.assertIn("Coverage-matched diagnostic", text)
        self.assertIn("Strict-Matched improves Strict PairAcc", text)
        self.assertIn("Best strict-family variant", text)
        self.assertIn("Use this as the main gated result", text)

    def test_strict_family_detection_includes_lambda_followups(self):
        self.assertTrue(compare_results.is_strict_family("Strict-Gated"))
        self.assertTrue(compare_results.is_strict_family("Strict-Matched"))
        self.assertTrue(compare_results.is_strict_family("Strict_lam=0.2"))
        self.assertTrue(compare_results.is_strict_family("Strict-Gated [lambda=0.2, strict_lam02]"))
        self.assertFalse(compare_results.is_strict_family("Naive Swap"))

    def test_best_variant_by_selects_highest_strict_pair_accuracy(self):
        results = {
            "Strict-Gated": {"strict_pair_accuracy": [0.80]},
            "Strict_lam=0.2": {"strict_pair_accuracy": [0.83]},
            "Strict-Matched": {"strict_pair_accuracy": [0.82]},
        }
        best = compare_results.best_variant_by(
            results,
            ["Strict-Gated", "Strict_lam=0.2", "Strict-Matched"],
            "strict_pair_accuracy",
        )
        self.assertEqual(best, ("Strict_lam=0.2", 0.83))

    def test_paired_delta_requires_matched_seed_count(self):
        self.assertEqual(compare_results.paired_delta([0.83, 0.84], [0.82, 0.83]), [0.010000000000000009, 0.010000000000000009])
        self.assertEqual(compare_results.paired_delta([0.83], [0.82, 0.83]), [])
        self.assertEqual(compare_results.paired_delta([0.83, float("nan")], [0.82, 0.83]), [])

    def test_naive_vs_best_gated_diagnostic_prints_seed_consistency(self):
        results = {
            "Naive Swap": {
                "f1": [0.79, 0.80, 0.78],
                "strict_pair_accuracy": [0.82, 0.83, 0.82],
                "strict_prob_gap": [0.020, 0.022, 0.021],
            },
            "Strict-Matched": {
                "f1": [0.795, 0.799, 0.781],
                "strict_pair_accuracy": [0.83, 0.84, 0.81],
                "strict_prob_gap": [0.018, 0.020, 0.023],
            },
        }

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_naive_vs_best_gated_diagnostic(results)
        text = out.getvalue()
        self.assertIn("Naive vs best gated paired diagnostic", text)
        self.assertIn("Best gated row: Strict-Matched", text)
        self.assertIn("Strict PairAcc", text)
        self.assertIn("2/3 matched seeds", text)
        self.assertIn("Strict ProbGap", text)

    def test_recommended_next_steps_request_matched_and_lambda_when_naive_wins(self):
        results = {
            "Baseline": {"f1": [0.79]},
            "Naive Swap": {"f1": [0.79], "strict_pair_accuracy": [0.83]},
            "Strict-Gated": {"f1": [0.79], "strict_pair_accuracy": [0.80]},
        }

        steps = compare_results.recommended_next_steps(results)
        joined = "\n".join(steps)
        self.assertIn("Run Strict-Matched", joined)
        self.assertIn("Strict_lam=0.15", joined)
        self.assertIn("validity-coverage tradeoff", joined)

    def test_recommended_next_steps_freezes_search_when_gated_wins(self):
        results = {
            "Baseline": {"f1": [0.79]},
            "Naive Swap": {"f1": [0.79], "strict_pair_accuracy": [0.82]},
            "Strict-Gated": {"f1": [0.795], "strict_pair_accuracy": [0.83]},
        }

        steps = compare_results.recommended_next_steps(results)
        self.assertEqual(len(steps), 1)
        self.assertIn("Freeze the method search", steps[0])

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
        text = out.getvalue()
        self.assertIn("| Baseline | 0.7900 |", text)
        self.assertIn("FPR minN", text)

    def test_metadata_warning_for_mixed_commits(self):
        metadata = [
            {"path": "a.json", "git_commit": "aaa", "gate_version": "v1", "model": "m"},
            {"path": "b.json", "git_commit": "bbb", "gate_version": "v1", "model": "m"},
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_metadata_warnings(metadata)
        self.assertIn("mix different git_commit", out.getvalue())

    def test_duplicate_experiment_names_are_renamed_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "strict_lam005.json"
            second = Path(tmpdir) / "strict_lam02.json"
            first.write_text(json.dumps({
                "_meta": {"git_commit": "aaa", "gate_version": "v1", "model": "m"},
                "Strict-Gated": {
                    "f1": [0.79],
                    "config": {"lambda": 0.05, "git_commit": "aaa", "gate_version": "v1", "model": "m"},
                },
            }), encoding="utf-8")
            second.write_text(json.dumps({
                "_meta": {"git_commit": "aaa", "gate_version": "v1", "model": "m"},
                "Strict-Gated": {
                    "f1": [0.81],
                    "config": {"lambda": 0.2, "git_commit": "aaa", "gate_version": "v1", "model": "m"},
                },
            }), encoding="utf-8")
            results, _ = compare_results.load_results_with_metadata([first, second])

        self.assertIn("Strict-Gated", results)
        renamed = [name for name in results if name.startswith("Strict-Gated [")]
        self.assertEqual(len(renamed), 1)
        self.assertEqual(results["Strict-Gated"]["f1"], [0.79])
        self.assertEqual(results[renamed[0]]["f1"], [0.81])

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_experiment_config_warnings(results)
        self.assertIn("duplicate experiment name", out.getvalue())

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
