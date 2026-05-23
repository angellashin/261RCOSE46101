import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "validity_gated_exp" / "experiment_utils.py"

spec = importlib.util.spec_from_file_location("experiment_utils", MODULE_PATH)
experiment_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(experiment_utils)


class ExperimentUtilsTest(unittest.TestCase):
    def test_coverage_matched_lambda_scales_to_reference_coverage(self):
        lam = experiment_utils.coverage_matched_lambda(
            base_lambda=0.1,
            reference_valid_count=100,
            target_valid_count=50,
            max_lambda=None,
        )
        self.assertAlmostEqual(lam, 0.2)

    def test_coverage_matched_lambda_uses_cap(self):
        lam = experiment_utils.coverage_matched_lambda(
            base_lambda=0.1,
            reference_valid_count=100,
            target_valid_count=10,
            max_lambda=0.3,
        )
        self.assertAlmostEqual(lam, 0.3)

    def test_coverage_matched_lambda_falls_back_on_empty_target(self):
        lam = experiment_utils.coverage_matched_lambda(
            base_lambda=0.1,
            reference_valid_count=100,
            target_valid_count=0,
        )
        self.assertAlmostEqual(lam, 0.1)

    def test_unique_result_name_includes_lambda_and_source(self):
        name = experiment_utils.unique_result_name(
            "Strict-Gated",
            {"Strict-Gated"},
            lambda_value=0.2,
            source="new_run",
        )
        self.assertEqual(name, "Strict-Gated [lambda=0.2, new_run]")

    def test_merge_result_maps_renames_different_duplicate_configs(self):
        existing = {
            "_meta": {"git_commit": "old"},
            "Strict-Gated": {
                "f1": [0.79],
                "config": {"lambda": 0.1},
            },
        }
        new = {
            "_meta": {"git_commit": "new"},
            "Strict-Gated": {
                "f1": [0.81],
                "config": {"lambda": 0.2},
            },
        }
        merged, renames = experiment_utils.merge_result_maps(existing, new, source="new_run")
        self.assertEqual(renames, [("Strict-Gated", "Strict-Gated [lambda=0.2, new_run]")])
        self.assertEqual(merged["Strict-Gated"]["f1"], [0.79])
        self.assertEqual(merged["Strict-Gated [lambda=0.2, new_run]"]["f1"], [0.81])
        self.assertEqual(merged["_meta"], {"git_commit": "new"})

    def test_merge_result_maps_overwrites_identical_config(self):
        existing = {
            "Strict-Gated": {
                "f1": [0.79],
                "config": {"lambda": 0.1},
            },
        }
        new = {
            "Strict-Gated": {
                "f1": [0.80],
                "config": {"lambda": 0.1},
            },
        }
        merged, renames = experiment_utils.merge_result_maps(existing, new)
        self.assertEqual(renames, [])
        self.assertEqual(merged["Strict-Gated"]["f1"], [0.80])

    def test_build_result_snapshot_records_partial_progress(self):
        snapshot = experiment_utils.build_result_snapshot(
            {"Baseline": {"f1": [0.79]}},
            {"git_commit": "abc123"},
            ["Baseline"],
            "after Baseline",
            is_final=False,
        )

        self.assertEqual(snapshot["Baseline"]["f1"], [0.79])
        self.assertEqual(snapshot["_meta"]["git_commit"], "abc123")
        self.assertEqual(snapshot["_meta"]["completed_experiments"], ["Baseline"])
        self.assertEqual(snapshot["_meta"]["save_stage"], "after Baseline")
        self.assertFalse(snapshot["_meta"]["is_final"])

    def test_repeated_snapshot_merge_against_same_base_is_stable(self):
        existing = {
            "Baseline": {
                "f1": [0.70],
                "config": {"lambda": 0.0, "git_commit": "old"},
            }
        }
        snapshot = experiment_utils.build_result_snapshot(
            {
                "Baseline": {
                    "f1": [0.79],
                    "config": {"lambda": 0.0, "git_commit": "new"},
                }
            },
            {"git_commit": "new"},
            ["Baseline"],
            "after Baseline",
            is_final=False,
        )

        first, first_renames = experiment_utils.merge_result_maps(existing, snapshot, source="new_run")
        second, second_renames = experiment_utils.merge_result_maps(existing, snapshot, source="new_run")
        self.assertEqual(first_renames, second_renames)
        self.assertEqual(sorted(first), sorted(second))
        self.assertIn("Baseline [lambda=0.0, new_run]", first)

    def test_parse_strict_lambda_tags_defaults_for_full_run(self):
        self.assertEqual(experiment_utils.parse_strict_lambda_tags(None), [0.05, 0.2])

    def test_parse_strict_lambda_tags_accepts_arbitrary_followups(self):
        tags = ["Naive Swap", "Strict_lam=0.15", "Strict_lam=0.25", "Strict_lam=0.15"]
        self.assertEqual(experiment_utils.parse_strict_lambda_tags(tags), [0.15, 0.25])

    def test_parse_strict_lambda_tags_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            experiment_utils.parse_strict_lambda_tags(["Strict_lam=abc"])
        with self.assertRaises(ValueError):
            experiment_utils.parse_strict_lambda_tags(["Strict_lam=0"])

    def test_unknown_experiment_tags_allows_known_and_strict_lambda(self):
        unknown = experiment_utils.unknown_experiment_tags(
            ["Baseline", "Strict_lam=0.15", "Typo"],
            {"Baseline", "Naive Swap"},
        )
        self.assertEqual(unknown, ["Typo"])

    def test_unknown_experiment_tags_allows_empty_request(self):
        self.assertEqual(experiment_utils.unknown_experiment_tags(None, {"Baseline"}), [])


if __name__ == "__main__":
    unittest.main()
