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


if __name__ == "__main__":
    unittest.main()
