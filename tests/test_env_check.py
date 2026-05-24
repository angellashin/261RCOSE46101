import argparse
import contextlib
import importlib.metadata
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "validity_gated_exp" / "env_check.py"

spec = importlib.util.spec_from_file_location("env_check", MODULE_PATH)
env_check = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = env_check
assert spec.loader is not None
spec.loader.exec_module(env_check)


def namespace(**overrides):
    defaults = {
        "disk_path": str(REPO_ROOT),
        "min_free_gb": 0.0,
        "require_cuda": False,
        "require_clean": False,
        "expected_commit": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class EnvCheckTest(unittest.TestCase):
    def test_check_package_reports_sklearn_pip_name_when_import_fails(self):
        def importer(name):
            raise ModuleNotFoundError(f"No module named {name!r}")

        def version_lookup(name):
            raise importlib.metadata.PackageNotFoundError(name)

        result = env_check.check_package(
            env_check.PackageCheck("sklearn", "scikit-learn", "python -m pip install scikit-learn"),
            importer=importer,
            version_lookup=version_lookup,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["import_name"], "sklearn")
        self.assertEqual(result["pip_name"], "scikit-learn")
        self.assertIn("scikit-learn", result["install_hint"])

    def test_check_package_accepts_successful_import(self):
        def importer(name):
            return ModuleType(name)

        def version_lookup(name):
            return "1.2.3"

        result = env_check.check_package(
            env_check.PackageCheck("numpy", "numpy", "python -m pip install numpy"),
            importer=importer,
            version_lookup=version_lookup,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["version"], "1.2.3")

    def test_disk_report_honors_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = env_check.disk_report(Path(tmpdir), min_free_gb=0.0)

        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["free_gb"], 0)

    def test_cuda_report_requires_cuda_when_requested(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                get_device_name=lambda index: "unused",
            )
        )

        report = env_check.cuda_report(
            require_cuda=True,
            importer=lambda name: fake_torch,
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["available"])

    def test_print_report_marks_failures_and_install_hint(self):
        report = {
            "python": "/tmp/python",
            "python_version": "3.12.0",
            "git": {"commit": "abc123", "dirty": False},
            "disk": {
                "free_gb": 20.0,
                "total_gb": 100.0,
                "path": Path("/tmp"),
                "min_free_gb": 15.0,
            },
            "cuda": {"available": False, "device": None, "error": ""},
            "packages": [
                {
                    "ok": False,
                    "import_name": "sklearn",
                    "pip_name": "scikit-learn",
                    "version": None,
                    "error": "ModuleNotFoundError",
                    "install_hint": "python -m pip install scikit-learn",
                }
            ],
            "failures": ["Package import failed: sklearn."],
            "warnings": [],
        }

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            env_check.print_report(report)

        text = out.getvalue()
        self.assertIn("FAIL sklearn [scikit-learn]", text)
        self.assertIn("python -m pip install -r validity_gated_exp/requirements-runtime.txt", text)
        self.assertIn("python -m pip install torch torchvision torchaudio", text)
        self.assertIn("python -m pip install -r validity_gated_exp/requirements.txt", text)
        self.assertIn("ENV CHECK FAIL", text)


if __name__ == "__main__":
    unittest.main()
