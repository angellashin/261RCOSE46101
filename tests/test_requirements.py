import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "validity_gated_exp"


def requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip()
        names.add(name)
    return names


class RequirementsTest(unittest.TestCase):
    def test_runtime_requirements_exclude_torch(self):
        runtime = requirement_names(EXP_DIR / "requirements-runtime.txt")

        self.assertNotIn("torch", runtime)
        self.assertNotIn("torchvision", runtime)
        self.assertNotIn("torchaudio", runtime)

    def test_runtime_requirements_cover_non_torch_training_imports(self):
        runtime = requirement_names(EXP_DIR / "requirements-runtime.txt")

        self.assertTrue(
            {
                "transformers",
                "datasets",
                "kiwipiepy",
                "scikit-learn",
                "scipy",
                "tqdm",
                "numpy",
            }.issubset(runtime)
        )

    def test_full_requirements_include_runtime_dependencies_and_torch(self):
        full = requirement_names(EXP_DIR / "requirements.txt")
        runtime = requirement_names(EXP_DIR / "requirements-runtime.txt")

        self.assertIn("torch", full)
        self.assertTrue(runtime.issubset(full))


if __name__ == "__main__":
    unittest.main()
