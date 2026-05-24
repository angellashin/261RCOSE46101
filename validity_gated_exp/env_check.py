"""
Runtime environment check for validity-gated experiments.

This script is intentionally stdlib-only at import time. It can be run before
the expensive training command to catch missing packages, broken torch installs,
low disk space, dirty report runs, and missing CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MIN_FREE_GB = 15.0


@dataclass(frozen=True)
class PackageCheck:
    import_name: str
    pip_name: str
    install_hint: str


PACKAGE_CHECKS = [
    PackageCheck(
        "torch",
        "torch",
        "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    ),
    PackageCheck(
        "transformers",
        "transformers",
        "python -m pip install transformers",
    ),
    PackageCheck("datasets", "datasets", "python -m pip install datasets"),
    PackageCheck("kiwipiepy", "kiwipiepy", "python -m pip install kiwipiepy"),
    PackageCheck("sklearn", "scikit-learn", "python -m pip install scikit-learn"),
    PackageCheck("scipy", "scipy", "python -m pip install scipy"),
    PackageCheck("tqdm", "tqdm", "python -m pip install tqdm"),
    PackageCheck("numpy", "numpy", "python -m pip install numpy"),
]


def package_version(
    pip_name: str,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
) -> str | None:
    try:
        return version_lookup(pip_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_package(
    check: PackageCheck,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, object]:
    version = package_version(check.pip_name, version_lookup)
    try:
        importer(check.import_name)
    except Exception as exc:
        return {
            "ok": False,
            "import_name": check.import_name,
            "pip_name": check.pip_name,
            "version": version,
            "error": f"{type(exc).__name__}: {exc}",
            "install_hint": check.install_hint,
        }

    return {
        "ok": True,
        "import_name": check.import_name,
        "pip_name": check.pip_name,
        "version": version or "unknown",
        "error": "",
        "install_hint": check.install_hint,
    }


def check_packages(
    checks: list[PackageCheck] | None = None,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
) -> list[dict[str, object]]:
    return [
        check_package(check, importer=importer, version_lookup=version_lookup)
        for check in (checks or PACKAGE_CHECKS)
    ]


def disk_report(path: Path, min_free_gb: float) -> dict[str, object]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    return {
        "ok": free_gb >= min_free_gb,
        "path": path,
        "free_gb": free_gb,
        "total_gb": total_gb,
        "min_free_gb": min_free_gb,
    }


def git_value(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def git_report(repo_root: Path) -> dict[str, object]:
    status = git_value(["status", "--porcelain"], repo_root)
    return {
        "commit": git_value(["rev-parse", "--short", "HEAD"], repo_root),
        "dirty": None if status is None else bool(status),
    }


def cuda_report(
    require_cuda: bool,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> dict[str, object]:
    try:
        torch = importer("torch")
    except Exception as exc:
        return {
            "ok": not require_cuda,
            "available": False,
            "device": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        available = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if available else None
    except Exception as exc:
        return {
            "ok": False,
            "available": False,
            "device": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "ok": available or not require_cuda,
        "available": available,
        "device": device,
        "error": "",
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    packages = check_packages()
    disk = disk_report(Path(args.disk_path).resolve(), args.min_free_gb)
    git = git_report(REPO_ROOT)
    cuda = cuda_report(args.require_cuda)

    failures: list[str] = []
    warnings: list[str] = []

    for package in packages:
        if not package["ok"]:
            failures.append(
                f"Package import failed: {package['import_name']} "
                f"(pip package: {package['pip_name']})."
            )

    if not disk["ok"]:
        failures.append(
            f"Only {disk['free_gb']:.1f}GB free at {disk['path']}; "
            f"need at least {disk['min_free_gb']:.1f}GB."
        )

    if args.expected_commit and git["commit"] != args.expected_commit:
        failures.append(f"Git commit {git['commit']} does not match {args.expected_commit}.")
    if args.require_clean and git["dirty"]:
        failures.append("Git worktree is dirty; do not use this as report-grade evidence.")
    elif git["dirty"]:
        warnings.append("Git worktree is dirty; record this if you run experiments now.")

    if not cuda["ok"]:
        failures.append("CUDA is not available but --require_cuda was set.")
    elif not cuda["available"]:
        warnings.append("CUDA is not available; full report run will be slow on CPU.")

    return {
        "packages": packages,
        "disk": disk,
        "git": git,
        "cuda": cuda,
        "failures": failures,
        "warnings": warnings,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
    }


def print_report(report: dict[str, object]) -> None:
    print("Environment check")
    print("-----------------")
    print(f"Python: {report['python']} ({report['python_version']})")

    git = report["git"]
    assert isinstance(git, dict)
    print(f"Git: commit={git.get('commit')} dirty={git.get('dirty')}")

    disk = report["disk"]
    assert isinstance(disk, dict)
    print(
        "Disk: "
        f"{disk['free_gb']:.1f}GB free / {disk['total_gb']:.1f}GB total "
        f"at {disk['path']} (min {disk['min_free_gb']:.1f}GB)"
    )

    cuda = report["cuda"]
    assert isinstance(cuda, dict)
    if cuda["available"]:
        print(f"CUDA: available ({cuda['device']})")
    elif cuda["error"]:
        print(f"CUDA: unavailable ({cuda['error']})")
    else:
        print("CUDA: unavailable")

    print("Packages:")
    for package in report["packages"]:
        assert isinstance(package, dict)
        status = "PASS" if package["ok"] else "FAIL"
        version = package["version"] or "not installed"
        print(f"- {status} {package['import_name']} [{package['pip_name']}] version={version}")
        if not package["ok"]:
            print(f"  error: {package['error']}")
            print(f"  install: {package['install_hint']}")

    if any(not package["ok"] for package in report["packages"]):
        print("Install missing non-torch dependencies when torch is already installed:")
        print("  python -m pip install -r validity_gated_exp/requirements-runtime.txt")
        print("Install or repair CUDA torch separately if torch failed:")
        print("  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("Fresh all-in-one install:")
        print("  python -m pip install -r validity_gated_exp/requirements.txt")

    for warning in report["warnings"]:
        print(f"WARN: {warning}")
    for failure in report["failures"]:
        print(f"FAIL: {failure}")

    if report["failures"]:
        print("ENV CHECK FAIL")
    else:
        print("ENV CHECK PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disk_path", default=str(REPO_ROOT))
    parser.add_argument("--min_free_gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--require_clean", action="store_true")
    parser.add_argument("--expected_commit", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    print_report(report)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
