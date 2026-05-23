"""
Compare experiment result JSON files produced by run_exp.py.

This script is intentionally dependency-light: it uses only the Python standard
library so it can run locally even when the training environment is not set up.

Usage:
    python validity_gated_exp/compare_results.py validity_gated_exp/results_core.json
    python validity_gated_exp/compare_results.py results_naive.json results_strict_lam02.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PRIMARY_METRICS = [
    ("f1", "F1", "higher"),
    ("pair_accuracy", "PairAcc", "higher"),
    ("strict_pair_accuracy", "S-PairAcc", "higher"),
    ("flip_rate", "Flip", "lower"),
    ("strict_flip_rate", "S-Flip", "lower"),
    ("prob_gap", "ProbGap", "lower"),
    ("strict_prob_gap", "S-ProbGap", "lower"),
    ("fpr_gap", "FPRGap", "lower"),
    ("train_valid_cf_ratio", "TrainCF%", "higher"),
]


def load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for name, metrics in data.items():
            if isinstance(metrics, dict) and "f1" in metrics:
                merged[name] = metrics
    return merged


def fmt(values: Any, scale: float = 1.0) -> str:
    if not isinstance(values, list) or not values:
        return "N/A"
    vals = [v * scale for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f}±{pstdev(vals):.4f}"


def mean_or_none(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return mean(vals) if vals else None


def delta_str(base: float | None, cur: float | None, direction: str) -> str:
    if base is None or cur is None:
        return "N/A"
    delta = cur - base
    good = delta > 0 if direction == "higher" else delta < 0
    marker = "+" if good else "-"
    return f"{delta:+.4f} {marker}"


def print_table(results: dict[str, dict[str, Any]]) -> None:
    name_w = max(12, *(len(k) for k in results))
    headers = ["Experiment"] + [label for _, label, _ in PRIMARY_METRICS]
    widths = [name_w] + [13] * len(PRIMARY_METRICS)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for name, metrics in results.items():
        row = [name.ljust(name_w)]
        for key, _, _ in PRIMARY_METRICS:
            scale = 100.0 if key == "train_valid_cf_ratio" else 1.0
            row.append(fmt(metrics.get(key), scale=scale).rjust(13))
        print("  ".join(row))


def print_baseline_deltas(results: dict[str, dict[str, Any]]) -> None:
    if "Baseline" not in results:
        return
    base = results["Baseline"]
    print("\nDelta vs Baseline")
    print("-----------------")
    for name, metrics in results.items():
        if name == "Baseline":
            continue
        print(f"\n{name}")
        for key, label, direction in PRIMARY_METRICS:
            if key == "train_valid_cf_ratio":
                continue
            b = mean_or_none(base.get(key))
            c = mean_or_none(metrics.get(key))
            print(f"  {label:<12} {delta_str(b, c, direction)}")


def print_interpretation_notes(results: dict[str, dict[str, Any]]) -> None:
    print("\nInterpretation guardrails")
    print("-------------------------")
    print("- Do not rank methods by flip rate alone; low flip can hide consistently wrong pairs.")
    print("- Prefer Macro-F1 + Strict PairAcc as the main claim when available.")
    print("- TrainCF% explains regularization strength: a stricter gate may lose because it sees fewer CF pairs.")

    naive = results.get("Naive Swap")
    strict = results.get("Strict-Gated")
    if naive and strict:
        naive_sp = mean_or_none(naive.get("strict_pair_accuracy"))
        strict_sp = mean_or_none(strict.get("strict_pair_accuracy"))
        if naive_sp is not None and strict_sp is not None:
            if strict_sp >= naive_sp:
                print("- Strict-Gated beats or matches Naive on Strict PairAcc: this supports the validity-gated claim.")
            else:
                print("- Naive beats Strict on Strict PairAcc: frame the result as an invariance-validity tradeoff.")
        else:
            print("- Strict/Naive PairAcc is missing for at least one method; rerun both with the same current code.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+", type=Path, help="result JSON path(s)")
    args = parser.parse_args()
    results = load_results(args.json)
    if not results:
        raise SystemExit("No valid experiment results found.")
    print_table(results)
    print_baseline_deltas(results)
    print_interpretation_notes(results)


if __name__ == "__main__":
    main()
