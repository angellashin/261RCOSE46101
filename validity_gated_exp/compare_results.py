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
    ("cons_batch_ratio", "ConsBatch%", "higher"),
    ("avg_valid_cf_per_batch", "ValidCF/B", "higher"),
]


def load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    results, _ = load_results_with_metadata(paths)
    return results


def load_results_with_metadata(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[str, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("_meta")
        if isinstance(meta, dict):
            metadata.append({"path": str(path), **meta})
        else:
            metadata.append({"path": str(path), "missing_meta": True})
        for name, metrics in data.items():
            if isinstance(metrics, dict) and "f1" in metrics:
                merged[name] = {**metrics, "_source_path": str(path)}
    return merged, metadata


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
            scale = 100.0 if key in ("train_valid_cf_ratio", "cons_batch_ratio") else 1.0
            row.append(fmt(metrics.get(key), scale=scale).rjust(13))
        print("  ".join(row))


def print_metadata_warnings(metadata: list[dict[str, Any]]) -> None:
    print("Result metadata")
    print("---------------")
    if not metadata:
        print("No metadata found.")
        return

    for meta in metadata:
        if meta.get("missing_meta"):
            print(f"- {meta['path']}: missing _meta (likely old result; avoid mixing in final tables)")
        else:
            print(
                f"- {meta['path']}: commit={meta.get('git_commit')} "
                f"gate={meta.get('gate_version')} model={meta.get('model')} "
                f"lambda={meta.get('lambda')} seeds={meta.get('seeds')}"
            )

    for key in ("git_commit", "gate_version", "model", "max_len"):
        vals = {m.get(key) for m in metadata if not m.get("missing_meta")}
        vals.discard(None)
        if len(vals) > 1:
            print(f"WARNING: result files mix different {key} values: {sorted(vals)}")
    dirty = [m["path"] for m in metadata if m.get("git_dirty")]
    if dirty:
        print(f"WARNING: result files were produced from dirty git state: {dirty}")


def print_experiment_config_warnings(results: dict[str, dict[str, Any]]) -> None:
    print("\nExperiment configs")
    print("------------------")
    configs: list[tuple[str, dict[str, Any]]] = []
    for name, metrics in results.items():
        config = metrics.get("config")
        if not isinstance(config, dict):
            print(f"- {name}: missing per-experiment config (likely old result)")
            continue
        configs.append((name, config))
        print(
            f"- {name}: mode={config.get('mode')} lambda={config.get('lambda')} "
            f"commit={config.get('git_commit')} dirty={config.get('git_dirty')} "
            f"gate={config.get('gate_version')} model={config.get('model')}"
        )

    for key in ("git_commit", "gate_version", "model", "max_len", "epochs", "batch_size", "lr"):
        vals = {c.get(key) for _, c in configs}
        vals.discard(None)
        if len(vals) > 1:
            print(f"WARNING: experiments mix different {key} values: {sorted(vals)}")
    dirty_methods = [name for name, config in configs if config.get("git_dirty")]
    if dirty_methods:
        print(f"WARNING: experiments were run from dirty git state: {dirty_methods}")


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
        naive_f1 = mean_or_none(naive.get("f1"))
        strict_f1 = mean_or_none(strict.get("f1"))
        naive_gap = mean_or_none(naive.get("strict_prob_gap"))
        strict_gap = mean_or_none(strict.get("strict_prob_gap"))
        naive_cf = mean_or_none(naive.get("train_valid_cf_ratio"))
        strict_cf = mean_or_none(strict.get("train_valid_cf_ratio"))
        naive_cb = mean_or_none(naive.get("cons_batch_ratio"))
        strict_cb = mean_or_none(strict.get("cons_batch_ratio"))
        naive_vb = mean_or_none(naive.get("avg_valid_cf_per_batch"))
        strict_vb = mean_or_none(strict.get("avg_valid_cf_per_batch"))
        if naive_sp is not None and strict_sp is not None:
            if strict_sp >= naive_sp:
                print("- Strict-Gated beats or matches Naive on Strict PairAcc: this supports the validity-gated claim.")
            else:
                print("- Naive beats Strict on Strict PairAcc: frame the result as an invariance-validity tradeoff.")
            print("\nPaper-claim suggestion")
            print("----------------------")
            f1_close = (
                naive_f1 is not None and strict_f1 is not None
                and abs(strict_f1 - naive_f1) <= 0.01
            )
            gap_better = (
                naive_gap is not None and strict_gap is not None
                and strict_gap <= naive_gap
            )
            if strict_sp >= naive_sp and f1_close:
                print("Use the strong claim: validity-gated CCR improves or matches Naive while preserving F1.")
            elif strict_sp < naive_sp and f1_close and gap_better:
                print("Use the tradeoff claim: Naive gives stronger hard-label invariance, Strict gives comparable F1 and softer probability stability.")
            elif strict_sp < naive_sp:
                print("Use the diagnostic claim: current strict gate is conservative; analyze TrainCF% and invalid-pair examples.")
            else:
                print("Use a cautious claim: identity-swap CCR helps, but gate benefits depend on metric choice.")
            if naive_cf is not None and strict_cf is not None:
                print(f"TrainCF coverage: Naive={100*naive_cf:.2f}% vs Strict={100*strict_cf:.2f}%.")
            if naive_cb is not None and strict_cb is not None:
                print(f"Regularized batches: Naive={100*naive_cb:.2f}% vs Strict={100*strict_cb:.2f}%.")
            if naive_vb is not None and strict_vb is not None:
                print(f"Valid CF per batch: Naive={naive_vb:.2f} vs Strict={strict_vb:.2f}.")
        else:
            print("- Strict/Naive PairAcc is missing for at least one method; rerun both with the same current code.")


def print_markdown_table(results: dict[str, dict[str, Any]]) -> None:
    print("\nMarkdown table")
    print("--------------")
    print("| Method | Macro-F1 | Pair Acc | Strict Pair Acc | Flip Rate | Strict Flip | Prob Gap | Strict Prob Gap | Train CF% | Cons Batch% |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in results.items():
        print(
            f"| {name} | {fmt(metrics.get('f1'))} | {fmt(metrics.get('pair_accuracy'))} | "
            f"{fmt(metrics.get('strict_pair_accuracy'))} | {fmt(metrics.get('flip_rate'))} | "
            f"{fmt(metrics.get('strict_flip_rate'))} | {fmt(metrics.get('prob_gap'))} | "
            f"{fmt(metrics.get('strict_prob_gap'))} | {fmt(metrics.get('train_valid_cf_ratio'), scale=100.0)} | "
            f"{fmt(metrics.get('cons_batch_ratio'), scale=100.0)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+", type=Path, help="result JSON path(s)")
    args = parser.parse_args()
    results, metadata = load_results_with_metadata(args.json)
    if not results:
        raise SystemExit("No valid experiment results found.")
    print_metadata_warnings(metadata)
    print_experiment_config_warnings(results)
    print()
    print_table(results)
    print_baseline_deltas(results)
    print_interpretation_notes(results)
    print_markdown_table(results)


if __name__ == "__main__":
    main()
