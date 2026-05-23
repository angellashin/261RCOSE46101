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

from experiment_utils import unique_result_name


PRIMARY_METRICS = [
    ("f1", "F1", "higher"),
    ("pair_accuracy", "PairAcc", "higher"),
    ("strict_pair_accuracy", "S-PairAcc", "higher"),
    ("flip_rate", "Flip", "lower"),
    ("strict_flip_rate", "S-Flip", "lower"),
    ("prob_gap", "ProbGap", "lower"),
    ("strict_prob_gap", "S-ProbGap", "lower"),
    ("fpr_gap", "FPRGap", "lower"),
    ("fpr_min_group_n", "FPR minN", "higher"),
    ("train_valid_cf_ratio", "TrainCF%", "higher"),
    ("cons_batch_ratio", "ConsBatch%", "higher"),
    ("avg_valid_cf_per_batch", "ValidCF/B", "higher"),
]


def load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    results, _ = load_results_with_metadata(paths)
    return results


def duplicate_result_name(name: str, metrics: dict[str, Any], path: Path, existing: set[str]) -> str:
    config = metrics.get("config")
    lam = config.get("lambda") if isinstance(config, dict) else None
    return unique_result_name(name, existing, lambda_value=lam, source=path.stem)


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
                result_name = name
                duplicate_of = None
                if result_name in merged:
                    duplicate_of = name
                    result_name = duplicate_result_name(name, metrics, path, set(merged))
                merged[result_name] = {
                    **metrics,
                    "_source_path": str(path),
                    "_original_name": duplicate_of or name,
                    "_renamed_duplicate": duplicate_of is not None,
                }
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


def fmt_num(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def delta_str(base: float | None, cur: float | None, direction: str) -> str:
    if base is None or cur is None:
        return "N/A"
    delta = cur - base
    good = delta > 0 if direction == "higher" else delta < 0
    marker = "+" if good else "-"
    return f"{delta:+.4f} {marker}"


def is_strict_family(name: str) -> bool:
    return (
        name == "Strict-Gated"
        or name == "Strict-Matched"
        or name.startswith("Strict_lam=")
        or name.startswith("Strict-Gated [")
        or name.startswith("Strict-Matched [")
    )


def best_variant_by(results: dict[str, dict[str, Any]], names: list[str], metric: str) -> tuple[str, float] | None:
    scored = []
    for name in names:
        value = mean_or_none(results[name].get(metric))
        if value is not None:
            scored.append((name, value))
    if not scored:
        return None
    return max(scored, key=lambda x: x[1])


def paired_delta(values_a: Any, values_b: Any) -> list[float]:
    if not isinstance(values_a, list) or not isinstance(values_b, list):
        return []
    if len(values_a) != len(values_b) or not values_a:
        return []
    deltas = []
    for a, b in zip(values_a, values_b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return []
        if math.isnan(a) or math.isnan(b):
            return []
        deltas.append(float(a) - float(b))
    return deltas


def has_strict_lambda_followup(results: dict[str, dict[str, Any]]) -> bool:
    return any(name.startswith("Strict_lam=") for name in results)


def recommended_next_steps(results: dict[str, dict[str, Any]]) -> list[str]:
    steps: list[str] = []
    missing_core = [name for name in ("Baseline", "Naive Swap", "Strict-Gated") if name not in results]
    if missing_core:
        steps.append(f"Run the missing core methods under the current commit: {', '.join(missing_core)}.")

    naive = results.get("Naive Swap")
    strict_names = [name for name in results if is_strict_family(name)]
    best_strict = best_variant_by(results, strict_names, "strict_pair_accuracy")
    if naive and best_strict:
        best_name, best_sp = best_strict
        naive_sp = mean_or_none(naive.get("strict_pair_accuracy"))
        best_f1 = mean_or_none(results[best_name].get("f1"))
        naive_f1 = mean_or_none(naive.get("f1"))
        f1_close = best_f1 is not None and naive_f1 is not None and abs(best_f1 - naive_f1) <= 0.01

        if naive_sp is not None and best_sp < naive_sp:
            if "Strict-Matched" not in results:
                steps.append("Run Strict-Matched to test whether the strict gate is simply under-regularized by lower CF coverage.")
            if not has_strict_lambda_followup(results):
                steps.append("Run targeted Strict_lam follow-ups, e.g. Strict_lam=0.15 and Strict_lam=0.25, before changing the method.")
            if f1_close:
                steps.append("If gated variants still trail Naive, keep the method as a validity-coverage tradeoff rather than forcing a stronger claim.")
        elif naive_sp is not None and best_sp >= naive_sp and f1_close:
            steps.append("Freeze the method search and move to error analysis/report writing; the gated result is strong enough.")

    if not steps:
        steps.append("No automatic follow-up is triggered; inspect metadata warnings and qualitative errors before launching more runs.")
    return steps


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
            f"strategy={config.get('lambda_strategy', 'unknown')} "
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
    renamed = [
        (name, metrics.get("_original_name"), metrics.get("_source_path"))
        for name, metrics in results.items()
        if metrics.get("_renamed_duplicate")
    ]
    for name, original, source in renamed:
        print(f"WARNING: duplicate experiment name '{original}' from {source} was renamed to '{name}'")


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
            if key in ("train_valid_cf_ratio", "cons_batch_ratio", "avg_valid_cf_per_batch", "fpr_min_group_n"):
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
    low_fpr_support = [
        (name, n) for name, metrics in results.items()
        if (n := mean_or_none(metrics.get("fpr_min_group_n"))) is not None and n < 20
    ]
    if low_fpr_support:
        details = ", ".join(f"{name}=minN {n:.1f}" for name, n in low_fpr_support)
        print(f"- FPR Gap has small normal-group support ({details}); keep FPR Gap as a secondary metric.")

    naive = results.get("Naive Swap")
    strict = results.get("Strict-Gated")
    matched = results.get("Strict-Matched")
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

    if strict and matched:
        strict_sp = mean_or_none(strict.get("strict_pair_accuracy"))
        matched_sp = mean_or_none(matched.get("strict_pair_accuracy"))
        strict_gap = mean_or_none(strict.get("strict_prob_gap"))
        matched_gap = mean_or_none(matched.get("strict_prob_gap"))
        strict_lam = mean_or_none(strict.get("lambda"))
        matched_lam = mean_or_none(matched.get("lambda"))
        if strict_sp is not None and matched_sp is not None:
            print("\nCoverage-matched diagnostic")
            print("---------------------------")
            print(f"Strict lambda={strict_lam} vs Strict-Matched lambda={matched_lam}.")
            if matched_sp > strict_sp:
                print("- Strict-Matched improves Strict PairAcc: Strict-Gated was likely under-regularized by lower CF coverage.")
            elif matched_gap is not None and strict_gap is not None and matched_gap < strict_gap:
                print("- Strict-Matched improves probability stability but not hard pair accuracy; report this as a soft-consistency gain.")
            else:
                print("- Strict-Matched does not improve Strict: the gate may be filtering useful signal, not merely reducing coverage.")

    strict_names = [name for name in results if is_strict_family(name)]
    best_strict = best_variant_by(results, strict_names, "strict_pair_accuracy")
    if best_strict:
        best_name, best_sp = best_strict
        best_f1 = mean_or_none(results[best_name].get("f1"))
        best_gap = mean_or_none(results[best_name].get("strict_prob_gap"))
        naive_sp = mean_or_none(naive.get("strict_pair_accuracy")) if naive else None
        naive_f1 = mean_or_none(naive.get("f1")) if naive else None
        naive_gap = mean_or_none(naive.get("strict_prob_gap")) if naive else None
        print("\nBest strict-family variant")
        print("--------------------------")
        print(
            f"{best_name}: Strict PairAcc={best_sp:.4f}, "
            f"F1={fmt_num(best_f1)}, Strict ProbGap={fmt_num(best_gap)}"
        )
        if naive_sp is not None:
            f1_close = best_f1 is not None and naive_f1 is not None and abs(best_f1 - naive_f1) <= 0.01
            if best_sp >= naive_sp and f1_close:
                print("- Use this as the main gated result: it matches/beats Naive on Strict PairAcc while preserving F1.")
            elif best_sp < naive_sp and f1_close:
                print("- Use this as the strongest gated result, but frame Naive vs gated as a validity-coverage tradeoff.")
            else:
                print("- Use cautiously: compare F1 and pair metrics before making this the main result.")
            if best_gap is not None and naive_gap is not None and best_gap < naive_gap:
                print("- It also improves Strict ProbGap over Naive, useful as a soft-consistency argument.")
        if "Strict-Matched" not in results and "Naive Swap" in results and best_name != "Strict-Matched":
            print("- If Naive still beats this variant, run Strict-Matched to separate low coverage from gate quality.")


def print_naive_vs_best_gated_diagnostic(results: dict[str, dict[str, Any]]) -> None:
    naive = results.get("Naive Swap")
    strict_names = [name for name in results if is_strict_family(name)]
    best_strict = best_variant_by(results, strict_names, "strict_pair_accuracy")
    if not naive or not best_strict:
        return

    best_name, _ = best_strict
    best_metrics = results[best_name]
    comparisons = [
        ("Strict PairAcc", "strict_pair_accuracy", "higher"),
        ("Macro-F1", "f1", "higher"),
        ("Strict ProbGap", "strict_prob_gap", "lower"),
    ]
    print("\nNaive vs best gated paired diagnostic")
    print("-------------------------------------")
    print(f"Best gated row: {best_name}")
    for label, key, direction in comparisons:
        deltas = paired_delta(best_metrics.get(key), naive.get(key))
        if not deltas:
            print(f"- {label}: paired seed comparison unavailable.")
            continue
        mean_delta = mean(deltas)
        wins = sum(d > 0 for d in deltas) if direction == "higher" else sum(d < 0 for d in deltas)
        sign = "+" if mean_delta >= 0 else ""
        good_word = "higher" if direction == "higher" else "lower"
        print(
            f"- {label}: gated-naive mean delta={sign}{mean_delta:.4f}; "
            f"gated is {good_word} on {wins}/{len(deltas)} matched seeds."
        )


def print_next_step_recommendations(results: dict[str, dict[str, Any]]) -> None:
    print("\nRecommended next steps")
    print("----------------------")
    for step in recommended_next_steps(results):
        print(f"- {step}")


def print_markdown_table(results: dict[str, dict[str, Any]]) -> None:
    print("\nMarkdown table")
    print("--------------")
    print("| Method | Macro-F1 | Pair Acc | Strict Pair Acc | Flip Rate | Strict Flip | Prob Gap | Strict Prob Gap | FPR Gap | FPR minN | Train CF% | Cons Batch% |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in results.items():
        print(
            f"| {name} | {fmt(metrics.get('f1'))} | {fmt(metrics.get('pair_accuracy'))} | "
            f"{fmt(metrics.get('strict_pair_accuracy'))} | {fmt(metrics.get('flip_rate'))} | "
            f"{fmt(metrics.get('strict_flip_rate'))} | {fmt(metrics.get('prob_gap'))} | "
            f"{fmt(metrics.get('strict_prob_gap'))} | {fmt(metrics.get('fpr_gap'))} | "
            f"{fmt(metrics.get('fpr_min_group_n'))} | {fmt(metrics.get('train_valid_cf_ratio'), scale=100.0)} | "
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
    print_naive_vs_best_gated_diagnostic(results)
    print_next_step_recommendations(results)
    print_markdown_table(results)


if __name__ == "__main__":
    main()
