"""Dependency-light helpers for experiment configuration."""

from __future__ import annotations

<<<<<<< HEAD
=======
import math

>>>>>>> main

ERROR_EXAMPLE_BUCKETS = (
    "flip",
    "strict_flip",
    "both_wrong",
    "strict_both_wrong",
    "orig_wrong_cf_right",
    "orig_right_cf_wrong",
    "false_positive_original",
    "false_positive_cf",
)

<<<<<<< HEAD
=======
AVAILABLE_EXPERIMENT_TAGS = (
    "Baseline",
    "Masking Cons Reg",
    "Naive Swap",
    "Validity-Gated",
    "Strict-Gated",
    "Strict-Matched",
)

CORE_REPORT_EXPERIMENTS = ("Baseline", "Naive Swap", "Strict-Gated")

RESUME_CONFIG_KEYS = (
    "tag",
    "mode",
    "use_cons",
    "lambda",
    "lambda_strategy",
    "epochs",
    "model",
    "max_len",
    "batch_size",
    "lr",
    "weight_decay",
    "gate_version",
    "git_commit",
    "git_dirty",
)

RESUME_METRIC_KEYS = (
    "f1",
    "flip_rate",
    "prob_gap",
    "pair_accuracy",
    "strict_flip_rate",
    "strict_prob_gap",
    "strict_pair_accuracy",
)

>>>>>>> main

def coverage_matched_lambda(
    base_lambda: float,
    reference_valid_count: int,
    target_valid_count: int,
    max_lambda: float | None = 0.3,
) -> float:
    """Scale lambda so a lower-coverage gate gets comparable CF signal.

    The reference is usually Naive Swap, where every generated swap is used.
    The target is usually Strict-Gated, where the validity gate filters pairs.
    A conservative cap prevents an accidental tiny target set from producing
    an unstable regularization weight.
    """
    if base_lambda <= 0 or reference_valid_count <= 0 or target_valid_count <= 0:
        return base_lambda
    matched = base_lambda * (reference_valid_count / target_valid_count)
    if max_lambda is not None:
        matched = min(matched, max_lambda)
    return matched


def unique_result_name(name: str, existing_names: set[str], lambda_value=None, source: str | None = None) -> str:
    """Return a non-overwriting result name for duplicate experiment tags."""
    if name not in existing_names:
        return name
    parts = []
    if lambda_value is not None:
        parts.append(f"lambda={lambda_value}")
    if source:
        parts.append(source)
    suffix = ", ".join(parts) if parts else "duplicate"
    candidate = f"{name} [{suffix}]"
    i = 2
    while candidate in existing_names:
        candidate = f"{name} [{suffix}, dup{i}]"
        i += 1
    return candidate


def merge_result_maps(existing: dict, new: dict, source: str = "new_run") -> tuple[dict, list[tuple[str, str]]]:
    """Merge result JSON maps without silently overwriting different configs."""
    merged = dict(existing)
    renames: list[tuple[str, str]] = []
    for name, metrics in new.items():
        if name == "_meta":
            continue
        result_name = name
        if result_name in merged and isinstance(merged.get(result_name), dict):
            old_metrics = merged[result_name]
            old_config = old_metrics.get("config") if isinstance(old_metrics, dict) else None
            new_config = metrics.get("config") if isinstance(metrics, dict) else None
            if old_config != new_config:
                result_name = unique_result_name(
                    name,
                    set(merged),
                    lambda_value=new_config.get("lambda") if isinstance(new_config, dict) else None,
                    source=source,
                )
                renames.append((name, result_name))
        merged[result_name] = metrics
    if "_meta" in new:
        merged["_meta"] = new["_meta"]
    return merged, renames


def build_result_snapshot(
    results: dict,
    run_meta: dict,
    completed_experiments: list[str],
    save_stage: str,
    is_final: bool,
) -> dict:
    """Attach run metadata to a partial or final result map."""
    snapshot = dict(results)
    snapshot["_meta"] = {
        **run_meta,
        "completed_experiments": list(completed_experiments),
        "save_stage": save_stage,
        "is_final": bool(is_final),
    }
    return snapshot


<<<<<<< HEAD
=======
def config_mismatches(existing_config: dict | None, expected_config: dict, keys=RESUME_CONFIG_KEYS) -> list[str]:
    """Return config keys that differ enough to prevent safe result reuse."""
    if not isinstance(existing_config, dict):
        return ["<missing config>"]
    return [
        key for key in keys
        if existing_config.get(key) != expected_config.get(key)
    ]


def completed_seed_set(metrics: dict) -> set[int] | None:
    """Return completed seeds when epoch_history records them, otherwise None."""
    histories = metrics.get("epoch_history")
    if not isinstance(histories, list) or not histories:
        return None
    seeds: set[int] = set()
    for item in histories:
        if not isinstance(item, dict) or "seed" not in item:
            return None
        try:
            seeds.add(int(item["seed"]))
        except (TypeError, ValueError):
            return None
    return seeds


def has_complete_metric_arrays(metrics: dict, required_count: int, keys=RESUME_METRIC_KEYS) -> bool:
    """Check that all core metric arrays contain at least the requested seed count."""
    for key in keys:
        values = metrics.get(key)
        if not isinstance(values, list) or len(values) < required_count:
            return False
    return True


def can_resume_result(
    existing_results: dict,
    tag: str,
    expected_config: dict,
    required_seeds: list[int],
) -> tuple[bool, str]:
    """Decide whether a saved experiment row is safe to reuse.

    Reusing partial report runs is useful, but only if the row matches the
    current method/config and has evidence for every requested seed.
    """
    metrics = existing_results.get(tag)
    if not isinstance(metrics, dict):
        return False, "no saved result row"
    if "f1" not in metrics:
        return False, "saved row has no F1 metrics"

    mismatches = config_mismatches(metrics.get("config"), expected_config)
    if mismatches:
        return False, "config mismatch: " + ", ".join(mismatches)

    required_count = len(required_seeds)
    if not has_complete_metric_arrays(metrics, required_count):
        return False, f"incomplete metric arrays for {required_count} requested seed(s)"

    completed_seeds = completed_seed_set(metrics)
    if completed_seeds is not None:
        missing = sorted(set(required_seeds) - completed_seeds)
        if missing:
            return False, f"missing requested seed(s): {missing}"

    return True, "complete compatible result"


>>>>>>> main
def parse_strict_lambda_tags(exp_tags: list[str] | None) -> list[float]:
    """Extract lambda values from --exp tags like Strict_lam=0.15.

    This keeps lambda follow-up runs explicit in the result table instead of
    reusing the generic Strict-Gated row name.
    """
    if not exp_tags:
        return [0.05, 0.2]

    values: list[float] = []
    seen: set[float] = set()
    for tag in exp_tags:
        if not tag.startswith("Strict_lam="):
            continue
        raw_value = tag.split("=", 1)[1]
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid Strict_lam tag: {tag}") from exc
<<<<<<< HEAD
        if value <= 0:
=======
        if not math.isfinite(value) or value <= 0:
>>>>>>> main
            raise ValueError(f"Strict_lam must be positive: {tag}")
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def unknown_experiment_tags(exp_tags: list[str] | None, known_tags: set[str]) -> list[str]:
    """Return requested --exp tags that are neither known ablations nor Strict_lam values."""
    if not exp_tags:
        return []
    return [
        tag for tag in exp_tags
        if tag not in known_tags and not tag.startswith("Strict_lam=")
    ]


<<<<<<< HEAD
=======
def resolve_requested_experiments(exp_tags: list[str] | None) -> tuple[list[str], list[float]]:
    """Resolve --exp tags into base experiment names and Strict_lam follow-ups."""
    known_tags = set(AVAILABLE_EXPERIMENT_TAGS)
    unknown = unknown_experiment_tags(exp_tags, known_tags)
    if unknown:
        valid = sorted(known_tags) + ["Strict_lam=<positive_float>"]
        raise ValueError(f"Unknown --exp tag(s): {unknown}. Valid choices: {valid}")

    strict_lambdas = parse_strict_lambda_tags(exp_tags)
    if not exp_tags:
        return list(AVAILABLE_EXPERIMENT_TAGS), strict_lambdas

    selected = [tag for tag in AVAILABLE_EXPERIMENT_TAGS if tag in exp_tags]
    if not selected and not strict_lambdas:
        valid = sorted(known_tags) + ["Strict_lam=<positive_float>"]
        raise ValueError(f"No experiments selected. Valid choices: {valid}")
    return selected, strict_lambdas


>>>>>>> main
def _round_float(value, digits: int):
    if value is None:
        return None
    return round(float(value), digits)


def collect_fairness_error_examples(
    pair_records: list[dict],
    limit_per_bucket: int = 5,
    digits: int = 4,
) -> dict[str, list[dict]]:
    """Collect capped, report-friendly examples explaining fairness metrics.

    Pair-level metrics are useful but easy to misread: a low flip rate can still
    hide pairs where both sides are consistently wrong. These buckets preserve a
    few concrete examples per seed so the final report can discuss whether a
    method improved robustness or simply became consistently over/under-sensitive.
    """
    buckets = {bucket: [] for bucket in ERROR_EXAMPLE_BUCKETS}

    def add(bucket: str, record: dict):
        if len(buckets[bucket]) >= limit_per_bucket:
            return
        buckets[bucket].append({
            "text": record.get("text"),
            "cf_text": record.get("cf_text"),
            "label": record.get("label"),
            "pred": record.get("pred"),
            "cf_pred": record.get("cf_pred"),
            "prob": _round_float(record.get("prob"), digits),
            "cf_prob": _round_float(record.get("cf_prob"), digits),
            "prob_gap": _round_float(record.get("prob_gap"), digits),
            "orig_term": record.get("orig_term"),
            "swap_term": record.get("swap_term"),
            "category": record.get("category"),
            "strict_valid": bool(record.get("strict_valid")),
        })

    for record in pair_records:
        if record.get("cf_pred") is None:
            continue

        label = record.get("label")
        pred = record.get("pred")
        cf_pred = record.get("cf_pred")
        strict_valid = bool(record.get("strict_valid"))

        orig_correct = pred == label
        cf_correct = cf_pred == label

        if pred != cf_pred:
            add("flip", record)
            if strict_valid:
                add("strict_flip", record)
        if not orig_correct and not cf_correct:
            add("both_wrong", record)
            if strict_valid:
                add("strict_both_wrong", record)
        elif not orig_correct and cf_correct:
            add("orig_wrong_cf_right", record)
        elif orig_correct and not cf_correct:
            add("orig_right_cf_wrong", record)

        if label == 0 and pred == 1:
            add("false_positive_original", record)
        if label == 0 and cf_pred == 1:
            add("false_positive_cf", record)

    return buckets
