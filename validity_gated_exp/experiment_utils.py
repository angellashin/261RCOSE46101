"""Dependency-light helpers for experiment configuration."""

from __future__ import annotations


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
