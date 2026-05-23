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
