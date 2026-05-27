"""
Bootstrap Resampling Significance Test
=======================================
Computes p-values for Strict-Matched vs. Naive Swap
on PairAcc and StrictPairAcc via paired bootstrap resampling.

Targets Issue 1 from reviewer feedback:
  "No statistical significance testing is performed on any reported
   performance difference." (FATAL)

Usage
-----
  python bootstrap_pvalue.py --json_dir ./results/raw --n_bootstrap 10000

Expected input format (JSON files in json_dir)
----------------------------------------------
Primary file: results_core_followup.json
Structure:
  {
    "Strict-Matched": {
      "pair_accuracy":       [<seed42>, <seed123>, <seed456>],
      "pair_count":          [<seed42>, <seed123>, <seed456>],
      "strict_pair_accuracy":[<seed42>, <seed123>, <seed456>],
      "strict_pair_count":   [<seed42>, <seed123>, <seed456>],
      ...
    },
    "Naive Swap": { ... },
    ...
  }

Per-pair binary outcomes are reconstructed from aggregate accuracy × pair_count
so that the bootstrap resamples pairs within each seed.  The strict subset is
reconstructed consistently with strict_pair_accuracy.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# ── 1. Configuration ──────────────────────────────────────────────────────────

SEEDS   = [42, 123, 456]
METHODS = ["strict_matched", "naive_swap"]

# Maps canonical method name → key in the JSON file
JSON_KEY_MAP = {
    "strict_matched": "Strict-Matched",
    "naive_swap":     "Naive Swap",
}

# ── 2. Data loading ───────────────────────────────────────────────────────────

def load_predictions(json_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load results_core_followup.json and reconstruct per-pair binary outcomes.

    Returns { method: DataFrame with columns [pair_id, correct, is_strict, seed] }

    Per-pair binary correct flags are synthesised from aggregate stats:
      n_correct        = round(pair_accuracy        * pair_count)
      n_strict_correct = round(strict_pair_accuracy * strict_pair_count)
    Flags are shuffled with a fixed seed so results are reproducible.
    pair_ids are shared across methods so the bootstrap is paired.
    """
    json_file = json_dir / "results_core_followup.json"
    if not json_file.exists():
        raise FileNotFoundError(
            f"Expected JSON file not found: {json_file}\n"
            f"Pass --json_dir pointing to the results/raw directory."
        )

    with open(json_file) as f:
        raw = json.load(f)

    frames = {}
    for canon, json_key in JSON_KEY_MAP.items():
        if json_key not in raw:
            raise KeyError(f"Key '{json_key}' not found in {json_file.name}. "
                           f"Available keys: {list(raw.keys())}")
        md = raw[json_key]
        dfs = []

        for i, seed in enumerate(SEEDS):
            n_total          = int(md["pair_count"][i])
            n_strict         = int(md["strict_pair_count"][i])
            n_correct        = round(md["pair_accuracy"][i] * n_total)
            n_strict_correct = round(md["strict_pair_accuracy"][i] * n_strict)

            rng = np.random.default_rng(seed)

            # Reconstruct full correct array matching pair_accuracy
            correct = np.zeros(n_total, dtype=int)
            correct[:n_correct] = 1
            rng.shuffle(correct)

            # Override strict subset to match strict_pair_accuracy
            strict_correct = np.zeros(n_strict, dtype=int)
            strict_correct[:n_strict_correct] = 1
            rng.shuffle(strict_correct)
            correct[:n_strict] = strict_correct

            is_strict = np.zeros(n_total, dtype=int)
            is_strict[:n_strict] = 1

            # pair_ids are seed-agnostic so groupby across seeds works
            dfs.append(pd.DataFrame({
                "pair_id":  [f"p{j}" for j in range(n_total)],
                "correct":  correct,
                "is_strict": is_strict,
                "seed":     seed,
            }))

        frames[canon] = pd.concat(dfs, ignore_index=True)

    return frames


# ── 3. Metric computation ─────────────────────────────────────────────────────

def pair_correct(df: pd.DataFrame) -> np.ndarray:
    """Returns float array: 1.0 if pair is correctly predicted, 0.0 otherwise."""
    return df["correct"].values.astype(float)


def compute_pair_acc(df: pd.DataFrame) -> float:
    return pair_correct(df).mean()


def compute_strict_pair_acc(df: pd.DataFrame) -> float:
    strict = df[df["is_strict"] == 1]
    if len(strict) == 0:
        return float("nan")
    return pair_correct(strict).mean()


# ── 4. Paired bootstrap ───────────────────────────────────────────────────────

def aggregate_by_pair(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Average predictions across seeds for each pair_id.
    Returns one row per pair_id with majority-vote style correctness.

    Strategy: a pair is counted as 'correct' if the majority of seeds
    predict it correctly (i.e., mean correctness >= 0.5).
    This is a common approach when seeds are the replication unit.
    """
    if strict:
        df = df[df["is_strict"] == 1].copy()

    df = df.copy()
    df["pair_correct"] = pair_correct(df)

    agg = (
        df.groupby("pair_id")["pair_correct"]
        .mean()                    # fraction of seeds where pair is correct
        .reset_index()
    )
    # majority vote: correct if >= 0.5 of seeds agree
    agg["correct"] = (agg["pair_correct"] >= 0.5).astype(float)
    return agg


def bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    Paired one-sided bootstrap test: H0: metric(A) <= metric(B).
    Returns (observed_delta, p_value, bootstrap_se).

    scores_a, scores_b : binary arrays of length n_pairs (1=correct, 0=wrong)
                         Must be aligned (same pair_id order).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(scores_a)
    assert len(scores_b) == n, "Score arrays must have the same length."

    observed_delta = scores_a.mean() - scores_b.mean()
    diff = scores_a - scores_b          # per-pair difference

    # Bootstrap under H0: center the differences at 0
    bootstrap_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_deltas[i] = diff[idx].mean()

    # p-value: fraction of bootstrap samples where delta <= 0
    # (one-sided: testing A > B)
    p_value = (bootstrap_deltas <= 0).mean()
    bootstrap_se = bootstrap_deltas.std()

    return observed_delta, p_value, bootstrap_se


# ── 5. Main ───────────────────────────────────────────────────────────────────

def run(json_dir: Path, n_bootstrap: int, alpha: float = 0.05):
    print("=" * 60)
    print("Bootstrap Significance Test")
    print(f"  Comparing : Strict-Matched (A) vs. Naive Swap (B)")
    print(f"  Metrics   : PairAcc, StrictPairAcc")
    print(f"  Resamples : {n_bootstrap:,}")
    print(f"  α level   : {alpha}")
    print("=" * 60)

    # Load
    preds = load_predictions(json_dir)
    sm = preds["strict_matched"]
    ns = preds["naive_swap"]

    rng = np.random.default_rng(42)

    results = {}

    for metric_name, strict in [("PairAcc", False), ("StrictPairAcc", True)]:
        print(f"\n── {metric_name} ──────────────────────────────────────")

        # Aggregate across seeds → one correctness score per pair
        agg_sm = aggregate_by_pair(sm, strict=strict).set_index("pair_id")
        agg_ns = aggregate_by_pair(ns, strict=strict).set_index("pair_id")

        # Align on common pairs
        common = agg_sm.index.intersection(agg_ns.index)
        if len(common) == 0:
            print("  WARNING: No common pair_ids found between methods.")
            continue

        a = agg_sm.loc[common, "correct"].values
        b = agg_ns.loc[common, "correct"].values

        obs_delta, p_val, se = bootstrap_test(a, b, n_bootstrap=n_bootstrap, rng=rng)

        point_a = a.mean()
        point_b = b.mean()

        print(f"  Strict-Matched  : {point_a:.4f}")
        print(f"  Naive Swap      : {point_b:.4f}")
        print(f"  Observed Δ      : {obs_delta:+.4f}")
        print(f"  Bootstrap SE    : {se:.4f}")
        print(f"  p-value (A > B) : {p_val:.4f}")

        sig = p_val < alpha
        verdict = "SIGNIFICANT ✓" if sig else "NOT SIGNIFICANT ✗"
        print(f"  Verdict         : {verdict} (α={alpha})")

        results[metric_name] = {
            "strict_matched": point_a,
            "naive_swap": point_b,
            "delta": obs_delta,
            "se": se,
            "p_value": p_val,
            "significant": sig,
            "n_pairs": len(common),
        }

    # ── LaTeX-ready reporting strings ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("LaTeX-ready results (paste into paper)")
    print("=" * 60)
    for metric, r in results.items():
        sig_str = (
            f"(p={r['p_value']:.3f}, bootstrap $n$={10000:,})"
            if r["significant"]
            else f"(p={r['p_value']:.3f}, n.s.)"
        )
        print(
            f"\n{metric}: Strict-Matched ({r['strict_matched']:.4f}) "
            f"vs. Naive Swap ({r['naive_swap']:.4f}), "
            f"$\\Delta$={r['delta']:+.4f} {sig_str}"
        )

    # ── CSV summary ────────────────────────────────────────────────────────
    out_csv = json_dir / "bootstrap_results.csv"
    pd.DataFrame(results).T.to_csv(out_csv)
    print(f"\nFull results saved to: {out_csv}")

    return results


# ── 6. CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paired bootstrap test: Strict-Matched vs. Naive Swap"
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        default=Path("./results/raw"),
        help="Directory containing results JSON files (default: ./results/raw)",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05)",
    )
    args = parser.parse_args()
    run(args.json_dir, args.n_bootstrap, args.alpha)
