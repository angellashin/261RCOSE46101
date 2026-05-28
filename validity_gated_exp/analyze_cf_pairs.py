"""
<<<<<<< HEAD
analyze_cf_pairs.py — cf_pairs_train.jsonl 데이터 분석 (GPU 불필요)

논문용 분석 3종:
  1. Overall pair count / pass rate
  2. Category-wise validity statistics
  3. Pass / Reject 예시 (qualitative)

Usage:
    python validity_gated_exp/analyze_cf_pairs.py
    python validity_gated_exp/analyze_cf_pairs.py --jsonl path/to/cf_pairs_train.jsonl
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--jsonl',  default='validity_gated_exp/data/cf_pairs_train.jsonl',
                    help='cf_pairs_train.jsonl 경로')
parser.add_argument('--train_total', type=int, default=172157,
                    help='전체 train 샘플 수 (check_data.py 결과값)')
parser.add_argument('--out', default=None,
                    help='결과 저장 경로 (생략 시 stdout만)')
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
jsonl_path = Path(args.jsonl)
if not jsonl_path.exists():
    raise FileNotFoundError(f'JSONL not found: {jsonl_path}')

pairs = []
with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        pairs.append(json.loads(line))

lines_out: list[str] = []

def pr(s=''):
    print(s)
    lines_out.append(s)

# ── Reject reason 결정 (strict gate 기준, priority order) ─────────────────────
REASON_LABELS = {
    'grammar':       'strict_valid_grammar',
    'semantics':     'strict_valid_semantics',
    'asym_pair':     'strict_label_preserving',
    'comparison':    'strict_no_comparison',
    'harmful_obj':   'strict_no_harmful_obj',
}

def get_reject_reason(p: dict) -> str:
    """strict_use_for_ccr=False인 pair에서 첫 번째 False 필드를 이유로 반환."""
    for reason, field in REASON_LABELS.items():
        if not p.get(field, True):
            return reason
    return 'unknown'

# ═══════════════════════════════════════════════════════════════════════════════
pr('=' * 65)
pr('  [1] CF Construction Statistics')
pr('=' * 65)

n_train    = args.train_total
n_swap     = len(pairs)
n_base     = sum(1 for p in pairs if p['base_use_for_ccr'])
n_strict   = sum(1 for p in pairs if p['strict_use_for_ccr'])

swap_rate   = n_swap   / n_train * 100
base_rate   = n_base   / n_swap  * 100
strict_rate = n_strict / n_swap  * 100

col = 32
pr(f'  {"Item":<{col}} Value')
pr(f'  {"-"*col} --------')
pr(f'  {"Train samples":<{col}} {n_train:,}')
pr(f'  {"Swappable samples":<{col}} {n_swap:,}  ({swap_rate:.1f}% of train)')
pr(f'  {"Base-valid pairs":<{col}} {n_base:,}  ({base_rate:.1f}% of swappable)')
pr(f'  {"Strict-valid pairs":<{col}} {n_strict:,}  ({strict_rate:.1f}% of swappable)')

pr()
pr('  Strict gate additionally filters:')
pr(f'  {"Rejected by strict (vs base)":<{col}} {n_base - n_strict:,}  ({(n_base-n_strict)/n_base*100:.1f}% of base-valid)')

# reject reason breakdown (중 strict에서 추가로 걸리는 것)
# base_valid=True but strict_valid=False → strict에서 추가로 거른 것
extra_rejected = [p for p in pairs if p['base_use_for_ccr'] and not p['strict_use_for_ccr']]
reason_cnt = Counter(get_reject_reason(p) for p in extra_rejected)
pr()
pr('  Strict-only rejection breakdown (base-valid → strict-rejected):')
for reason, cnt in reason_cnt.most_common():
    pr(f'    {reason:<20}: {cnt:,}  ({cnt/len(extra_rejected)*100:.1f}%)')

# ═══════════════════════════════════════════════════════════════════════════════
pr()
pr('=' * 65)
pr('  [2] Category-wise Validity Statistics')
pr('=' * 65)

cats = ['gender', 'ethnicity', 'religion', 'age', 'sexuality', 'disability']

# header
pr(f'  {"Category":<12} {"Swappable":>10} {"Base-valid":>11} {"Strict-valid":>13} '
   f'{"Base%":>7} {"Strict%":>8}')
pr(f'  {"-"*12} {"-"*10} {"-"*11} {"-"*13} {"-"*7} {"-"*8}')

cat_stats: dict[str, dict] = defaultdict(lambda: {'swap': 0, 'base': 0, 'strict': 0})
for p in pairs:
    c = p['category']
    cat_stats[c]['swap']   += 1
    cat_stats[c]['base']   += int(p['base_use_for_ccr'])
    cat_stats[c]['strict'] += int(p['strict_use_for_ccr'])

for cat in cats:
    s = cat_stats.get(cat, {'swap': 0, 'base': 0, 'strict': 0})
    if s['swap'] == 0:
        pr(f'  {cat:<12} {"N/A":>10}')
        continue
    bp = s['base']   / s['swap'] * 100
    sp = s['strict'] / s['swap'] * 100
    pr(f'  {cat:<12} {s["swap"]:>10,} {s["base"]:>11,} {s["strict"]:>13,} '
       f'{bp:>6.1f}% {sp:>7.1f}%')

# ═══════════════════════════════════════════════════════════════════════════════
pr()
pr('=' * 65)
pr('  [3] Qualitative Pass / Reject Examples')
pr('=' * 65)

# ── Pass 예시: strict-valid 중 각 category에서 1개씩 ──────────────────────────
pr()
pr('  [PASS examples — strict-valid=True]')
pr()
shown_cats: set[str] = set()
pass_shown = 0
for p in pairs:
    if not p['strict_use_for_ccr']:
        continue
    cat = p['category']
    if cat in shown_cats:
        continue
    shown_cats.add(cat)
    pr(f'  Category : {cat}')
    pr(f'  Original : {p["original"][:80]}')
    pr(f'  CF       : {p["cf"][:80]}')
    pr(f'  Label    : {p["label"]}')
    pr()
    pass_shown += 1
    if pass_shown >= 4:
        break

# ── Reject 예시: strict_use_for_ccr=False, 이유별 1개씩 ──────────────────────
pr('  [REJECT examples — strict-valid=False]')
pr()

reason_order = ['semantics', 'asym_pair', 'comparison', 'harmful_obj', 'grammar']
reason_labels_kor = {
    'semantics':   'semantic blacklist',
    'asym_pair':   'asymmetric pair (label not preserved)',
    'comparison':  'comparison expression',
    'harmful_obj': 'harmful object/event context',
    'grammar':     'grammar check failed',
}

shown_reasons: set[str] = set()
for p in pairs:
    if p['strict_use_for_ccr']:
        continue
    reason = get_reject_reason(p)
    if reason in shown_reasons or reason not in reason_order:
        continue
    shown_reasons.add(reason)
    pr(f'  Category : {p["category"]}')
    pr(f'  Original : {p["original"][:80]}')
    pr(f'  CF       : {p["cf"][:80]}')
    pr(f'  Reason   : {reason_labels_kor[reason]}')
    pr(f'  (base_valid={p["base_use_for_ccr"]}, strict_valid={p["strict_use_for_ccr"]})')
    pr()
    if len(shown_reasons) >= 5:
        break

# ═══════════════════════════════════════════════════════════════════════════════
pr('=' * 65)
pr('  완료.')
pr('=' * 65)

# ── 파일 저장 ──────────────────────────────────────────────────────────────────
if args.out:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_out))
    print(f'\n결과 저장 → {out_path}')
=======
Analyze generated counterfactual pairs without GPU dependencies.

Report-useful outputs:
  1. Overall pair count / pass rate
  2. Category-wise validity statistics
  3. Strict-only rejection breakdown
  4. Reason-by-category rejection matrix
  5. Pass / reject examples

Usage:
    python validity_gated_exp/analyze_cf_pairs.py
    python validity_gated_exp/analyze_cf_pairs.py --jsonl validity_gated_exp/data/cf_pairs_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REASON_LABELS = {
    "grammar": "strict_valid_grammar",
    "semantics": "strict_valid_semantics",
    "asym_pair": "strict_label_preserving",
    "comparison": "strict_no_comparison",
    "harmful_obj": "strict_no_harmful_obj",
    "age_context": "strict_no_age_contradiction",
}
REASON_ORDER = ["semantics", "asym_pair", "comparison", "harmful_obj", "age_context", "grammar", "unknown"]
REASON_LABELS_READABLE = {
    "semantics": "semantic blacklist",
    "asym_pair": "asymmetric pair (label not preserved)",
    "comparison": "comparison expression",
    "harmful_obj": "harmful object/event context",
    "age_context": "explicit age context contradiction",
    "grammar": "grammar check failed",
    "unknown": "unknown",
}
DEFAULT_CATEGORIES = ["gender", "ethnicity", "religion", "age", "sexuality", "disability"]


def pct(num: int, denom: int) -> float:
    return (num / denom * 100.0) if denom else 0.0


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def get_reject_reason(pair: dict[str, Any]) -> str:
    """Return the first failed strict-gate field by priority order."""
    for reason, field in REASON_LABELS.items():
        if not pair.get(field, True):
            return reason
    return "unknown"


def load_pairs(jsonl_path: Path) -> list[dict[str, Any]]:
    pairs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def analyze_pairs(pairs: list[dict[str, Any]], train_total: int) -> dict[str, Any]:
    n_swap = len(pairs)
    n_base = sum(1 for p in pairs if p.get("base_use_for_ccr"))
    n_strict = sum(1 for p in pairs if p.get("strict_use_for_ccr"))
    extra_rejected = [p for p in pairs if p.get("base_use_for_ccr") and not p.get("strict_use_for_ccr")]

    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"swap": 0, "base": 0, "strict": 0})
    reason_cnt: Counter[str] = Counter()
    reason_by_cat: dict[str, Counter[str]] = defaultdict(Counter)
    gate_versions = Counter(p.get("gate_version", "missing") for p in pairs)

    for p in pairs:
        cat = p.get("category", "unknown")
        cat_stats[cat]["swap"] += 1
        cat_stats[cat]["base"] += int(bool(p.get("base_use_for_ccr")))
        cat_stats[cat]["strict"] += int(bool(p.get("strict_use_for_ccr")))
        if p.get("base_use_for_ccr") and not p.get("strict_use_for_ccr"):
            reason = get_reject_reason(p)
            reason_cnt[reason] += 1
            reason_by_cat[cat][reason] += 1

    return {
        "train_total": train_total,
        "n_swap": n_swap,
        "n_base": n_base,
        "n_strict": n_strict,
        "gate_versions": gate_versions,
        "extra_rejected": extra_rejected,
        "reason_cnt": reason_cnt,
        "cat_stats": cat_stats,
        "reason_by_cat": reason_by_cat,
    }


def ordered_categories(cat_stats: dict[str, dict[str, int]]) -> list[str]:
    extras = sorted(c for c in cat_stats if c not in DEFAULT_CATEGORIES)
    return DEFAULT_CATEGORIES + extras


def build_report_lines(
    pairs: list[dict[str, Any]],
    train_total: int,
    examples_per_reason: int,
    max_chars: int,
) -> list[str]:
    stats = analyze_pairs(pairs, train_total)
    n_swap = stats["n_swap"]
    n_base = stats["n_base"]
    n_strict = stats["n_strict"]
    extra_rejected = stats["extra_rejected"]
    reason_cnt = stats["reason_cnt"]
    cat_stats = stats["cat_stats"]
    reason_by_cat = stats["reason_by_cat"]
    gate_versions = stats["gate_versions"]

    lines: list[str] = []

    def pr(s: str = "") -> None:
        lines.append(s)

    pr("=" * 65)
    pr("  [1] CF Construction Statistics")
    pr("=" * 65)
    col = 32
    pr(f'  {"Item":<{col}} Value')
    pr(f'  {"-" * col} --------')
    pr(f'  {"Train samples":<{col}} {train_total:,}')
    pr(f'  {"Swappable samples":<{col}} {n_swap:,}  ({pct(n_swap, train_total):.1f}% of train)')
    pr(f'  {"Base-valid pairs":<{col}} {n_base:,}  ({pct(n_base, n_swap):.1f}% of swappable)')
    pr(f'  {"Strict-valid pairs":<{col}} {n_strict:,}  ({pct(n_strict, n_swap):.1f}% of swappable)')
    pr(f'  {"Gate versions":<{col}} {dict(gate_versions)}')
    if "missing" in gate_versions:
        pr()
        pr("  WARNING: gate_version is missing. This JSONL may have been generated by an older gate;")
        pr("           rerun check_data.py or run_exp.py before using construction stats in the report.")

    pr()
    pr("  Strict gate additionally filters:")
    pr(
        f'  {"Rejected by strict (vs base)":<{col}} {n_base - n_strict:,}  '
        f'({pct(n_base - n_strict, n_base):.1f}% of base-valid)'
    )

    pr()
    pr("  Strict-only rejection breakdown (base-valid -> strict-rejected):")
    if extra_rejected:
        for reason, cnt in reason_cnt.most_common():
            pr(f"    {reason:<20}: {cnt:,}  ({pct(cnt, len(extra_rejected)):.1f}%)")
    else:
        pr("    none                : 0  (0.0%)")

    pr()
    pr("=" * 65)
    pr("  [2] Category-wise Validity Statistics")
    pr("=" * 65)
    pr(f'  {"Category":<12} {"Swappable":>10} {"Base-valid":>11} {"Strict-valid":>13} {"Base%":>7} {"Strict%":>8}')
    pr(f'  {"-" * 12} {"-" * 10} {"-" * 11} {"-" * 13} {"-" * 7} {"-" * 8}')
    for cat in ordered_categories(cat_stats):
        s = cat_stats.get(cat, {"swap": 0, "base": 0, "strict": 0})
        if s["swap"] == 0:
            pr(f"  {cat:<12} {'N/A':>10}")
            continue
        pr(
            f'  {cat:<12} {s["swap"]:>10,} {s["base"]:>11,} {s["strict"]:>13,} '
            f'{pct(s["base"], s["swap"]):>6.1f}% {pct(s["strict"], s["swap"]):>7.1f}%'
        )

    pr()
    pr("=" * 65)
    pr("  [3] Strict-only Rejection Matrix")
    pr("=" * 65)
    reasons_seen = [r for r in REASON_ORDER if r in reason_cnt]
    if not reasons_seen:
        reasons_seen = ["none"]
    header = f'  {"Category":<12}' + "".join(f" {reason[:12]:>12}" for reason in reasons_seen) + f" {'Total':>8}"
    pr(header)
    pr("  " + "-" * (len(header) - 2))
    for cat in ordered_categories(cat_stats):
        if cat_stats.get(cat, {}).get("swap", 0) == 0:
            continue
        total = sum(reason_by_cat.get(cat, Counter()).values())
        row = f"  {cat:<12}"
        for reason in reasons_seen:
            row += f" {reason_by_cat.get(cat, Counter()).get(reason, 0):>12,}"
        row += f" {total:>8,}"
        pr(row)

    pr()
    pr("=" * 65)
    pr("  [4] Qualitative Pass / Reject Examples")
    pr("=" * 65)
    pr()
    pr("  [PASS examples — strict-valid=True]")
    pr()
    shown_cats: set[str] = set()
    for p in pairs:
        if not p.get("strict_use_for_ccr"):
            continue
        cat = p.get("category", "unknown")
        if cat in shown_cats:
            continue
        shown_cats.add(cat)
        pr(f"  Category : {cat}")
        pr(f'  Original : {shorten(p.get("original", ""), max_chars)}')
        pr(f'  CF       : {shorten(p.get("cf", ""), max_chars)}')
        pr(f'  Label    : {p.get("label")}')
        pr()
        if len(shown_cats) >= 4:
            break

    pr("  [REJECT examples — strict-valid=False]")
    pr()
    shown_by_reason: Counter[str] = Counter()
    for p in extra_rejected:
        reason = get_reject_reason(p)
        if shown_by_reason[reason] >= examples_per_reason:
            continue
        shown_by_reason[reason] += 1
        pr(f'  Category : {p.get("category", "unknown")}')
        pr(f'  Original : {shorten(p.get("original", ""), max_chars)}')
        pr(f'  CF       : {shorten(p.get("cf", ""), max_chars)}')
        pr(f"  Reason   : {REASON_LABELS_READABLE.get(reason, reason)}")
        pr(f'  (base_valid={p.get("base_use_for_ccr")}, strict_valid={p.get("strict_use_for_ccr")})')
        pr()
        if all(shown_by_reason[r] >= examples_per_reason for r in reason_cnt):
            break

    pr("=" * 65)
    pr("  완료.")
    pr("=" * 65)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default="validity_gated_exp/data/cf_pairs_train.jsonl",
        help="cf_pairs_train.jsonl path",
    )
    parser.add_argument("--train_total", type=int, default=172157, help="train sample count")
    parser.add_argument("--out", default=None, help="write report text to this path")
    parser.add_argument("--examples_per_reason", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=120)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    pairs = load_pairs(jsonl_path)
    lines = build_report_lines(
        pairs,
        train_total=args.train_total,
        examples_per_reason=args.examples_per_reason,
        max_chars=args.max_chars,
    )
    print("\n".join(lines))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n결과 저장 -> {out_path}")


if __name__ == "__main__":
    main()
>>>>>>> main
