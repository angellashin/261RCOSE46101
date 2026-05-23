"""
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
