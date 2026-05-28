"""
check_data.py — K-HATERS 로딩 + CF pair 생성 확인용 스크립트.
실험 전 데이터셋이 정상적으로 만들어지는지 확인한다.

Usage:
    cd c:/nlp_project
    python validity_gated_exp/check_data.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from datasets import load_dataset
from dataset import (
<<<<<<< HEAD
=======
    GATE_VERSION,
>>>>>>> main
    load_khaters, to_binary,
    find_swap, make_swap,
    compute_validity, compute_validity_strict,
    load_cf_pairs,
)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CF_PATH  = os.path.join(BASE_DIR, 'data', 'cf_pairs_train.jsonl')
os.makedirs(os.path.dirname(CF_PATH), exist_ok=True)

# ── 1. 원본 label 분포 확인 ───────────────────────────────────────────────────
print('='*60)
print('  [1] K-HATERS 원본 label 분포')
print('='*60)
raw_ds = load_dataset('humane-lab/K-HATERS', split='train')
raw_label_dist = Counter(row['label'] for row in raw_ds)
print(f'  원본 labels: {dict(raw_label_dist)}')
for lbl, cnt in raw_label_dist.most_common():
    binary = to_binary(lbl)
    print(f'    {lbl!r:<15} → binary={binary}  (n={cnt})')

# ── 2. binarize 후 분포 ───────────────────────────────────────────────────────
print()
print('='*60)
print('  [2] binarize 후 split별 분포')
print('='*60)
for split in ('train', 'validation', 'test'):
    examples = load_khaters(split, subset=0)
    pos = sum(l for _, l, _ in examples)
    neg = len(examples) - pos
    print(f'  {split:<12}: total={len(examples)}  pos={pos} ({100*pos/len(examples):.1f}%)  neg={neg}')

# ── 3. CF pair 생성 (train) ───────────────────────────────────────────────────
print()
print('='*60)
print('  [3] CF pair 생성 (train)')
print('='*60)
raw_train = load_khaters('train', subset=0)
cat_cnt   = Counter()
cf_pairs  = []

for text, label, targets in raw_train:
    orig_term, swap_term, cat = find_swap(text)
    if orig_term is None:
        continue
    cat_cnt[cat] += 1
    cf_text  = make_swap(text, orig_term, swap_term)
    base_v   = compute_validity(text, cf_text, orig_term, swap_term, cat)
    strict_v = compute_validity_strict(text, cf_text, orig_term, swap_term, cat)
    cf_pairs.append({
        'original':  text,
        'cf':        cf_text,
<<<<<<< HEAD
=======
        'gate_version': GATE_VERSION,
>>>>>>> main
        'orig_term': orig_term,
        'swap_term': swap_term,
        'category':  cat,
        'label':     label,
        'targets':   targets,
        **{f'base_{k}': v for k, v in base_v.items()},
        **{f'strict_{k}': v for k, v in strict_v.items()},
    })

n_swap         = len(cf_pairs)
n_base_valid   = sum(1 for p in cf_pairs if p['base_use_for_ccr'])
n_strict_valid = sum(1 for p in cf_pairs if p['strict_use_for_ccr'])

print(f'  swappable   : {n_swap} / {len(raw_train)} ({100*n_swap/len(raw_train):.1f}%)')
print(f'  base_valid  : {n_base_valid} / {n_swap} ({100*n_base_valid/n_swap:.1f}%)')
print(f'  strict_valid: {n_strict_valid} / {n_swap} ({100*n_strict_valid/n_swap:.1f}%)')
print()
print('  category distribution:')
for cat, cnt in cat_cnt.most_common():
    print(f'    {cat:<12}: {cnt}')

# ── 4. JSONL 저장 & 재로딩 확인 ──────────────────────────────────────────────
print()
print('='*60)
print('  [4] JSONL 저장 및 재로딩 확인')
print('='*60)
with open(CF_PATH, 'w', encoding='utf-8') as f:
    for p in cf_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
print(f'  저장 완료 → {CF_PATH}')

cf_lookup = load_cf_pairs(CF_PATH)
print(f'  재로딩    : {len(cf_lookup)} entries')

# ── 5. 샘플 출력 ──────────────────────────────────────────────────────────────
print()
print('='*60)
print('  [5] 샘플 5개 (base_valid=True 우선)')
print('='*60)
samples = [p for p in cf_pairs if p['base_use_for_ccr']][:5]
for i, p in enumerate(samples, 1):
    print(f'  [{i}] cat={p["category"]}  label={p["label"]}')
    print(f'       orig  : {p["original"][:60]}')
    print(f'       cf    : {p["cf"][:60]}')
    print(f'       base_valid={p["base_use_for_ccr"]}  strict_valid={p["strict_use_for_ccr"]}')

print()
print('='*60)
print('  완료. 위 숫자가 모두 자연스러우면 실험 진행 가능.')
print('='*60)
