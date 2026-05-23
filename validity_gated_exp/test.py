"""
test.py — 빠른 sanity check용 test run.
λ=0.1 / seed 2개 / epoch 3 / λ sensitivity 없음.

Usage:
    cd c:/nlp_project/validity_gated_exp
    python test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── test 설정 override ────────────────────────────────────────────────────────
os.environ.setdefault('EXP_DIR', 'validity_gated_exp')

import run_exp
run_exp.SEEDS   = [42, 123]
run_exp.EPOCHS  = 3
run_exp.LAMBDA  = 0.1

# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
from dataset import load_khaters, load_cf_pairs, compute_validity_strict
from collections import Counter
import json, os

print('\n' + '='*60)
print('  TEST RUN  (λ=0.1, seeds=[42,123], epochs=3)')
print('='*60)

print('\n--- Loading K-HATERS ---')
raw_train = load_khaters('train',      0)
raw_val   = load_khaters('validation', 0)
raw_test  = load_khaters('test',       0)

pos_rate = sum(l for _, l, _ in raw_train) / len(raw_train)
print(f'train={len(raw_train)}  val={len(raw_val)}  test={len(raw_test)}')
print(f'train positive rate: {pos_rate:.3f}')

# label 분포 sanity check
raw_ds = __import__('datasets').load_dataset('humane-lab/K-HATERS', split='train')
label_dist = Counter(row['label'] for row in raw_ds)
print(f'\n[Sanity] K-HATERS train label distribution: {dict(label_dist)}')

# ── CF pair 저장 ──────────────────────────────────────────────────────────────
from dataset import find_swap, make_swap, compute_validity
BASE_DIR = os.environ.get('EXP_DIR', 'validity_gated_exp')
cf_path  = os.path.join(BASE_DIR, 'data', 'cf_pairs_train.jsonl')
os.makedirs(os.path.dirname(cf_path), exist_ok=True)

cat_cnt = Counter()
cf_pairs = []
for text, label, targets in raw_train:
    orig_term, swap_term, cat = find_swap(text)
    if orig_term is None:
        continue
    cat_cnt[cat] += 1
    cf_text  = make_swap(text, orig_term, swap_term)
    base_v   = compute_validity(text, cf_text, orig_term, swap_term, cat)
    strict_v = compute_validity_strict(text, cf_text, orig_term, swap_term, cat)
    cf_pairs.append({
        'original': text, 'cf': cf_text,
        'orig_term': orig_term, 'swap_term': swap_term,
        'category': cat, 'label': label, 'targets': targets,
        **{f'base_{k}': v for k, v in base_v.items()},
        **{f'strict_{k}': v for k, v in strict_v.items()},
    })
with open(cf_path, 'w', encoding='utf-8') as f:
    for p in cf_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
n_swap         = len(cf_pairs)
n_base_valid   = sum(1 for p in cf_pairs if p['base_use_for_ccr'])
n_strict_valid = sum(1 for p in cf_pairs if p['strict_use_for_ccr'])
print(f'\nswappable: {n_swap}/{len(raw_train)} ({100*n_swap/len(raw_train):.1f}%)')
print(f'base_valid={n_base_valid}  strict_valid={n_strict_valid}')
print('category distribution:')
for cat, cnt in cat_cnt.most_common():
    print(f'  {cat}: {cnt}')

cf_lookup = load_cf_pairs(cf_path)
print(f'CF lookup loaded: {len(cf_lookup)} entries')

run_exp.train_data = raw_train
run_exp.val_data   = raw_val
run_exp.test_data  = raw_test

# ── Ablation 실행 ─────────────────────────────────────────────────────────────
ABLATIONS = [
    dict(tag='Baseline',         mode='none',   use_cons=False, lam=0.0),
    dict(tag='Masking Cons Reg', mode='mask',   use_cons=True,  lam=run_exp.LAMBDA),
    dict(tag='Naive Swap',       mode='swap',   use_cons=True,  lam=run_exp.LAMBDA),
    dict(tag='Validity-Gated',   mode='gated',  use_cons=True,  lam=run_exp.LAMBDA),
    dict(tag='Strict-Gated',     mode='strict', use_cons=True,  lam=run_exp.LAMBDA),
]

import numpy as np
all_results = {}
for exp in ABLATIONS:
    print(f"\n{'#'*60}\n  Experiment: {exp['tag']}\n{'#'*60}")
    all_results[exp['tag']] = run_exp.run_experiment(**exp, cf_lookup=cf_lookup)

# ── 체크 목록 출력 ─────────────────────────────────────────────────────────────
def _fmt(lst): return f'{np.mean(lst):.4f}±{np.std(lst):.4f}' if lst else 'N/A'

print('\n' + '='*110)
print(f"  {'Model':<22} {'F1':>14} {'Flip Rate':>14} {'Prob Gap':>14} {'S-Flip':>14} {'S-ProbGap':>14} {'ConsLoss':>10}")
print('='*110)
for name, r in all_results.items():
    cl_list = []
    for seed_hist in r.get('epoch_history', []):
        last_ep = seed_hist['epochs'][-1]
        cl_list.append(last_ep.get('cons_loss', 0.0))
    cl_str = f'{np.mean(cl_list):.4f}' if cl_list else 'N/A'
    print(f"  {name:<22}  {_fmt(r['f1']):>14}  {_fmt(r['flip_rate']):>14}  "
          f"{_fmt(r['prob_gap']):>14}  {_fmt(r['strict_flip_rate']):>14}  "
          f"{_fmt(r['strict_prob_gap']):>14}  {cl_str:>10}")

print('\n[체크 목록]')
base = all_results.get('Baseline', {})
strict = all_results.get('Strict-Gated', {})

b_flip = np.mean(base.get('flip_rate', [0]))
s_flip = np.mean(strict.get('flip_rate', [0]))
b_f1   = np.mean(base.get('f1', [0]))
s_f1   = np.mean(strict.get('f1', [0]))

print(f'  Baseline cons_loss == 0    : ', end='')
bl_cons = [ep['cons_loss']
           for h in base.get('epoch_history', [])
           for ep in h['epochs']]
print('OK' if bl_cons and max(bl_cons) < 1e-6 else f'WARNING ({max(bl_cons) if bl_cons else "?"})')

print(f'  Strict-Gated cons_loss > 0 : ', end='')
sg_cons = [ep['cons_loss']
           for h in strict.get('epoch_history', [])
           for ep in h['epochs']]
print('OK' if sg_cons and max(sg_cons) > 1e-6 else 'WARNING (cons_loss stays 0 — cf_valid 확인 필요)')

print(f'  Flip Rate 방향 (Strict < Baseline): ', end='')
print(f'OK ({b_flip:.4f} → {s_flip:.4f})' if s_flip < b_flip else f'확인 필요 ({b_flip:.4f} → {s_flip:.4f})')

print(f'  F1 유지 여부 (drop < 0.02): ', end='')
drop = b_f1 - s_f1
print(f'OK (drop={drop:.4f})' if drop < 0.02 else f'확인 필요 (drop={drop:.4f})')

# 결과 저장
out_path = os.path.join(BASE_DIR, 'results_test.json')
import json as _json
with open(out_path, 'w', encoding='utf-8') as f:
    _json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f'\nTest results saved → {out_path}')
