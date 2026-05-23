"""
eval_summary.py
  1. prob_gap 중심 3-seed 표
  2. Bootstrap 95% CI (flip_rate, 455 test pairs)
  3. val+test 합산 평가 (~910 pairs)
"""
import os, sys, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from collections import defaultdict
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

from dataset import load_khaters, find_swap, make_swap, compute_validity_strict

BASE_DIR   = 'validity_gated_exp'
CKPT_DIR   = os.path.join(BASE_DIR, 'checkpoints')
MODEL_NAME = 'klue/roberta-base'
MAX_LEN    = 128
BATCH_SIZE = 256
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPERIMENTS = {
    'Baseline':     'Baseline',
    'Naive Swap':   'Naive_Swap',
    'Strict-Gated': 'Strict-Gated',
}
SEEDS = [42, 123, 456]
N_BOOTSTRAP = 10000

# ── Model ──────────────────────────────────────────────────────────────────────
class HateDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(MODEL_NAME)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))


def load_model(ckpt_path):
    model = HateDetector().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    return model


def batch_infer(model, tokenizer, texts):
    probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(texts[i:i+BATCH_SIZE], max_length=MAX_LEN,
                        padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        probs.extend(F.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return probs


def eval_fairness_raw(model, tokenizer, examples):
    """per-example 결과 반환 (bootstrap용)"""
    meta = []
    for text, label, _ in examples:
        orig_term, swap_term, cat = find_swap(text)
        cf_text = make_swap(text, orig_term, swap_term) if orig_term else None
        meta.append((text, label, orig_term, swap_term, cat, cf_text))

    orig_probs = batch_infer(model, tokenizer, [m[0] for m in meta])
    swap_idx   = [i for i, m in enumerate(meta) if m[5] is not None]
    cf_map     = {}
    if swap_idx:
        cf_probs = batch_infer(model, tokenizer, [meta[i][5] for i in swap_idx])
        cf_map   = dict(zip(swap_idx, cf_probs))

    rows = []
    for i, (text, label, orig_term, swap_term, cat, cf_text) in enumerate(meta):
        prob    = orig_probs[i]
        pred    = int(prob >= 0.5)
        cf_prob = cf_map.get(i)
        cf_pred = int(cf_prob >= 0.5) if cf_prob is not None else None
        rows.append({'label': label, 'pred': pred, 'prob': prob,
                     'cf_pred': cf_pred, 'cf_prob': cf_prob, 'cat': cat,
                     'orig_term': orig_term, 'swap_term': swap_term,
                     'text': text, 'cf_text': cf_text})

    swap_rows   = [r for r in rows if r['cf_pred'] is not None]
    strict_rows = [r for r, m in zip(rows, meta)
                   if m[5] is not None and
                   compute_validity_strict(m[0], m[5], m[2], m[3], m[4])['use_for_ccr']]

    f1 = f1_score([r['label'] for r in rows], [r['pred'] for r in rows], average='macro')

    return f1, swap_rows, strict_rows


def aggregate(swap_rows, strict_rows):
    n_swap   = len(swap_rows)
    n_strict = len(strict_rows)
    flip     = sum(r['pred'] != r['cf_pred'] for r in swap_rows) / n_swap if n_swap else 0
    pgap     = sum(abs(r['prob'] - r['cf_prob']) for r in swap_rows) / n_swap if n_swap else 0
    sflip    = sum(r['pred'] != r['cf_pred'] for r in strict_rows) / n_strict if n_strict else 0
    spgap    = sum(abs(r['prob'] - r['cf_prob']) for r in strict_rows) / n_strict if n_strict else 0
    return flip, pgap, sflip, spgap, n_swap, n_strict


def bootstrap_ci(flip_arr, n_boot=N_BOOTSTRAP, alpha=0.05):
    """flip_arr: binary list (1=flipped, 0=not). returns (mean, lo, hi)"""
    arr  = np.array(flip_arr, dtype=float)
    mean = arr.mean()
    boot = np.array([np.random.choice(arr, len(arr), replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return mean, lo, hi


# ── Main ───────────────────────────────────────────────────────────────────────
print("K-HATERS 로딩...")
val_data  = load_khaters('validation', 0)
test_data = load_khaters('test', 0)
combined  = test_data + val_data
print(f"val={len(val_data)}  test={len(test_data)}  combined={len(combined)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

results = {}  # exp → {seed → {test, val, combined}}

for exp_name, ckpt_prefix in EXPERIMENTS.items():
    print(f"\n{'='*60}\n  {exp_name}\n{'='*60}")
    results[exp_name] = {}
    for seed in SEEDS:
        ckpt = os.path.join(CKPT_DIR, f'{ckpt_prefix}_seed{seed}.pt')
        if not os.path.exists(ckpt):
            print(f"  [SKIP] {ckpt} 없음")
            continue
        print(f"  seed={seed} 로딩...")
        model = load_model(ckpt)

        f1_t, sw_t, st_t = eval_fairness_raw(model, tokenizer, test_data)
        f1_v, sw_v, st_v = eval_fairness_raw(model, tokenizer, val_data)
        sw_c = sw_t + sw_v
        st_c = st_t + st_v

        results[exp_name][seed] = {
            'test':     (f1_t, sw_t, st_t),
            'val':      (f1_v, sw_v, st_v),
            'combined': (None, sw_c, st_c),
        }
        fl, pg, sfl, spg, ns, nst = aggregate(sw_t, st_t)
        print(f"    test  | n_swap={ns} n_strict={nst} | "
              f"flip={fl:.4f} prob_gap={pg:.4f} s_flip={sfl:.4f}")
        fl, pg, sfl, spg, ns, nst = aggregate(sw_v, st_v)
        print(f"    val   | n_swap={ns} n_strict={nst} | "
              f"flip={fl:.4f} prob_gap={pg:.4f} s_flip={sfl:.4f}")
        fl, pg, sfl, spg, ns, nst = aggregate(sw_c, st_c)
        print(f"    comb  | n_swap={ns} n_strict={nst} | "
              f"flip={fl:.4f} prob_gap={pg:.4f} s_flip={sfl:.4f}")

        del model; torch.cuda.empty_cache()


# ── 표 1: prob_gap 중심 3-seed 요약 ───────────────────────────────────────────
print("\n\n" + "="*100)
print("  표 1. prob_gap 중심  (test set, 3-seed mean ± std)")
print("="*100)
print(f"  {'Model':<16} {'F1':>14} {'Prob Gap':>14} {'S-Prob Gap':>14} "
      f"{'Flip Rate':>14} {'S-Flip':>14}  n_swap")
print("-"*100)

for exp_name in EXPERIMENTS:
    if exp_name not in results: continue
    f1s, pgs, spgs, flips, sflips = [], [], [], [], []
    n_swap_list = []
    for seed in SEEDS:
        if seed not in results[exp_name]: continue
        f1_t, sw_t, st_t = results[exp_name][seed]['test']
        fl, pg, sfl, spg, ns, nst = aggregate(sw_t, st_t)
        f1s.append(f1_t); pgs.append(pg); spgs.append(spg)
        flips.append(fl); sflips.append(sfl); n_swap_list.append(ns)

    def fm(lst): return f'{np.mean(lst):.4f}±{np.std(lst):.4f}' if len(lst)>1 else f'{lst[0]:.4f}'
    ns_str = f'{int(np.mean(n_swap_list))}'
    print(f"  {exp_name:<16}  {fm(f1s):>14}  {fm(pgs):>14}  {fm(spgs):>14}  "
          f"{fm(flips):>14}  {fm(sflips):>14}  {ns_str}")


# ── 표 2: val+test 합산 ────────────────────────────────────────────────────────
print("\n\n" + "="*100)
print("  표 2. val+test 합산  (3-seed mean ± std, ~910 pairs)")
print("="*100)
print(f"  {'Model':<16} {'Prob Gap':>14} {'S-Prob Gap':>14} "
      f"{'Flip Rate':>14} {'S-Flip':>14}  n_swap")
print("-"*100)

for exp_name in EXPERIMENTS:
    if exp_name not in results: continue
    pgs, spgs, flips, sflips, nslist = [], [], [], [], []
    for seed in SEEDS:
        if seed not in results[exp_name]: continue
        _, sw_c, st_c = results[exp_name][seed]['combined']
        fl, pg, sfl, spg, ns, nst = aggregate(sw_c, st_c)
        pgs.append(pg); spgs.append(spg); flips.append(fl); sflips.append(sfl)
        nslist.append(ns)

    def fm(lst): return f'{np.mean(lst):.4f}±{np.std(lst):.4f}' if len(lst)>1 else f'{lst[0]:.4f}'
    print(f"  {exp_name:<16}  {fm(pgs):>14}  {fm(spgs):>14}  "
          f"{fm(flips):>14}  {fm(sflips):>14}  {int(np.mean(nslist))}")


# ── Bootstrap CI ──────────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("  Bootstrap 95% CI  (flip_rate, test set, N_boot=10000)")
print("="*80)
print(f"  {'Model':<16} {'seed':>6}  {'mean':>8}  {'95% CI':>20}  n_flips/n_swap")
print("-"*80)

for exp_name in ['Naive Swap', 'Strict-Gated']:
    if exp_name not in results: continue
    all_flip_arr = []
    for seed in SEEDS:
        if seed not in results[exp_name]: continue
        _, sw_t, _ = results[exp_name][seed]['test']
        flip_arr = [int(r['pred'] != r['cf_pred']) for r in sw_t]
        mean, lo, hi = bootstrap_ci(flip_arr)
        n_flip = sum(flip_arr)
        print(f"  {exp_name:<16} {seed:>6}  {mean:.4f}   [{lo:.4f}, {hi:.4f}]"
              f"   {n_flip}/{len(flip_arr)}")
        all_flip_arr.extend(flip_arr)
    # pooled
    mean, lo, hi = bootstrap_ci(all_flip_arr)
    print(f"  {exp_name:<16} {'pooled':>6}  {mean:.4f}   [{lo:.4f}, {hi:.4f}]"
          f"   {sum(all_flip_arr)}/{len(all_flip_arr)}")
    print()

print("\n[해석] Naive vs Strict의 95% CI가 겹치면 → 통계적으로 유의미한 차이 없음")

# JSON 저장
out = {}
for exp_name in EXPERIMENTS:
    if exp_name not in results: continue
    out[exp_name] = {'test': {}, 'val': {}, 'combined': {}}
    for seed in SEEDS:
        if seed not in results[exp_name]: continue
        for split in ['test', 'val', 'combined']:
            f1_v, sw, st = results[exp_name][seed][split]
            fl, pg, sfl, spg, ns, nst = aggregate(sw, st)
            out[exp_name][split][str(seed)] = {
                'f1': f1_v, 'flip': fl, 'prob_gap': pg,
                'strict_flip': sfl, 'strict_prob_gap': spg,
                'n_swap': ns, 'n_strict': nst,
            }

with open(os.path.join(BASE_DIR, 'eval_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n결과 저장 → {BASE_DIR}/eval_summary.json")
