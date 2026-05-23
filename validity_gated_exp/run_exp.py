"""
Validity-Gated Counterfactual Consistency Regularization
for Korean Abusive/Offensive Language Detection

Ablation 4종:
  Baseline          : L_cls only
  Masking Cons Reg  : L_cls + KL(orig || [MASK])
  Naive Swap        : L_cls + KL(orig || swap, no gate)
  Validity-Gated    : L_cls + KL(orig || swap, same-category gate)
  Strict-Gated      : L_cls + KL(orig || swap, strict validity gate)
"""

import os, sys, json, random, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from scipy import stats
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

from dataset import (
    SWAP_PAIRS_BY_CAT, SWAP_MAP, SWAP_KEYS, kiwi,
    find_swap, find_swap_naive, make_swap, make_swap_naive,
    compute_validity, compute_validity_strict,
    load_khaters, save_cf_pairs, load_cf_pairs, HatersDataset,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = 'validity_gated_exp'   # relative; override with env var EXP_DIR
MODEL_NAME  = 'klue/roberta-base'
MAX_LEN     = 128
BATCH_SIZE  = 256    # L4 24GB: 64 OK
EPOCHS      = 3
LR          = 3e-5
WEIGHT_DECAY= 0.01
LAMBDA      = 0.1
SEEDS       = [42, 123, 456]
SUBSET      = 0      # 0 = full 172K

BASE_DIR = os.environ.get('EXP_DIR', BASE_DIR)

CKPT_DIR    = os.path.join(BASE_DIR, 'checkpoints')
RESULT_PATH = os.path.join(BASE_DIR, 'results_final.json')
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
print(f'Model  : {MODEL_NAME}')
print(f'Swap terms: {len(SWAP_KEYS)}개 ({len(SWAP_PAIRS_BY_CAT)} categories)')

# ── Seed ─────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Model ─────────────────────────────────────────────────────────────────────
class HateDetector(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))

    def probs(self, input_ids, attention_mask):
        return F.softmax(self.forward(input_ids, attention_mask), dim=-1)


# ── Loss ──────────────────────────────────────────────────────────────────────
def sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp(min=1e-8)
    q = q.clamp(min=1e-8)
    return (F.kl_div(q.log(), p, reduction='batchmean') +
            F.kl_div(p.log(), q, reduction='batchmean')) / 2


# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, use_cons: bool, lam: float):
    model.train()
    s_total = s_cls = s_cons = 0.0
    for batch in tqdm(loader, desc='  train', leave=False):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y    = batch['label'].to(device)
        valid = batch['cf_valid'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits   = model(ids, mask)
            cls_loss = F.cross_entropy(logits, y)
            loss     = cls_loss
            c_val    = torch.tensor(0.0, device=device)

            if use_cons and 'cf_input_ids' in batch and valid.any():
                cf_ids  = batch['cf_input_ids'].to(device)
                cf_mask = batch['cf_attention_mask'].to(device)
                p_o = model.probs(ids[valid],   mask[valid])
                p_c = model.probs(cf_ids[valid], cf_mask[valid])
                c_val = sym_kl(p_o, p_c)
                loss  = loss + lam * c_val

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        s_total += loss.item(); s_cls += cls_loss.item(); s_cons += c_val.item()

    n = len(loader)
    return s_total / n, s_cls / n, s_cons / n


def eval_f1(model, loader) -> float:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval', leave=False):
            logits = model(batch['input_ids'].to(device),
                           batch['attention_mask'].to(device))
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(batch['label'].tolist())
    return f1_score(labels, preds, average='macro')


# ── Fairness eval on test set ─────────────────────────────────────────────────
def eval_fairness(model, test_examples, tokenizer):
    """
    Returns:
      flip_rate, mean_prob_gap,
      fpr_gap (max FPR across target groups - min FPR),
      per_group_fpr dict
    """
    model.eval()

    # Pre-compute swap info and CF texts once (avoids kiwi calls inside loop)
    meta = []
    for text, label, _ in test_examples:
        orig_term, swap_term, cat = find_swap(text)
        cf_text = make_swap(text, orig_term, swap_term) if orig_term else None
        meta.append((text, label, orig_term, swap_term, cat, cf_text))

    # Batched inference: original texts
    def batch_infer(texts):
        probs_all = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch_texts, max_length=MAX_LEN, padding='max_length',
                            truncation=True, return_tensors='pt')
            with torch.no_grad():
                logits = model(enc['input_ids'].to(device),
                               enc['attention_mask'].to(device))
            probs_all.extend(F.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        return probs_all

    orig_probs = batch_infer([m[0] for m in meta])

    # CF inference only for swappable examples (reuse their index)
    swap_indices = [i for i, m in enumerate(meta) if m[5] is not None]
    cf_probs_map: dict[int, float] = {}
    if swap_indices:
        cf_texts = [meta[i][5] for i in swap_indices]
        cf_probs_list = batch_infer(cf_texts)
        cf_probs_map = dict(zip(swap_indices, cf_probs_list))

    results = []
    for i, (text, label, orig_term, swap_term, cat, cf_text) in enumerate(meta):
        prob = orig_probs[i]
        pred = int(prob >= 0.5)
        cf_prob = cf_probs_map.get(i)
        cf_pred = int(cf_prob >= 0.5) if cf_prob is not None else None
        results.append({
            'label': label, 'pred': pred, 'prob': prob,
            'cf_pred': cf_pred, 'cf_prob': cf_prob,
            'cat': cat,
        })

    # Flip rate & prob gap (all swappable pairs)
    swap_res = [r for r in results if r['cf_pred'] is not None]
    flip_rate = (sum(r['pred'] != r['cf_pred'] for r in swap_res) / len(swap_res)
                 if swap_res else 0.0)
    mean_prob_gap = (sum(abs(r['prob'] - r['cf_prob']) for r in swap_res) / len(swap_res)
                     if swap_res else 0.0)

    # Strict-valid subset: label-preserving pair만
    strict_res = [
        r for r, m in zip(results, meta)
        if m[5] is not None and compute_validity_strict(
            m[0], m[5], m[2], m[3], m[4])['use_for_ccr']
    ]
    strict_flip_rate = (sum(r['pred'] != r['cf_pred'] for r in strict_res) / len(strict_res)
                        if strict_res else 0.0)
    strict_prob_gap  = (sum(abs(r['prob'] - r['cf_prob']) for r in strict_res) / len(strict_res)
                        if strict_res else 0.0)

    # Per-CATEGORY FPR using lexicon-based group assignment
    # K-HATERS의 target_label은 label=1에만 존재 → FPR 계산에 사용 불가
    # 대신: 문장 내 identity term이 속한 카테고리 기준으로 그룹 분류
    # FPR = P(predict hate | actually normal), 그룹 = 언급된 identity 카테고리
    group_fp = defaultdict(int)
    group_tn = defaultdict(int)
    for r in results:
        if r['label'] == 0:   # normal 예제만 FPR 대상
            grp = r['cat'] if r['cat'] else 'none'
            if r['pred'] == 1:
                group_fp[grp] += 1
            else:
                group_tn[grp] += 1

    per_group_fpr = {}
    for grp in set(list(group_fp.keys()) + list(group_tn.keys())):
        denom = group_fp[grp] + group_tn[grp]
        per_group_fpr[grp] = group_fp[grp] / denom if denom else 0.0

    # FPR gap: identity 그룹들 사이의 격차 ('none' 제외)
    identity_fprs = {k: v for k, v in per_group_fpr.items() if k != 'none'}
    fpr_vals = list(identity_fprs.values())
    fpr_gap  = (max(fpr_vals) - min(fpr_vals)) if len(fpr_vals) >= 2 else 0.0

    return flip_rate, mean_prob_gap, strict_flip_rate, strict_prob_gap, fpr_gap, per_group_fpr


# ── Experiment runner ─────────────────────────────────────────────────────────
def run_experiment(tag: str, mode: str, use_cons: bool, lam: float = LAMBDA,
                   seeds=None, n_epochs: int = EPOCHS, cf_lookup: dict | None = None):
    if seeds is None:
        seeds = SEEDS

    metrics = {
        'f1': [], 'flip_rate': [], 'prob_gap': [],
        'strict_flip_rate': [], 'strict_prob_gap': [],
        'fpr_gap': [],
        'epoch_history': [],   # [{seed, epochs: [{ep, val_f1, total_loss, cls_loss, cons_loss}]}]
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    va_ds = HatersDataset(val_data, tokenizer, MAX_LEN, mode='none')
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=torch.cuda.is_available())

    for seed in seeds:
        print(f'\n[{tag}] seed={seed}  lam={lam}')
        set_seed(seed)

        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)
            random.seed(seed + worker_id)

        g = torch.Generator()
        g.manual_seed(seed)

        tr_ds = HatersDataset(train_data, tokenizer, MAX_LEN, mode=mode,
                              cf_lookup=cf_lookup)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=torch.cuda.is_available(),
                           worker_init_fn=worker_init_fn, generator=g)

        model = HateDetector(MODEL_NAME).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
        total_steps  = len(tr_dl) * n_epochs
        warmup_steps = max(1, int(0.06 * total_steps))
        scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        scaler = torch.cuda.amp.GradScaler()

        best_f1 = 0.0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        seed_epochs = []
        for ep in range(1, n_epochs + 1):
            tl, cl, cons = train_epoch(model, tr_dl, opt, scheduler, scaler, use_cons, lam)
            vf1 = eval_f1(model, va_dl)
            print(f'  ep{ep}: total={tl:.4f} cls={cl:.4f} cons={cons:.4f} | val_F1={vf1:.4f}')
            seed_epochs.append({
                'ep': ep, 'val_f1': round(vf1, 6),
                'total_loss': round(tl, 6), 'cls_loss': round(cl, 6), 'cons_loss': round(cons, 6),
            })
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state,
                           os.path.join(CKPT_DIR,
                                        f"{tag.replace(' ', '_')}_seed{seed}.pt"))

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_f1 = eval_f1(model,
                          DataLoader(HatersDataset(test_data, tokenizer, MAX_LEN,
                                                   mode='none'),
                                     batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=4))
        flip, lgap, sflip, sgap, fpr_gap, per_grp = eval_fairness(model, test_data, tokenizer)

        print(f'  test F1={test_f1:.4f}  flip={flip:.4f}  prob_gap={lgap:.4f}  '
              f'strict_flip={sflip:.4f}  strict_prob_gap={sgap:.4f}  fpr_gap={fpr_gap:.4f}')
        print(f'  per-group FPR: ' +
              '  '.join(f'{k}={v:.3f}' for k, v in sorted(per_grp.items())))

        metrics['f1'].append(test_f1)
        metrics['flip_rate'].append(flip)
        metrics['prob_gap'].append(lgap)
        metrics['strict_flip_rate'].append(sflip)
        metrics['strict_prob_gap'].append(sgap)
        metrics['fpr_gap'].append(fpr_gap)
        metrics['epoch_history'].append({'seed': seed, 'epochs': seed_epochs})

        del model; gc.collect(); torch.cuda.empty_cache()

    def _s(lst): return f'{np.mean(lst):.4f}±{np.std(lst):.4f}' if lst else 'N/A'
    print(f'\n{"="*60}')
    print(f'  [{tag}]  {len(seeds)}-seed summary')
    print(f'  Test Macro-F1      : {_s(metrics["f1"])}')
    print(f'  Flip Rate ↓        : {_s(metrics["flip_rate"])}')
    print(f'  Prob Gap ↓         : {_s(metrics["prob_gap"])}')
    print(f'  Strict Flip Rate ↓ : {_s(metrics["strict_flip_rate"])}')
    print(f'  Strict Prob Gap ↓  : {_s(metrics["strict_prob_gap"])}')
    print(f'  FPR Gap ↓          : {_s(metrics["fpr_gap"])}')
    print(f'{"="*60}')
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',   nargs='+', default=None,
                        help='실행할 실험 tag (예: Strict-Gated "Naive Swap")')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='사용할 seed 목록 (예: 42 123)')
    args = parser.parse_args()
    if args.seeds:
        SEEDS = args.seeds

    print('\n--- Loading K-HATERS ---')
    raw_train = load_khaters('train',      SUBSET)
    raw_val   = load_khaters('validation', 0)
    raw_test  = load_khaters('test',       0)

    # stats
    hate_rate = sum(l for _, l, _ in raw_train) / len(raw_train)
    print(f'train={len(raw_train)}  val={len(raw_val)}  test={len(raw_test)}')
    print(f'train hate rate: {hate_rate:.3f}')

    # single pass: swappable ratio + category distribution + CF pair saving
    cf_path = os.path.join(BASE_DIR, 'data', 'cf_pairs_train.jsonl')
    os.makedirs(os.path.dirname(cf_path), exist_ok=True)
    cat_cnt: Counter = Counter()
    cf_pairs = []
    for text, label, targets in raw_train:
        orig_term, swap_term, cat = find_swap(text)
        if orig_term is None:
            continue
        cat_cnt[cat] += 1
        cf_text        = make_swap(text, orig_term, swap_term)
        base_v         = compute_validity(text, cf_text, orig_term, swap_term, cat)
        strict_v       = compute_validity_strict(text, cf_text, orig_term, swap_term, cat)
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
    n_swap        = len(cf_pairs)
    n_base_valid  = sum(1 for p in cf_pairs if p['base_use_for_ccr'])
    n_strict_valid= sum(1 for p in cf_pairs if p['strict_use_for_ccr'])
    print(f'swappable train samples: {n_swap} / {len(raw_train)} '
          f'({100*n_swap/len(raw_train):.1f}%)')
    print(f'CF pairs saved → {cf_path}  '
          f'(total={n_swap}, base_valid={n_base_valid}, strict_valid={n_strict_valid})')
    print('swap category distribution (train):')
    for cat, cnt in cat_cnt.most_common():
        print(f'  {cat}: {cnt}')

    train_data, val_data, test_data = raw_train, raw_val, raw_test

    # pre-computed CF pairs 로딩 (있으면 kiwi find_swap+make_swap 생략)
    cf_lookup = None
    if os.path.exists(cf_path):
        cf_lookup = load_cf_pairs(cf_path)
        print(f'Pre-computed CF pairs loaded: {len(cf_lookup)} entries → kiwi skipped for swap/gated/strict')

    ABLATIONS = [
        dict(tag='Baseline',        mode='none',   use_cons=False, lam=0.0),
        dict(tag='Masking Cons Reg',mode='mask',   use_cons=True,  lam=LAMBDA),
        dict(tag='Naive Swap',      mode='swap',   use_cons=True,  lam=LAMBDA),
        dict(tag='Validity-Gated',  mode='gated',  use_cons=True,  lam=LAMBDA),
        dict(tag='Strict-Gated',    mode='strict', use_cons=True,  lam=LAMBDA),
    ]

    # --exp 인자로 특정 실험만 선택
    run_ablations = ABLATIONS
    lam_targets = [0.05, 0.2]
    if args.exp:
        run_ablations = [e for e in ABLATIONS if e['tag'] in args.exp]
        lam_targets   = [l for l in lam_targets if f'Strict_lam={l}' in args.exp]

    all_results = {}
    for exp in run_ablations:
        print(f"\n{'#'*60}\n  Experiment: {exp['tag']}\n{'#'*60}")
        all_results[exp['tag']] = run_experiment(**exp, cf_lookup=cf_lookup)

    # λ sensitivity (Strict-Gated; lam=0.1 is already in ABLATIONS)
    for lam in lam_targets:
        key = f'Strict_lam={lam}'
        all_results[key] = run_experiment(
            tag=key, mode='strict', use_cons=True, lam=lam, n_epochs=3,
            cf_lookup=cf_lookup)

    # Summary table
    def _fmt(lst): return f'{np.mean(lst):.4f}±{np.std(lst):.4f}' if lst else 'N/A'
    print('\n' + '=' * 110)
    print(f"  {'Model':<22} {'F1':>14} {'Flip Rate':>14} {'Prob Gap':>14} {'S-Flip Rate':>14} {'S-Prob Gap':>14}")
    print('=' * 110)
    for name, r in all_results.items():
        print(f"  {name:<22}  {_fmt(r['f1']):>14}  {_fmt(r['flip_rate']):>14}  "
              f"{_fmt(r['prob_gap']):>14}  {_fmt(r['strict_flip_rate']):>14}  "
              f"{_fmt(r['strict_prob_gap']):>14}")

    # Paired t-test: Baseline vs Strict-Gated (flip rate)
    for target in ['Strict-Gated', 'Validity-Gated', 'Naive Swap', 'Masking Cons Reg']:
        if 'Baseline' in all_results and target in all_results:
            b = all_results['Baseline']['flip_rate']
            g = all_results[target]['flip_rate']
            if len(b) == len(g) and len(b) > 1:
                t, p = stats.ttest_rel(b, g)
                print(f'\n  [t-test] Baseline vs {target} flip_rate  '
                      f't={t:.4f}  p={p:.4f}  {"*significant*" if p < 0.05 else "n.s."} (α=0.05)'
                      f'  (n={len(b)}, mean±std: {np.mean(b):.4f}±{np.std(b):.4f} → '
                      f'{np.mean(g):.4f}±{np.std(g):.4f})')

    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f'\nResults saved → {RESULT_PATH}')

    import csv, statistics
    csv_path = RESULT_PATH.replace('.json', '.csv')
    rows = []
    for exp_tag, metrics in all_results.items():
        if not isinstance(metrics, dict) or 'f1' not in metrics:
            continue
        def _m(vals): return round(statistics.mean(vals), 4) if vals else ''
        def _s(vals): return round(statistics.stdev(vals), 4) if len(vals) > 1 else ''
        rows.append({
            'experiment': exp_tag,
            'f1_mean': _m(metrics['f1']), 'f1_std': _s(metrics['f1']),
            'flip_rate_mean': _m(metrics['flip_rate']), 'flip_rate_std': _s(metrics['flip_rate']),
            'prob_gap_mean': _m(metrics['prob_gap']), 'prob_gap_std': _s(metrics['prob_gap']),
            'strict_flip_mean': _m(metrics['strict_flip_rate']), 'strict_flip_std': _s(metrics['strict_flip_rate']),
            'fpr_gap_mean': _m(metrics['fpr_gap']), 'fpr_gap_std': _s(metrics['fpr_gap']),
        })
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'CSV saved → {csv_path}')
