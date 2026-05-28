"""
run_strict_only.py — Strict-Gated 단독 실행 (ipynb Baseline과 동일 조건)

LR=3e-5, BATCH_SIZE=256, fp16, num_workers=4, warmup 6%+linear decay

Usage:
    python validity_gated_exp/run_strict_only.py
    python validity_gated_exp/run_strict_only.py --cf_path /path/to/cf_pairs.jsonl
"""
import os, sys, json, random, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings('ignore')

from dataset import (
    find_swap, make_swap,
    compute_validity_strict,
    load_khaters, load_cf_pairs, HatersDataset,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--cf_path',     default=None)
parser.add_argument('--result_path', default=None)
parser.add_argument('--ckpt_dir',    default=None)
args = parser.parse_args()

# ── Config (ipynb Baseline과 동일) ────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME   = 'klue/roberta-base'
MAX_LEN      = 128
BATCH_SIZE   = 256
EPOCHS       = 3
LR           = 3e-5
WEIGHT_DECAY = 0.01
LAMBDA       = 0.1
SEEDS        = [42, 123, 456]

CF_PATH     = args.cf_path     or os.path.join(BASE_DIR, 'data', 'cf_pairs_train_colab.jsonl')
RESULT_PATH = args.result_path or os.path.join(BASE_DIR, 'data', 'results_strict.json')
CKPT_DIR    = args.ckpt_dir    or os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
print(f'CF_PATH: {CF_PATH}')
print(f'RESULT : {RESULT_PATH}')

# ── Seed ──────────────────────────────────────────────────────────────────────
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

# ── Train ─────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, lam: float):
    model.train()
    s_total = s_cls = s_cons = 0.0
    for batch in tqdm(loader, desc='  train', leave=False):
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        y     = batch['label'].to(device)
        valid = batch['cf_valid'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits   = model(ids, mask)
            cls_loss = F.cross_entropy(logits, y)
            loss     = cls_loss
            c_val    = torch.tensor(0.0, device=device)

            if 'cf_input_ids' in batch and valid.any():
                cf_ids  = batch['cf_input_ids'].to(device)
                cf_mask = batch['cf_attention_mask'].to(device)
                p_o = model.probs(ids[valid],    mask[valid])
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

# ── Eval ──────────────────────────────────────────────────────────────────────
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


def eval_fairness(model, test_examples, tokenizer):
    model.eval()
    meta = []
    for text, label, _ in test_examples:
        orig_term, swap_term, cat = find_swap(text)
        cf_text = make_swap(text, orig_term, swap_term) if orig_term else None
        meta.append((text, label, orig_term, swap_term, cat, cf_text))

    def batch_infer(texts):
        probs_all = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(texts[i:i+BATCH_SIZE], max_length=MAX_LEN,
                            padding='max_length', truncation=True, return_tensors='pt')
            with torch.no_grad():
                logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            probs_all.extend(F.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        return probs_all

    orig_probs   = batch_infer([m[0] for m in meta])
    swap_indices = [i for i, m in enumerate(meta) if m[5] is not None]
    cf_probs_map: dict = {}
    if swap_indices:
        cf_probs_list = batch_infer([meta[i][5] for i in swap_indices])
        cf_probs_map  = dict(zip(swap_indices, cf_probs_list))

    results = []
    for i, (text, label, orig_term, swap_term, cat, cf_text) in enumerate(meta):
        prob    = orig_probs[i]
        pred    = int(prob >= 0.5)
        cf_prob = cf_probs_map.get(i)
        cf_pred = int(cf_prob >= 0.5) if cf_prob is not None else None
        results.append({'label': label, 'pred': pred, 'prob': prob,
                        'cf_pred': cf_pred, 'cf_prob': cf_prob, 'cat': cat})

    swap_res      = [r for r in results if r['cf_pred'] is not None]
    flip_rate     = sum(r['pred'] != r['cf_pred'] for r in swap_res) / len(swap_res) if swap_res else 0.0
    mean_prob_gap = sum(abs(r['prob'] - r['cf_prob']) for r in swap_res) / len(swap_res) if swap_res else 0.0

    strict_res = [
        r for r, m in zip(results, meta)
        if m[5] is not None and compute_validity_strict(m[0], m[5], m[2], m[3], m[4])['use_for_ccr']
    ]
    strict_flip_rate = sum(r['pred'] != r['cf_pred'] for r in strict_res) / len(strict_res) if strict_res else 0.0
    strict_prob_gap  = sum(abs(r['prob'] - r['cf_prob']) for r in strict_res) / len(strict_res) if strict_res else 0.0

    group_fp, group_tn = defaultdict(int), defaultdict(int)
    for r in results:
        if r['label'] == 0:
            grp = r['cat'] if r['cat'] else 'none'
            (group_fp if r['pred'] == 1 else group_tn)[grp] += 1
    per_group_fpr = {}
    for grp in set(list(group_fp) + list(group_tn)):
        d = group_fp[grp] + group_tn[grp]
        per_group_fpr[grp] = group_fp[grp] / d if d else 0.0
    id_fprs  = {k: v for k, v in per_group_fpr.items() if k != 'none'}
    fpr_vals = list(id_fprs.values())
    fpr_gap  = (max(fpr_vals) - min(fpr_vals)) if len(fpr_vals) >= 2 else 0.0

    return flip_rate, mean_prob_gap, strict_flip_rate, strict_prob_gap, fpr_gap, per_group_fpr

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Strict-Gated 단독 실행  (LR=3e-5, fp16, num_workers=4)')
    print('='*60)

    print('\nK-HATERS 로딩...')
    train_data = load_khaters('train',      0)
    val_data   = load_khaters('validation', 0)
    test_data  = load_khaters('test',       0)
    print(f'train={len(train_data)}  val={len(val_data)}  test={len(test_data)}')

    cf_lookup = load_cf_pairs(CF_PATH)
    print(f'CF lookup: {len(cf_lookup)} entries')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    metrics = {
        'f1': [], 'flip_rate': [], 'prob_gap': [],
        'strict_flip_rate': [], 'strict_prob_gap': [],
        'fpr_gap': [], 'epoch_history': [],
    }

    va_dl = DataLoader(
        HatersDataset(val_data, tokenizer, MAX_LEN, mode='none'),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=torch.cuda.is_available())

    for seed in SEEDS:
        print(f'\n[Strict-Gated] seed={seed}  lam={LAMBDA}')
        set_seed(seed)

        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)
            random.seed(seed + worker_id)

        g = torch.Generator()
        g.manual_seed(seed)

        tr_dl = DataLoader(
            HatersDataset(train_data, tokenizer, MAX_LEN, mode='strict', cf_lookup=cf_lookup),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn, generator=g)

        model        = HateDetector(MODEL_NAME).to(device)
        opt          = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps  = len(tr_dl) * EPOCHS
        warmup_steps = max(1, int(0.06 * total_steps))
        scheduler    = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        scaler       = torch.cuda.amp.GradScaler()

        best_f1, best_state, seed_epochs = 0.0, {}, []
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for ep in range(1, EPOCHS + 1):
            tl, cl, cons = train_epoch(model, tr_dl, opt, scheduler, scaler, LAMBDA)
            vf1 = eval_f1(model, va_dl)
            print(f'  ep{ep}: total={tl:.4f} cls={cl:.4f} cons={cons:.4f} | val_F1={vf1:.4f}')
            seed_epochs.append({'ep': ep, 'val_f1': round(vf1, 6),
                                 'total_loss': round(tl, 6), 'cls_loss': round(cl, 6),
                                 'cons_loss': round(cons, 6)})
            if vf1 > best_f1:
                best_f1    = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(CKPT_DIR, f'Strict-Gated_seed{seed}.pt'))

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_f1 = eval_f1(model, DataLoader(
            HatersDataset(test_data, tokenizer, MAX_LEN, mode='none'),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4))
        flip, lgap, sflip, sgap, fgap, per_grp = eval_fairness(model, test_data, tokenizer)

        print(f'  test F1={test_f1:.4f}  flip={flip:.4f}  prob_gap={lgap:.4f}  '
              f'strict_flip={sflip:.4f}  strict_prob_gap={sgap:.4f}')

        metrics['f1'].append(test_f1)
        metrics['flip_rate'].append(flip)
        metrics['prob_gap'].append(lgap)
        metrics['strict_flip_rate'].append(sflip)
        metrics['strict_prob_gap'].append(sgap)
        metrics['fpr_gap'].append(fgap)
        metrics['epoch_history'].append({'seed': seed, 'epochs': seed_epochs})

        del model; gc.collect(); torch.cuda.empty_cache()

    def _s(lst): return f'{np.mean(lst):.4f}±{np.std(lst):.4f}' if lst else 'N/A'
    print(f'\n{"="*60}')
    print(f'  [Strict-Gated]  3-seed summary')
    print(f'  Macro-F1           : {_s(metrics["f1"])}')
    print(f'  Flip Rate ↓        : {_s(metrics["flip_rate"])}')
    print(f'  Prob Gap ↓         : {_s(metrics["prob_gap"])}')
    print(f'  Strict Flip Rate ↓ : {_s(metrics["strict_flip_rate"])}')
    print(f'  Strict Prob Gap ↓  : {_s(metrics["strict_prob_gap"])}')
    print(f'{"="*60}')

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'Strict-Gated': metrics}, f, ensure_ascii=False, indent=2)
    print(f'\n결과 저장 → {RESULT_PATH}')
