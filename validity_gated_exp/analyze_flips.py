"""
Naive Swap vs Strict-Gated flip 방향 분석 + 예시 추출
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from dataset import load_khaters, find_swap, make_swap, compute_validity_strict

BASE_DIR   = 'validity_gated_exp'
MODEL_NAME = 'klue/roberta-base'
MAX_LEN    = 128
BATCH_SIZE = 256
SEED       = 42
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_NAIVE  = os.path.join(BASE_DIR, 'checkpoints', f'Naive_Swap_seed{SEED}.pt')
CKPT_STRICT = os.path.join(BASE_DIR, 'checkpoints', f'Strict-Gated_seed{SEED}.pt')


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


def load_model(ckpt_path):
    model = HateDetector(MODEL_NAME).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    return model


def batch_infer(model, tokenizer, texts):
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


def main():
    print(f'Device: {device}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print('Loading test data...')
    raw_test = load_khaters('test')
    print(f'test={len(raw_test)}')

    # 스왑 가능한 테스트 샘플만
    print('Computing CF pairs for test set...')
    samples = []
    for text, label, targets in tqdm(raw_test):
        orig_term, swap_term, cat = find_swap(text)
        if orig_term is None:
            continue
        cf_text = make_swap(text, orig_term, swap_term)
        is_strict = compute_validity_strict(text, cf_text, orig_term, swap_term, cat)['use_for_ccr']
        samples.append({
            'text': text, 'cf_text': cf_text,
            'label': label, 'cat': cat,
            'orig_term': orig_term, 'swap_term': swap_term,
            'is_strict': is_strict,
        })
    print(f'Swappable test samples: {len(samples)}')

    orig_texts = [s['text'] for s in samples]
    cf_texts   = [s['cf_text'] for s in samples]

    print('\nLoading Naive Swap model...')
    base_model  = load_model(CKPT_NAIVE)
    base_orig_p = batch_infer(base_model, tokenizer, orig_texts)
    base_cf_p   = batch_infer(base_model, tokenizer, cf_texts)
    del base_model; torch.cuda.empty_cache()

    print('Loading Strict-Gated model...')
    strict_model  = load_model(CKPT_STRICT)
    strict_orig_p = batch_infer(strict_model, tokenizer, orig_texts)
    strict_cf_p   = batch_infer(strict_model, tokenizer, cf_texts)
    del strict_model; torch.cuda.empty_cache()

    # 분류
    categories = {'base_flip_strict_no': [], 'base_no_strict_flip': [],
                  'both_flip': [], 'both_no_large_gap': []}

    flip_dirs = {'base_1to0': 0, 'base_0to1': 0, 'strict_1to0': 0, 'strict_0to1': 0}
    base_flip_total = strict_flip_total = 0

    for i, s in enumerate(samples):
        bp  = base_orig_p[i];   bcp  = base_cf_p[i]
        sp  = strict_orig_p[i]; scp  = strict_cf_p[i]
        bp_pred  = int(bp  >= 0.5); bcp_pred  = int(bcp  >= 0.5)
        sp_pred  = int(sp  >= 0.5); scp_pred  = int(scp  >= 0.5)
        base_flip   = bp_pred != bcp_pred
        strict_flip = sp_pred != scp_pred
        base_gap    = abs(bp - bcp)
        strict_gap  = abs(sp - scp)

        if base_flip:
            base_flip_total += 1
            if bp_pred == 1 and bcp_pred == 0: flip_dirs['base_1to0'] += 1
            else: flip_dirs['base_0to1'] += 1
        if strict_flip:
            strict_flip_total += 1
            if sp_pred == 1 and scp_pred == 0: flip_dirs['strict_1to0'] += 1
            else: flip_dirs['strict_0to1'] += 1

        entry = {**s, 'bp': bp, 'bcp': bcp, 'sp': sp, 'scp': scp,
                 'base_gap': base_gap, 'strict_gap': strict_gap,
                 'base_flip': base_flip, 'strict_flip': strict_flip,
                 'bp_pred': bp_pred, 'bcp_pred': bcp_pred,
                 'sp_pred': sp_pred, 'scp_pred': scp_pred}

        if base_flip and not strict_flip:
            categories['base_flip_strict_no'].append(entry)
        elif not base_flip and strict_flip:
            categories['base_no_strict_flip'].append(entry)
        elif base_flip and strict_flip:
            categories['both_flip'].append(entry)
        elif not base_flip and not strict_flip and (base_gap > 0.1 or strict_gap > 0.1):
            categories['both_no_large_gap'].append(entry)

    # ── 결과 출력 ──────────────────────────────────────────────────────────────

    print('\n' + '='*70)
    print('FLIP 방향 분석')
    print('='*70)
    n = len(samples)
    print(f'총 swappable: {n}')
    print(f'\nNaive Swap  flip_rate: {base_flip_total/n:.4f}  ({base_flip_total}/{n})')
    print(f'  pos_to_neg (1→0): {flip_dirs["base_1to0"]/n:.4f}  ({flip_dirs["base_1to0"]})')
    print(f'  neg_to_pos (0→1): {flip_dirs["base_0to1"]/n:.4f}  ({flip_dirs["base_0to1"]})')
    print(f'\nStrict-Gated flip_rate: {strict_flip_total/n:.4f}  ({strict_flip_total}/{n})')
    print(f'  pos_to_neg (1→0): {flip_dirs["strict_1to0"]/n:.4f}  ({flip_dirs["strict_1to0"]})')
    print(f'  neg_to_pos (0→1): {flip_dirs["strict_0to1"]/n:.4f}  ({flip_dirs["strict_0to1"]})')

    # 카테고리별 개수
    print('\n' + '='*70)
    print('카테고리별 샘플 수')
    print('='*70)
    for k, v in categories.items():
        print(f'  {k}: {len(v)}')

    # 예시 출력 함수
    def show_examples(entries, title, n=10):
        print(f'\n{"="*70}')
        print(f'{title} (상위 {min(n, len(entries))}개)')
        print('='*70)
        # gap 차이 큰 순으로 정렬
        entries_sorted = sorted(entries, key=lambda x: abs(x['base_gap'] - x['strict_gap']), reverse=True)
        for e in entries_sorted[:n]:
            print(f'\n[label={e["label"]} | cat={e["cat"]}]')
            print(f'  원문: {e["text"][:80]}')
            print(f'  CF  : {e["cf_text"][:80]}')
            print(f'  Baseline   orig_p={e["bp"]:.3f}({e["bp_pred"]}) → cf_p={e["bcp"]:.3f}({e["bcp_pred"]})  gap={e["base_gap"]:.3f}  flip={e["base_flip"]}')
            print(f'  Strict     orig_p={e["sp"]:.3f}({e["sp_pred"]}) → cf_p={e["scp"]:.3f}({e["scp_pred"]})  gap={e["strict_gap"]:.3f}  flip={e["strict_flip"]}')

    show_examples(categories['base_flip_strict_no'],  '개선 사례 (Naive flip, Strict no flip)')
    show_examples(categories['base_no_strict_flip'],  '악화 사례 (Naive no flip, Strict flip)')
    show_examples(categories['both_flip'],            '여전히 어려운 사례 (both flip)')
    show_examples(categories['both_no_large_gap'],    'confidence 흔들림 (both no flip, large prob gap)')


if __name__ == '__main__':
    main()
