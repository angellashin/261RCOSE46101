"""
dataset.py
K-HATERS 로딩, identity swap pair 정의, counterfactual 생성,
validity 판정, HatersDataset 클래스.
"""

import os, json, random
from datasets import load_dataset
from kiwipiepy import Kiwi

import torch
from torch.utils.data import Dataset

# ── Identity swap pairs ────────────────────────────────────────────────────────
SWAP_PAIRS_BY_CAT: dict[str, list[tuple[str, str]]] = {
    'gender':     [('여성', '남성'), ('여자', '남자'), ('여성들', '남성들'),
                   ('여자들', '남자들'), ('페미니스트', '남성우월주의자'),
                   ('페미', '한남'), ('메갈', '한남'), ('한녀', '한남')],
    'religion':   [('무슬림', '기독교인'), ('이슬람', '기독교'),
                   ('무슬림', '천주교인'), ('이슬람교도', '기독교인')],
    'ethnicity':  [('조선족', '한국인'), ('외국인', '내국인'),
                   ('탈북민', '남한사람'), ('베트남인', '한국인'),
                   ('일본인', '한국인'), ('재일교포', '한국인'),
                   ('동남아인', '한국인')],
    'age':        [('노인', '청년'), ('노년층', '청년층'),
                   ('할머니', '젊은여자'), ('할아버지', '젊은남자')],
    'sexuality':  [('동성애자', '이성애자'), ('게이', '이성애자'),
                   ('레즈비언', '이성애자'), ('성소수자', '이성애자'),
                   ('트랜스젠더', '이성애자'), ('퀴어', '이성애자')],
    'disability': [('장애인', '비장애인'), ('정신장애인', '비장애인'),
                   ('지적장애인', '비장애인')],
}

# term → (counterpart, category)
SWAP_MAP: dict[str, tuple[str, str]] = {}
for _cat, _pairs in SWAP_PAIRS_BY_CAT.items():
    for _a, _b in _pairs:
        SWAP_MAP[_a] = (_b, _cat)

# 길이 내림차순 정렬 (긴 term 먼저 매칭)
SWAP_KEYS: list[str] = sorted(SWAP_MAP.keys(), key=len, reverse=True)

# ── 형태소 분석기 ──────────────────────────────────────────────────────────────
kiwi = Kiwi()

# ── Swap 탐지 ─────────────────────────────────────────────────────────────────
def find_swap(text: str) -> tuple[str | None, str | None, str | None]:
    """형태소 기반 탐지.
    - identity term이 독립 토큰으로 등장해야 함
    - 서로 다른 identity term이 2개 이상이면 None (CF 의미 오염 방지)
    """
    tokens = kiwi.tokenize(text)
    token_forms = [t.form for t in tokens]
    found = [term for term in SWAP_KEYS if term in token_forms]
    if len(set(found)) >= 2:
        return None, None, None
    if found:
        term = found[0]
        counterpart, cat = SWAP_MAP[term]
        return term, counterpart, cat
    return None, None, None


def find_swap_naive(text: str) -> tuple[str | None, str | None, str | None]:
    """단순 substring 매칭 — 형태소 경계/다중 term 미검사 (Naive Swap 조건)."""
    for term in SWAP_KEYS:
        if term in text:
            counterpart, cat = SWAP_MAP[term]
            return term, counterpart, cat
    return None, None, None


# ── 조사 교정 헬퍼 ─────────────────────────────────────────────────────────────
def _has_batchim(char: str) -> bool:
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):
        return False
    return (code - 0xAC00) % 28 != 0

def _ends_with_rieul(char: str) -> bool:
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):
        return False
    return (code - 0xAC00) % 28 == 8

_JOSA_VOWEL = {
    '이': '가', '을': '를', '은': '는',
    '과': '와', '아': '야',
    '이나': '나', '이랑': '랑', '이든': '든',
    '이라고': '라고', '이라며': '라며', '이라는': '라는',
    '이라서': '라서', '이라도': '라도', '이라면': '라면',
    '이란': '란', '이야': '야',
}
_JOSA_CONS = {v: k for k, v in _JOSA_VOWEL.items()}
_ALT_JOSA  = set(_JOSA_VOWEL) | set(_JOSA_CONS) | {'으로', '로'}

def _adjust_josa(new_term: str, josa: str) -> str:
    if not new_term:
        return josa
    last = new_term[-1]
    if josa in ('으로', '로'):
        return '으로' if (_has_batchim(last) and not _ends_with_rieul(last)) else '로'
    if _has_batchim(last):
        return _JOSA_CONS.get(josa, josa)
    return _JOSA_VOWEL.get(josa, josa)


# ── Swap 교체 ─────────────────────────────────────────────────────────────────
def make_swap(text: str, orig_term: str, new_term: str) -> str:
    """형태소 경계 기반 전체 교체 + 뒤따르는 조사 자동 조정."""
    tokens = kiwi.tokenize(text)
    subs: list[tuple[int, int, str]] = []
    for i, t in enumerate(tokens):
        if t.form == orig_term:
            subs.append((t.start, t.start + t.len, new_term))
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if str(nxt.tag).startswith('J') and nxt.form in _ALT_JOSA:
                    fixed = _adjust_josa(new_term, nxt.form)
                    if fixed != nxt.form:
                        subs.append((nxt.start, nxt.start + nxt.len, fixed))
    if not subs:
        return text
    for start, end, repl in sorted(subs, key=lambda x: x[0], reverse=True):
        text = text[:start] + repl + text[end:]
    return text


def make_swap_naive(text: str, orig_term: str, new_term: str) -> str:
    """단순 str.replace (Naive Swap 조건)."""
    return text.replace(orig_term, new_term)


# ── Validity 판정 ─────────────────────────────────────────────────────────────
_SEMANTIC_BLACKLIST: dict[str, list[str]] = {
    'ethnicity': ['입국', '방역', '국경', '불법체류', '이민', '귀화', '출입국',
                  '난민', '밀입국', '추방', '체류'],
    'gender':    ['위안부', '임신', '출산', '생식', '여학생', '남학생', '자궁',
                  '성폭력', '성폭행', '성매매', '강간', '몰카'],
    'religion':  ['부르카', '히잡', '테러', '지하드', '성전', '탈레반', '샤리아'],
    'age':       ['위안부', '전쟁', '일제', '역사적'],
    'sexuality': ['결혼', '입양', '헌혈', '군대', '병역'],
}

# Strict gate용 확장 blacklist (기존 항목 + 추가)
_SEMANTIC_BLACKLIST_STRICT: dict[str, list[str]] = {
    'ethnicity': _SEMANTIC_BLACKLIST['ethnicity'] + ['식민지', '침략', '전쟁범죄'],
    'gender':    _SEMANTIC_BLACKLIST['gender']    + ['생리', '군대', '병역', '군필'],
    'religion':  _SEMANTIC_BLACKLIST['religion']  + ['이단', '사이비', '교주', '세뇌'],
    'sexuality': _SEMANTIC_BLACKLIST['sexuality'] + ['에이즈', 'HIV', '성전환'],
    'age':       _SEMANTIC_BLACKLIST['age']       + ['60대', '70대', '80대', '90대',
                                                      '고령', '은퇴', '노후', '요양', '치매'],
    'disability': [],
}

_ASYMMETRIC_PAIRS: set[tuple[str, str]] = {
    ('게이', '이성애자'), ('이성애자', '게이'),
    ('레즈비언', '이성애자'), ('이성애자', '레즈비언'),
    ('트랜스젠더', '이성애자'), ('이성애자', '트랜스젠더'),
    ('성소수자', '이성애자'), ('이성애자', '성소수자'),
    ('퀴어', '이성애자'), ('이성애자', '퀴어'),
    ('할머니', '젊은여자'), ('젊은여자', '할머니'),
    ('할아버지', '젊은남자'), ('젊은남자', '할아버지'),
}

# Strict gate: pair-level asymmetric set은 기존 기준을 유지하고,
# ethnicity/religion은 pair 전면 차단 대신 context blacklist로 처리한다.
_ASYMMETRIC_PAIRS_STRICT: set[tuple[str, str]] = set(_ASYMMETRIC_PAIRS)

# 비교 구문 조사/부사 (④)
_COMPARISON_TOKENS: set[str] = {
    '보다', '처럼', '만큼', '대비', '반면', '달리',
    '비해', '비해서', '비하면',
}

# age swap: 문장에 명시적 나이대 표현이 있으면 의미 모순 가능 (예: 60대 청년)
_AGE_DECADE_PATTERNS: list[str] = [
    '10대', '20대', '30대', '40대', '50대',
    '60대', '70대', '80대', '90대',
]

def has_age_decade_context(text: str) -> bool:
    return any(p in text for p in _AGE_DECADE_PATTERNS)


# 목적어 + 사건 맥락 키워드 (③ 제한 적용)
_EVENT_OBJ_KEYWORDS: set[str] = {
    '폭행', '살해', '강간', '임신', '출산', '생식', '피해',
    '고소', '처벌', '신고', '학대', '착취',
}

def _check_grammar(cf: str, swap_term: str) -> bool:
    if not swap_term:
        return True
    last = swap_term[-1]
    has_batchim = _has_batchim(last)
    if has_batchim:
        bad = [swap_term + j for j in _JOSA_VOWEL.values()]
    else:
        bad = [swap_term + j for j in _JOSA_VOWEL.keys()]
    return not any(p in cf for p in bad)


def compute_validity(
    original: str, cf: str,
    orig_term: str, swap_term: str, cat: str,
) -> dict[str, bool]:
    """4가지 validity 기준 판정."""
    valid_grammar   = _check_grammar(cf, swap_term)
    blacklist       = _SEMANTIC_BLACKLIST.get(cat, [])
    valid_semantics = not any(kw in original + ' ' + cf for kw in blacklist)
    label_preserving = (
        (orig_term, swap_term) not in _ASYMMETRIC_PAIRS
        and valid_semantics
    )
    use_for_ccr = valid_grammar and valid_semantics and label_preserving
    return {
        'same_category':    True,
        'valid_grammar':    valid_grammar,
        'valid_semantics':  valid_semantics,
        'label_preserving': label_preserving,
        'use_for_ccr':      use_for_ccr,
    }


def compute_validity_strict(
    original: str, cf: str,
    orig_term: str, swap_term: str, cat: str,
) -> dict[str, bool]:
    """Strict validity gate.
    기존 기준 + ① asymmetric pair 확장 + ② semantic blacklist 보강
              + ④ 비교 구문 필터 + ③ 목적어+사건맥락 제한 필터
    """
    valid_grammar = _check_grammar(cf, swap_term)

    # ② 확장된 blacklist
    blacklist = _SEMANTIC_BLACKLIST_STRICT.get(cat, [])
    valid_semantics = not any(kw in original + ' ' + cf for kw in blacklist)

    # ① 확장된 asymmetric pairs
    label_preserving = (
        (orig_term, swap_term) not in _ASYMMETRIC_PAIRS_STRICT
        and valid_semantics
    )

    # ④ 비교 구문 필터: 토큰 기준 + raw text 기준 모두 체크
    tokens = kiwi.tokenize(original)
    token_forms = [t.form for t in tokens]
    no_comparison = (
        not any(p in token_forms for p in _COMPARISON_TOKENS)
        and not any(p in original for p in _COMPARISON_TOKENS)
    )

    # ③ 목적어+사건맥락 필터 (전면 적용 아님 — 목적어이면서 사건 키워드 동반 시만)
    no_harmful_obj = True
    for i, t in enumerate(tokens):
        if t.form == orig_term and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if str(nxt.tag).startswith('J') and nxt.form in ('을', '를'):
                if any(kw in original for kw in _EVENT_OBJ_KEYWORDS):
                    no_harmful_obj = False
                    break

    # age swap: 문장에 명시적 나이대 표현 있으면 의미 모순 가능
    no_age_contradiction = True
    if cat == 'age' and has_age_decade_context(original + ' ' + cf):
        no_age_contradiction = False

    use_for_ccr = (
        valid_grammar and valid_semantics and label_preserving
        and no_comparison and no_harmful_obj and no_age_contradiction
    )
    return {
        'same_category':    True,
        'valid_grammar':    valid_grammar,
        'valid_semantics':  valid_semantics,
        'label_preserving': label_preserving,
        'no_comparison':    no_comparison,
        'no_harmful_obj':   no_harmful_obj,
        'use_for_ccr':      use_for_ccr,
    }


# ── K-HATERS 로딩 ─────────────────────────────────────────────────────────────
ABUSIVE_LABELS = {'offensive', 'l1_hate', 'l2_hate'}

def to_binary(label: str) -> int:
    return 1 if label.strip().lower() in ABUSIVE_LABELS else 0


def load_khaters(split: str, subset: int = 0, seed: int = 42) -> list:
    ds = load_dataset('humane-lab/K-HATERS', split=split)
    examples = []
    for row in ds:
        text = row['text'].strip()
        if not text:
            continue
        label   = to_binary(row['label'])
        targets = row.get('target_label') or []
        examples.append((text, label, targets))

    if subset:
        rng = random.Random(seed)
        pos = [e for e in examples if e[1] == 1]
        neg = [e for e in examples if e[1] == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        examples = pos[:subset // 2] + neg[:subset // 2]
        rng.shuffle(examples)

    return examples


def load_cf_pairs(path: str) -> dict[str, tuple[str, str, str, str]]:
    """JSONL → {original_text: (cf_text, orig_term, swap_term, cat)}
    pre-computed CF pairs를 로딩해 kiwi find_swap+make_swap 호출을 생략한다.
    """
    lookup: dict[str, tuple[str, str, str, str]] = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            lookup[p['original']] = (p['cf'], p['orig_term'], p['swap_term'], p['category'])
    return lookup


def save_cf_pairs(examples: list, path: str) -> None:
    """CF 쌍 전체를 JSONL로 저장 (검수 및 재현성용)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pairs = []
    for text, label, targets in examples:
        orig_term, swap_term, cat = find_swap(text)
        if orig_term is None:
            continue
        cf_text  = make_swap(text, orig_term, swap_term)
        base_v   = compute_validity(text, cf_text, orig_term, swap_term, cat)
        strict_v = compute_validity_strict(text, cf_text, orig_term, swap_term, cat)
        pairs.append({
            'original':  text,
            'cf':        cf_text,
            'orig_term': orig_term,
            'swap_term': swap_term,
            'category':  cat,
            'label':     label,
            'targets':   targets,
            **{f'base_{k}': v for k, v in base_v.items()},
            **{f'strict_{k}': v for k, v in strict_v.items()},
        })
    with open(path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    n_base   = sum(1 for p in pairs if p['base_use_for_ccr'])
    n_strict = sum(1 for p in pairs if p['strict_use_for_ccr'])
    print(f'CF pairs saved → {path}  '
          f'(total={len(pairs)}, base_valid={n_base}, strict_valid={n_strict})')


# ── Dataset ───────────────────────────────────────────────────────────────────
class HatersDataset(Dataset):
    """
    mode: 'none'   → original only (Baseline)
          'mask'   → original + [MASK] replacement (Masking Cons Reg)
          'swap'   → original + identity swap, no gate (Naive Swap)
          'gated'  → original + identity swap + validity gate (Validity-Gated)
          'strict' → original + identity swap + strict validity gate (Strict-Gated)

    cf_lookup: load_cf_pairs()로 로딩한 dict {text: (cf_text, orig_term, swap_term, cat)}
               제공 시 find_swap + make_swap(kiwi) 호출을 생략 → 대규모 데이터에서 빠름.
               mask 모드는 make_swap을 여전히 사용하나 find_swap은 스킵됨.
    """
    def __init__(self, examples: list, tokenizer, max_len: int,
                 mode: str = 'none',
                 cf_lookup: dict | None = None):
        assert mode in ('none', 'mask', 'swap', 'gated', 'strict')
        self.tok, self.max_len, self.mode = tokenizer, max_len, mode
        self.items = []
        for text, label, targets in examples:
            orig_term, swap_term, cat = None, None, None
            cf_text  = None
            cf_valid = False

            if mode != 'none':
                # fast path: pre-computed CF lookup (kiwi find_swap + make_swap 생략)
                if cf_lookup is not None:
                    entry = cf_lookup.get(text)
                    if entry is not None:
                        precomp_cf, orig_term, swap_term, cat = entry
                        has_swap = True
                    else:
                        has_swap = False
                    precomp_cf = precomp_cf if has_swap else None
                else:
                    orig_term, swap_term, cat = find_swap(text)
                    has_swap = orig_term is not None
                    precomp_cf = None

                if has_swap and mode == 'swap':
                    cf_text  = precomp_cf or make_swap(text, orig_term, swap_term)
                    cf_valid = True
                elif has_swap and mode == 'gated':
                    cf_text  = precomp_cf or make_swap(text, orig_term, swap_term)
                    validity = compute_validity(text, cf_text, orig_term, swap_term, cat)
                    cf_valid = validity['use_for_ccr']
                elif has_swap and mode == 'strict':
                    cf_text  = precomp_cf or make_swap(text, orig_term, swap_term)
                    validity = compute_validity_strict(text, cf_text, orig_term, swap_term, cat)
                    cf_valid = validity['use_for_ccr']
                elif mode == 'mask' and has_swap:
                    # mask CF는 pre-computed에 없으므로 make_swap 필요
                    cf_text  = make_swap(text, orig_term, tokenizer.mask_token)
                    cf_valid = True

            self.items.append({
                'text': text, 'label': label,
                'targets': targets,
                'cf_text': cf_text,
                'cf_valid': cf_valid,
                'orig_term': orig_term,
                'swap_term': swap_term,
                'cat': cat,
            })

    def _enc(self, text: str):
        e = self.tok(text, max_length=self.max_len,
                     padding='max_length', truncation=True, return_tensors='pt')
        return e['input_ids'].squeeze(0), e['attention_mask'].squeeze(0)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it  = self.items[idx]
        ids, mask = self._enc(it['text'])
        out = {
            'input_ids':      ids,
            'attention_mask': mask,
            'label':          torch.tensor(it['label'], dtype=torch.long),
            'cf_valid':       torch.tensor(it['cf_valid'], dtype=torch.bool),
        }
        cf_src = it['cf_text'] if it['cf_text'] is not None else it['text']
        cf_ids, cf_mask = self._enc(cf_src)
        out['cf_input_ids']      = cf_ids
        out['cf_attention_mask'] = cf_mask
        return out
