# Validity-Gated CCR Experiment Runbook

이 문서는 `Validity-Gated Counterfactual Consistency Regularization` 실험을 같은 조건으로 반복 실행하기 위한 체크리스트입니다.

## 1. Environment

새 서버/Jupyter 환경에서는 이 repo만 clone해서 사용합니다.

```bash
cd ~
git clone https://github.com/angellashin/nlp-korean.git
cd nlp-korean
```

이미 clone되어 있으면:

```bash
cd ~/nlp-korean
git pull
```

설치 전 디스크 여유 공간을 확인합니다. PyTorch, KLUE model, HuggingFace cache 때문에 최소 15GB 이상이 안전합니다.

```bash
df -h
```

이전에 깨진 venv나 cache 때문에 공간이 부족하면 정리합니다.

```bash
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
```

venv를 만들고 PyTorch를 먼저 설치한 뒤 나머지 패키지를 설치합니다.

```bash
cd ~/nlp-korean
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r validity_gated_exp/requirements.txt
```

Jupyter에서 이 venv를 커널로 쓰려면:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name nlp-korean --display-name "nlp-korean"
jupyter notebook
```

노트북에서는 상단 커널을 `nlp-korean`으로 선택한 뒤 확인합니다.

```python
import sys
print(sys.executable)
```

주의: 노트북의 `!source .venv/bin/activate`는 다음 셀에 유지되지 않습니다. 커널을 venv로 선택하거나, 매번 `.venv/bin/python`을 직접 호출하세요.

## 2. Data Sanity Checks

먼저 데이터와 counterfactual pair 생성이 정상인지 확인합니다.

```bash
python validity_gated_exp/check_data.py 2>&1 | tee check_data.log
python validity_gated_exp/analyze_cf_pairs.py \
  --jsonl validity_gated_exp/data/cf_pairs_train.jsonl \
  --out validity_gated_exp/data/cf_analysis.txt \
  2>&1 | tee analyze_cf_pairs.log
```

보고서에는 최소한 다음 숫자를 기록합니다.

- train/validation/test 크기와 hate label 비율
- swappable train sample 수
- base-valid, strict-valid pair 수와 비율
- category별 strict-valid 비율
- strict gate가 reject한 대표 예시

## 3. Smoke Test

전체 실험 전에 작은 subset으로 코드가 끝까지 도는지 확인합니다.

```bash
python validity_gated_exp/run_exp.py \
  --exp Baseline \
  --seeds 42 \
  --subset 512 \
  --epochs 1 \
  --batch_size 8 \
  --num_workers 0 \
  --result_path validity_gated_exp/results_smoke.json \
  2>&1 | tee smoke.log
```

노트북 셀에서 실행할 때는:

```python
!.venv/bin/python validity_gated_exp/run_exp.py --exp Baseline --seeds 42 --subset 512 --epochs 1 --batch_size 8 --num_workers 0 --result_path validity_gated_exp/results_smoke.json 2>&1 | tee smoke.log
```

## 4. Core Experiments

논문용 기본 비교는 같은 seeds, epochs, batch size로 돌립니다.

```bash
python validity_gated_exp/run_exp.py \
  --exp Baseline "Naive Swap" Validity-Gated Strict-Gated "Masking Cons Reg" \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_core.json \
  2>&1 | tee train_core.log
```

단일 실험만 다시 돌릴 때:

```bash
python validity_gated_exp/run_exp.py \
  --exp "Naive Swap" \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_naive.json \
  2>&1 | tee train_naive.log
```

사용자가 말한 형태도 가능합니다.

```bash
python validity_gated_exp/run_exp.py --exp "Naive Swap" 2>&1 | tee train_new2.log
```

다만 보고서용 결과라면 `--seeds`, `--epochs`, `--batch_size`, `--result_path`를 명시하는 편이 재현성에 좋습니다.

## 5. Lambda Sensitivity

Strict-Gated가 중심 방법이면 lambda sensitivity를 작게 추가합니다.

```bash
python validity_gated_exp/run_exp.py \
  --exp Strict-Gated \
  --lambda 0.05 \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_strict_lam005.json \
  2>&1 | tee train_strict_lam005.log

python validity_gated_exp/run_exp.py \
  --exp Strict-Gated \
  --lambda 0.2 \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_strict_lam02.json \
  2>&1 | tee train_strict_lam02.log
```

Naive가 Strict보다 강하게 나오는 경우, Strict가 보는 valid CF 수가 더 적어서 regularization signal이 약한지 확인해야 합니다. 이때는 `train_valid_cf_ratio`를 보고 Strict lambda를 조금 키운 ablation을 추가합니다.

```bash
python validity_gated_exp/run_exp.py \
  --exp Strict_lam=0.15 \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_strict_lam015.json \
  2>&1 | tee train_strict_lam015.log
```

`Strict_lam=<값>` 형식은 임의의 positive lambda 값을 받습니다. 예를 들어 `Strict_lam=0.12`, `Strict_lam=0.25`도 가능합니다. 이 방식으로 돌리면 결과 행 이름이 lambda 값을 포함하므로 나중에 비교표에서 덜 헷갈립니다.

더 좋은 진단용 follow-up은 `Strict-Matched`입니다. 이 조건은 Strict-valid pair 수가 Naive-valid pair 수보다 적은 만큼 lambda를 자동으로 키워, 낮은 coverage 때문에 Strict가 약해진 것인지 분리합니다. 계산식은 `min(0.3, base_lambda * naive_valid_count / strict_valid_count)`입니다.

```bash
python validity_gated_exp/run_exp.py \
  --exp Strict-Matched \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_strict_matched.json \
  2>&1 | tee train_strict_matched.log
```

해석:

- `Strict-Matched > Strict-Gated`: Strict의 약점은 gate 자체보다 regularization signal coverage 부족일 가능성이 큽니다.
- `Strict-Matched <= Strict-Gated`: gate가 useful pair까지 버리거나, strict pair만으로는 hard-label consistency를 올리기 어렵다는 분석이 가능합니다.
- `Strict-Matched > Naive Swap`: 가장 좋은 결과입니다. validity filtering을 유지하면서 Naive 수준 이상의 consistency를 얻었다고 주장할 수 있습니다.

## 6. Compare Results

여러 JSON을 한 번에 비교합니다.

```bash
python validity_gated_exp/compare_results.py \
  validity_gated_exp/results_core.json
```

파일이 나뉘어 있으면:

```bash
python validity_gated_exp/compare_results.py \
  validity_gated_exp/results_naive.json \
  validity_gated_exp/results_core_followup.json \
  validity_gated_exp/results_strict_lam015.json
```

이 스크립트는 콘솔용 비교표, Baseline 대비 delta, paper claim suggestion, Markdown 표를 같이 출력합니다. 보고서 표 초안은 `Markdown table` 섹션을 가져가면 됩니다.
`TrainCF%`, `ConsBatch%`, `ValidCF/B`는 Strict가 Naive보다 약하게 나왔을 때 regularization signal coverage 차이를 설명하는 데 씁니다.
`FPR minN`은 identity category별 FPR을 계산할 때 가장 작은 normal-sample group 크기입니다. 이 값이 작으면 `FPR Gap`은 보조 지표로만 해석합니다.
`Best strict-family variant`는 `Strict-Gated`, `Strict-Matched`, `Strict_lam=*` 중 Strict Pair Acc가 가장 높은 gated 계열 결과를 골라줍니다. 보고서 대표 gated 결과를 고를 때 이 섹션을 먼저 확인합니다.
`Naive vs best gated paired diagnostic`은 같은 seed끼리 best gated와 Naive를 비교합니다. 평균 차이가 작으면 몇 개 seed에서 방향이 유지되는지까지 확인한 뒤 claim 강도를 정합니다.
`Recommended next steps`는 현재 결과 기준으로 후속 실험이 더 필요한지 알려줍니다. Naive가 gated 계열보다 강하면 `Strict-Matched`와 `Strict_lam=*` follow-up을 먼저 돌리고, gated가 충분히 강하면 method search를 멈추고 error analysis/report로 넘어갑니다.

비교 출력 맨 위의 `Result metadata`와 `Experiment configs`를 먼저 확인합니다.

- `missing _meta` 또는 `missing per-experiment config`가 뜨는 old result는 final table에 섞지 않습니다.
- `mix different git_commit`, `gate_version`, `model`, `max_len` 경고가 뜨면 같은 표에 직접 비교하지 않습니다.
- `dirty=True`가 보이면 uncommitted local code로 돌린 결과이므로, 보고서용 결과로 쓰기 전에 commit된 상태에서 다시 실행합니다.
- lambda follow-up은 가능하면 별도 `--result_path`에 저장하고, `compare_results.py`로 여러 JSON을 함께 읽습니다.
- 같은 `--result_path`에 follow-up을 저장하더라도 기존 실험과 config가 다르면 새 행 이름으로 저장되어 덮어쓰지 않습니다. 그래도 보고서용으로는 fresh path를 권장합니다.
- 여러 JSON 안에 같은 experiment name이 있으면 두 번째부터 자동으로 `Strict-Gated [lambda=..., file]`처럼 이름이 바뀝니다. 이 경고가 보이면 표에 들어갈 행 이름을 수동으로 확인합니다.

## 7. Hardware Notes

- CUDA GPU: start with `--batch_size 64`. If memory is enough, try 128.
- Mac MPS or CPU: use `--batch_size 8` or `16`, and smoke test first.
- `--num_workers 0` is safer in notebooks. If stable, use `2` or `4`.
- `2>&1 | tee file.log` means stderr와 stdout을 합쳐서 화면과 로그 파일에 동시에 저장한다는 뜻입니다.

## 8. Metrics To Trust

주요 지표는 아래 순서로 해석합니다.

1. `Macro-F1`: 기존 hate detection 성능이 유지되는지.
2. `Strict Pair Acc`: strict-valid counterfactual pair에서 원문과 CF를 모두 맞히는지.
3. `Pair Acc`: 전체 swappable pair에서 둘 다 맞히는지.
4. `Flip Rate` / `Strict Flip Rate`: 예측이 바뀌는 비율. 낮을수록 좋지만, 둘 다 틀려도 낮아질 수 있으므로 단독 주장에는 쓰지 않습니다.
5. `FPR Gap`: identity category별 false positive 격차. category별 표본 수가 작으면 보조 지표로만 해석합니다.
6. `FPR minN`: FPR Gap에 들어간 identity group 중 가장 작은 normal sample 수. 작을수록 FPR Gap 해석이 불안정합니다.

보고서의 핵심 claim은 `Macro-F1 유지 + Strict Pair Acc 개선 + invalid counterfactual filtering 근거`로 잡는 것이 가장 안전합니다.

Strict-Gated가 Naive Swap보다 항상 좋아진다는 보장은 없습니다. Naive가 이기면 실패로 처리하지 말고, `TrainCF%`, `ConsBatch%`, `ValidCF/B`, strict rejection breakdown을 근거로 "strong invariance vs validity filtering" trade-off로 해석합니다.
