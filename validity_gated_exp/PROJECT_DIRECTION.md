# Project Direction: Validity-Gated Counterfactual Consistency Regularization

## Recommended Claim

한국어 혐오 표현 탐지에서 identity term swap을 consistency regularization에 쓰되, 모든 swap을 학습 신호로 넣는 방법과 validity gate를 통과한 counterfactual pair만 regularization에 사용하는 방법을 비교한다.

핵심 주장은 다음 형태가 가장 안전하다.

1. Identity swap CCR은 hate detection 성능을 크게 해치지 않으면서 counterfactual consistency를 개선할 수 있다.
2. Ungated/Naive swap은 더 강한 invariance를 줄 수 있지만 invalid counterfactual도 학습 신호로 쓸 위험이 있다.
3. Validity-gated CCR은 consistency gain과 counterfactual validity 사이의 trade-off를 분석하는 방법이다.
4. 한국어에서는 조사, 형태소 경계, 문화/사건 특정 맥락 때문에 English-style counterfactual augmentation을 그대로 쓰기 어렵고, 이 점이 방법론적 novelty다.

## Why This Is Better Than Evaluation-Only Pair Testing

단순히 synthetic/test pair를 만들고 KOLD나 K-HATERS 모델을 평가하는 방식은 결과 분석에 가깝다. 반면 이 방향은 모델 학습 목적함수에 개입한다.

- Baseline: K-HATERS fine-tuning
- Naive Swap: 형태소 경계와 단일 identity 제약은 유지하되, validity gate 없이 모든 generated swap에 KL consistency 적용
- Validity-Gated: same-category/grammar/semantic gate 통과 pair만 적용
- Strict-Gated: 비교 구문, 사건 목적어, age contradiction까지 제한한 conservative gate
- Masking Cons Reg: identity masking과 비교하는 sanity baseline

Implementation note: consistency KL은 원문을 다시 forward하지 않고 classification loss에서 이미 계산한 original logits를 anchor로 재사용한다. 이렇게 해야 train-time dropout이 원문 쪽에 한 번 더 들어가면서 KL signal이 불필요하게 noisy해지는 것을 줄일 수 있다.

따라서 보고서에서는 "새 평가셋 만들기"보다 "counterfactual regularization에서 invariance strength와 validity filtering의 trade-off를 분석하는 방법"으로 쓰는 편이 좋다.

## Critical Risks

1. Flip rate만 보면 안 된다.

   Flip rate는 원문과 counterfactual을 둘 다 틀리게 예측해도 낮아질 수 있다. 핵심 지표는 `Strict Pair Acc`와 `Pair Acc`로 두고, flip/prob gap은 보조 지표로 둔다.

2. Validity gate는 oracle이 아니다.

   `후쿠시마`, `방사능`, `노인이 돼도` 같은 event-specific or age-transition context가 pass되면 방법의 설득력이 약해진다. strict gate는 conservative하게 유지하고, 보고서에는 pass/reject 예시와 rejection breakdown을 반드시 넣는다.

3. Swap map은 방향성이 있다.

   `한국인`처럼 여러 minority identity의 counterpart로 쓰이는 항목은 reverse swap이 모호하다. 임의로 symmetric map을 만들면 더 큰 오류가 생길 수 있으므로, 현재는 one-directional controlled swaps로 두고 limitation에 명시한다.

4. FPR gap은 표본 수에 민감하다.

   K-HATERS test에서 identity category별 normal sample 수가 작으면 FPR gap이 불안정하다. category counts를 같이 보고, 주요 결론은 pair accuracy 중심으로 둔다.

5. 기존 `results_final.json`에는 old Baseline만 있다.

   이 baseline에는 새로 추가된 `pair_accuracy`, `strict_pair_accuracy`가 없다. 논문용 결과는 `results_core.json`처럼 새 파일에 Baseline부터 전부 다시 실행해야 한다.

6. Naive Swap이 Strict-Gated보다 잘 나올 수 있다.

   이 경우 프로젝트가 실패한 것이 아니다. Naive가 더 많은 CF pair를 학습하기 때문에 stronger regularization을 받는다는 해석이 가능하다. 새 코드에서는 `train_valid_cf_ratio`, `pair_count`, `strict_pair_count`를 저장하므로, Strict가 지면 "gate가 너무 보수적이어서 useful signal을 줄인다"는 분석으로 전환한다.
   추가로 `cons_batch_ratio`와 `avg_valid_cf_per_batch`를 확인해 Strict가 실제로 더 적은 batch/CF에서만 consistency loss를 받았는지 검증한다.
   이때 `Strict-Matched`를 후속 실험으로 돌린다. 이 ablation은 `lambda = min(0.3, base_lambda * naive_valid_count / strict_valid_count)`로 설정해서 Strict의 낮은 CF coverage를 보정한다. `Strict-Matched`가 개선되면 coverage 부족이 원인이고, 개선되지 않으면 strict gate가 useful signal까지 줄였다는 해석이 가능하다.

## Minimum Experiment Set

보고서용 최소 실험:

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

결과 비교:

```bash
python validity_gated_exp/compare_results.py validity_gated_exp/results_core.json
```

추가 ablation:

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

python validity_gated_exp/run_exp.py \
  --exp Strict-Matched \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_strict_matched.json \
  2>&1 | tee train_strict_matched.log
```

## Report Table

Main table:

| Method | Macro-F1 | Pair Acc | Strict Pair Acc | Flip Rate | Strict Flip Rate | FPR Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | | | | | | |
| Masking Cons Reg | | | | | | |
| Naive Swap | | | | | | |
| Validity-Gated | | | | | | |
| Strict-Gated | | | | | | |
| Strict-Matched | | | | | | |

Construction analysis table:

| Category | Swappable | Base-valid | Strict-valid | Strict-valid % |
| --- | ---: | ---: | ---: | ---: |
| gender | | | | |
| ethnicity | | | | |
| religion | | | | |
| age | | | | |
| sexuality | | | | |
| disability | | | | |

## Interpretation Rules

- Best outcome: Strict-Gated keeps Macro-F1 within roughly 1 point of Baseline and improves Strict Pair Acc over Baseline/Naive Swap.
- Acceptable outcome: Strict-Gated improves pair metrics but slightly lowers F1; frame as robustness-accuracy tradeoff.
- Trade-off outcome: Naive Swap beats Strict-Gated on Strict Pair Acc or Flip Rate, while Strict-Gated has comparable F1 or lower Prob Gap. Then report an invariance-validity tradeoff.
- Bad outcome: Naive Swap beats Strict-Gated on every metric. Then the current gate is too conservative or wrong; analyze valid pair coverage and category distribution and try `Strict-Matched` or a larger `--lambda` for Strict-Gated.
- Do not claim fairness improvement from lower flip rate alone.
