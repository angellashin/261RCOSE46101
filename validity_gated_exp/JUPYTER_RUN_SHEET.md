# Jupyter Run Sheet

Use this as the copy-paste checklist for the report-grade run.

## 0. Sync

```bash
cd ~/nlp-korean
git pull
git rev-parse --short HEAD
git status --short --branch
```

Expected:

- commit: `a916ac5` or newer
- status: no local changes

## 1. Environment Check

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
df -h
```

If CUDA is false, stop before the full run unless you intentionally want a slow CPU run.

## 2. Smoke Run

Run this only if the current server/venv has not already completed a smoke test.

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

Expected:

- finishes end-to-end
- writes `validity_gated_exp/results_smoke.json`
- no CUDA/package/dataset errors

## 3. Report-Grade Preflight

```bash
python validity_gated_exp/preflight_run.py \
  --exp Baseline "Naive Swap" Strict-Gated Strict-Matched Strict_lam=0.15 Strict_lam=0.25 \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_core_followup.json \
  --require_core \
  --require_clean \
  --fresh_result_path
```

Expected:

- `PREFLIGHT PASS`
- `dirty=False`
- `Planned model fits: 18`
- `Planned train epochs: 54`

If `results_core_followup.json` already exists, use a new result path rather than overwriting report evidence.

## 4. Report-Grade Run

```bash
python validity_gated_exp/run_exp.py \
  --exp Baseline "Naive Swap" Strict-Gated Strict-Matched Strict_lam=0.15 Strict_lam=0.25 \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_core_followup.json \
  2>&1 | tee train_core_followup.log
```

The script checkpoint-saves after each experiment. If the run stops midway, inspect the partial JSON before restarting.

## 5. Compare

```bash
python validity_gated_exp/compare_results.py \
  validity_gated_exp/results_core_followup.json \
  --show_examples \
  --example_bucket both_wrong \
  --example_bucket strict_flip \
  --example_bucket false_positive_original \
  --max_examples 2
```

Read these sections in order:

1. `Result metadata`: same commit, clean state.
2. `Report readiness audit`: no `FAIL`.
3. `Claim assessment`: choose the final paper claim strength.
4. `Best strict-family variant`: choose the representative gated row.
5. `Saved qualitative examples`: use for discussion/error analysis.

## 6. Claim Decision

- `strong_gated`: use the best gated method as the main positive result.
- `soft_consistency_tradeoff`: report Naive as stronger hard invariance, gated as validity-filtered soft consistency.
- `validity_coverage_tradeoff`: emphasize invalid-CF filtering and reduced training coverage.
- `diagnostic_only`: do not claim a positive gated method; analyze why it failed.
- `incomplete`: rerun missing or mismatched experiments before writing the result section.
