# M2 Iteration Note — Interrupt Yield Fix

**Date:** 2026-05-21

## Why a second pass

The first distilled checkpoint was already strong on most classes, but the raw scenario suite still missed `yield_on_interrupt`. That was the last remaining weak spot.

## What changed

1. Added an interrupt-specific hard-negative template to the data generator:
   - `interrupt_mid_sentence_yield`
2. Regenerated a larger hard-negative set:
   - `results/m2_data_hardneg_v2.jsonl`
3. Retrained the distilled policy:
   - `results/m2_model_checkpoints/distilled_v3/`

## Before

Distilled v2 raw scenario-suite accuracy:
- 83.3%
- missed `yield_on_interrupt`

Distilled v2 guarded scenario-suite accuracy:
- 100.0%

## After

Distilled v3 raw scenario-suite accuracy:
- 100.0%

Distilled v3 guarded scenario-suite accuracy:
- 100.0%

## Dataset effect

The larger hard-negative set increased coverage of the yield class and gave the model the exact interrupt pattern it was missing.

## Training/eval outcome

Final v3 checkpoint:
- Dataset accuracy: 98.63%
- Per-action validation accuracy:
  - speak: 100.0%
  - wait: 98.64%
  - clarify: 86.02%
  - repair: 93.20%
  - yield: 100.0%
- Scenario-suite accuracy: 100.0% raw and guarded

## Interpretation

The remaining failure was not an architecture problem; it was a data coverage problem.
The second pass resolved it by explicitly teaching the model the interrupt-yield pattern.

## Files

- `scripts/m2_collect_data.py`
- `scripts/m2_train.py`
- `scripts/m2_eval.py`
- `scripts/m2_guarded_policy.py`
- `results/m2_data_hardneg_v2.jsonl`
- `results/m2_model_checkpoints/distilled_v3/`
- `results/m2_eval_distilled_v3_raw.json`
- `results/m2_eval_distilled_v3_guarded.json`
