# Milestone 2 — Close-Out Document

**Status:** Complete
**Date:** 2026-05-21

## Goal

Milestone 2 tested whether the M1 speaking policy could be compressed into a small learned model without losing the behavior that matters.

The core idea was:
- keep the deterministic policy as the safety baseline
- collect behavior from the baseline plus a teacher model
- train a small policy model on the resulting data
- wrap the learned model with deterministic safety rails

## What was built

### Data collection

A dedicated collection pipeline was added:
- `scripts/m2_collect_data.py`

It supports three sources of training signal:
- deterministic baseline trajectories
- teacher-labeled disagreement cases
- hard-negative templates for edge cases

The hard-negative generator was especially important because class imbalance in the baseline data was extreme.

### Training

A small MLP policy trainer was added:
- `scripts/m2_train.py`

It supports:
- behavioral cloning
- distillation-style training
- weighted sampling to address class imbalance
- checkpointing and training history

### Evaluation

A reproducible evaluator was added:
- `scripts/m2_eval.py`

A deterministic safety wrapper was also added:
- `scripts/m2_guarded_policy.py`

This lets the learned model run as the policy body while hard safety rules remain enforced at runtime.

## Data collected

Final training inputs used in the successful second pass:
- `results/m2_data_deterministic.jsonl`
- `results/m2_data_teacher.jsonl`
- `results/m2_data_hardneg_v2.jsonl`

Teacher signal came from Claude 3.5 Haiku and was especially useful in the disagreement cases where the deterministic policy defaulted to `wait`.

## Iteration history

### First pass

The initial distilled checkpoint was already strong, but one scenario still failed in the raw model:
- `yield_on_interrupt`

That mattered because it is a safety-critical turn-taking case.

### Second pass

To close that gap, the data generator was updated with an interrupt-specific hard-negative template:
- `interrupt_mid_sentence_yield`

That expanded the yield class coverage and taught the model the exact interruption pattern it had missed.

## Final results

### Distilled v3 checkpoint

- Checkpoint: `results/m2_model_checkpoints/distilled_v3/`
- Final model: `results/m2_model_checkpoints/distilled_v3/final_model.pt`

### Dataset metrics

- Dataset accuracy: 98.63%
- Per-action validation accuracy:
  - speak: 100.0%
  - wait: 98.64%
  - clarify: 86.02%
  - repair: 93.20%
  - yield: 100.0%

### Scenario-suite metrics

Raw model:
- 100.0% accuracy

Guarded model:
- 100.0% accuracy

The six scenario cases were all correct in the final state:
- speak_on_silence
- wait_while_user_speaks
- yield_on_interrupt
- yield_on_collision
- clarify_after_silence
- repair_in_repair_state

## What this means

The main lesson from Milestone 2 is that the learned policy is viable, but only if it is treated as a policy distribution rather than as an uncapped authority.

The correct architecture is:
1. the learned model handles the normal policy surface
2. deterministic rails enforce safety-critical turn-taking
3. hard-negative data keeps the rare cases from collapsing into the majority class

The project does not need a pure LLM policy to succeed here. It needs a compact learned policy plus guardrails.

## Key decision

Milestone 2 is complete.

We have enough evidence to say:
- the policy can be distilled into a small model
- safety-critical behavior should remain deterministic at runtime
- clarify can be learned once class imbalance is handled properly
- interrupt/yield must stay guarded even if the learned model is strong

## Files

Core code:
- `scripts/m2_collect_data.py`
- `scripts/m2_train.py`
- `scripts/m2_eval.py`
- `scripts/m2_guarded_policy.py`

Reports:
- `docs/m2_eval_report.md`
- `docs/m2_iteration_note.md`
- `results/m2_eval_distilled_v3_raw.json`
- `results/m2_eval_distilled_v3_guarded.json`

Training outputs:
- `results/m2_model_checkpoints/distilled_v3/`
- `results/m2_data_hardneg_v2.jsonl`
