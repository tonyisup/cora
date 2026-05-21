# M2 Distilled Policy Evaluation

**Date:** 2026-05-21
**Checkpoint:** `results/m2_model_checkpoints/distilled_v2/final_model.pt`
**Guarded evaluation:** `results/m2_eval_distilled_v2_guarded.json`

## Summary

The distilled policy is now functioning as intended when wrapped with deterministic safety rails.

- Dataset accuracy: 94.1%
- Scenario-suite accuracy (raw): 83.3%
- Scenario-suite accuracy (guarded): 100.0%

The key improvement from the guard layer is that it restores the safety-critical interrupt/yield behavior without changing the learned model.

## What the model learned

On the collected datasets:
- speak: 97.7% / 93.3% depending on report
- wait: ~94.1%
- clarify: 100%
- repair: 100%
- yield: 100%

The model is strongest on rare, explicit classes after class balancing and teacher augmentation.

## What still needs the guard

The learned policy still misses the `yield_on_interrupt` case in the raw scenario suite.
That is acceptable because the runtime wrapper enforces this rule deterministically:
- if user and agent are both speaking, yield immediately
- if already in repair state, repair takes precedence

## Scenario-suite results

Guarded policy results:
- speak_on_silence: correct
- wait_while_user_speaks: correct
- yield_on_interrupt: correct
- yield_on_collision: correct
- clarify_after_silence: correct
- repair_in_repair_state: correct

## Interpretation

This is the right shape for the project:
- the learned model captures the policy distribution
- the deterministic wrapper preserves safety-critical actions
- the benchmark confirms that clarify can be learned when the class imbalance is handled correctly

## Files

- `scripts/m2_collect_data.py`
- `scripts/m2_train.py`
- `scripts/m2_eval.py`
- `scripts/m2_guarded_policy.py`
- `results/m2_data_deterministic.jsonl`
- `results/m2_data_teacher.jsonl`
- `results/m2_data_hardneg.jsonl`
- `results/m2_model_checkpoints/distilled_v2/`
- `results/m2_eval_distilled_v2_guarded.json`
