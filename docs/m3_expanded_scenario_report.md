# M3 Expanded Scenario Benchmark

**Date:** 2026-05-21
**Checkpoint:** `results/m3_model_checkpoints/trace_mix_scaled/final_model.pt`

## Why this benchmark exists

The original six canonical M2 scenarios were saturated at 100% once the distilled policy stabilized. To test whether the M3 local-user traces improved robustness, I added an expanded scenario set with more ambiguous clarify / wait / yield / repair boundaries.

## Expanded suite

The new benchmark contains 13 scenarios:
- 6 canonical cases from M2
- 7 ambiguous extensions probing borderline states

Ambiguous cases include:
- idle/no-queue wait vs clarify
- short-silence speak
- long interrupt after speech
- overlap with queued utterance
- long silence with no queue
- question-marking / hesitant clarification
- repair-state repetition

## Evaluation result

On the scaled M3 checkpoint:
- raw scenario accuracy: 92.31%
- guarded scenario accuracy: 92.31%
- canonical subset: 100%
- ambiguous subset: 85.71%

Only one scenario failed:
- `wait_in_idle_no_queue` → predicted `clarify` instead of `wait`

## Interpretation

This is the first benchmark that meaningfully separates the current policy from the canonical six-case ceiling.

The result says:
- the model is solid on the original canonical cases
- the expanded ambiguous cases are harder and do expose a real boundary issue
- the model is over-eager to clarify when it sees a silence/no-queue case with no obvious user speech

That is useful because it gives us a concrete next target rather than just a saturated perfect score.

## Recommendation

Treat `wait_in_idle_no_queue` as the new hard-negative case to add to training / evaluation.

If we want to push further, the next best step is:
1. add a targeted hard-negative template for idle/no-queue silence,
2. retrain on the scaled mix,
3. rerun this expanded suite.

## Files

- `scripts/m3_scenario_benchmark.py`
- `results/m3_scenario_benchmark_scaled.json`
- `results/m3_scenario_benchmark_scaled_guarded.json`
