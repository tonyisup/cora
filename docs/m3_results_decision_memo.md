# M3 Results / Decision Memo — Scaled Local-User Traces

**Date:** 2026-05-21

## Summary

The scaled local-user collection succeeded and produced a materially larger trace set, but it did not improve the scenario-suite ceiling because the policy was already at 100% raw/guarded accuracy on the six canonical cases.

What the scaling did do is increase the amount of rare-action evidence in the training mix, especially clarify / yield / repair, while still keeping the collection fully local-first.

## Scaled collection

Run:
- `results/m3_data_local_user_scaled.jsonl`
- 25 trials
- 21,901 frames total
- local model: `gemma-4-e4b-it-4bit`
- timeout: 20s per turn

Action distribution:
- wait: 21,676
- speak: 75
- yield: 62
- repair: 62
- clarify: 26

This is still wait-heavy, but it is meaningfully richer than the 2-trial smoke batch.

## Retraining result

Checkpoint:
- `results/m3_model_checkpoints/trace_mix_scaled/final_model.pt`

Training mix:
- deterministic M2 data
- teacher-labeled M2 data
- M2 hard negatives
- M3 scaled local-user traces

Training outcome:
- dataset accuracy: 96.14%
- validation accuracy: 93.88%
- per-action validation accuracy:
  - speak: 100.0%
  - wait: 96.14%
  - clarify: 94.62%
  - repair: 91.16%
  - yield: 100.0%

## Scenario evaluation

Raw scenario-suite accuracy:
- 100%

Guarded scenario-suite accuracy:
- 100%

All six canonical cases remained correct:
- speak_on_silence
- wait_while_user_speaks
- yield_on_interrupt
- yield_on_collision
- clarify_after_silence
- repair_in_repair_state

## Comparison to the smoke mix

Compared to `trace_mix_smoke`, the scaled run:
- increased total training evidence from the local user agent
- improved the absolute count of rare actions
- lowered validation accuracy slightly, which is consistent with adding more diverse / harder examples
- did not change the canonical scenario suite because that suite was already solved

So the scaling was useful as a data-distribution experiment, not as a benchmark gain.

## Interpretation

The important result is that the local-user pipeline is now at a point where it can:
1. generate a much larger dataset,
2. preserve the runtime safety rails,
3. retrain the policy without breaking the canonical suite.

But the canonical suite is no longer the right discriminator for M3. It is saturated.

The next useful evaluation target should be one of:
- a larger, more diverse scenario bank that includes more ambiguous clarify / correction / overlap cases,
- a dedicated diversity report over the generated traces,
- or a decision memo on whether M3 has reached the point of diminishing returns and the project should shift toward the next milestone.

## Recommendation

Do not keep scaling blindly.

The local-user path works, but the current six-case scenario suite is too small to prove that more M3 data is improving policy behavior. The next step should be either:
- expand the scenario bank to stress the rare actions more directly, or
- stop M3 here and transition to the next milestone/pivot analysis.

## Files

- `results/m3_data_local_user_scaled.jsonl`
- `results/m3_model_checkpoints/trace_mix_scaled/`
- `results/m3_eval_trace_mix_scaled_raw.json`
- `results/m3_eval_trace_mix_scaled_guarded.json`
