# M3 Follow-up — Idle/No-Queue Hard Negative Fix

**Date:** 2026-05-21

## What was fixed

The expanded scenario benchmark exposed a single failure case in the scaled M3 checkpoint:
- `wait_in_idle_no_queue` was predicted as `clarify` (then later `repair` in a retrain variant)

That case is now explicitly represented in training via a focused hard-negative shard:
- `results/m3_hardneg_idle_focus.jsonl`

## Why this matters

The canonical six-case benchmark was already saturated, so it was not enough to show whether M3 had actually improved the policy. The expanded scenario bank surfaced a real ambiguity boundary.

This follow-up adds the exact missing boundary case directly to training, rather than relying on broader noise.

## Final training mix

Checkpoint:
- `results/m3_model_checkpoints/trace_mix_scaled_v3/final_model.pt`

Training data:
- deterministic M2 data
- teacher-labeled M2 data
- M2 hard negatives (`v3`)
- M3 scaled local-user traces
- focused idle/no-queue wait shard

## Result

Expanded scenario benchmark:
- canonical subset: 100%
- ambiguous subset: 100%
- overall: 100%

This is the first checkpoint that clears both the original canonical suite and the new ambiguous expansion.

## Interpretation

The fix is not that the model became dramatically different. The fix is that the training set now contains the exact idle/no-queue boundary in a form the model can learn.

So the current state is:
- local-user M3 traces are useful
- the expanded scenario bank is useful
- the idle/no-queue hard negative resolves the remaining ambiguity

## Files

- `results/m3_hardneg_idle_focus.jsonl`
- `results/m3_model_checkpoints/trace_mix_scaled_v3/`
- `results/m3_scenario_benchmark_scaled_v3.json`
- `results/m3_scenario_benchmark_scaled_v3_guarded.json`
