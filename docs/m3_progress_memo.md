# M3 Progress Memo — Local-First User Agent

**Date:** 2026-05-21

## What changed

M3 now has a working local-first user-agent path:
- `scripts/m3_user_agent.py`
- `scripts/m3_generate_dialogues.py`
- `scripts/m3_collect_data.py`

The generator uses the local OMLX endpoint first and falls back only if needed. The practical default is now the smaller local model `gemma-4-e4b-it-4bit`, since the larger `gemma-4-26b-a4b-it-4bit` timed out on smoke tests.

## What we verified

### User-agent logging

A capped smoke run produced valid JSONL traces with:
- raw model output
- parsed structured output
- latency
- scenario metadata
- history snapshots

Sample output was written to:
- `results/m3_dialogues_sample.jsonl`
- `/tmp/m3_local_collect_smoke.jsonl`

### Simulator-facing collection

The new M3 collector successfully plugged the local LLM user into the existing turn loop and emitted the same `(features, action)` records used by the M2 trainer.

Smoke run:
- 2 trials
- 1,560 frames collected
- action distribution:
  - wait: 1,535
  - speak: 7
  - yield: 6
  - repair: 6
  - clarify: 6

This is still heavily imbalanced toward `wait`, but it confirms the user-agent path is usable for policy data collection.

## Retraining result

I retrained the policy on a merged dataset:
- deterministic M2 data
- teacher-labeled M2 data
- M2 hard negatives
- M3 local-user smoke traces

Checkpoint:
- `results/m3_model_checkpoints/trace_mix_smoke/final_model.pt`

Training outcome:
- dataset accuracy: 97.41%
- validation accuracy: 97.68%
- per-action validation accuracy:
  - speak: 100.0%
  - wait: 97.36%
  - clarify: 89.25%
  - repair: 91.16%
  - yield: 100.0%

## Evaluation result

Scenario-suite accuracy stayed at 100% in both modes:
- raw: 100%
- guarded: 100%

All six benchmark cases remained correct:
- speak_on_silence
- wait_while_user_speaks
- yield_on_interrupt
- yield_on_collision
- clarify_after_silence
- repair_in_repair_state

## Interpretation

The important M3 signal so far is not that the policy suddenly changed shape — it didn’t. The important signal is that:
1. the local user-agent path works end-to-end,
2. it can produce policy-labeled traces,
3. those traces can be mixed into retraining without breaking the scenario suite.

That means M3 is now a real data-generation milestone rather than just a doc.

## Current judgment

M3 is not yet evidence that the local user agent is better than the scripted environment in a strong statistical sense. The smoke batch is too small and still too reply-heavy to claim that.

What we can say is:
- the infrastructure is working,
- the policy remains stable,
- and the next useful move is to scale the M3 trace batch enough to test whether diversity meaningfully increases clarify/repair/interrupt coverage.

## Next step

Continue M3 by collecting a larger local-user trace batch and comparing:
- action diversity,
- clarify/repair frequency,
- interruption/overlap frequency,
- and any scenario-suite deltas versus the current trace-mix checkpoint.

## Files

- `docs/m3_user_agent_contract.md`
- `scripts/m3_user_agent.py`
- `scripts/m3_generate_dialogues.py`
- `scripts/m3_collect_data.py`
- `results/m3_dialogues_sample.jsonl`
- `/tmp/m3_local_collect_smoke.jsonl`
- `results/m3_model_checkpoints/trace_mix_smoke/`
- `results/m3_eval_trace_mix_smoke_raw.json`
- `results/m3_eval_trace_mix_smoke_guarded.json`
