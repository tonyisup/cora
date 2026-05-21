# Milestone 3 — Close-Out Document

**Status:** Complete
**Date:** 2026-05-21

## Goal

Milestone 3 tested whether the M2 distilled speaking policy could be stressed with a local-first LLM user agent and a more realistic, more ambiguous scenario distribution.

The intent was not to chase a bigger model or spend tokens on a remote loop. The intent was to make the conversation distribution broader while keeping the runtime local and reproducible.

## What was built

### Local-first user agent

Implemented a local-first user-agent path using the local OMLX endpoint, with a practical default of `gemma-4-e4b-it-4bit`.

Files:
- `scripts/m3_user_agent.py`
- `scripts/m3_generate_dialogues.py`
- `scripts/m3_collect_data.py`

The larger local model (`gemma-4-26b-a4b-it-4bit`) was tested, but it timed out on the smoke run and was not used as the default.

### Trace generation

Collected both a small smoke batch and a scaled local-user batch:
- `results/m3_dialogues_sample.jsonl`
- `results/m3_data_local_user_scaled.jsonl`

The scaled batch contained 21,901 frames across 25 trials.

### Retraining

Retrained the M2 distilled policy on:
- deterministic M2 data
- teacher-labeled M2 data
- M2 hard negatives
- M3 local-user traces
- a focused idle/no-queue hard-negative shard

Final checkpoint:
- `results/m3_model_checkpoints/trace_mix_scaled_v3/final_model.pt`

## Evaluation summary

### Canonical benchmark

The original six canonical M2 scenarios remained at 100% throughout the M3 work.

### Expanded benchmark

A new expanded benchmark was created to stress ambiguous boundaries that the canonical suite could not expose.

File:
- `scripts/m3_scenario_benchmark.py`

Initial expanded-suite result on the scaled trace mix:
- canonical subset: 100%
- ambiguous subset: 85.71%
- overall: 92.31%

The failure case was:
- `wait_in_idle_no_queue` → predicted `clarify` or `repair` depending on the checkpoint variant

### Targeted fix

A focused hard-negative shard was added for the idle/no-queue wait boundary:
- `results/m3_hardneg_idle_focus.jsonl`

After retraining with that shard, the final v3 checkpoint reached:
- canonical subset: 100%
- ambiguous subset: 100%
- overall: 100%

This was verified in both raw and guarded modes.

## Key outcome

M3 confirmed three things:

1. The local-user-agent pipeline works end-to-end.
2. The expanded benchmark is necessary; the canonical six-case suite was too easy.
3. The remaining ambiguity could be fixed with a targeted hard-negative rather than a larger model or remote token-heavy loop.

## Final judgment

M3 is complete and worthwhile.

The project now has:
- a local-first user-agent generator,
- a scaled trace collection path,
- a larger ambiguous scenario bank,
- and a distilled policy that clears both the canonical and expanded suites.

The important lesson is that the local user agent did not just create more data. It exposed a real policy boundary, and that boundary was resolved by adding the exact missing hard negative.

## Recommended next step

Move on to the next milestone or pivot analysis rather than continuing to scale M3 blindly.

If the project continues on this track, the next useful step is likely:
- broader scenario diversification,
- or a higher-level policy layer built on top of the now-stable distilled core.

## File index

Documentation:
- `docs/m3_progress_memo.md`
- `docs/m3_results_decision_memo.md`
- `docs/m3_expanded_scenario_report.md`
- `docs/m3_followup_idle_no_queue_fix.md`

Code:
- `scripts/m3_user_agent.py`
- `scripts/m3_generate_dialogues.py`
- `scripts/m3_collect_data.py`
- `scripts/m3_scenario_benchmark.py`

Key outputs:
- `results/m3_data_local_user_scaled.jsonl`
- `results/m3_hardneg_idle_focus.jsonl`
- `results/m3_model_checkpoints/trace_mix_scaled_v3/`
- `results/m3_scenario_benchmark_scaled_v3.json`
- `results/m3_scenario_benchmark_scaled_v3_guarded.json`
