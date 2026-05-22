# M4 Multi-Step Broadening Report

**Date:** 2026-05-21

## Why this was needed

The original M4 PPO setup was still effectively one-step-per-scenario. That made the canonical and expanded suites too easy: the distilled baseline already saturated the benchmark, so PPO had nowhere obvious to improve.

## What changed

Added a multi-step environment that chains short turn-taking sequences:
- idle/no-queue boundary sequence
- interrupt/yield sequence
- repair recovery sequence
- silence-to-speak sequence

Files:
- `scripts/m4_multistep_env.py`
- `scripts/m4_multistep_traces.py`
- `scripts/m4_multistep_benchmark.py`
- `scripts/m4_multistep_train_ppo.py`
- `docs/m4_multistep_plan.md`

## Smoke-test result

The multi-step environment produced 12 transitions across 4 episodes in the smoke run.

The broader benchmark now exposes real horizon effects:
- heuristic policy mean episode return: 3.175
- distilled v3 mean episode return: 1.9625
- PPO smoke mean episode return: 1.9625

So the new horizon is informative, but the current PPO smoke run does not yet beat the distilled baseline.

## Key observation

The multi-step benchmark exposed a real weakness in the distilled baseline:
- the `repair_chain` episode includes a negative intermediate step (`bad_yield=-1.00`)
- the overall sequence return is lower than the heuristic baseline

That is the first concrete sign that the broader environment is doing useful work.

## Interpretation

This broadening succeeded in the most important way:
- it turned M4 into a sequence problem instead of isolated action scoring
- it created delayed consequences that the one-step benchmark could not show
- it gave PPO a real horizon to optimize against

But the current PPO smoke run is not yet long enough or tuned enough to beat the baseline.

## Recommendation

Continue the broader-environment path before considering live evaluation.

The next useful step is either:
1. run a longer PPO training pass on the multi-step environment, or
2. refine the multi-step reward so the repair/interrupt boundaries create a stronger learning gradient.

## Artifacts

Benchmark outputs:
- `results/m4_multistep_benchmark_heuristic.json`
- `results/m4_multistep_benchmark_distilled.json`
- `results/m4_multistep_benchmark_ppo.json`

Trace output:
- `/tmp/m4_multistep_smoke.jsonl`

## Bottom line

Broadening first was the right call. The environment is now sufficiently sequence-shaped to reveal regressions and delayed consequences, but the PPO loop still needs more tuning before it can outperform the distilled baseline.
