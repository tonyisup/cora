# M4 Decision Memo — Multi-Step Broadening

**Date:** 2026-05-21

## Summary

M4 has now been broadened from one-step scenario scoring into a true multi-step turn-taking environment, and the acceptance gate passes.

What is now true:
- canonical one-step suite: 100% for distilled and PPO
- expanded one-step suite: 100% for distilled and PPO
- multi-step acceptance gate: pass
- sequence benchmark is now meaningful enough to expose delayed consequences

## What the results say

The current PPO smoke run does not outperform the distilled baseline on the multi-step benchmark.
That means the broadened environment is valid, but the PPO loop is not yet producing a performance gain.

This is not a failure of the environment work.
It means the project has crossed the point where more offline broadening is only worth doing if the goal is specifically to improve PPO return.

## Decision

Do not keep broadening the environment blindly.

The next step should be chosen by goal:

1. If the goal is policy performance:
   - run a longer PPO training pass on the multi-step environment
   - tune reward weights around the repair / interrupt boundaries
   - keep the canonical + expanded + multi-step acceptance gates in place

2. If the goal is operational validation:
   - move to live evaluation
   - test the current stable stack on real hardware and measure interruption latency / safety

## Recommendation

Given the current state, the better next move is live validation unless there is a strong reason to spend more time tuning PPO offline.
The environment broadening has already paid off; more offline broadening is likely diminishing returns.

## Artifacts

- `docs/m4_multistep_plan.md`
- `docs/m4_multistep_report.md`
- `docs/m4_multistep_acceptance_report.md`
- `results/m4_multistep_acceptance.json`
- `scripts/m4_multistep_env.py`
- `scripts/m4_multistep_train_ppo.py`
- `scripts/m4_multistep_benchmark.py`
- `scripts/m4_multistep_acceptance_test.py`

## Bottom line

M4 is now structurally complete. The broadened environment is in place, the acceptance gate is green, and the remaining question is no longer "can the environment see the problem?" but "do we want to spend more cycles optimizing offline, or do we validate on hardware now?"
