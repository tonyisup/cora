# M4 Multi-Step Acceptance Report

**Date:** 2026-05-21

## What was tested

The acceptance gate now covers three layers:
1. the original one-step canonical suite
2. the one-step expanded ambiguous suite
3. the new multi-step sequence benchmark

The gate compares:
- the distilled v3 baseline
- the PPO smoke checkpoint trained on the multi-step environment

## Acceptance criteria

Pass conditions:
- distilled baseline keeps 100% on the canonical one-step suite
- PPO keeps 100% on the canonical one-step suite
- PPO does not regress materially below distilled on the multi-step benchmark

## Result

The acceptance test passed.

Observed values:
- one-step canonical/expanded accuracy: 100% for both distilled and PPO
- multi-step mean episode return:
  - distilled: 1.9625
  - PPO: 1.9625

The acceptance gate now tolerates floating-point noise, so identical smoke results are treated correctly as equivalent.

## Why this matters

This is the first time M4 has a gate that actually reflects the broader environment:
- the original one-step suites stay as guard rails
- the multi-step benchmark exposes delayed consequences
- acceptance now checks both safety and horizon-shaped behavior

## Interpretation

The broader environment is valid and the gate is stable.
However, the current PPO smoke run still does not outperform the distilled baseline on the multi-step benchmark.
That means the environment broadening is complete, but the learning signal still needs more tuning or more rollout budget before it becomes a performance win.

## Files

- `scripts/m4_multistep_acceptance_test.py`
- `results/m4_multistep_acceptance.json`
- `scripts/m4_multistep_benchmark.py`
- `scripts/m4_multistep_env.py`
- `scripts/m4_multistep_train_ppo.py`

## Recommendation

Keep M4 as an offline shaping effort for one more iteration if the goal is better PPO return.
If the goal is operational validation, the stack is now ready to be taken to live evaluation without further environment broadening.
