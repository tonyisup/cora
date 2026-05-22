# M4 Results Memo — Shaped-Reward PPO

**Date:** 2026-05-21

## Summary

M4 now has a complete shaped-reward RL scaffold:
- explicit reward function
- reward trace logging
- rollout environment
- PPO trainer
- evaluation and acceptance tests

The PPO smoke run completed successfully, but it did not improve the policy behavior on the current benchmark because the distilled v3 baseline was already saturating the canonical and expanded suites.

## What was tested

### Baseline distilled policy
- canonical + expanded suite accuracy: 100%
- total shaped reward: +15.85

### PPO smoke checkpoint
- canonical + expanded suite accuracy: 100%
- total shaped reward: +15.85

### Acceptance result
- PASS
- both distilled and PPO policies satisfy the current acceptance gates

## Interpretation

This is a useful outcome even though the PPO checkpoint did not outperform the baseline on the current scenario bank.

It tells us:
1. The RL plumbing works end-to-end.
2. The shaped reward is stable and auditable.
3. The current benchmark is now too easy to show a PPO gain.

In other words, the infrastructure is correct, but the evaluation is not yet rich enough to expose where PPO would help.

## Recommendation

Do not blindly scale PPO on the same scenario bank.

The next useful step is either:
- broaden the reward-bearing environment with longer-horizon or multi-step episodes, or
- move toward live evaluation if the current goal is to validate the safety and latency of the already-stable policy stack.

## Files

### Code
- `scripts/m4_reward.py`
- `scripts/m4_collect_reward_traces.py`
- `scripts/m4_rollout_env.py`
- `scripts/m4_train_ppo.py`
- `scripts/m4_eval.py`
- `scripts/m4_acceptance_test.py`

### Reports
- `results/m4_ppo_smoke/`
- `results/m4_acceptance_test.json` (if saved via CLI)

## Conclusion

M4 is complete as an engineering milestone: the PPO loop exists, it runs, and it passes acceptance. But it does not yet produce a measurable gain on the current benchmark because the distilled policy is already at the ceiling for the current scenario set.
