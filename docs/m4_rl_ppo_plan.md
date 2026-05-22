# M4 — Shaped-Reward RL/PPO Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Train a speaking-policy agent with shaped-reward RL/PPO on top of the stabilized distilled policy, while preserving the deterministic guardrails that keep turn-taking safe.

**Architecture:** M4 keeps the M3 distilled policy as the initialization point and introduces a reward-shaped environment that scores turn continuation, interruption avoidance, collision handling, and recovery. The learning loop remains offline/simulated first, using the same canonical and expanded scenario suites as acceptance gates before any live audio work.

**Tech Stack:** Python, PyTorch, the existing simulator harness in `scripts/m1_speaking_policy_sim.py`, the distilled-policy checkpoints from M2/M3, JSONL trace logging, and a small in-repo PPO-style trainer/evaluator.

---

## M4 Scope and Non-Goals

M4 is:
- a shaped reward environment for speech-policy learning
- an RL/PPO training loop initialized from the distilled policy
- an evaluation harness that compares raw policy, guarded policy, and PPO-tuned policy on canonical + expanded scenarios
- a conservative step toward better turn-taking and recovery

M4 is not:
- raw waveform generation
- a replacement for the deterministic guard rails
- a live-audio deployment milestone
- a move to the sound-first motor-learning track

## Success Criteria

M4 is successful if:
1. The reward function is explicit, logged, and reproducible.
2. The PPO-trained policy improves or matches the distilled policy on ambiguous turn-taking cases.
3. The canonical suite remains at 100%.
4. The expanded ambiguous suite does not regress.
5. The runtime remains safe: yield/collision handling stays guarded.

---

## Task 1: Codify the shaped reward and transition log schema

**Objective:** Define an explicit reward function for turn-taking events and a JSONL schema for reward traces.

**Files:**
- Create: `scripts/m4_reward.py`
- Modify: `scripts/m1_speaking_policy_sim.py` (only if a light transition hook is needed)
- Modify: `scripts/m4_collect_reward_traces.py`

**Step 1: Write the reward tests first**

Create a small test file or a script-level self-check that exercises the core cases:
- speak on silence: positive reward
- wait while user is speaking: neutral/slightly positive
- yield on interrupt/collision: positive reward
- miss an interrupt: negative reward
- clarify after long silence: small positive reward
- premature clarify on idle/no-queue silence: negative reward

The reward cases should be driven by the same canonical and ambiguous scenario descriptions used in M2/M3.

**Step 2: Run the checks to verify the reward mapping is explicit**

Run: `python -m py_compile scripts/m4_reward.py`
Expected: pass

Run a small reward sanity script over the canonical and expanded benchmark cases.
Expected: each case produces the intended reward sign.

**Step 3: Implement the minimal reward function**

Example shape:

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class RewardConfig:
    speak_on_silence: float = 1.0
    wait_while_user_speaks: float = 0.25
    yield_on_interrupt: float = 2.0
    yield_on_collision: float = 2.0
    clarify_after_silence: float = 0.5
    premature_clarify: float = -1.0
    missed_interrupt: float = -2.0
    repair_success: float = 1.5


def compute_reward(prev_features: Dict[str, Any], action: str, next_features: Dict[str, Any], cfg: RewardConfig) -> float:
    ...
```

Keep it deterministic and easy to inspect. Prefer a small set of high-signal terms over a complicated reward model.

**Step 4: Run the reward sanity checks again**

Expected: the intended actions score higher than the known failure actions for the same scenario class.

**Step 5: Commit**

```bash
git add scripts/m4_reward.py scripts/m4_collect_reward_traces.py scripts/m1_speaking_policy_sim.py
git commit -m "feat: add shaped reward scaffolding for M4"
```

---

## Task 2: Build the RL training loop around the distilled policy

**Objective:** Add a PPO-style trainer that starts from the distilled checkpoint and learns from reward-shaped simulated traces.

**Files:**
- Create: `scripts/m4_train_ppo.py`
- Create: `scripts/m4_rollout_env.py`
- Modify: `scripts/m2_train.py` only if a shared policy/model helper is needed

**Step 1: Write a minimal rollout test**

The rollout test should:
- load the `results/m3_model_checkpoints/trace_mix_scaled_v3/final_model.pt` checkpoint
- run a short simulated episode
- collect `(state, action, reward, done)` tuples
- confirm the rollout terminates and returns a finite reward

**Step 2: Run the rollout test**

Run: `python scripts/m4_rollout_env.py --checkpoint results/m3_model_checkpoints/trace_mix_scaled_v3/final_model.pt --episodes 2`
Expected: two finite episodes, no NaNs, no crashes.

**Step 3: Implement the PPO update loop**

Start small:
- shared policy architecture from the distilled model
- clipped policy gradient objective
- value head for advantage estimation
- short rollouts
- conservative learning rate

Do not optimize for final performance yet. Optimize for correct, inspectable learning behavior.

**Step 4: Run a tiny training smoke test**

Run: `python scripts/m4_train_ppo.py --checkpoint ... --episodes 20 --save-dir /tmp/m4_ppo_smoke`
Expected: training runs end-to-end and writes a checkpoint.

**Step 5: Commit**

```bash
git add scripts/m4_train_ppo.py scripts/m4_rollout_env.py
git commit -m "feat: add PPO training scaffold for M4"
```

---

## Task 3: Add evaluation and acceptance tests

**Objective:** Prove that PPO did not break the canonical behavior while improving or preserving ambiguous-case handling.

**Files:**
- Create: `scripts/m4_eval.py`
- Modify: `scripts/m3_scenario_benchmark.py` if it should accept multiple checkpoints more cleanly
- Create or update: `scripts/m4_acceptance_test.py`

**Step 1: Define the acceptance gates**

Minimum gates:
- canonical suite: 100%
- expanded suite: no regression vs the current v3 distilled checkpoint
- reward trace sanity: finite rewards, no repeated invalid transitions
- guard rails still dominate collision/yield handling

**Step 2: Run the evaluation on the baseline distilled checkpoint first**

This establishes the comparison line before PPO training meaningfully starts.

**Step 3: Add PPO-vs-baseline comparison output**

Report:
- canonical accuracy
- expanded ambiguity accuracy
- per-case deltas
- reward trend over training

**Step 4: Commit**

```bash
git add scripts/m4_eval.py scripts/m4_acceptance_test.py
git commit -m "feat: add M4 evaluation and acceptance gates"
```

---

## Task 4: Document the results and decide whether to go live

**Objective:** Produce a concise M4 result memo and decide whether the project is ready for live audio hardware evaluation.

**Files:**
- Create: `docs/m4_results_memo.md`
- Update: `docs/m1_speaking_policy_milestone.md` if the milestone ladder needs revision

**What to include:**
- baseline vs PPO results
- reward behavior
- whether ambiguous cases improved
- whether the canonical suite stayed stable
- whether the system is ready for live eval or should keep iterating offline

**Exit criteria:**
- either the RL loop is worth pushing to live evaluation
- or the data says to pivot again before spending time on hardware

---

## Implementation Order

1. Reward + trace schema
2. Rollout environment
3. PPO training loop
4. Evaluation/acceptance tests
5. Results memo

## Practical Notes

- Keep the deterministic guard rails intact.
- Don’t let PPO directly decide collision/yield if the guard can harden that decision more safely.
- Use the expanded scenario suite as the primary discriminator, not the canonical six-case suite alone.
- If the PPO loop becomes unstable, stop and tighten the reward or rollout shape before scaling up.

## Recommended First Step

Implement Task 1: codify the shaped reward and transition log schema.

That gives the RL work a concrete foundation and ensures every later comparison is grounded in the same reward semantics.
