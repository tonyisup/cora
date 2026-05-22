# M4b — Multi-Step Reward Environment Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Broaden the M4 speaking-policy RL setup from single-scenario episodes into multi-step turn-taking episodes so PPO can learn long-horizon behavior instead of only one-step action selection.

**Architecture:** Keep the existing shaped reward function, but wrap it in a deterministic multi-step episode generator that chains scenario states across turns. Each episode should expose a short sequence of observations, actions, transitions, and rewards so the trainer can see delayed consequences such as missed interrupts, premature clarifies, repair recovery, and returning to idle without queue. The canonical and expanded scenario suites remain the acceptance baseline; the new long-horizon suite adds sequence-level regressions.

**Tech Stack:** Python, PyTorch, the current distilled policy checkpoint, the M4 shaped reward module, and the existing benchmark/scenario definitions from M3.

---

## M4b Scope and Non-Goals

M4b is:
- a multi-step episode wrapper around the shaped reward environment
- a trace logger for state/action/reward trajectories
- a sequence-level benchmark to expose delayed reward effects
- the foundation for a broader PPO pass

M4b is not:
- live audio hardware evaluation
- a replacement for the deterministic guard rails
- a new speech recognition or TTS stack
- a large new action space

## Success Criteria

M4b is successful if:
1. Episodes contain multiple decision steps with state transitions, not just one-shot scoring.
2. The reward trace logs show how one action affects later observations.
3. The multi-step benchmark can expose regressions that the one-step benchmark cannot.
4. The distilled baseline still passes the canonical and expanded one-step suites.
5. PPO can be trained against the broader environment without breaking the existing safety rails.

---

## Task 1: Define the multi-step episode schema

**Objective:** Introduce a compact episode/step schema for chained turn-taking transitions.

**Files:**
- Create: `scripts/m4_multistep_env.py`
- Create: `scripts/m4_multistep_traces.py`

**Step 1: Write a failing smoke test / self-check first**

Add a small self-check that expects:
- an episode to contain at least 3 steps
- a reward trace to include `observation`, `action`, `reward`, `next_observation`, `done`, and `reason`
- at least one delayed consequence (e.g. a premature clarify in step 1 causing a repair or lower reward in step 2)

Run: `python -m py_compile scripts/m4_multistep_env.py`
Expected: initially fail until the scaffold exists.

**Step 2: Implement the minimal multi-step episode wrapper**

Start with a deterministic episode builder that chains a small number of states per episode. Example step sequence:
- step 1: idle/listen with queue/silence boundary
- step 2: user speech or overlap introduced based on previous action
- step 3: repair or settle back to idle

Suggested skeleton:

```python
@dataclass
class MultiStepTransition:
    obs: dict
    action: str
    reward: float
    next_obs: dict
    done: bool
    info: dict

class MultiStepTurnEnv:
    def reset(self, episode_id: int) -> dict: ...
    def step(self, action: str) -> tuple[dict, float, bool, dict]: ...
    def rollout(self, policy) -> list[MultiStepTransition]: ...
```

**Step 3: Run the smoke test**

Run: `python scripts/m4_multistep_env.py`
Expected: a short sample episode printed with at least 3 transitions.

**Step 4: Commit**

```bash
git add scripts/m4_multistep_env.py scripts/m4_multistep_traces.py
git commit -m "feat: add multi-step episode scaffold for M4"
```

---

## Task 2: Log multi-step reward traces

**Objective:** Produce audit-friendly traces that show delayed reward effects over a sequence.

**Files:**
- Create: `scripts/m4_multistep_trace_logger.py`
- Modify: `scripts/m4_reward.py` only if a tiny helper is needed for transition scoring

**Step 1: Add a trace logger test**

The logger should emit JSONL rows with fields:
- `episode_id`
- `step_idx`
- `observation`
- `action`
- `reward`
- `next_observation`
- `done`
- `reason`
- `components`

**Step 2: Generate a small trace batch**

Run: `python scripts/m4_multistep_trace_logger.py --episodes 5 --output /tmp/m4_multistep_smoke.jsonl`
Expected: multiple rows per episode; at least one delayed consequence case.

**Step 3: Verify the trace shape**

Read the first few lines and confirm that the later steps reflect earlier actions.

**Step 4: Commit**

```bash
git add scripts/m4_multistep_trace_logger.py
git commit -m "feat: add multi-step reward trace logging"
```

---

## Task 3: Add a sequence-level benchmark

**Objective:** Evaluate policies on short action sequences, not only on isolated scenarios.

**Files:**
- Create: `scripts/m4_multistep_benchmark.py`
- Create: `docs/m4_multistep_report.md`

**Step 1: Define sequence cases**

Include sequences like:
- idle wait -> user interrupts -> yield
- silent queue -> speak -> user overlap -> yield -> repair
- long silence -> clarify -> user responds -> wait
- premature clarify -> repair -> settle back to wait

**Step 2: Run the baseline distilled policy on the sequence suite**

Expected: the baseline should stay safe but may not maximize cumulative reward.

**Step 3: Compare against the PPO policy after retraining**

Report both:
- sequence-level return
- step-level correctness

**Step 4: Commit**

```bash
git add scripts/m4_multistep_benchmark.py docs/m4_multistep_report.md
git commit -m "feat: add multi-step benchmark for M4"
```

---

## Task 4: Hook the PPO trainer to multi-step rollouts

**Objective:** Replace the one-step rollout assumption with short multi-step episodes.

**Files:**
- Modify: `scripts/m4_train_ppo.py`
- Modify: `scripts/m4_rollout_env.py` or replace with `scripts/m4_multistep_env.py`

**Step 1: Write a small failing rollout test**

The trainer should confirm:
- trajectories have length > 1
- returns are accumulated across steps
- advantages are computed from multi-step returns

**Step 2: Implement discounted returns and GAE-lite**

Use a modest discount factor and a simple advantage estimator. Keep it readable and conservative.

**Step 3: Smoke-test the broader PPO loop**

Run a tiny training job with a few episodes and verify that the multi-step environment produces finite returns.

**Step 4: Commit**

```bash
git add scripts/m4_train_ppo.py scripts/m4_multistep_env.py
git commit -m "feat: broaden M4 PPO to multi-step episodes"
```

---

## Task 5: Results memo and decision gate

**Objective:** Decide whether the broader environment produces a measurable gain or whether the project should pivot toward live evaluation.

**Files:**
- Create: `docs/m4_multistep_results_memo.md`

**Include:**
- baseline vs PPO on the sequence suite
- canonical suite preservation
- ambiguous sequence improvements or regressions
- whether multi-step broadening actually exposed new learning signal

**Exit criteria:**
- either the broader environment produces a clear improvement
- or the data says the next useful step is live validation rather than more offline RL

---

## Recommended first step

Implement Task 1: the multi-step episode schema and environment scaffold.

That is the minimum change that turns M4 from one-step scoring into a true longer-horizon learning problem.
