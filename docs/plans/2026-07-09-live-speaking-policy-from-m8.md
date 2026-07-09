# Live Speaking Policy from og-cora M8 Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a safe, test-backed path from cora-infans' simulation-only self-prediction work to eventual live playback by porting the *policy/acceptance shape* of og-cora M8 first, not the live speaker path.

**Architecture:** Start with an offline SPEAK-state policy layer that consumes deterministic conversation/self-audio events and emits explicit actions (`speak`, `wait`, `yield`, `repair`, `clarify`). The first implementation must remain simulation-only: it may schedule or authorize a would-be utterance, but it must not play audio. Only after offline policy + promotion gates pass should we add a separate live SPEAK/mic-suppression milestone.

**Tech Stack:** Python 3.13, numpy, pytest, existing `cora_infans` package, existing `self_suppression.py`, `motor_plan.py`, `live_listener.py`, and `theta_gamma.py`. No scipy, no live audio playback, no extra dependencies in the first slice.

---

## Source reference

Primary reference:
- `/Users/juicebox/Projects/cora/.worktrees/og-cora/docs/m8_speaking_policy.md`

Important M8 lesson:
- M8 is not “play audio now.”
- M8 is a policy/acceptance ladder: `ConversationTracker` → deterministic `SpeakingPolicy` → reward → offline simulator → acceptance gate → only then live validation.

Current cora-infans baseline:
- Simulation-only signal self-suppression: `src/cora_infans/io/self_suppression.py`
- Forward-model learning from prediction error: same file
- Robustness sweep: same file
- Promotion gate: same file
- Live listener intentionally logs motor-imitation trials without playback: `src/cora_infans/io/live_listener.py`
- Scheduling-level corollary-discharge proxy: `src/cora_infans/theta_gamma.py`

Critical boundary:
- Do not add live playback in this plan.
- The plan ends at `live_playback_allowed=false` plus a future design doc for live SPEAK state.

---

## Recommended milestone sequence

### M8-infans-A — Offline speaking-policy scaffold

Purpose: create the policy shape and episode replay harness without playback.

Files:
- Create: `src/cora_infans/speaking_policy.py`
- Create: `tests/test_speaking_policy.py`
- Modify later: `src/cora_infans/cli.py`

Core types:
- `SpeakingAction = Literal["speak", "wait", "yield", "repair", "clarify"]`
- `ConversationState`: frozen dataclass with state features
- `HeuristicSpeakingPolicy.decide(features) -> SpeakingAction`

Initial action rules, adapted from og-cora M8:
1. If `is_user_speaking` → `wait`
2. If `is_agent_speaking and was_collided` → `yield`
3. If `was_interrupted and has_deferred_output` → `repair`
4. If `needs_clarification` → `clarify`
5. If `queued_output and silence_ms >= 2000` → `speak`
6. Else → `wait`

Acceptance for this milestone:
- No hardware.
- No playback.
- Unit tests prove every action branch.
- CLI/describe lists `simulation_only_speaking_policy` as implemented.

### M8-infans-B — Deterministic conversation tracker

Purpose: convert scripted events into policy features.

Files:
- Create: `src/cora_infans/conversation_tracker.py`
- Create: `tests/test_conversation_tracker.py`

Inputs:
- `user_speech_start`
- `user_speech_end`
- `agent_output_queued`
- `agent_speech_start`
- `agent_speech_end`
- `overlap_detected`
- `user_correction`

Output:
- `ConversationFeatures` / dict with:
  - `is_user_speaking`
  - `is_agent_speaking`
  - `queued_output`
  - `has_deferred_output`
  - `needs_clarification`
  - `was_interrupted`
  - `was_collided`
  - `silence_ms`
  - `repair_count`
  - `clarification_count`
  - `turn_count`

Acceptance:
- Tracker is a pure observer.
- It has no side effects and no audio I/O.
- Tests cover transitions and reset behavior.

### M8-infans-C — Offline episode simulator + reward

Purpose: replay scripted episodes, ask the policy for actions, and score behavior.

Files:
- Create: `src/cora_infans/speaking_simulator.py`
- Create: `tests/test_speaking_simulator.py`
- Create: `src/cora_infans/speaking_reward.py`
- Create: `tests/test_speaking_reward.py`

Episode shape:
```python
{
    "episode_id": "interrupt_then_repair",
    "events": [
        {"timestamp_ms": 0, "kind": "agent_output_queued", "text": "...", "duration_ms": 1200},
        {"timestamp_ms": 500, "kind": "user_speech_start"},
        {"timestamp_ms": 650, "kind": "overlap_detected"},
        {"timestamp_ms": 1200, "kind": "user_speech_end"},
    ],
}
```

Reward terms, adapted from og-cora M8:
- `successful_repair`: +1.5
- `successful_clarify`: +1.0
- `smooth_turn_transition`: +0.5
- `missed_repair`: -1.5
- `unnecessary_clarify`: -0.5
- `stalled_conversation`: -0.2 per second after 3 seconds

Acceptance:
- Simulator emits JSONable steps: timestamp, features, action, reward.
- Canonical episodes are deterministic.
- No LLM, no live hardware, no TTS playback.

### M8-infans-D — Speaking-policy acceptance gate

Purpose: produce a single promotion/hold verdict for the offline policy, analogous to the self-forward-model promotion gate.

Files:
- Create or extend: `src/cora_infans/speaking_simulator.py`
- Create: `tests/test_speaking_acceptance.py`
- Modify: `src/cora_infans/cli.py`

CLI:
```bash
PYTHONPATH=src python3 -m cora_infans --promote-speaking-policy
```

Gate criteria:
1. Canonical reward ≥ 80% of oracle/expected reward.
2. All overlap episodes yield within the modeled window.
3. No false repairs when no collision occurred.
4. No unnecessary clarification in clean episodes.
5. No stalled-conversation penalty > 2 seconds.
6. `live_playback_allowed` remains false.

Output:
```json
{
  "mode": "simulation_only_speaking_policy_promotion_gate",
  "verdict": "promote_simulation_only" | "hold",
  "live_playback": false,
  "live_playback_allowed": false,
  "criteria": {...},
  "episode_reports": [...]
}
```

Acceptance:
- Full pytest suite passes.
- CLI smoke returns `promote_simulation_only` for canonical fixtures.
- Failing-threshold test can force `hold`.

### M8-infans-E — Integration with self-prediction reports, still offline

Purpose: connect speaking policy to the self-suppression/prediction-error evidence without playing sound.

Files:
- Modify: `src/cora_infans/speaking_simulator.py`
- Modify: `src/cora_infans/io/self_suppression.py` only if needed for shared reporting helpers
- Create/extend: `tests/test_speaking_self_prediction_integration.py`

Behavior:
- Episode can include a `self_forward_model_gate` result.
- Speaking policy must not authorize `speak` unless the self-forward-model gate passed simulation-only promotion.
- Even when authorized, the output is `would_speak`, not playback.

Acceptance:
- If self-forward-model promotion is `hold`, speaking policy returns `wait` or `repair`, not `speak`.
- If promotion passes and silence/queued-output criteria are met, policy returns `speak_authorized_simulation_only` or reports `speak` with `live_playback_allowed=false`.

### M8-infans-F — Live SPEAK-state design doc, not implementation

Purpose: before any live playback code, write the design boundary explicitly.

Files:
- Create: `docs/plans/YYYY-MM-DD-live-speak-state-and-mic-suppression.md`

Must answer:
1. What is the SPEAK state?
2. What captures are suppressed or tagged during SPEAK?
3. How is efference PCM retained?
4. How is roundtrip latency measured?
5. What is the no-feedback safety policy?
6. How does `LiveListener.listen()` opt in?
7. What command name makes the risk explicit?
8. What are the live smoke acceptance criteria?

Non-goal:
- Do not implement playback in this doc task.

---

## First implementation slice recommendation

Start with M8-infans-A only.

Why:
- It ports the highest-value M8 idea: explicit speaking policy before live action.
- It is safe, simulation-only, and small.
- It creates the action vocabulary needed for later SPEAK-state work.
- It does not disturb the just-completed self-forward-model simulation gate.

---

## Detailed first slice tasks

### Task 1: Add speaking action policy tests

**Objective:** Define the policy contract and action branches.

**Files:**
- Create: `tests/test_speaking_policy.py`
- Later create: `src/cora_infans/speaking_policy.py`

**Test cases:**
- user speaking → `wait`
- agent collided while speaking → `yield`
- interrupted with deferred output → `repair`
- needs clarification → `clarify`
- queued output after 2s silence → `speak`
- no queued output → `wait`

**Run:**
```bash
python3 -m pytest tests/test_speaking_policy.py -q
```

**Expected RED:** import error or missing class.

### Task 2: Implement `speaking_policy.py`

**Objective:** Make the Task 1 tests pass with pure deterministic logic.

**Files:**
- Create: `src/cora_infans/speaking_policy.py`

**Implementation notes:**
- Use dataclasses and `typing.Literal` only.
- No dependency on live listener.
- No subprocess/audio calls.

**Run:**
```bash
python3 -m pytest tests/test_speaking_policy.py -q
```

**Expected GREEN:** all speaking policy tests pass.

### Task 3: Add CLI describe surfacing

**Objective:** Make the new subsystem visible.

**Files:**
- Modify: `src/cora_infans/cli.py`
- Modify: `tests/test_cli.py`

**Behavior:**
`--describe` includes `simulation_only_speaking_policy` in `implemented_subsystems`.

**Run:**
```bash
python3 -m pytest tests/test_cli.py -q
```

### Task 4: Full verification and commit

**Run:**
```bash
python3 -m pytest tests -q
```

**Expected:** existing count + new tests, all passing.

**Commit:**
```bash
git add src/cora_infans/speaking_policy.py tests/test_speaking_policy.py src/cora_infans/cli.py tests/test_cli.py docs/plans/2026-07-09-live-speaking-policy-from-m8.md
git commit -m "Plan and start simulation-only speaking policy from og-cora M8"
```

---

## Do not do yet

- Do not play audio.
- Do not add a live `--speak-live` command.
- Do not open the mic and speaker at the same time for self-playback.
- Do not port og-cora PPO/distillation yet.
- Do not add torch/librosa/scipy dependencies.
- Do not claim biological corollary discharge beyond the simulation and scheduling boundaries already documented.

---

## Strategic recommendation

The right next implementation is M8-infans-A through D: build the speaking-policy ladder offline. Only after that should we write the live SPEAK/mic-suppression design doc. Live playback should remain gated behind both:

1. `--promote-self-forward-model` passing, and
2. `--promote-speaking-policy` passing.

That gives us two independent simulation gates before any embodied playback: one for self-audio prediction, one for conversational action selection.
