# M8 — Live Speaking Policy Validation

> **Status:** Scoped. M0–M7 complete. M8 is the first milestone of the self-speaking pivot.

## Goal

Take the completed self-listening stack (streaming runtime + M4 arbitration + controller + SAG) and validate it in live conversational scenarios. Measure whether Cora can make good speaking decisions — not just whether it can attribute self vs. world, but whether it can *act* on that attribution to sustain a conversation.

This is the bridge between "I can hear myself" (M0–M7) and "I can speak well" (M8+).

## Architecture

M8 reuses the existing streaming runtime and controller almost unchanged. The new pieces are:

```
                    ┌─────────────────────────────────┐
                    │         Perception Layer         │
                    │  (cochlea → thalamus → STG)      │
                    │  Already done: M0–M4             │
                    └──────────────┬──────────────────┘
                                   │ AttributionResult
                                   ▼
                    ┌─────────────────────────────────┐
                    │       Speech Policy Layer        │  ← NEW (M8)
                    │  Observes conversation state     │
                    │  Decides: speak/wait/clarify/    │
                    │    repair/yield                  │
                    └──────────────┬──────────────────┘
                                   │ ControllerDecision
                                   ▼
                    ┌─────────────────────────────────┐
                    │        Agentive Controller       │
                    │  (pfc/agentive_controller.py)    │
                    │  Already done: M3                │
                    │  Gates TTS, manages state        │
                    └──────────────┬──────────────────┘
                                   │ PCM output
                                   ▼
                    ┌─────────────────────────────────┐
                    │        Execution Layer           │
                    │  (broca/tts.py)                  │
                    │  Already done: M0                │
                    └─────────────────────────────────┘
```

### What's new in M8

1. **ConversationTracker** — observes frame logs and maintains a conversation state machine (idle → listening → speaking → repair). Runs as a post-hoc observer at first (like M6 SelfModel), then moves inline.

2. **SpeakingPolicy** — the decision function. Starts as deterministic heuristic (rules over conversation state), then distilled, then RL. Same progression as M2→M4.

3. **M8 reward function** — shaped reward for conversational quality. Extends `m4_reward.py` with new signals for repair success, clarification efficiency, and turn continuation.

4. **Offline simulator** — deterministic replay of recorded or scripted conversational episodes. TheSpeakingPolicy makes decisions, the simulator scores them. Enables rapid iteration before live hardware.

5. **Acceptance ladder** — replay + live harness with pass/fail criteria.

## Action Set

Small and explicit (from the pivot doc):

| Action | Meaning |
|--------|---------|
| `speak` | Emit the queued utterance now |
| `wait` | Stay silent, keep listening |
| `clarify` | Ask for clarification ("Sorry, could you repeat that?") |
| `repair` | Acknowledge collision and yield ("Sorry, go ahead.") |
| `yield` | Stop speaking, defer the rest of the utterance |

This is the same action set the controller already understands. M8 adds the policy layer that *chooses* among these actions based on conversation state, rather than relying purely on the reactive BOTH-frame overlap detection.

## Deterministic First (M8a)

Before any learning, build the evaluation harness:

**Step 1: Conversation state tracker**

- File: `packages/pfc/conversation_tracker.py`
- Reads frame logs (same interface as M6 SelfModel)
- Maintains: `ConversationState` (idle, listening, speaking, repair, clarification)
- Exposes: `get_conversation_features()` → dict of booleans/numeric features for the policy
- Features: `is_user_speaking`, `is_agent_speaking`, `seconds_since_last_user_speech`, `seconds_since_last_agent_speech`, `was_interrupted`, `was_collided`, `repair_count`, `clarification_count`, `turn_count`

**Step 2: Heuristic speaking policy**

- File: `packages/pfc/speaking_policy.py`
- `HeuristicSpeakingPolicy` — deterministic rules:
  - If user is speaking → `wait`
  - If silence > 2s and agent has queued output → `speak`
  - If BOTH detected and agent is speaking → `yield`
  - If agent was interrupted and has deferred output → `repair`
  - If user clarification detected → `clarify`
- `SpeakingPolicy` interface:
  ```python
  class SpeakingPolicy(ABC):
      @abstractmethod
      def decide(self, features: dict) -> str: ...  # returns one of ACTION_NAMES
      @abstractmethod
      def reset(self) -> None: ...
  ```

**Step 3: Offline conversation simulator**

- File: `scripts/m8_conversation_simulator.py`
- Deterministic replay of scripted conversational episodes
- Each episode is a sequence of (timestamp, event) pairs:
  - `user_speech_start`, `user_speech_end`
  - `agent_output_queued` (text + duration)
  - `user_interrupt`
  - `user_correction`
- The simulator feeds events into the policy, collects decisions, scores them
- Output: JSONL of (timestamp, features, action, reward) per step

**Step 4: M8 reward function**

- File: `scripts/m8_reward.py`
- Extends M4 reward with conversational metrics:
  - `successful_repair`: +1.5 (user continues after repair)
  - `successful_clarify`: +1.0 (user clarifies and conversation continues)
  - `smooth_turn_transition`: +0.5 (natural handoff, no gap > 1.5s)
  - `missed_repair`: -1.5 (should have repaired but didn't)
  - `unnecessary_clarify`: -0.5 (clarified when not needed)
  - `stalled_conversation`: -0.2 per second of dead air > 3s after agent was expected to speak

**Step 5: Acceptance criteria (M8a)**

File: `scripts/m8_acceptance_ladder.py`

Pass conditions on the scripted episode suite:
1. Heuristic policy achieves ≥ 80% of optimal reward on canonical episodes
2. No missed interrupts (all overlaps detected and yielded)
3. No false repairs (repair only triggered when collision actually occurred)
4. All episodes complete without `stalled_conversation` penalty > 2s

## Distillation (M8b)

Once the deterministic harness works:

- Script: `scripts/m8_collect_demonstrations.py` — run heuristic policy on expanded episode suite, collect (features, action) pairs
- Script: `scripts/m8_train_distilled.py` — train a small MLP to imitate the heuristic
- Model: `packages/m8_policy/distilled.py` — same architecture pattern as M2 distilled policy
- Gate: distilled policy must match heuristic ≥ 90% on canonical suite, ≥ 80% on expanded

## RL (M8c)

Only after distillation gates pass:

- Reuse `scripts/m4_train_ppo.py` infrastructure (already has multi-step env)
- Replace reward function with `m8_reward.py`
- Train on the multi-step conversation environment (already built for M4b)
- Gate: PPO must beat distilled on expanded suite by ≥ 10% reward improvement

## Evaluation Metrics

| Metric | Target | How measured |
|--------|--------|-------------|
| Turn-taking accuracy | ≥ 95% | Offline simulator: correct action at each step |
| Repair success rate | ≥ 90% | Post-repair: user continues within 2s |
| Clarification efficiency | ≥ 80% | Post-clarify: user provides needed info |
| Interruption response | 100% | All overlaps trigger yield within 1.5s |
| Conversation stall rate | < 5% | Episodes with > 3s dead air when action expected |
| End-to-end latency (speak decision) | < 500ms | From trigger event to action decision (offline) |

## Test Plan

New test files:

| File | What it tests |
|------|--------------|
| `tests/unit/test_conversation_tracker.py` | State machine transitions, feature extraction |
| `tests/unit/test_speaking_policy.py` | Heuristic rules, action selection |
| `tests/unit/test_m8_reward.py` | Reward computation for all signal types |
| `tests/unit/test_m8_simulator.py` | Episode replay, event sequencing |
| `tests/integration/test_m8_acceptance_ladder.py` | End-to-end acceptance criteria |
| `tests/integration/test_m8_live_smoke.py` | Live hardware smoke with heuristic policy |

## Execution Order

1. **ConversationTracker** + unit tests — pure observer, no behavior change
2. **HeuristicSpeakingPolicy** + unit tests — deterministic baseline
3. **M8 reward function** + unit tests
4. **Offline simulator** + scripted episodes
5. **M8a acceptance ladder** — must pass before M8b
6. Data collection + distillation
7. **M8b acceptance** — must pass before M8c
8. PPO training
9. **M8c acceptance** — must pass before declaring M8 complete
10. Live hardware validation (end-to-end with the learned policy)

## Open Risks

- **Offline-to-live gap:** The simulator is deterministic. Real conversations are messy. Expect the distilled policy to degrade on live hardware. Plan at least one "reality check" live run after M8b before going to M8c.
- **Reward shaping sensitivity:** The M8 reward weights will need tuning. The M4 experience (PPO smoke = distilled return at 1.9625) suggests the reward landscape is flat in places. Start conservative.
- **Action set too small?** The 5-action set may not cover all conversational situations. If the simulator reveals gaps, expand to include `restate` and `emphasize` before M8c.

## Non-Goals (M8)

- Sound-first motor learning (babbling, acoustic imitation) — that's Track 2 / M9
- Prosody control — the policy decides *what* action, not *how* to render it
- LLM as online policy — LLM is teacher/critic only in M8, same as M4
- Full SAG acoustic expansion — words-only SAG is sufficient for M8

## Files Summary

New files:
- `packages/pfc/conversation_tracker.py`
- `packages/pfc/speaking_policy.py`
- `packages/m8_policy/` (package for distilled + PPO policies)
- `scripts/m8_conversation_simulator.py`
- `scripts/m8_reward.py`
- `scripts/m8_collect_demonstrations.py`
- `scripts/m8_train_distilled.py`
- `scripts/m8_acceptance_ladder.py`
- `docs/m8_acceptance_report.md`

Existing files reused:
- `packages/pfc/streaming_runtime.py` — unchanged
- `packages/pfc/agentive_controller.py` — unchanged
- `scripts/m4_train_ppo.py` — reuse PPO infrastructure
- `scripts/m4_multistep_env.py` — reuse multi-step episode wrapper
- `scripts/m4_reward.py` — extend for M8 signals
- `packages/m2_policy/` — policy interface pattern reference
