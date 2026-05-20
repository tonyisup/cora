# Track 1 — Milestone 1: Offline Speaking Policy (Deterministic Baseline)

## Goal

Build a deterministic speaking policy that chooses among a small action set
(`speak`, `wait`, `clarify`, `repair`, `yield`) in an offline simulated
environment. Success means the policy measurably improves conversation flow
across repeated trials.

## Scope

- **Offline only.** No live microphone or speakers. All audio is synthetic.
- **Deterministic or prompt-based policy.** No learned model in v1.
- **Simulated user.** A scripted user agent that responds to the speaking
  agent with realistic timing and content. LLM-based users are deferred.
- **Measurable outcomes.** Every run produces metrics: interruption rate,
  turn continuation rate, repair success rate, average gap timing.

## Action Set

| Action   | Meaning                                              |
|----------|------------------------------------------------------|
| `speak`  | Emit the next queued utterance via TTS               |
| `wait`   | Stay silent; keep listening                          |
| `clarify`| Ask a clarification question                         |
| `repair` | Acknowledge collision / yield floor ("Sorry, go ahead") |
| `yield`  | Stop speaking immediately; transition to LISTEN      |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Offline Simulator (scripts/m1_speaking_policy_sim.py)      │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────┐  │
│  │  Simulated    │   │  Speaking    │   │  Controllable  │  │
│  │  User Agent   │──▶│  Policy      │──▶│  TTS (Broca)  │  │
│  │  (scripted/   │   │  (determin-  │   │  edge-tts      │  │
│  │   LLM-based)  │   │   istic or   │   │  rate, pitch   │  │
│  │              ◀──│   prompt-     │◀──│  params)       │  │
│  │              │   │   based)     │   │               │  │
│  └──────┬───────┘   └──────┬───────┘   └───────────────┘  │
│         │                  │                                │
│         ▼                  ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Perception Layer                                     │  │
│  │  - VAD energy (cochlea.vad)                          │  │
│  │  - Silence duration                                   │  │
│  │  - Attribution verdict (thalamus.attributor)          │  │
│  │  - ASR transcript (stg.asr)                          │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Feedback / Metrics Layer                             │  │
│  │  - interruption count                                 │  │
│  │  - turn continuation rate                             │  │
│  │  - repair success rate                                │  │
│  │  - avg response gap (ms)                              │  │
│  │  - clarification efficiency                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Reuse Map

| Component                  | Source Package            | Role in Milestone 1                     |
|----------------------------|---------------------------|-----------------------------------------|
| TTS synthesis              | `packages/broca/tts.py`   | Audio output (rate/pitch controllable)  |
| VAD                        | `packages/cochlea/vad.py` | Speech/silence detection                |
| Attribution                | `packages/thalamus/attributor.py` | Self vs world verdict           |
| ASR                        | `packages/stg/asr.py`     | Transcribe user speech                  |
| AgentiveController         | `packages/pfc/agentive_controller.py` | Turn-taking state machine   |
| StreamingRuntime (adapted) | `packages/pfc/streaming_runtime.py` | Synthetic frame injection   |
| Simulated loop             | `packages/pfc/simulated_loop.py` | Template for offline orchestration |
| M2 policy pattern          | `packages/m2_policy/`     | Conservative action selection template  |
| Bus events                 | `packages/bus/events.py`  | Event schemas for simulator             |
| M7 latency test            | `scripts/m7_interruption_latency_test.py` | Latency measurement template |

## Policy Logic (Deterministic Baseline)

The first implementation is a deterministic state machine. This makes every
decision inspectable and debuggable.

```
State: IDLE
  On user_speech_start → transition to LISTEN

State: LISTEN
  On user_speech_end + silence > 300ms:
    If agent has queued utterance → action = speak
    Else → action = wait
  On user_speech_continues > 2s without pause → action = wait
  On user_interrupts_agent → action = yield

State: SPEAK
  On user_speech_detected (BOTH frame, energy_ratio > threshold):
    If overlap_duration < 200ms → action = yield (collision repair)
    Else → action = yield + queue repair utterance
  On agent_utterance_complete:
    If user_responds_within_2s → transition to LISTEN
    Else → action = clarify

State: REPAIR
  After repair_utterance_complete → transition to LISTEN
```

### Prompt-Based Variant

Once the deterministic baseline is validated, swap the decision logic for an
LLM prompt that receives the same context. This is a calibration baseline,
not the product policy.

```
Context:
- Current state: {state}
- Last user utterance: {user_text}
- Last agent utterance: {agent_text}
- Silence duration: {silence_ms}ms
- VAD energy: {energy_db} dB
- Attribution: {verdict}
- Conversation history (last 3 turns): {history}

Available actions: speak, wait, clarify, repair, yield

Choose the best action and explain why.
```

The LLM's action choice is executed; its explanation is logged for analysis.
This gives us a strong baseline to distill from later.

## Simulated User Agent

Two tiers:

### Tier 1: Scripted (Deterministic)

A finite state machine that responds to the agent with pre-recorded or
synthesized utterances. Behavior rules:

- Responds to agent speech after a random gap of 200-800ms (human-like)
- Interrupts the agent with 10% probability when agent speaks > 3s
- Asks for clarification if agent speech is ambiguous (scripted trigger)
- Provides correction if agent says something factually wrong (scripted trigger)

### Tier 2: LLM-Based (Stochastic)

This is intentionally deferred until a later milestone. Once the scripted user
and deterministic policy are stable, an LLM-based user agent can be introduced
as a stress test for diversity and brittleness.

## Metrics

All metrics are computed per-run and aggregated across N trials (default N=50).

| Metric                    | Definition                                              | Target (v1)        |
|---------------------------|---------------------------------------------------------|--------------------|
| Interruption rate         | # times user interrupts agent / # agent speaking turns  | < 15%              |
| Turn continuation rate    | # smooth turn-takes / total turns                       | > 70%              |
| Repair success rate       | # successful repairs / # repairs attempted              | > 80%              |
| Avg response gap (ms)     | Mean silence between user speech end and agent speech   | 300-800ms          |
| Clarification efficiency  | # clarifications leading to successful turn / total     | > 60%              |
| Collision rate            | # BOTH-frame collisions / # agent speaking turns        | < 10%              |

## Implementation Plan

### Phase 1: Harness (1-2 days)

1. Create `scripts/m1_speaking_policy_sim.py`
2. Implement scripted user agent (Tier 1)
3. Wire Broca TTS for both agent and user
4. Reuse `pfc/simulated_loop.py` pattern for orchestration
5. Output metrics as JSON

### Phase 2: Deterministic Policy (1-2 days)

1. Implement the state machine described above
2. Wire VAD + silence detection as input features
3. Wire attribution verdicts from thalamus
4. Run 50 trials, collect metrics
5. Tune thresholds (silence_ms, energy_ratio, etc.)

### Phase 3: Prompt-Based Policy (1 day)

1. Replace state machine with LLM prompt
2. Log LLM explanations alongside actions
3. Run 50 trials, compare metrics vs deterministic baseline
4. Document where LLM outperforms and where it doesn't

### Phase 4: Analysis & Distillation Prep (1 day)

1. Compare deterministic vs prompt-based metrics
2. Identify decision patterns suitable for distillation
3. Document findings in `docs/m1_speaking_policy_results.md`
4. Define distillation spec for Milestone 2

## File Map

```
scripts/
  m1_speaking_policy_sim.py          # Main simulator harness
  m1_speaking_policy_eval.py         # Batch eval runner (N trials)
  m1_speaking_policy_report.py       # Metrics aggregation + plotting

docs/
  m1_speaking_policy_results.md      # Results after each phase

tests/
  integration/
    test_m1_speaking_policy.py       # Acceptance tests for policy
```

## Success Criteria

- [ ] Harness runs 50 trials without errors
- [ ] Deterministic policy achieves targets on all 6 metrics
- [ ] Prompt-based policy matches or exceeds deterministic on >= 3 metrics
- [ ] All decision logic is logged and inspectable
- [ ] Results doc compares both policies with concrete numbers

## Next Milestones (Preview)

**Milestone 2:** Distill the deterministic/prompt policy into a small
classifier (e.g., logistic regression or small MLP) trained on logged
(state, action) pairs from Milestone 1.

**Milestone 3:** Replace scripted user with LLM user agent (Tier 2).
Retrain distilled policy on more diverse conversations.

**Milestone 4:** Introduce RL/PPO with shaped reward:
  reward = turn_continuation_bonus - interruption_penalty - collision_penalty
Only after simulator, action space, and reward are stable.

**Milestone 5:** Live evaluation — connect to real audio hardware,
measure interruption latency using M7 harness.
