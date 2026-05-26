# M8 Live Hardware Validation

> **Status:** In progress. Offline M8a/b/c complete. Live harness is the final step.

## Goal

Validate the M8 speaking policy on real audio hardware end-to-end. The streaming runtime handles real-time audio I/O, VAD, attribution, and TTS gating. The speaking policy decides *what action to take* (speak/wait/clarify/repair/yield) based on conversation state. Real TTS audio flows through the actual acoustic path.

## Architecture

```
  ┌──────────────────────────────────────────────────────────────┐
  │                  Live Episode Orchestrator                   │
  │                                                              │
  │  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
  │  │ Conversation │───▶│  Speaking    │───▶│  TTS Synthesis │  │
  │  │  Tracker     │    │  Policy      │    │  (edge-tts)    │  │
  │  │ (observes   │    │  (M8 policy  │    │                │  │
  │  │  frame logs)│    │   decides    │    │                │  │
  │  │             │    │   action)    │    │                │  │
  │  └─────────────┘    └──────┬───────┘    └───────┬────────┘  │
  │                            │                     │           │
  │                     action="speak"          tts_pcm, text    │
  │                     + text                                     │
  │                            │                     │           │
  │                            ▼                     ▼           │
  │                     ┌──────────────────────────────────┐    │
  │                     │     StreamingRuntime              │    │
  │                     │  (audio callback loop)            │    │
  │                     │  VAD→Attribution→Controller→Gate  │    │
  │                     └──────────────┬───────────────────┘    │
  │                                    │                        │
  └────────────────────────────────────┼────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                                     │
              ┌─────▼─────┐                         ┌─────▼─────┐
              │  Speaker   │                         │   Mic     │
              │  (TTS out) │                         │  (input)  │
              └─────┬──────┘                         └─────┬─────┘
                    │                                      │
                    └──────────────┬───────────────────────┘
                                   │
                             ┌─────▼──────┐
                             │  2nd Device │
                             │  (human     │
                             │   voice)    │
                             └────────────┘
```

The speaking policy does NOT run in the audio callback. It runs in the orchestrator loop, reads the ConversationTracker's post-hoc features, decides an action, and queues the next utterance. The streaming runtime handles real-time gating.

## Episode Protocol

Each live episode:

1. **Setup:** Load a script (agent utterances + user voice timing + expected policy actions)
2. **Start:** Begin streaming runtime with the first agent utterance queued
3. **Run:** Orchestrator monitors frame logs, runs speaking policy between utterances
4. **Simulate user:** Play pre-recorded human voice on a second device (like M7 two-speaker test)
5. **Log:** Record all policy decisions, controller transitions, timestamps
6. **Evaluate:** Compare live decisions against scripted ground truth

## Two Harnesses

### Harness A: Policy Decision Validation
- Pre-defined episode scripts with ground-truth actions at each step
- Speaking policy runs in "open loop" — its decisions don't affect playback timing
- Measures: decision accuracy, decision latency, action match rate
- No TTS generation latency (pre-synthesized audio)

### Harness B: Full End-to-End
- Speaking policy runs in "closed loop" — policy chooses actions that affect TTS output
- Real TTS synthesis between utterances (latency ~200-500ms with edge-tts)
- Simulated user voice from second device
- Measures: end-to-end conversation flow, interruption detection + yield, repair behavior

## Hardware Requirements
- Mic (e.g., BL-2021USB)
- Primary speaker (e.g., Mac mini Speakers) for agent TTS
- Second audio source (second speaker, phone, or laptop) for simulated human voice
- Same setup as M7 live interruption latency test

## Acceptance Criteria

Harness A (Policy Decision):
1. Decision accuracy >= 90% on canonical episodes (match offline heuristic)
2. Decision accuracy >= 75% on expanded episodes
3. Decision latency < 50ms (policy inference + feature extraction, excluding TTS)

Harness B (Full E2E):
1. All utterances complete without audio collisions
2. Correct yield on interruption (every overlap detected and handled)
3. No stalled conversations (> 3s dead air when speaking was expected)
4. Total episode completion within 120% of expected duration

## Action Routing

The speaking policy produces action strings. The orchestrator maps them to streaming runtime behavior:

| Policy Action | Orchestrator Behavior |
|---------------|----------------------|
| `speak` | Synthesize next utterance TTS → queue to streaming runtime |
| `wait` | Do nothing, keep monitoring |
| `repair` | Queue collision repair text ("Sorry, go ahead.") |
| `yield` | Signal streaming runtime to defer remaining TTS |
| `clarify` | Synthesize clarification utterance |

## Implementation Order

1. **Scripted live episode player** — plays pre-synthesized TTS + human voice on two devices, logs frame data. Reuse M7 two-speaker infrastructure.
2. **Offline decision evaluator** — runs speaking policy on recorded frame logs, compares decisions to ground truth
3. **Full E2E harness** — closed-loop with real-time policy decisions driving TTS queue
4. **Acceptance report** — per-episode metrics, decision accuracy, latency distribution

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/m8_live_harness.py` | Main live episode player (two-speaker, pre-synthesized audio) |
| `scripts/m8_live_e2e.py` | Full E2E with closed-loop policy + real TTS |
| `results/m8_live_acceptance.json` | Acceptance report |

## Files Summary

New files:
- `scripts/m8_live_harness.py`
- `scripts/m8_live_e2e.py`
- `docs/m8_live_acceptance_report.md`

Existing files reused:
- `packages/pfc/streaming_runtime.py` — audio callback loop (unchanged)
- `packages/pfc/conversation_tracker.py` — frame log observer
- `packages/pfc/speaking_policy.py` — policy interface + heuristic
- `packages/m8_policy/model.py` — distilled policy
- `packages/m8_policy/scripted_env.py` — scripted episode generator
- `scripts/m7_live_interruption_latency.py` — two-speaker setup pattern
