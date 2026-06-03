# Cora Status — 2026-06-03

## Test Health

```
588 passed, 1 skipped, 0 failed
```

- 59 M8 tests (tracker, speaking policy, reward, model)
- The M2/M3 artifact replay tests now pass after restoring imported fixtures from the corollary copy; those assets are optional for fresh clones, but present in this checkout for local replay coverage
- Added an M9 live-capture retrain smoke harness so the current vowel path can be regression-tested without hardware

## Milestone Status

| Milestone | Status | Evidence |
|-----------|--------|----------|
| M0 | Done | Simulated loop closure, SVSR >= 20 dB |
| M1a | Done | Hardware-validated variable vowel probes |
| M1b | Done | Syllable repertoire, trust-gated re-adaptation |
| M1c | Done | Dual-input fusion, SELF/WORLD/BOTH hardware validation |
| M2 | Operational | Calibrated harness, M4 metadata, room-local scoring |
| M3 | Live hardware validated | Streaming runtime, gated playback, collision repair |
| M4 | Promoted | 100% self-recall, wired into streaming runtime, multi-step env + acceptance gate |
| M5 | Acceptance passed | Hippocampus + consolidation + multi-speaker world model |
| M6 | Live validated | SAG introspective queries functional |
| M7 | Complete | Trust-gated NLMS + DriftAlarm + IR bank + device profile |
| M8 | Complete | Speaking policy: all 3 phases PASS |

## M8 Progress

**M8a (Deterministic Harness) — DONE:**
- ConversationTracker: state machine over frame logs, extracts 20+ features
- SpeakingPolicy interface + HeuristicSpeakingPolicy (5-action: speak/wait/clarify/repair/yield)
- M8 reward function: repair success (+1.5), clarify efficiency (+1.0), stall penalty (-0.2/s), resume after yield (+0.8)
- Offline conversation simulator with 6 scripted episodes
- M8a acceptance ladder: PASS on all 6 episodes
- 44 new unit tests, all passing

**M8b (Distillation) — DONE:**
- 9012 demonstration samples collected (6 episodes x 5 perturbed seeds)
- M8DistilledPolicy MLP (21 -> 64 -> 64 -> 5, 5893 params)
- Training: 98.67% validation accuracy, early stopping at epoch 13
- Acceptance: 100% canonical match, 89.14% expanded match (target >=80%)
- Checkpoint: data/m8_distilled_policy.pt

**M8c (RL) — DONE:**
- ScriptedEpisodeEnv: gym-like wrapper around 6 scripted episodes for PPO training
- M8ActorCritic (13->64->32->5, ~4K params) trained with PPO + M8-only reward
- Training: 5.83 -> 19.83 best eval return (50 iterations, 128 eps/iter)
- Acceptance: +3062% improvement over distilled on expanded episodes (target >=10%)
- Distilled avg (expanded): +0.60 | PPO avg (expanded): +18.98
- Checkpoint: results/m8_ppo_scripted_v2/best_policy.pt
- Total M8 tests: 59 (44 original + 15 model tests)

**M8 Live (Hardware Validation) — DONE:**
- 3 scripted episodes on real audio hardware (2-speaker setup)
- Heuristic policy: end-to-end decisions from live frame logs; yield on interruption detected
- Known limitation: post-segment evaluation (not inline), crosstalk with muted speakers
- Acceptance report: `docs/m8_acceptance_report.md`
- Harness: `scripts/m8_live_harness.py` (supports heuristic, distilled, PPO; dry-run + live modes)

## Recent Commits (newest first)

- `2ce8b9d` feat(m9d): text association with bidirectional word-sound edges
- `e284ed2` feat(m9c): weak terminal confirmation and label attachment
- `0c51931` feat(m9b): repeatability and convergence with acoustic prototype refinement
- `6a27849` feat(m9): add acoustic SAG nodes and edge helpers
- `ab4dbd2` feat(m9a): sound imitation primitive — complete M9a implementation
- `c790af8` feat(m9a): sound imitation primitive — oscillator, substrate, controller, producer
- `0508411` docs: add storage/memory assessment to M9 spec

## Key Metrics

- Self/not-self attribution: 100% recall, 16.4% FP (same-family voices)
- Interruption latency: 1390ms (comparable to human ~1500ms)
- Pipeline overhead: ~330ms (2 callbacks + 3-frame classifier)
- Loop index (self-trigger rate): 0
- M8 test coverage: 59 tests (tracker, policy, reward, model)
- M8 scripted episodes: 6 (speak_on_silence, yield_on_interrupt, resume_after_yield, wait_while_user_speaks, repair_after_collision, user_silence_then_speak)
- M8 action set: 5 (speak, wait, clarify, repair, yield)
- M9a smoke: 10 trials, mean similarity 0.5460, std 0.0056
- M9b smoke: avg CV 0.0000, mean similarity 0.7181, PASS
- M9c smoke: baseline pass rate 0.67, terminal-confirmed pass rate 1.00, final similarity 0.5433
- M9d smoke: word->sound 1.00, sound->word 1.00, 20 nodes, 20 edges, PASS

## Open Items

- 5 data-missing test failures (M2/M3 artifacts) — need artifact regeneration or test skip markers
- `packages/salience/` is a stub (empty `__init__.py`)
- M8 known gap: clarify/repair actions not triggered by heuristic in current episode types — distilled never learned them
- M9 acoustic imitation: working on real hardware with clear temporal patterns
  - Substrate pre-trained on live audio (frequency classification, 87.5% acc)
  - Controller trained on 2-tone sequences (13 pairs x 5 repeats, 1690 samples)
  - 2-tone stimulus -> matching 2-tone imitation, scores 0.88-0.94 on live hardware
  - Single frequency: 220-660 Hz range, scores 0.92-0.99
  - Key breakthroughs: phase continuity, noise filtering, small controller, EMA smoothing
  - See docs/m9_troubleshooting_log.md for detailed iteration history
  - Next: formant synthesis for speech-like sounds, text association