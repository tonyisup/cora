# Cora Status — 2026-05-23

## Test Health

```
538 passed, 1 skipped, 5 failed (data-missing)
```

- 59 M8 tests (tracker, speaking policy, reward, model)
- 5 unit tests fail with FileNotFoundError — all are M2/M3 artifact replay tests that need data files not present in local data/
- These are environment/data availability issues, not code regressions

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

## Recent Commits

- `b557863` docs: update M8 status — all phases complete (M8a/b/c PASS)
- `e3deb64` docs: M8c acceptance PASS — update README status
- `f57368b` feat(m8c): ScriptedEpisodeEnv bridges PPO to scripted episodes — M8c PASS
- `76ad1fc` docs: update M8 status — M8a/b PASS, M8c partial
- `bd2ffa5` feat(m8c): PPO training pipeline with M8 reward
- `03bc541` feat(m8b): distilled speaking policy — 98.7% val accuracy
- `61c231f` feat(m8): conversation tracker, speaking policy, reward, simulator
- `4fa49ff` feat(m8): acceptance ladder passes on 6 scripted episodes
- `e3e0651` docs: update test count to 523, M8a acceptance status

## Key Metrics

- Self/not-self attribution: 100% recall, 16.4% FP (same-family voices)
- Interruption latency: 1390ms (comparable to human ~1500ms)
- Pipeline overhead: ~330ms (2 callbacks + 3-frame classifier)
- Loop index (self-trigger rate): 0
- M8 test coverage: 59 tests (tracker, policy, reward, model)
- M8 scripted episodes: 6 (speak_on_silence, yield_on_interrupt, resume_after_yield, wait_while_user_speaks, repair_after_collision, user_silence_then_speak)
- M8 action set: 5 (speak, wait, clarify, repair, yield)

## Open Items

- 5 data-missing test failures (M2/M3 artifacts) — need artifact regeneration or test skip markers
- `packages/salience/` is a stub (empty `__init__.py`)
- M8 known gap: clarify/repair actions not triggered by heuristic in current episode types — distilled never learned them
- M9: scoped next milestone (Track 2 sound-first motor learning from `docs/pivot_learns_to_speak.md`)
