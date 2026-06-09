# Cora Status — 2026-06-04

## Test Health

```
627 passed, 1 skipped, 0 failed
```

- 628 total tests across M8/M9 and earlier suites
- The M2/M3 artifact replay tests now pass after restoring imported fixtures from the corollary copy; those assets are optional for fresh clones, but present in this checkout for local replay coverage
- Added an M9 live-capture retrain smoke harness so the current vowel path can be regression-tested without hardware
- Added an M9 temporal-sequence smoke harness over the saved temporal controller/substrate pair
- Added a noisy-room regression for the temporal smoke path so small additive input noise stays stable
- Added a diphthong-family temporal regression with a longer 5-tone transition family
- Added a wide-sweep temporal regression with a 7-tone transition family
- Added a device-profile regression for the temporal smoke path so capture high-pass/gain variation stays stable
- Added a live-capture wrapper so the temporal sequence can be played through hardware and written back out as rendered/captured WAV artifacts; `phase_triad` is now a stable clean 3-transition chain with longer spacing, and `phase_arc` adds a second stable 2-transition live family

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

- `62c7dad` feat(m9): add live temporal capture wrapper
- `e1be563` feat(m9): add wav export for temporal smoke
- `acbee80` docs: sync status after temporal regression
- `f77dbfa` feat(m9): add wide sweep temporal regression
- `96652c3` feat(m9): add device profile temporal regression
- `82e87d7` feat(m9): add diphthong temporal family
- `ddff5a8` feat(m9): add noisy temporal smoke regression
- `5f426c9` feat(m9): add temporal sequence smoke harness
- `c0660f3` feat(m9): add live retrain smoke harness

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

- M9 temporal path now includes device-profile regression, live-capture validation wrapper, live vowel-transition phase-closure capture wrapper, the live vowel-transition phase-closure gate, and a new main live vowel demo that routes through the latest phase-closure path; `phase_triad` is now a clean 3-transition chain with longer segment spacing for live stability, and `phase_arc` adds a second stable 2-transition family. The current live imitation phase is ready for use under the highpass profile at threshold 0.9.
- M8 known gap: clarify/repair actions not triggered by heuristic in current episode types — distilled never learned them
- `packages/salience/` remains a stub (empty `__init__.py`)
- M9 next: broader formant-rich temporal coverage beyond the long_arc, formant_rich, dense_cycle, ultra_dense, hyper_dense, and mega_dense slices