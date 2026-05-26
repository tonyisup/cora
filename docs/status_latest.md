# Cora Status — 2026-05-23

## Test Health

```
523 passed, 1 skipped, 5 failed (data-missing)
```

- All 74 integration tests pass
- 449 of 52 unit tests pass (44 new M8 tests added)
- 5 unit tests fail with `FileNotFoundError` — all are M2/M3 artifact replay tests that need data files not present in local `data/`:
  - `test_pfc_m2_artifact_replay.py::test_artifact_backed_speak_interruption_replay` — needs `data/m2_baseline/m2_energy_forward_tol75_matrix_20260429_093302/`
  - `test_pfc_m3_acceptance_ladder.py::test_m3_acceptance_ladder_passes_current_slice` — same M2 baseline
  - `test_pfc_m3_artifact_ladder.py::test_ladder_clean_vowel_overlap_defers_and_resumes` — needs `data/m2_policy/family_policy_validation_20260429_130530/`
  - `test_pfc_m3_artifact_ladder.py::test_ladder_self_only_control_does_not_defer` — same M2 baseline
  - `test_thalamus_attributor.py::test_room_local_score_with_exported_filter` — needs `data/m3_broca_room_threshold_calibration_broca_controls_reexport.npy`
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
|| M8 | In Progress | M8a PASS, M8b PASS — distilled 89% expanded match, 100% canonical |

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
- M8DistilledPolicy MLP (21 → 64 → 64 → 5, 5893 params)
- Training: 98.67% validation accuracy, early stopping at epoch 13
- Acceptance: 100% canonical match, 89.14% expanded match (target ≥80%)
- Checkpoint: data/m8_distilled_policy.pt

**M8c (RL) — Next:**
- Reuse M4 PPO infrastructure with M8 reward
- Gate: PPO beats distilled by >= 10% on expanded suite

## Recent Commits

- `4fa49ff` feat(m8): acceptance ladder passes on 6 scripted episodes
- `61c231f` feat(m8): add conversation tracker, speaking policy, reward, and simulator
- `291b804` docs: spec M8 speaking policy milestone

## Key Metrics

- Self/not-self attribution: 100% recall, 16.4% FP (same-family voices)
- Interruption latency: 1390ms (comparable to human ~1500ms)
- Pipeline overhead: ~330ms (2 callbacks + 3-frame classifier)
- Loop index (self-trigger rate): 0
- M8 test coverage: 44 new tests (tracker, policy, reward)

## Open Items

- 5 data-missing test failures (M2/M3 artifacts) — need artifact regeneration or test skip markers
- `packages/salience/` is a stub (empty `__init__.py`)
- M8b: distilled policy training + expanded episode suite
- M8c: PPO with M8 reward after M8b gates pass
