# Cora Status — 2026-05-23

## Test Health

```
479 passed, 1 skipped, 5 failed (data-missing)
```

- All 74 integration tests pass
- 405 of 52 unit tests pass
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
| M8 | Scoped | Speaking policy — deterministic harness → distilled → RL |

## Recent Changes (last 3 commits)

- `7646870` docs: record m4 live smoke tests
- `74e9db3` feat: port M4 self-voice arbitration
- `4bf9d8f` fix: learn WORLD adapter profiles from attribution confidence

## Key Metrics

- Self/not-self attribution: 100% recall, 16.4% FP (same-family voices)
- Interruption latency: 1390ms (comparable to human ~1500ms)
- Pipeline overhead: ~330ms (2 callbacks + 3-frame classifier)
- Loop index (self-trigger rate): 0

## Open Items

- 5 data-missing test failures (M2/M3 artifacts) — need artifact regeneration or test skip markers
- `packages/salience/` is a stub (empty `__init__.py`)
- `docs/plan.md` referenced in README layout but does not exist
- M4 multi-step PPO does not yet beat distilled v3 baseline (both 1.9625 return) — per decision memo, next step is live validation
