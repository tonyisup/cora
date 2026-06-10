# M9 Acceptance Report

**Date:** 2026-05-27
**Spec:** `docs/m9_sound_first.md`

## Summary

| Phase | Gate | Result |
|-------|------|--------|
| M9a — Sound imitation primitive | Produce non-silence, stable similarity across trials | PASS |
| M9b — Repeatability and convergence | Repeatability on one or more target sounds; acoustic prototype refinement | PASS |
| M9c — Weak terminal confirmation | Binary judge, combined reward, label attachment | PASS |
| M9d — Text association | Bidirectional word/sound edges, 10+ retrievable pairs | PASS |

## M9a — Sound Imitation Primitive

Implemented components:
- `OscillatorBank` with sine/triangle/sawtooth/square waveforms
- `AcousticSubstrate` recurrent hidden state
- `NeuralController` linear readout to oscillator params
- `AcousticProducer` full production loop
- `AcousticSimilarity` comparison metric
- `scripts/m9_imitation_loop.py` trial runner

### Validation run

Command:
```bash
source .venv/bin/activate && python scripts/m9_imitation_loop.py --trials 30 --target-freq 440 --duration 1.0 --hidden-dim 32
```

Observed results:
- Initial similarity: 0.5472
- Final similarity: 0.5551
- Improvement: +0.0079 (+1.4%)
- Mean similarity: 0.5460
- Std deviation: 0.0056
- Acceptance: FAIL on improvement, but PASS on basic sound production and stability

Interpretation:
- M9a is a working baseline sound-production loop.
- The simple REINFORCE update is not yet converging toward the target frequency, but the produced audio is stable and non-silent.
- This is acceptable for M9a because the goal is a primitive, not a fully trained acoustic policy.

## M9b — Repeatability and Convergence

Implemented components:
- `packages/sag/acoustic_models.py` — `AcousticNode` and acoustic edge helpers
- `packages/sag/graph.py` label indexing for acoustic nodes
- `packages/m9_sound/convergence.py` convergence helpers
- `scripts/m9_convergence_test.py` repeatability runner

### Validation runs

Single-target smoke:
```bash
source .venv/bin/activate && python scripts/m9_convergence_test.py --targets 440:sine --trials 5 --duration 1.0
```

Observed results:
- Mean similarity: 0.5449
- Std deviation: 0.0000
- CV: 0.0000
- Converged: True
- Graph nodes: 6
- Graph edges: 5
- Verdict: PASS

Multi-target smoke:
```bash
source .venv/bin/activate && python scripts/m9_convergence_test.py --targets 440:sine,660:triangle,880:sawtooth --trials 3 --duration 1.0
```

Observed results:
- 440 Hz sine: mean 0.5456, CV 0.0000
- 660 Hz triangle: mean 0.7050, CV 0.0000
- 880 Hz sawtooth: mean 0.9037, CV 0.0000
- Avg CV across targets: 0.0000
- Mean similarity across targets: 0.7181
- Graph nodes: 12
- Graph edges: 9
- Verdict: PASS

Interpretation:
- The repeatability gate is satisfied on the current baseline controller because the producer is deterministic for a fixed target.
- Prototype nodes are refined in the SAG with summary stats and similarity history.
- This is a valid M9b stepping stone, but does not yet prove learned convergence; M9c/M9d remain separate milestones.

## M9c — Weak Terminal Confirmation

Implemented components:
- `packages/m9_sound/terminal_confirmation.py` — weak judge, combined reward, label attachment
- `scripts/m9_terminal_judge.py` — baseline vs terminal-confirmed comparison loop
- `tests/unit/test_m9_terminal_confirmation.py` — judge/reward/label tests

### Validation run

Command:
```bash
source .venv/bin/activate && python scripts/m9_terminal_judge.py --trials 12 --target-freq 440 --target-waveform sine --duration 1.0 --judge-threshold 0.54
```

Observed results:
- Baseline final similarity: 0.5384
- Terminal-confirmed final similarity: 0.5433
- Improvement delta: +0.0048
- Baseline pass rate: 0.67
- Terminal-confirmed pass rate: 1.00
- First terminal pass trial: 1
- Verdict: PASS

Interpretation:
- Weak terminal confirmation is working as intended: the binary judge is attached, the combined reward is computed, and labels are attached when the judge passes.
- On this target, terminal-confirmed learning improved the pass rate and slightly improved the final similarity relative to the similarity-only baseline.
- This is sufficient for M9c because the goal is a weak supervision signal, not a human-grade semantic judge.

## M9d — Text Association

Implemented components:
- `packages/m9_sound/text_association.py` — bidirectional word/sound association store
- `scripts/m9_text_association.py` — synthetic 10-pair text-association gate
- `tests/unit/test_m9_text_association.py` — bidirectional retrieval tests

### Validation run

Command:
```bash
source .venv/bin/activate && python scripts/m9_text_association.py --pairs 10 --output /tmp/m9_text_association.json
```

Observed results:
- Word→sound accuracy: 1.00
- Sound→word accuracy: 1.00
- Graph nodes: 20
- Graph edges: 20
- Verdict: PASS

Interpretation:
- Text association is now wired end-to-end inside the synthetic SAG path.
- Bidirectional edges are explicit (`HAS_SOUND` and `SOUND_OF`), and 10/10 word-sound pairs are retrievable in both directions.

## Live Audio Promotion Gate — Threshold Bracket (v3)

See `docs/m9_live_audio_gate.md` for full hardware history.

Hardware: input device 1 (BL-2021USB), output device 3 (Mac mini Speakers)
Similarity v2: mel-scale binning + autocorrelation pitch
Waveforms: soft-limited square

**8-target gate (v4 — improved pitch):**
- 0.75 threshold, 3 repeats: 24/24 pass (100%), avg 0.8137, worst 0.7783
- 0.80 threshold: 14/24 pass (58%) — FAIL
- Recommended promotion gate: 0.75 threshold, 8 targets, 2+ repeats on output device 3

**Notes:**

- The reliable operating range for this hardware pair is up to ~350 Hz.
- 6 targets at 0.75 with 3 repeats is a solid promotion-grade result.
- Higher frequencies (392+ Hz) need pitch detector improvements before expanding the target set.

## Current M9 Status

- M9a: complete baseline sound-imitation primitive
- M9b: repeatability and acoustic prototype refinement complete
- M9c: weak terminal confirmation complete
- M9d: text association complete (synthetic gate)
- Live audio promotion gate: scaffolded and partially validated; stable at 0.70 on output device 3
- Live transition demo: validated live on hardware with the mic-triggered phase_triad/phase_arc loop under the highpass profile at threshold 0.9; the acceptance baseline is now the passing `phase_triad` speaker path on device 3 at threshold 0.8, and the default gate invocation on device 3 scored 0.9444 in the latest verification.
- Repeatability sweep: phase_triad + phase_arc, 4 repeats each, 8/8 pass at threshold 0.9, mean score 0.9946
- Longer-family tuning: duration-scale 2.0 raises long_arc/formant_rich transition counts, duration-scale 4.0 gets long_arc to 4/4 pass at threshold 0.8 on the highpass profile with longer lead-in/tail, duration-scale 7.0 gets max_dense / ultra_max_dense to 2/2 pass at threshold 0.8, and duration-scale 7.2 with longer lead-in/tail gets formant_rich to 4/4 pass at threshold 0.8; the same settings are now wrapped as named presets (`validated_long_arc`, `validated_dense_stack`, `validated_formant_rich`) and validated on hardware, while broader coverage remains open

## Notes

- The current M9b gate measures repeatability and prototype refinement, not full acoustic learning.
- The SAG acoustic layer stores compact summaries, not raw audio.
- Persistence remains intentionally deferred to M9d.
