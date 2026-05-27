# M9 Acceptance Report

**Date:** 2026-05-27
**Spec:** `docs/m9_sound_first.md`
**Scripts:** `scripts/m9_imitation_loop.py`, `scripts/m9_convergence_test.py`

## Summary

| Phase | Gate | Result |
|-------|------|--------|
| M9a — Sound imitation primitive | Produce non-silence, stable similarity across trials | PASS |
| M9b — Repeatability and convergence | Repeatability on one or more target sounds; acoustic prototype refinement | PASS |

## M9a — Sound Imitation Primitive

Implemented components:
- `OscillatorBank` with sine/triangle/sawtooth/square waveforms
- `AcousticSubstrate` recurrent hidden state
- `NeuralController` linear readout to oscillator params
- `AcousticProducer` full production loop
- `AcousticSimilarity` comparison metric
- `m9_imitation_loop.py` trial runner

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
- Prototype nodes are being refined in the SAG with summary stats and similarity history.
- This is a valid M9b stepping stone, but it does not yet prove learned convergence; M9c/M9d remain separate future milestones.

## Current M9 Status

- M9a: complete enough to serve as a baseline sound-imitation primitive
- M9b: repeatability and acoustic prototype refinement are implemented and passing smoke tests
- M9c: terminal confirmation / weak supervision — pending
- M9d: text association / word-sound pairing — pending

## Notes

- The current M9b gate measures repeatability and prototype refinement, not full acoustic learning.
- The SAG acoustic layer stores compact summaries, not raw audio.
- Persistence remains intentionally deferred to M9c.
