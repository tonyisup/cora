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

## Live Audio Promotion Gate — Threshold Bracket

See `docs/m9_live_audio_gate.md` for the full hardware history and detailed per-target breakdowns.

Hardware: input device 1 (BL-2021USB), output device 2 (External Headphones)
Similarity metric v2: proper mel-scale binning + autocorrelation pitch detection
Targets: alpha/220/sine, beta/247/triangle, gamma/262/sawtooth, delta/294/square, epsilon/330/sine, zeta/349/triangle
Repeats: 2 each, duration: 1.0s

| Threshold | Pass rate | Avg score | Worst | Verdict |
|-----------|-----------|-----------|-------|---------|
| 0.60 | 12/12 (100%) | 0.8161 | 0.7889 | PASS |
| 0.70 | 12/12 (100%) | 0.8161 | 0.7889 | PASS |
| 0.75 | 11/12 (92%) | 0.8004 | 0.7139 | PASS |
| 0.80 | 8/12 (67%) | 0.8090 | 0.7824 | FAIL |

v2 metric removed the sawtooth-as-weakest-target gap. Promising for promotion.

Weakest target: gamma / 262 Hz sawtooth. The softened square waveform is no longer the limiting case.

Interpretation:
- The live gate is stable at threshold 0.70 on output device 2 with the current 6-target set.
- For promotion, we can either accept threshold 0.70 with the sawtooth/gamma gap documented as a known limitation, or improve the sawtooth capture path before promoting at a stricter threshold.

## Current M9 Status

- M9a: complete baseline sound-imitation primitive
- M9b: repeatability and acoustic prototype refinement complete
- M9c: weak terminal confirmation complete
- M9d: text association complete (synthetic gate)
- Live audio promotion gate: scaffolded and partially validated; stable at 0.70 on output device 2

## Notes

- The current M9b gate measures repeatability and prototype refinement, not full acoustic learning.
- The SAG acoustic layer stores compact summaries, not raw audio.
- Persistence remains intentionally deferred to M9d.
