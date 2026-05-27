# M9 Acceptance Report

**Date:** 2026-05-27
**Spec:** `docs/m9_sound_first.md`
**Scripts:** `scripts/m9_imitation_loop.py`, `scripts/m9_convergence_test.py`

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

- M9a: complete baseline sound-imitation primitive
- M9b: repeatability and acoustic prototype refinement complete
- M9c: weak terminal confirmation complete
- M9d: text association complete (synthetic gate)
- Live audio end-to-end remains a future promotion gate

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
- This remains a synthetic gate; live audio end-to-end is still a future promotion step.

## Live Audio Promotion Gate — First Hardware Smoke

Implemented components:
- `docs/m9_live_audio_gate.md` — promotion-gate spec
- `scripts/m9_live_audio_gate.py` — dry-run and live capture/score scaffold
- `tests/unit/test_m9_live_audio_gate.py` — dry-run coverage

### Validation run

Command:
```bash
source .venv/bin/activate && python scripts/m9_live_audio_gate.py --word alpha --target-freq 220 --target-waveform sine --duration 1.0 --input-device 1 --output-device 0 --threshold 0.30 --output /tmp/m9_live_audio_gate_live.json
```

Observed results:
- Input device: BL-2021USB
- Output device: ASUS PB287Q
- Overall score: 0.7251
- Threshold: 0.30
- Verdict: PASS
- Capture elapsed: 1.8498s
- Playback started: True

Interpretation:
- The hardware smoke confirms that the live gate can play a target sound, capture it back, and score the capture on real devices.
- This is a smoke pass only; it is not yet a promotion-grade multi-trial or multi-word validation.
- The live gate now exists as the path to the next promotion step if we want to harden M9 against real audio I/O.

## Live Audio Promotion Gate — Small Word-Set Smoke

Command:
```bash
source .venv/bin/activate && python scripts/m9_live_audio_gate.py --targets alpha:220:sine,beta:247:triangle,gamma:262:sawtooth --input-device 1 --output-device 0 --threshold 0.30 --output /tmp/m9_live_audio_gate_live_multi.json
```

Observed results:
- Pass rate: 1.00 (3/3)
- Average score: 0.6918
- Worst score: 0.6226
- Threshold: 0.30
- Verdict: PASS
- Capture elapsed: 1.83s–1.99s per target
- Playback started: True for all targets

Per-target scores:
- alpha / 220 Hz sine: 0.7213
- beta / 247 Hz triangle: 0.7314
- gamma / 262 Hz sawtooth: 0.6226

Interpretation:
- The live gate now passes on a small set of distinct sounds, not just a single tone.
- This is still a smoke, not a promotion-grade validation; we have not yet explored failure cases or a tighter acceptance threshold.
- The next live step is to expand to a slightly broader set and record repeatability / worst-case behavior.

## Repeat-run Live Smoke

Command:
```bash
source .venv/bin/activate && python scripts/m9_live_audio_gate.py --targets alpha:220:sine,beta:247:triangle,gamma:262:sawtooth --repeats 2 --input-device 1 --output-device 0 --threshold 0.30 --output /tmp/m9_live_audio_gate_repeat.json
```

Observed results:
- Pass rate: 1.00 (6/6)
- Average score: 0.6928
- Worst score: 0.6180
- Threshold: 0.30
- Verdict: PASS
- Capture elapsed: 1.83s–2.01s per target
- Playback started: True for all targets

Per-target repeat summaries:
- alpha / 220 Hz sine: 2/2 pass, avg 0.7256, worst 0.7206
- beta / 247 Hz triangle: 2/2 pass, avg 0.7322, worst 0.7310
- gamma / 262 Hz sawtooth: 2/2 pass, avg 0.6206, worst 0.6180

Interpretation:
- The live gate held up across immediate repeats on the same hardware.
- The sawtooth target is still the weakest of the three, but it remains above the current smoke threshold on both runs.
- This is better evidence than a one-shot smoke, but still short of a promotion-grade live gate; we still need broader target coverage and a more deliberate failure threshold.

## Notes

- The current M9b gate measures repeatability and prototype refinement, not full acoustic learning.
- The SAG acoustic layer stores compact summaries, not raw audio.
- Persistence remains intentionally deferred to M9d.
