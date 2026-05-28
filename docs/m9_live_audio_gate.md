# M9 Live Audio Promotion Gate

> Status: planned. M9 synthetic validation is complete. Live audio end-to-end is a future promotion gate, not a blocker for M9 completion.

## Goal

Prove that the M9 sound/text pipeline survives real audio I/O:
- synthesize a target sound from a word or sound prompt
- play it through a real output device
- capture the result on a microphone or loopback input
- score the capture against the target
- record pass rate, latency, and failure modes

## Why this exists

The synthetic M9 gates prove that the learning loop, acoustic graph, and text association all work in-memory. They do not prove that the audio survives:
- OS audio routing
- device latency
- microphone capture quirks
- room acoustics
- output-device mismatch

This gate is the first place where those issues matter.

## Proposed validation modes

### 1) Dry run
No hardware required.
- builds the target sound
- exercises the scoring and reporting path
- validates config and file output

### 2) Loopback / mic smoke
Requires audio hardware.
- plays a target sound to an output device
- records a simultaneous input capture
- compares captured audio to the target
- reports similarity and latency

### 3) Small word set
Requires audio hardware and the M9d synthetic word/sound map.
- choose 3–10 words
- map each word to a target sound
- run the loopback/mic smoke for each word
- report word-level pass rate and worst-case capture quality

## Suggested acceptance criteria

These are intentionally conservative and can be tightened later.

- Dry run completes successfully with no hardware
- Loopback/mic smoke returns a similarity score above the chosen threshold on at least 80% of trials
- Worst-case capture latency stays within an acceptable band for the room/device setup
- No device-specific failure becomes the default case
- Results are saved to JSON for later comparison

## Current recommendation

Do not block M9 on this gate.

Treat the live-audio gate as the next promotion step after M9 synthetic validation, and use it to answer the remaining systems question: does the M9 pipeline still work when sound leaves the model and enters the real world?

## First live hardware smoke

Command:
```bash
source .venv/bin/activate && python scripts/m9_live_audio_gate.py --word alpha --target-freq 220 --target-waveform sine --duration 1.0 --input-device 1 --output-device 0 --threshold 0.30 --output /tmp/m9_live_audio_gate_live.json
```

Observed results:
- Input device: `BL-2021USB`
- Output device: `ASUS PB287Q`
- Overall score: 0.7251
- Threshold: 0.30
- Verdict: PASS
- Capture elapsed: 1.8498s
- Playback started: True

Interpretation:
- The live gate successfully exercised real audio I/O on the hardware.
- This is a smoke pass, not a full promotion-grade validation yet.
- The remaining work is to broaden this from one word/sound pair to a small live set and establish a stable promotion threshold.

## Repeat-run live smoke

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

## Output-device-2 strict sweep

Command:
```bash
source .venv/bin/activate && python scripts/m9_live_audio_gate.py --targets alpha:220:sine,beta:247:triangle,gamma:262:sawtooth,delta:294:square --repeats 2 --input-device 1 --output-device 2 --threshold 0.60 --output /tmp/m9_live_audio_gate_headphones.json
```

Observed results:
- Pass rate: 1.00 (8/8)
- Average score: 0.7533
- Worst score: 0.6845
- Threshold: 0.60
- Verdict: PASS
- Capture elapsed: 1.83s–1.96s per target
- Playback started: True for all targets

Per-target repeat summaries:
- alpha / 220 Hz sine: 2/2 pass, avg 0.7768, worst 0.7746
- beta / 247 Hz triangle: 2/2 pass, avg 0.7717, worst 0.7653
- gamma / 262 Hz sawtooth: 2/2 pass, avg 0.7204, worst 0.6845
- delta / 294 Hz square: 2/2 pass, avg 0.7442, worst 0.7285

Interpretation:
- Switching from device 0 to output device 2 materially improved the capture scores.
- The softened square waveform also removed the prior failure boundary at 0.60.
- We now have a clearer promotion signal: the current live gate is stable at 0.60 on output device 2, with the square waveform no longer the limiting case.

## Notes

- The current M9 live gate can handle a small set of target sounds on the actual devices available in this environment.
- Immediate repeats stayed stable at the easy threshold.
- With output device 2 and a softened square waveform, the stricter 4-target 0.60 sweep passed, so the previous failure boundary moved upward.
- We still need broader target coverage if we want a promotion-grade claim, but the device-2 path is materially stronger than the earlier monitor-speaker path.

## Broader 6-target sweep

Command:
```bash
source .venv/bin/activate && python scripts/m9_live_audio_gate.py --targets alpha:220:sine,beta:247:triangle,gamma:262:sawtooth,delta:294:square,epsilon:330:sine,zeta:349:triangle --repeats 2 --input-device 1 --output-device 2 --threshold 0.60 --output /tmp/m9_live_audio_gate_broader.json
```

Observed results:
- Pass rate: 1.00 (12/12)
- Average score: 0.7518
- Worst score: 0.6754
- Threshold: 0.60
- Verdict: PASS
- Capture elapsed: 1.81s–1.85s per target
- Playback started: True for all targets

Per-target repeat summaries:
- alpha / 220 Hz sine: 2/2 pass, avg 0.7908, worst 0.7685
- beta / 247 Hz triangle: 2/2 pass, avg 0.7521, worst 0.7253
- gamma / 262 Hz sawtooth: 2/2 pass, avg 0.6804, worst 0.6754
- delta / 294 Hz square: 2/2 pass, avg 0.7292, worst 0.7210
- epsilon / 330 Hz sine: 2/2 pass, avg 0.7881, worst 0.7858
- zeta / 349 Hz triangle: 2/2 pass, avg 0.7701, worst 0.7595

Interpretation:
- The broader 6-target sweep stayed above threshold on both repeats for every target.
- The sawtooth target remains the weakest of the set, but it is still comfortably above threshold on output device 2.
- This is the strongest live evidence so far, and it suggests the device-2 path is a plausible promotion candidate, subject to a larger repeat set and any future failure cases.
