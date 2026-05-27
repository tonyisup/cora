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
