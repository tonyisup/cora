# M9 Live Audio Promotion Gate

> Status: active. M9 synthetic validation is complete. Live audio end-to-end is a promotion gate, not a blocker for M9 completion.

## Goal

Prove that the M9 sound/text pipeline survives real audio I/O:
- synthesize a target sound from a word or sound prompt
- play it through a real output device
- capture the result on a microphone or loopback input
- score the capture against the target
- record pass rate, latency, and failure modes

## Current hardware

- input: BL-2021USB (device 1)
- output: External Headphones (device 2)
- waveform improvement: square uses soft-limited approximation

## Threshold bracket

We swept the 6-target live set at increasing thresholds on output device 2 to find the promotion boundary.

Targets: alpha/220/sine, beta/247/triangle, gamma/262/sawtooth, delta/294/square, epsilon/330/sine, zeta/349/triangle
Repeats: 2 each, duration: 1.0s

### 0.60 threshold — PASS
- 12/12 (100%)
- avg 0.7518, worst 0.6754
- all targets pass both repeats

### 0.70 threshold — PASS
- 10/12 (83%)
- avg 0.7434, worst 0.6730
- gamma/sawtooth fails both repeats; all others pass

### 0.75 threshold — FAIL
- 9/12 (75%, below 80% gate)
- avg 0.7741, worst 0.6990
- gamma/sawtooth fails both repeats; beta/triangle fails one repeat

### Per-target stability (at 0.70 threshold)
- alpha / 220 Hz sine: 2/2 pass, avg 0.7570, worst 0.7561
- beta / 247 Hz triangle: 2/2 pass, avg 0.7265, worst 0.7134
- gamma / 262 Hz sawtooth: 0/2 pass, avg 0.6756, worst 0.6730
- delta / 294 Hz square: 2/2 pass, avg 0.7329, worst 0.7287
- epsilon / 330 Hz sine: 2/2 pass, avg 0.7957, worst 0.7922
- zeta / 349 Hz triangle: 2/2 pass, avg 0.7727, worst 0.7496

## Interpretation

- The live gate is stable at threshold 0.70 on output device 2 with the current 6-target set.
- The sawtooth waveform at 262 Hz is the consistent weak point across thresholds.
- The softened square waveform is no longer the limiting case.
- For promotion, we can either:
  - accept threshold 0.70 with the sawtooth/gamma gap documented as a known limitation, or
  - improve the sawtooth capture path before promoting at a stricter threshold.
