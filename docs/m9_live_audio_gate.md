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

## Threshold bracket (v3 — improved pitch detector)

8 targets, 3 repeats, output device 2:

| Threshold | Pass rate | Avg | Worst | Verdict |
|-----------|-----------|-----|-------|---------|
| 0.75 (6 targets) | 18/18 (100%) | 0.809 | 0.790 | PASS |
| 0.75 (8 targets) | 18/24 (75%) | 0.785 | 0.677 | FAIL |

6-target sweet spot (alpha through zeta, up to 349 Hz):
- all pass 3/3 repeats at 0.75
- avg scores: 0.778–0.850
- weakest: beta/247/triangle at 0.762

Higher-frequency failures:
- eta/392/sawtooth: 0/3 pass (pitch detection unreliable above ~350 Hz)
- iota/440/square: 0/3 pass (same)

## Interpretation

- The reliable operating range for this hardware pair is up to ~350 Hz.
- 6 targets at 0.75 with 3 repeats is a solid promotion-grade result on output device 2.
- Higher frequencies need either a better pitch detector or relaxed scoring above 350 Hz.
- For promotion, we can either:
  - accept the 6-target/0.75 gate with the high-frequency limitation documented, or
  - fix the pitch detector for higher frequencies before expanding the target set.
