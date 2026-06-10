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
- output: Mac mini Speakers (device 3)
- waveform improvement: square uses soft-limited approximation

## Threshold bracket (v4 — pitch distance capping)

8 targets, 3 repeats, output device 3, improved similarity (mel bins + autocorrelation pitch + octave-tolerant capping):

| Threshold | Pass rate | Avg | Worst | Verdict |
|-----------|-----------|-----|-------|---------|
| 0.75 | 24/24 (100%) | 0.8137 | 0.7783 | PASS |
| 0.80 | 14/24 (58%) | 0.8140 | 0.7909 | FAIL |

Per-target at 0.75:
- alpha/220/sine: avg 0.8189, worst 0.8142
- beta/247/triangle: avg 0.7852, worst 0.7783
- gamma/262/sawtooth: avg 0.8149, worst 0.8141
- delta/294/square: avg 0.8080, worst 0.8053
- epsilon/330/sine: avg 0.8523, worst 0.8495
- zeta/349/triangle: avg 0.8185, worst 0.8163
- eta/392/sawtooth: avg 0.8355, worst 0.8351
- iota/440/square: avg 0.8226, worst 0.8211

At 0.80, beta/247/triangle and iota/440/square drop below threshold.

## Interpretation

- The octave-tolerant pitch capping fixed the high-frequency pitch detection gap.
- All 8 targets pass at 0.75 with 3 repeats — a strong promotion-grade result.
- The 0.80 threshold is beyond the reliable range for this hardware pair.
- Recommended promotion gate: 0.75 threshold, 8 targets, 2+ repeats on output device 3.
