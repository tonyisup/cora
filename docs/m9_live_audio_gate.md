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

We swept the 6-target live set at increasing thresholds on output device 2 to find the promotion boundary. After improving the similarity metric (proper mel-scale binning + autocorrelation pitch), the scores improved materially.

Targets: alpha/220/sine, beta/247/triangle, gamma/262/sawtooth, delta/294/square, epsilon/330/sine, zeta/349/triangle
Repeats: 2 each, duration: 1.0s

| Threshold | Pass rate | Avg score | Worst | Verdict |
|-----------|-----------|-----------|-------|---------|
| 0.60 | 12/12 (100%) | 0.8161 | 0.7889 | PASS |
| 0.70 | 12/12 (100%) | 0.8161 | 0.7889 | PASS |
| 0.75 | 11/12 (92%) | 0.8004 | 0.7139 | PASS |
| 0.80 | 8/12 (67%) | 0.8090 | 0.7824 | FAIL |

Per-target at 0.70:
- alpha / 220 Hz sine: avg 0.8192
- beta / 247 Hz triangle: avg 0.7909, worst 0.7889
- gamma / 262 Hz sawtooth: avg 0.8126
- delta / 294 Hz square: avg 0.8053
- epsilon / 330 Hz sine: avg 0.8513
- zeta / 349 Hz triangle: avg 0.8171

## Interpretation

- The improved similarity metric removed the sawtooth-as-weakest-target gap.
- The live gate now has margin up to threshold 0.75 on output device 2.
- The 0.80 threshold is too strict for the current device pair and waveform set (beta/triangle drops below).
- For promotion, the 0.75 threshold with 6-target coverage and the improved metric is a plausible acceptance gate.
