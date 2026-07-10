# Structured Continuous-Audio Prediction Protocol

## Purpose

Test H2 on a controlled continuous acoustic stream after isolated natural clips
failed to support another predictor iteration. The task asks whether a bounded,
locally learned acoustic-state transition model can outperform persistence and
a causal moving-state baseline after playback, room transfer, and microphone
capture.

This is a mechanism experiment, not evidence of natural speech prediction or
biological fidelity.

## Safety and authorization boundary

- Default execution plans the protocol and performs no playback or capture.
- Fake I/O may synthesize an explicitly non-real captured WAV for plumbing and
  deterministic analysis tests.
- Real I/O requires both `--real-io` and an explicit bounded-playback
  authorization flag.
- Real playback is capped at `-18 dB`, one trial at a time, with no autonomous
  repetition beyond the fixed manifest.
- Capture remains the XVF3800's two fused host channels; no four-channel claim
  is permitted.

## Trial design

Two stimulus families are evaluated independently:

1. four pure tones (`300`, `500`, `800`, and `1200 Hz`);
2. four synthesized vowels (`a`, `e`, `i`, and `u`) at a shared F0.

Each trial contains:

- a short alignment chirp and silence;
- one unlabeled acoustic presentation of each of the four tokens, used only to
  initialize four codebook centroids;
- twenty grammar tokens generated from a fixed first-order branching process;
- `200 ms` per token and `50 ms` silence between tokens.

Three development and three held-out sequences per family use disjoint seeds.
The scheduled token identities are retained as post-hoc ground truth only. They
may not enter codebook fitting, transition fitting, or held-out prediction.

## Learner

For each aligned token window, the lower level measures a normalized spectral
signature directly from captured PCM. A four-state competitive codebook is
initialized from the four calibration observations and updated one observation
at a time on development captures only.

The higher level stores a smoothed `4 x 4` transition-count table. Its predicted
next acoustic state is consumed by the lower level, which records a one-hot
prediction residual when the next measured state arrives. There is no global
backpropagation, label-conditioned state assignment, or retained waveform in
the learner.

## Baselines and metrics

Held-out next-state accuracy is compared with:

- persistence: predict the current acoustic state again;
- development unigram mode;
- a causal exponential moving distribution over observed held-out states.

Reports include per-trial and pooled accuracy, improvement over the best
baseline, codebook occupancy, post-hoc token/cluster purity, bounded state size,
alignment provenance, and all capture artifact hashes.

## Promotion boundary

Fake-I/O success proves plumbing only. Before any real capture, a separate
evidence artifact must freeze the exact manifest and acceptance thresholds.
A real pass would support locally learned transition prediction for these two
structured acoustic families. It would not pass H2 for unconstrained speech;
that requires a later natural-stream transfer test.
