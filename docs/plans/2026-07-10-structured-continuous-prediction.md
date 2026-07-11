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

## First real-I/O attempt

The first authorized attempt stopped on `tones_dev_1103` before speaker
playback. AVFoundation was recording, but ffmpeg's raw muxer buffered less than
roughly 256 KiB, so the raw-file byte count remained zero until after the
three-second readiness deadline. No trial completed, no capture was retained,
and no held-out observation was spent.

The I/O runner now requests per-packet flushing. A microphone-only synchronized
diagnostic publishes 2,048 frames after 128 ms and verifies overlap using the
raw byte count. The failed attempt and fix are retained separately; the fixed
runner has a new preregistration and cannot execute the protocol again without
fresh explicit retry authorization.

## Authorized retry result

The separately frozen retry completed all 12 trials with real synchronized
playback/capture and 24 verified WAV hashes. Tones passed every preregistered
gate (`0.860` transition accuracy versus `0.263` for the best baseline), but
vowels failed (`0.596` tied the best baseline, three occupied states, purity
`0.600`). The required two-family verdict is therefore failure.

Post-hoc confusion counts show the spectral states merged `a/e` and mostly
merged `i/u`. This characterizes the spent result; it does not authorize tuning
or confirmation on these captures. A future speech-representation experiment
must use development-only selection and fresh confirmation data.

## Calibrated fresh-seed follow-up

A development-only design audit tested spectral resolution, linear bands,
cepstra, LPC formants, token margins, and trial-local codebooks. Only one
candidate beat the best causal baseline in every leave-one-development-trial-
out fold: featurewise z-scoring from the four unlabeled same-trial calibration
observations, using a fixed `0.30` scale floor and unit normalization.

That representation, six entirely new seeds, and H2-aligned predictive gates
were frozen before live I/O. The real 12-trial confirmation passed both
families. Tones improved `0.614` over the best baseline; vowels improved
`0.421`; every held-out trial improved at least `0.368`. The result promotes the
controlled bounded mechanism only. The remaining Stage-3 boundary is robustness
and transfer beyond one synthetic voice, grammar, geometry, and calibration
schedule.

## Condition robustness result

A later protocol froze three fresh held-out conditions: unseen synthetic F0 at
`200 Hz`, the same F0 with `10 dB` additive noise present during calibration and
grammar, and the same F0 with clean calibration plus `10%` grammar-only dropout.
Every tone and vowel condition beat its best causal baseline by at least
`0.316`. F0 shift and noise retained four vowel states.

The dropout condition achieved `0.842` accuracy versus `0.263` for its best
baseline, but all `u` tokens entered the `i`-dominated state and occupancy fell
to three. The frozen per-condition occupancy rule makes the overall robustness
verdict a failure. The result supports controlled predictive robustness, not
dropout-stable representation or natural-stream H2.

## Dropout-aware temporal aggregation stop

Median aggregation over 60 ms spectral subwindows passed a development-only
audit at `10–20%` dropout, but failed its required fresh synthetic gate before
hardware. Pooled vowel improvement was `0.193`; the `15%` and `20%` conditions
improved only `0.105`. The variant is retained as a falsified design, and its
real-I/O CLI path is explicitly blocked. Further window/dropout-percentage
tuning on this family is retired; a new attempt requires an explicit missing-
observation mechanism or a different research rationale.
