# Evidence Pipeline

## Purpose

This architecture prevents annotations, synthetic fixtures, and internal
promotion gates from being mistaken for empirical auditory evidence.

## Real-audio flow

```text
Uncompressed WAV samples (integer PCM or IEEE float)
  -> channel-preserving decoder
  -> fixed-duration measured frames
  -> RMS + acoustic feature extraction
  -> transparent saliency score
  -> predictions / decisions
  -> metrics joined with annotations by timestamp
```

Annotations enter only at the last step. `build_wav_corpus_frames` accepts a
`CorpusClip` for its path but never reads `clip.annotations`.

Every real frame records:

- timestamp;
- actual sample rate;
- actual channel count;
- per-channel RMS;
- peak RMS and spatial contrast;
- dominant frequency;
- pitch and autocorrelation confidence;
- spectral centroid and spectral flux;
- `source=pcm_wav` provenance.

## Development and held-out separation

Corpus manifests assign whole clips to `dev` or `held_out`. Baseline comparison:

1. decodes every clip from WAV;
2. selects a threshold independently for peak RMS and spatial contrast using
   development F1;
3. freezes each threshold;
4. evaluates the corresponding held-out frames once;
5. reports both development and held-out confusion metrics.

For register discrimination, `positive_labels` identifies the target condition
(for example `infant_directed_speech`). Other annotated conditions remain
negative controls; an annotation no longer implies positivity by itself.

The first implementation does not perform cross-validation or claim statistical
significance.

## Predictive evaluation

The local hierarchy is evaluated prequentially: it predicts the next feature
before observing it, records the residual, then applies local updates. It retains
a fixed five scalar state values regardless of sequence length.

The lower level is a bounded local trend predictor. The higher level estimates
the clipped residual trend with a local update. Its prediction becomes a
top-down correction consumed by the lower level on the next frame.

Every report compares the hierarchy with:

- persistence (`next = previous`);
- an exponential moving average.

A passing synthetic sequence is not a real-audio promotion. Corpus reports state
the held-out clip win rate explicitly.

For repeated aligned protocols, a bounded higher-level phase template may be
fit on development clips. Each phase value is updated by a local running mean;
the blend with lower-level persistence is selected by leave-one-development-
clip-out validation. The frozen template is then evaluated on whole held-out
clips. Reports keep online-local and phase-template wins separate.

## Analysis-by-synthesis evidence

The clean synthetic gate crosses:

- five vowel identities;
- four source F0 values;
- 16 kHz and 48 kHz sample rates.

It reports the full confusion matrix and per-formant errors. Recorded voices are
a separate future corpus and acceptance gate.

## Performance boundary

`benchmark_event_pipeline` measures only in-process `AudioFrame` to
`WorkspaceEvent` processing. The report explicitly excludes capture, XMOS DSP,
USB transport, decoding, playback, and energy. End-to-end claims require a
separate instrumented live benchmark.

`LiveListener` additionally measures the host call boundary from immediately
before `capture_frame()` through the workspace decision. This includes the
capture adapter's blocking work but not a device-reported ADC onset timestamp,
XMOS internal DSP latency, or later USB beam steering.

The CLI listener prefers a session-scoped native PortAudio/CoreAudio stream via
`sounddevice`. A session-scoped `ffmpeg` raw PCM stream remains the explicit
fallback, and single-frame capture retains its bounded file-based path. Both
continuous backends avoid reopening CoreAudio for every frame. Latency summaries
separate first-frame cold start from steady state and report strict plus
tolerance-adjusted deadline misses. Auto fallback provenance is part of the
listener report.

For native streams, PortAudio's nominal input latency is recorded separately.
Cora adds that nominal value to the host call distribution to provide an
estimated ADC-to-workspace distribution, while retaining the actual host-call
numbers and marking the estimate as distinct from per-buffer device timestamps.

## Workspace ablation

The deterministic ablation compares three policies:

- flat admission;
- bounded fixed-window admission;
- theta/gamma admission with identity-aware self suppression.

Temporal overlap is not evidence that audio is self-generated. Suppression
requires `expected_self_audio=true`; when both sides provide `efference_id`, the
identities must match. This prevents the scheduler from deleting legitimate
world audio merely because it shares a motor gamma bin.
