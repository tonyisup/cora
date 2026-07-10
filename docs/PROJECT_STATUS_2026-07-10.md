# Project Status — 2026-07-10

## Evidence-first pivot

This checkpoint turns the prior architecture description into a falsifiable
research contract and implements the first vertical evidence path.

## Completed

### Vision and claim discipline

- Replaced the incomplete vision with staged hypotheses, promotion criteria,
  hardware boundaries, and definitions for implemented/measured/validated/live.
- Explicitly prioritizes real auditory evidence over additional conversational
  policy surfaces.
- Corrects the hardware claim to four physical microphones with two fused host
  channels on the current macOS path.

### Real uncompressed WAV evaluation

- Added channel-preserving 8/16/24/32-bit integer PCM plus 32/64-bit IEEE-float
  WAV decoding.
- Added direct frame measurement from samples; annotations never generate input.
- Extracts peak RMS, spatial contrast, dominant frequency, pitch and confidence,
  spectral centroid, and spectral flux.
- Reports TP/FP/TN/FN, precision, recall, F1, false-positive rate, sample rate,
  channel count, source provenance, and feature summaries.
- Real WAV is the CLI default. Synthetic fixture evaluation requires explicit
  `--input-mode synthetic_fixture` and reports non-real evidence provenance.

### Development/held-out saliency comparison

- Corpus manifests accept clip-level `dev` and `held_out` splits.
- `--compare-baselines` selects peak-RMS and spatial-contrast thresholds using
  development F1 only, freezes them, then reports held-out performance.
- No feature-based parentese advantage is claimed yet.

### External IDS corpus preparation

- Selected the Music Lab cross-cultural human-vocalization corpus as the first
  target (DOI `10.5281/zenodo.5525161`, CC BY-NC-SA 4.0).
- Added a deterministic manifest builder for extracted data; it does not
  silently download the approximately 5.8 GB edited archive.
- Maps infant/adult-directed speech conditions from documented filenames.
- Splits by participant hash so one speaker cannot appear in both development
  and held-out partitions.
- Added target-label evaluation so adult-directed annotated clips can serve as
  explicit negative controls instead of being treated as positive events.
- Feature baselines now include pitch, pitch confidence, spectral centroid, and
  spectral flux in addition to peak RMS and spatial contrast.
- Zenodo supports byte-range reads from the archive, enabling a small auditable
  subset without retaining the full 5.8 GB download.
- The corpus mixes integer PCM and IEEE-float WAV (including float64). The first
  attempted run exposed that Python's standard WAV reader rejects format 3;
  Cora now decodes uncompressed integer PCM and IEEE float explicitly.

### IDS-vs-ADS pilot

- Fetched 12 paired speech clips from six participants, with four participants
  assigned to development and two to held-out by the manifest's deterministic
  participant hash. Audio remains outside the repository.
- Thresholds were selected using development frames only and frozen before the
  held-out report.
- Peak RMS: held-out F1 `0.4286`, FPR `0.2083`.
- Pitch confidence: held-out F1 `0.7200`, FPR `0.8333`.
- This is a negative gate result. The feature's F1 gain comes with excessive
  false positives, revealing that an F1-only criterion is too permissive.
- Before a fresh participant set is evaluated, the confirmatory gate is fixed:
  feature F1 must exceed peak RMS by at least `0.05`, and feature FPR may be no
  more than `0.05` above peak RMS. The six pilot participants are spent.

### IDS-vs-ADS confirmation

- After fixing the gate, fetched 12 new Beijing speech clips from six entirely
  fresh participants: four development and two held out, paired IDS/ADS.
- Peak RMS held-out: F1 `0.5102`, FPR `0.5472`.
- Pitch confidence held-out: F1 `0.5755`, FPR `0.5943`.
- The F1 improvement is `0.0653`; the FPR increase is `0.0472`. Both clear the
  pre-specified narrow thresholds, so the participant-disjoint confirmation
  passes.
- This is not broad validation: both absolute FPRs are high, all confirmatory
  speakers are from one fieldsite, and music/noise controls are absent.

### Developmental-speech predictive probe

- On the fresh Beijing protocol, the development-trained phase template beat
  persistence/EMA on 4/4 held-out clips for pitch confidence and spectral flux.
- The online local trend/residual hierarchy won 0/4 for all four tested
  features, so it remains a failed research hypothesis in its current form.
- A harder transfer trained the phase prior on ACO development participants and
  evaluated Beijing held-out participants. Spectral centroid won 3/4 clips with
  per-clip RMSE changes of `+1.1%`, `+12.5%`, `+2.3%`, and `-1.0%`; peak RMS,
  pitch confidence, and spectral flux failed the win-rate rule.
- Interpretation: onset-relative acoustic structure sometimes transfers, but
  simple amplitude features are not normalized across collection sites and the
  evidence is not semantic or free-running prediction.

### Prior-art real-capture replay

- `AGENTS.md` identified the Corollary project as prior art. Its archived M2
  `observed.npy` arrays are retained real speaker/microphone captures.
- Added a no-copy manifest builder and measured-array provider with explicit
  `recorded_npy` provenance and sample rate.
- First replay used six aligned captures from the energy-onset matrix: four
  development clips (seeds 0/1) and two held-out clips (seed 2), with self-only
  captures as negative controls and mixed world+self captures as positives.
- Held-out saliency result: spectral-flux F1 `0.7143`, FPR `0.4694`; peak-RMS
  F1 `0.6508`, FPR `0.7347`. The feature advantage is real but the task is not
  parentese discrimination.
- The original online local hierarchy lost 0/2 held-out clips and was unstable
  around abrupt frequency-bin changes. Replaced its unconstrained NLMS update
  with bounded local trend and residual updates; it no longer explodes but still
  does not beat the simple baselines on this data.
- Added a bounded higher phase template learned through local running means on
  development captures, with blend selected by leave-one-development-clip-out
  validation. It wins 2/2 held-out clips for peak RMS, pitch confidence,
  spectral centroid, and spectral flux. Spectral-centroid RMSE improvement over
  the best persistence/EMA baseline is `68.5%` and `52.9%` by clip.
- Scope: repeated temporally aligned capture protocol. This is top-down phase
  prediction evidence, not semantic prediction or cross-corpus generalization.

### Local predictive evidence

- Added a bounded local trend predictor with robust residual clipping.
- Added a higher residual trend whose output is consumed as the next top-down
  lower-level correction.
- Added prequential comparisons against persistence and EMA.
- Held-out corpus reports expose clip win rate and do not auto-promote from a
  synthetic test.

### Analysis-by-synthesis repair

- Replaced smoothed-FFT peak picking with LPC pole estimation.
- Increased fixed F1 synthesis excitation so lip-radiation tilt does not erase
  the first formant.
- Added a clean synthetic confusion-matrix gate across 5 vowels, 4 F0 values,
  and 16/48 kHz: 40/40 correct in the current deterministic run.
- Reports F1, F2, and F3 mean absolute error separately.
- Scope remains clean synthetic only; recorded-speaker validation is open.

### Performance measurement

- Added an in-process `AudioFrame -> WorkspaceEvent` microbenchmark.
- Reports mean/p50/p95/p99/max latency, CPU and wall time, real-time factor,
  traced memory, environment, and resident workspace size.
- Explicitly excludes capture, XMOS DSP, USB, decoding, playback, and energy.

### Live latency boundary

- Every `LiveListener` session now measures capture-call latency, pipeline
  decision latency, combined capture-to-workspace latency, deadline misses, and
  measured-path real-time factor.
- Reports whether the capture backend is a real `LiveMicArray` or injected fake.
- The timestamp quality is explicitly host call-boundary timing, not ADC onset,
  XMOS internal latency, or a CoreAudio backend timestamp.
- Real measurement found the original adapter reopened `ffmpeg` for every 250 ms
  frame: two-frame capture-to-workspace p95 was approximately `673.7 ms`, with
  both frames missing their deadlines.
- Replaced per-frame startup with a session-scoped raw PCM stream.
- Real eight-frame remeasurement on the XVF3800: cold start `524.1 ms`;
  steady-state mean `250.6 ms`, p95 `264.1 ms`, and real-time factor `1.003`.
  Two of eight frames exceeded the operational `262.5 ms` limit. The path is
  materially improved but is not yet promoted as deadline-safe.
- A 25 ms / 40-frame probe produced cold start `322.9 ms`, steady-state mean
  `25.34 ms`, p95 `37.72 ms`, and real-time factor `1.013`. Raw reads arrived in
  roughly 35–38 ms backend bursts followed by near-zero buffered reads. This
  suggests the ffmpeg/CoreAudio transport cannot guarantee one decision per
  25 ms gamma-sized window even though average throughput is close to real time;
  a native timestamped CoreAudio adapter or explicit paced ring buffer is needed
  for a strict sub-40 ms deadline claim.
- Prior-art review (`AGENTS.md`) identified Corollary's callback-based
  `sounddevice`/PortAudio runtime. Added it as Cora's preferred continuous
  capture backend while retaining explicit ffmpeg fallback and the existing
  `AudioFrame` contract.
- Real native-CoreAudio 25 ms / 40-frame result: cold start `58.1 ms`,
  steady-state mean `24.70 ms`, p95 `31.34 ms`, and real-time factor `0.988`.
  Nine of 40 frames exceeded the operational 30 ms limit and 31 exceeded the
  strict 25 ms boundary, reflecting callback-quantum jitter. This is a clear
  latency improvement and sub-40 ms p95 result, not a hard deadline guarantee.
- PortAudio reports nominal input latency `95.33 ms` on this device despite the
  low-latency request. Adding that declared backend latency to the steady host
  call distribution yields an estimated ADC-to-workspace p95 around `126.7 ms`.
  Reports expose both distributions and do not mislabel the estimate as a
  device timestamp.
- `--capture-backend auto|native|ffmpeg` makes selection auditable. Auto mode
  records any native-to-ffmpeg fallback reason and never falls back to synthetic
  audio.

### Workspace ablation and correction

- Added flat, fixed-window, and theta-gamma comparisons on two deterministic
  tasks: aligned self echo and legitimate world audio overlapping a motor bin.
- The first ablation exposed that phase coincidence alone wrongly suppressed
  legitimate world audio.
- Corrected the scheduler to require explicit expected-self evidence and a
  matching efference identity when identities are available.
- Corrected scheduler result: aligned self-echo F1 `1.0` versus flat `0.667`,
  while legitimate-overlap recall remains `1.0`.
- Evidence scope remains synthetic known timing; synchronized real-audio
  validation is still required.

## Verification

```text
pytest tests -q
266 passed in 17.41s
```

Representative 100-iteration event-core run on the current host:

```text
p50 0.00435 ms
p95 0.00584 ms
p99 0.00871 ms
real-time factor 0.000297
```

These numbers are a microbenchmark, not end-to-end auditory latency.

## Next evidence required

1. Repeat the fixed parentese criterion across additional fieldsites without
   reusing pilot or confirmatory participants.
2. Add music, impulsive noise, and diffuse-background negative controls.
3. Replace or materially rethink the online local predictor, which loses to
   persistence/EMA on every current held-out developmental-speech clip.
4. Add recorded-vowel speakers not used to tune LPC selection and run a separate
   confusion matrix.
5. Instrument the live capture path for microphone-to-event latency and dropped
   frame rate before making a real-time efficiency claim.
