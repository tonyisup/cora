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

### Cross-fieldsite saliency falsification

- Wrote the protocol before fetching new audio, then held the Beijing RMS and
  pitch-confidence thresholds fixed.
- Fetched 24 clips from six unseen participants across Krakow and Mbendjele
  BaYaka: paired infant/adult speech plus infant/adult song.
- Krakow: pitch F1 `0.3121` versus RMS `0.3836`; pitch FPR `0.8244` versus RMS
  `0.9979`. Gate failed.
- Mbendjele: pitch F1 `0.1265` versus RMS `0.1492`; pitch FPR `0.8643` versus RMS
  `0.9982`. Gate failed.
- Pitch-confidence activation on song ranged from `75.8%` to `91.7%` depending
  on fieldsite and direction. The feature is voiced/prosodic, not register-
  specific.
- Conclusion: the narrow Beijing confirmation does not transport. A useful
  scorer needs site normalization and explicit speech-vs-song structure.

### Speaker-adaptive register scorer

- Added long-window acoustic summaries spanning pitch and YIN pitch variation,
  energy, spectral change, modulation, and formant-related proxies.
- Added a transparent speech/song ridge gate followed by an IDS/ADS ridge gate
  over unlabeled within-speaker normalized summaries. Fitting, JSON
  serialization, grouped-fieldsite evaluation, and frozen-model CLI evaluation
  are reproducible.
- The first fresh Jenu Kuruba/Quechua confirmation failed: pooled F1 `0.4286`
  and FPR `0.2778`.
- An extraction audit against the source study found strong agreement for pitch
  median (`r=0.984`) but weak agreement for pitch IQR (`r=0.093`). An
  overlapping FFT-YIN estimator improved pitch-IQR agreement to `r=0.689`.
- The resulting v2 model still failed on fresh Mescalero/Tsimane' clips: pooled
  F1 `0.4444`; Tsimane' F1 `0.0`. Those clips were marked spent.
- Inspection of the published analysis code then exposed a semantic mismatch:
  the source standardized within both speaker and vocal mode. Cora had
  standardized all four speech/song conditions together.
- The corrected v3 scorer was frozen before fetching six unseen Menominee and
  Toposa speakers. It passed every preregistered gate: pooled F1 `0.6667`, FPR
  `0.1111`, zero song false positives, and F1 `0.6667` on each fieldsite. The
  better frozen baseline reached only F1 `0.4138`.
- Scope: each speaker supplies two unlabeled clips per vocal mode. This is
  batch calibration, not speaker identification, cold start, or strictly causal
  live detection.

### Environmental controls and vocal front gate

- Registered 16 ESC-50 fold-1 diffuse/impulsive controls before fetching them.
  The frozen register speech/song gate mislabeled `87.5%` as speech, proving it
  is not a vocal/non-vocal detector.
- Added a transparent upstream gate based on the fraction of five-second YIN
  windows with a credible pitch estimate.
- Threshold `0.14` was registered before fresh Spanish vocal clips and ESC-50
  fold-2 controls. Vocal recall was `1.0`, but environmental FPR was `0.3125`,
  so the gate failed and the fold was spent.
- Threshold `0.20` was then fixed from spent scores before fetching 12 Hadza
  vocal clips and 16 ESC-50 fold-3 controls. All preregistered gates passed:
  vocal recall `1.0`, overall environmental FPR `0.0625`, diffuse FPR `0.125`,
  and impulsive FPR `0.0`.
- One crackling-fire clip scored as vocal. The claim is limited to a five-second
  front gate on these conditions; it is not general or frame-level VAD.
- The confirmed front gate and v3 register scorer are combined in a hash-pinned
  v4 composition. It preserves the earlier Menominee/Toposa metrics, but the
  two gates were confirmed separately rather than in one untouched joint test.
- The v4 composition was then frozen before selecting 12 Vanuatu vocal clips
  and 16 ESC-50 fold-4 controls. The vocal surface passed cleanly: recall `1.0`
  and environmental pass-through `0.0` for both diffuse and impulsive groups.
- The register surface failed: F1 `0.400`, recall `0.333`, FPR `0.111`, with one
  of three IDS clips promoted. This fails the frozen `0.65` F1 gate and blocks
  Stage-2 promotion despite the earlier Menominee/Toposa pass.
- The active weakness is now IDS/ADS generalization rather than environmental
  rejection. All Vanuatu and fold-4 clips are spent.
- A v5 candidate selected its speech cutoff strictly from the original ten
  training fieldsites. On fresh Nyangatom clips, F1 was `0.857`, FPR `0.111`,
  and vocal recall `1.0`, but one adult-directed song produced song FPR `0.167`
  and failed the frozen `0.15` ceiling.
- The final same-family v6 refit used all 176 now-spent vocal clips across 16
  fieldsites, with unchanged feature families. Its 16-fieldsite LOFO threshold
  selection produced F1 `0.785`, FPR `0.105`, and minimum fieldsite F1 `0.500`.
- v6 was frozen before fetching six speakers from two untouched societies.
  Enga F1 was `0.500`; Turku F1 was `0.667`; pooled F1 `0.600` failed the
  required `0.65`. Baseline improvement, pooled FPR `0.056`, song FPR `0.083`,
  vocal recall `1.0`, and calibration gates all passed.
- The preregistered stop rule is active: Wellington, San Diego, and Toronto will
  not be spent on another threshold/refit iteration of the same paired-summary
  feature family.
- Music Lab audio remains external under CC BY-NC-SA 4.0; ESC-50 audio remains
  external under CC BY-NC 3.0. This personal, non-commercial use and the
  repository boundary are recorded in `docs/DATA_POLICY.md`.

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

### Safe online predictor redesign

- Replaced always-on local trend correction with a bounded ten-scalar mixture
  of persistence, EMA, and top-down trend experts. Each weight updates from its
  own prequential squared error.
- Fixed a scale bug: the old minimum correction bound of `1.0` was enormous for
  RMS and spectral-flux values near `0.01`, causing avoidable overshoot.
- Registered the safety criterion before fetching eight rural-Poland IDS/ADS
  clips from four new participants.
- All four features passed: mean regret against the best baseline was `2.38%`
  to `4.47%`; worst single-clip regret was `13.67%`, below the `15%` ceiling.
- Every feature still won `0/8` clips. The redesign validates safe degradation,
  not H2 predictive superiority.

### Longer-timescale predictive design audit

- Used only 200 already-spent clips across 18 fieldsites; no new confirmation
  audio was fetched.
- Tested duration-normalized phase templates, bounded local velocity over one
  to eight frames, fixed persistence/EMA interpolation, and quantized acoustic
  state transitions.
- The initial duration-normalized log-flux result appeared to win `198/200`
  clips. Audit found two non-auditory shortcuts: the extractor forces the first
  flux frame to zero, and duration normalization exposes the clip boundary.
- After discarding warm-up, excluding edge bins, and adding a causal running-
  mean baseline, the template won `56/200` clips and mean RMSE was `12.59%`
  worse than the strongest baseline.
- Every local-velocity and fixed-mean-reversion setting regressed on average.
- A 12-state, one-second transition model using batch codebook optimization had
  a modest `+4.7` point accuracy ceiling, but a one-pass local competitive
  codebook regressed by `3.0–4.5` points. It therefore fails H2's local-learning
  boundary and was not implemented or promoted.
- Conclusion: the current short isolated clips and frame summaries do not
  support another justified predictor iteration. Stage 3 needs a different
  observation protocol or locally learnable representation.

### Structured continuous-audio prerequisite

- Added a fixed 12-trial protocol spanning tone and synthesized-vowel families,
  with three development and three held-out seeds per family.
- Each continuous trial contains an alignment chirp, four unlabeled codebook
  calibration observations, and 20 grammar observations. Scheduled token
  identities are available only for post-hoc purity scoring.
- Added a bounded four-state competitive spectral codebook and smoothed local
  transition counts. The higher-level next-state prediction is consumed by the
  lower level and produces an explicit one-hot residual.
- Baselines are state persistence, development unigram, and a causal moving
  state distribution.
- The fake acoustic channel passes deterministically for both families:
  transition accuracy `0.912` versus best-baseline accuracy `0.281`, with four
  occupied states and purity `1.0`. This is synthetic plumbing, not evidence.
- The synchronized real runner is implemented through injected playback/capture
  interfaces. It requires explicit bounded authorization and complete geometry
  before CLI wiring touches devices, stops after the first failure, recovers a
  chirp alignment, and hashes every efference/capture artifact.
- A noiseless injected runner test exposed and fixed an absolute FFT-floor bug
  that had made state geometry depend on background noise.
- Real acceptance thresholds, manifest, and geometry were hash-frozen. The
  first authorized attempt stopped before playback with zero completed trials:
  ffmpeg's raw muxer buffered the XVF3800 stream beyond the readiness deadline.
- The runner now flushes raw packets so byte count is a live stream position.
  A microphone-only synchronized diagnostic verified readiness at 128 ms and
  verified overlap. Attempt 1 and its original hashes are retained; attempt 2
  was separately frozen and explicitly authorized.
- Attempt 2 completed all 12 real synchronized trials and retained 24 hashed
  WAVs. Tones passed every gate: occupancy `4`, purity `0.967`, transition
  accuracy `0.860` versus best baseline `0.263`, and improvement `0.596`.
- Vowels failed: occupancy `3`, purity `0.600`, transition accuracy `0.596`,
  best baseline `0.596`, and improvement `0.000`. Post-hoc confusion counts
  show `a/e` merged and `i/u` mostly merged under the measured spectral states.
  The overall two-family verdict is failure; H2 remains open and all 12 real
  captures are spent.
- A development-only follow-up audit rejected higher-resolution spectra,
  cepstra, LPC formants, token margins, and trial-local resets. Featurewise
  z-scoring from the four unlabeled same-trial calibration observations, with a
  fixed `0.30` standard-deviation floor and unit norm, was the only candidate
  to beat every causal baseline in every leave-one-development-trial-out fold.
- That representation and six new seeds were frozen before a new real run.
  All 12 synchronized captures completed and all predictive gates passed.
  Tones: accuracy `0.895`, baseline `0.281`, improvement `0.614`, purity `1.0`.
  Vowels: accuracy `0.702`, baseline `0.281`, improvement `0.421`, purity
  `0.800`. Minimum held-out-trial improvement was `0.368`.
- This promotes the controlled two-family mechanism only. The representation
  still partially mixes `i/u`, uses known calibration slots and one synthetic
  voice, and has not passed noise, dropout, speaker, or natural-stream transfer.
- A subsequent fresh robustness protocol used an unseen `200 Hz` synthetic F0,
  same-condition `10 dB` noise, and grammar-only `10%` dropout. Every condition
  beat its best causal baseline; pooled vowel improvement was `0.561`.
- F0 shift and noise retained all four vowel states. Dropout scored accuracy
  `0.842` versus baseline `0.263` but mapped every `u` token to the `i`-dominated
  state, leaving occupancy `3`. The frozen per-condition occupancy gate fails,
  so predictive robustness is positive but representation robustness is mixed.
- A development-only temporal-aggregation audit then selected median 60 ms
  spectra at a 30 ms hop, but an entirely fresh synthetic condition matrix
  failed before hardware: pooled vowel improvement `0.193`; `15%` and `20%`
  dropout improved only `0.105`. The CLI blocks real I/O for this variant. No
  new live capture was spent, and short-window/dropout-percentage tuning stops.
- Added a reproducible label-blind natural-stream design harness and reacquired
  only eight already-spent Beijing development clips by corrected ZIP64 byte
  ranges; every SHA-256 matched the earlier evidence.
- Across four participant-held-out folds, 18 combinations of full/temporal-
  median spectrum, `100/200/400 ms` frames, and `1/2/4`-frame horizons all
  failed. The best won `2/8` clips, mean improvement was `0.0036`, and worst-
  clip improvement was `-0.0989`. No fresh natural participant was consumed.
- Reframed the observation target as causal acoustic events—quiet, onset or
  recovery, stable active, and spectral change—rather than another calibrated
  fixed-frame codebook. A spent-only Beijing grid selected `50 ms` frames,
  quiet quantile `0.40`, and novelty quantile `0.75`, winning `8/8` participant-
  held-out clips with mean improvement `0.0952`.
- Froze that design, implementation hashes, Wellington participants, audio-hash
  requirement, and pass/fail thresholds before retrieving held-out audio. The
  official archive's mixed ZIP64 offsets were handled by exact HTTP byte ranges;
  all eight WAVs verified against uncompressed size, CRC32, and SHA-256.
- The untouched Beijing-to-Wellington transfer passed every gate: `8/8` clip
  wins, mean improvement `0.0751`, worst improvement `0.0107`, minimum state
  occupancy `3`, zero participant overlap, top-down predictions consumed, and
  bounded state of 16 scalars. B/D labels entered neither learning nor scoring.
- This is positive cross-fieldsite natural event-dynamics evidence. It does not
  establish semantic/content prediction, IDS/ADS recognition, arbitrary-
  language transfer, or repair the separate controlled-dropout collapse.
- On the now-spent Wellington confirmation clips, post-confirmation component
  ablations retained the original four-state target. The full mechanism beat
  no bottom-up event context and no learned top-down consumption on `8/8` clips
  each. Hiding quiet/saliency context reduced mean accuracy from `0.7600` to
  `0.6688`, with the full path winning `7/8`. This supports causal anatomy only;
  it is not another promotion and does not stand in for H3 workspace evidence.
- Preregistered an independent Beijing-only to ARA replication using the four
  lowest paired ARA participant IDs, unchanged observation/learner code, and
  the Wellington gate. All eight fetched WAVs verified by size, CRC32, and
  SHA-256, but scoring was blocked: `ARA01B` is exactly `2.000` seconds and
  `ARA03B` is `1.632` seconds, too short for the frozen 2-second calibration
  plus an evaluated frame. The failed eligibility result is retained with no
  participant substitution or calibration change. It does not count as
  predictive failure or replication.

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
285 passed in 21.75s
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

1. Preserve the vocal front gate, but retire threshold/refit iteration of the
   current paired-summary IDS model. A future register attempt needs a genuinely
   different temporal representation or protocol before using another fieldsite.
2. Separately design a causal calibration path before any cold-start or live
   parentese claim.
3. Keep the safe expert mixture as a fallback. The calibrated structured
   mechanism passes clean, F0-shift, and noise conditions, but dropout collapses
   one vowel state despite strong prediction. Any dropout-aware revision must
   be designed on development/synthetic data and confirmed on fresh conditions;
   recorded-speaker transfer remains a separate later gate. Natural event-
   dynamics transfer now passes Beijing-to-Wellington, while the earlier fixed-
   frame codebook remains retired; the next H2 work should be the required
   hierarchy/workspace/saliency ablations and a second independent natural
   context, not threshold retuning on Wellington.
   The first independent ARA attempt exposed a duration-eligibility omission;
   any replacement replication must preregister sequential duration screening
   and participant replacement before fetching candidate audio.
4. Add recorded-vowel speakers not used to tune LPC selection and run a separate
   confusion matrix.
5. Instrument the live capture path for microphone-to-event latency and dropped
   frame rate before making a real-time efficiency claim.
