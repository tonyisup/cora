# Project Cora-Infans

## Vision

Cora-Infans is an audio-centric research system for testing whether a small,
stateful, embodied learner can acquire useful speech representations through
local prediction, attention, and analysis by synthesis.

The long-term hypothesis is inspired by infant auditory development: useful
speech structure may be learned from a continuous acoustic stream without a
large globally trained sequence model. The project combines a microphone-array
front end, acoustic saliency, local predictive learning, a bounded event
workspace, and a perception-to-production loop.

This document is a research contract. Terms such as *predictive coding*,
*theta-gamma scheduling*, and *analysis by synthesis* name hypotheses and
interfaces. They are not evidence of biological fidelity by themselves.

## Hardware boundary

The target front end is the Seeed Studio ReSpeaker XMOS XVF3800. Its four
physical microphones feed on-board XMOS DSP. On the current macOS path, the
host receives two fused USB channels rather than four independent microphone
signals.

Therefore:

- the project may claim a four-microphone DSP front end;
- it may not claim host-side four-channel learning or beamforming;
- two-channel energy imbalance is only a coarse lateral cue;
- firmware source direction is usable only when the firmware emits a finite,
  speech-classified bearing;
- obtaining raw per-microphone signals requires a different firmware, transport,
  or device and is a separate milestone.

## Research hypotheses

### H1 — Acoustic saliency

Infant-directed speech has measurable acoustic structure that can improve an
attention gate beyond raw loudness alone. Candidate observations include RMS
energy, spatial contrast, pitch and pitch confidence, pitch movement, spectral
change, formant estimates, and speech/non-speech confidence.

The current peak-RMS gate is the baseline. It is not a parentese detector.

### H2 — Local predictive learning

An online learner using local prediction errors can reduce held-out error for
auditory observations and focus trajectories without global backpropagation.

An exponential frequency update or last-label repeat predictor is a baseline,
not completion of this hypothesis. A successful hierarchy must demonstrate:

- lower held-out prediction error than persistence and moving-average baselines;
- information flowing both bottom-up as residuals and top-down as predictions;
- stability under noise, dropouts, and changes in speaker;
- bounded memory and update cost.

### H3 — Bounded event workspace

A capacity-limited event workspace can preserve useful temporal bindings while
reducing interference between concurrent or self-generated events.

Timestamp bucketing into nominal theta and gamma periods is an implementation
mechanism. The hypothesis is supported only if scheduling improves a defined
binding, tracking, or interference metric relative to a flat workspace.

### H4 — Analysis by synthesis

Perception and production can share an acoustic-articulatory representation so
that perceived speech primitives generate motor targets and synthesized output
can be perceived again.

The first primitive set is the five-vowel inventory `a/e/i/o/u`. A fixed adult
formant table is a calibration scaffold, not infant motor learning or a DIVA
implementation.

### H5 — Efficient embodied operation

Offloading low-level DSP and using bounded local state can provide useful
auditory behavior with low latency and modest compute.

This remains unproven until wall-clock latency, real-time factor, CPU, memory,
and—where measurable—energy are reported on a named machine and hardware path.

## Evidence ladder

Each stage must pass before the next stage can be described as more than a
scaffold. Tests of plumbing are necessary but do not satisfy empirical gates.

### Stage 0 — Reproducible substrate

Status: implemented.

Required evidence:

- deterministic unit and integration tests;
- synthetic and live adapters share explicit runtime contracts;
- real-I/O failures never silently fall back to synthetic data;
- reports preserve configuration and measurement provenance.

### Stage 1 — Real-audio evaluation

Status: passed for prior-art replay and licensed developmental-speech subsets;
broader condition coverage remains open.

Required evidence:

- decode real uncompressed integer-PCM and IEEE-float WAV files into timestamped
  frames;
- annotations are used only as ground truth, never to generate evaluated input;
- report precision, recall, F1, false-positive rate, and per-label results;
- maintain separate development and held-out clip lists;
- record sample rate, channel count, frame size, score definition, and thresholds.

Promotion criterion:

- at least one checked-in or reproducibly generated corpus manifest completes
  end to end with non-empty real frames and auditable metrics.

### Stage 2 — Parentese-relevant saliency

Status: mixed and not promoted. A five-second, speaker-and-mode batch-calibrated
scorer passed on two fresh fieldsites but failed a later joint untouched
Vanuatu test. The environmental front gate has repeated positive evidence;
cold-start, causal, and robust cross-fieldsite register operation remain open.

Required evidence:

- extract pitch/prosody, spectral-change, energy, and formant-related features
  from real frames;
- compare peak RMS, spatial contrast, and feature-based gates on held-out clips;
- choose thresholds using development clips only;
- on a participant-disjoint confirmatory set, exceed peak-RMS held-out F1 by at
  least `0.05` while keeping false-positive rate no more than `0.05` above the
  peak-RMS false-positive rate;
- report failure cases across speech, music, impulsive noise, and diffuse noise.

Current non-parentese evidence:

- thresholds were selected on four real development captures and frozen for two
  seed-held-out captures;
- spectral flux achieved held-out F1 `0.714` versus peak-RMS F1 `0.651`, while
  reducing false-positive rate from `0.735` to `0.469`;
- the task distinguishes mixed world+self audio from self-only audio and does
  not satisfy the parentese promotion criterion.

Current developmental-speech pilot:

- 12 real speech clips from six Music Lab participants were fetched by byte
  range from the licensed archive; four participants were development and two
  were held out, with paired infant- and adult-directed speech;
- pitch confidence reached held-out F1 `0.720` versus peak-RMS F1 `0.429`, but
  its false-positive rate was `0.833` versus peak RMS `0.208`;
- the result exposed that F1 alone rewards an almost-always-positive gate and
  therefore fails the stricter promotion criterion above;
- these six pilot participants are spent and cannot satisfy the confirmatory
  gate.

Current confirmation:

- the criterion above was fixed before fetching a fresh 12-clip Beijing subset;
- four new participants supplied development clips and two new participants
  supplied held-out clips, each with paired infant- and adult-directed speech;
- pitch-confidence F1 was `0.575` versus peak-RMS F1 `0.510`, an improvement of
  `0.065`; false-positive rate was `0.594` versus `0.547`, an increase of
  `0.047` and therefore inside the fixed `0.05` ceiling;
- this passes the narrow participant-disjoint rule, but the high absolute false-
  positive rates and single-fieldsite scope prohibit a general parentese claim.

Current cross-fieldsite result:

- Beijing thresholds were frozen before fetching 24 clips from six unseen
  Krakow and Mbendjele BaYaka participants;
- paired infant/adult speech and infant/adult song made song an explicit
  negative control;
- pitch-confidence F1 fell below peak RMS by `0.071` in Krakow and `0.023` in
  Mbendjele, so both fieldsites failed the preregistered gate;
- pitch-confidence activated on `76–92%` of song frames, while the frozen RMS
  threshold activated on essentially every negative frame;
- the supported conclusion is that a single absolute threshold is not a
  cross-cultural parentese detector. Register evidence must separate speech
  from song and normalize collection-site effects.

Current speaker-adaptive register result:

- a transparent ridge scorer now separates speech from song before comparing
  infant- and adult-directed speech using unlabeled, within-speaker summaries;
- the first fresh test on Jenu Kuruba and Quechua speakers failed (pooled F1
  `0.429`, FPR `0.278`), and a YIN pitch upgrade then failed on Tsimane' despite
  low pooled FPR;
- inspection of the source analysis exposed the adaptation error: acoustic
  summaries must be standardized within both speaker and vocal mode, rather
  than across speech and song together;
- that correction was frozen before evaluating 24 clips from six unseen Menominee
  and Toposa speakers. Pooled F1 was `0.667` versus `0.414` for the better frozen
  baseline, FPR was `0.111`, and song FPR was `0.0`; both fieldsites met the
  preregistered F1 floor;
- calibration uses two unlabeled observations from the same speaker and mode.
  It does not identify speakers and is neither cold-start nor strictly causal.

Current environmental-control result:

- the register model's speech/song gate alone mislabeled `87.5%` of selected
  ESC-50 environmental clips as speech, confirming that vocal activity is a
  separate upstream decision;
- a preregistered YIN-coverage threshold of `0.14` retained all fresh Spanish
  vocal clips but admitted `31.25%` of fresh fold-2 controls and failed;
- a stricter `0.20` threshold was then frozen before fetching 12 Hadza vocal
  clips and 16 fold-3 ESC-50 controls. Vocal recall was `1.0`; environmental
  FPR was `0.0625` overall, `0.125` for diffuse controls, and `0.0` for
  impulsive controls;
- one crackling-fire clip remained a false positive. The supported claim is a
  five-second vocal-activity front gate on these conditions, not general VAD;
- the front gate and register scorer are composed with hash-pinned artifacts;
- on a later jointly untouched Vanuatu/ESC-50 fold-4 set, the front gate retained
  all 12 vocal clips and rejected all 16 environmental clips, but register F1
  fell to `0.400` and failed its frozen `0.65` gate;
- the Menominee/Toposa pass remains bounded evidence. The latest result blocks
  Stage-2 cross-cultural promotion and points to IDS/ADS generalization—not
  environmental rejection—as the active failure.
- a speech cutoff selected only from the original ten-fieldsite grouped
  predictions restored F1 to `0.857` on fresh Nyangatom clips, but one of six
  songs was promoted and failed the frozen song-FPR ceiling;
- a final same-family refit used all 176 spent clips from 16 fieldsites and was
  frozen before two untouched Enga/Turku fieldsites. Pooled F1 was `0.600`
  versus the required `0.65`; Enga F1 was `0.500` and Turku F1 `0.667`;
- the final refit still improved over frozen absolute-feature baselines and kept
  FPR `0.056`, song FPR `0.083`, and vocal recall `1.0`, but low cross-site
  recall activates the preregistered stop rule. The remaining Wellington, San
  Diego, and Toronto fieldsites will not be consumed by another threshold/refit
  iteration of this feature family.

### Stage 3 — Predictive hierarchy

Status: mixed and not promoted. Phase-template evidence remains positive in
bounded settings; a safer online learner passed a fresh-fieldsite regret gate
but failed superiority. The real structured-stream mechanism passed tones but
failed synthesized vowels and therefore failed its two-family promotion rule.

Required evidence:

- an online locally updated acoustic predictor;
- a higher-level predictor consuming lower-level state or residuals;
- explicit top-down prediction consumed by the lower level;
- held-out comparison against persistence and exponential-moving-average
  baselines;
- ablations for hierarchy depth, workspace state, and saliency gating.

Current evidence:

- a bounded online trend/residual learner is stable but does not beat baselines
  on the first held-out real captures;
- a higher-level phase template learned by local running means on development
  captures sends top-down predictions into the lower feature stream;
- on two held-out captures it beats persistence/EMA for peak RMS, pitch
  confidence, spectral centroid, and spectral flux;
- spectral-centroid RMSE improves by `68.5%` and `52.9%` per held-out clip;
- because the capture protocol is temporally aligned and repeated, this is
  evidence for learned phase structure, not semantic attractors or unconstrained
  speech prediction.
- on fresh, variable-duration Beijing speech clips, the phase template beat the
  best baseline on 4/4 held-out clips for pitch confidence and spectral flux;
- in an ACO-development to Beijing-held-out transfer, spectral centroid won 3/4
  clips with improvements from `-1.0%` to `12.5%`, while the other features
  failed the win-rate gate and RMS suffered severe recording-level shift;
- the online local trend/residual learner won 0/4 throughout. The evidence is
  for a bounded onset-phase prior, not a successful free-running local learner.
- after that failure, the always-on trend was replaced by a ten-scalar local
  mixture of persistence, EMA, and top-down trend experts weighted by their own
  recent errors;
- on eight fresh rural-Poland clips, every feature passed the preregistered
  safety rule: mean regret versus the best baseline was `2.4–4.5%` and worst
  single-clip regret was `13.7%`;
- it won `0/8` clips for every feature. This validates conservative bounded
  behavior, not predictive advantage, and H2 remains open.
- a later spent-data design audit tested duration-normalized phase templates,
  local velocity at `0.25–2.0 s` horizons, fixed mean reversion, and quantized
  acoustic-state transitions across 200 clips from 18 fieldsites;
- an apparent `198/200` spectral-flux phase result was traced to the extractor's
  deterministic zero first frame and clip-end decay. After removing warm-up,
  edge bins, and adding a causal running-mean baseline, the audited template won
  only `56/200` clips and regressed mean RMSE by `12.6%`;
- the other scalar candidates also failed. Batch-optimized acoustic states had
  a modest transition ceiling, but a one-pass local competitive codebook
  regressed accuracy by `3.0–4.5` points;
- no candidate cleared the design gate, so no fresh fieldsite was consumed.
  H2 now requires a different observation protocol or locally learnable
  representation rather than another predictor tuned to these clips.
- a bounded structured-stream protocol is now implemented as that next
  observation scaffold: disjoint development/held-out tone and synthesized-
  vowel sequences, label-blind spectral states, a locally updated four-state
  codebook, and local transition counts with explicit top-down consumption;
- the frozen 12-trial protocol then completed on real synchronized playback and
  XVF3800 capture. All 24 WAV hashes, stream positions, and chirp alignments
  verified;
- tones passed every gate: four occupied states, purity `0.967`, transition
  accuracy `0.860` versus best-baseline accuracy `0.263`, and minimum per-trial
  improvement `0.526`;
- vowels failed: only three states were occupied, purity was `0.600`, and
  transition accuracy `0.596` tied the best baseline. Post-hoc analysis shows
  the measured state representation merged `a/e` and mostly merged `i/u`;
- because both families were required to pass, the preregistered verdict is
  failure and H2 remains open. All 12 real captures are spent and cannot confirm
  a revised representation or predictor.

Promotion criterion:

- statistically or practically meaningful held-out error reduction across more
  than one recording or synthetic family, without unbounded state growth.

Controlled structured-stream success, if obtained on real playback/capture,
would validate the mechanism on those families only. H2 still requires a later
transfer to an unconstrained natural acoustic stream.

### Stage 4 — Workspace binding

Status: synthetic identity-aware ablation passed; real-audio task not demonstrated.

Required evidence:

- a task with competing events or self-generated audio;
- a binding/interference metric defined before evaluation;
- comparison of flat, fixed-window, and theta-gamma-style scheduling;
- improvement that is not explained solely by discarding more events.

Current synthetic result:

- phase coincidence alone was shown to suppress legitimate world audio;
- the scheduler now requires an explicit matching self-audio/efference identity;
- this improves aligned self-echo F1 over flat/windowed handling while preserving
  legitimate-overlap recall in deterministic ablations;
- promotion still requires the same result from synchronized real audio.

### Stage 5 — Perception-production closure

Status: mechanism demonstrated; acoustic fidelity gate not passed.

Required evidence:

- a five-by-five vowel confusion matrix from synthesized and recorded vowels;
- at least 90% identity preservation on clean synthesized vowels;
- formant error reported separately for F1, F2, and F3;
- held-out voices or formant targets not used to tune the estimator;
- motor plans learned or adapted from error before claiming motor learning.

### Stage 6 — Real-time efficiency

Status: software-core and host call-boundary timing measured on real hardware;
device-onset timestamps and energy comparison remain open.

The native CoreAudio path has a measured sub-40 ms *host capture-call* p95 at a
25 ms frame size. PortAudio reports an additional nominal input latency near
95 ms, yielding an estimated ADC-to-workspace p95 near 127 ms. Callback-quantum
jitter still fails the stricter 30 ms host-call deadline on some frames. This is
low-latency evidence, not yet a hard real-time guarantee or a device-timestamped
measurement.

Required evidence:

- end-to-end frame latency distribution;
- real-time factor for live and offline paths;
- peak resident memory and CPU utilization;
- dropped-frame and deadline-miss rates;
- comparison with at least one conventional baseline on the same hardware.

## Claim discipline

The following terms have narrow meanings in project reports:

- **implemented**: code exists and its contract is tested;
- **measured**: a named experiment produced an artifact and metric;
- **validated**: the result passed a stated acceptance criterion on data not
  generated from its annotations;
- **biologically inspired**: a design analogy, not biological equivalence;
- **learned**: parameters were selected from observations without placing the
  hidden generating parameters directly in the candidate set;
- **live**: actual device input or output occurred and the report records the
  device and timing provenance.

Historical debug results may remain in status documents, but retracted or
quarantined artifacts cannot satisfy a promotion gate.

## Near-term priority

Until Stages 1–3 pass, new conversational-policy features are secondary. The
priority order is:

1. real-audio measurement;
2. saliency baselines and parentese-relevant features;
3. locally learned predictive representations;
4. workspace ablations;
5. reliable perception-production closure;
6. latency and efficiency evidence.
