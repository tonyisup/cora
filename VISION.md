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
- the first genuinely order-preserving retry represented each utterance as six
  successive prosodic phases and evaluated all 200 already-spent clips with
  leave-one-fieldsite-out fitting. Its zero-threshold F1 was only `0.302` with
  FPR `0.225`; selecting cutoffs solely from the out-of-field predictions met
  the FPR ceilings but reduced F1 to `0.289`, recall to `0.224`, and retained a
  minimum fieldsite F1 of `0.0`. Vocal recall remained `0.995`. This temporal
  candidate is falsified, and Wellington, San Diego, and Toronto remain
  untouched.

### Stage 3 — Predictive hierarchy

Status: H2 supported within tested boundaries. Controlled and natural held-out
prediction, noise, known-efference dropout, speaker/fieldsite transfer, bounded
state, and bottom-up/top-down flow have positive evidence.
Phase-template evidence remains positive in bounded settings, and a safer
online learner passed a fresh-fieldsite regret gate but failed superiority. The
first real structured-stream mechanism passed tones but failed vowels; a
development-only calibrated representation then passed a fresh two-family run.

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
- a spent-development audit rejected higher-resolution bands, cepstra, LPC
  vectors, window changes, and trial-local resets. One candidate advanced:
  featurewise normalization from four unlabeled same-trial calibration
  observations, with a fixed `0.30` scale floor and unit norm;
- that design beat the best causal baseline in all three leave-one-development-
  trial-out folds by at least `0.211`, so a new manifest with six fresh seeds
  was frozen before another real run;
- the fresh calibrated run completed 12 synchronized captures and passed every
  predictive-advantage gate. Tones reached accuracy `0.895` versus baseline
  `0.281` (`+0.614`); vowels reached `0.702` versus `0.281` (`+0.421`). Every
  held-out trial improved by at least `0.368`, all four states were occupied,
  top-down predictions were consumed, and state remained bounded at 68 scalars;
- vowel purity was `0.800`: calibration eliminated the prior `a/e` collapse,
  while `i/u` remained partially mixed. This passes the controlled two-family
  mechanism gate, not natural or semantic speech prediction;
- the calibrated-v2 captures are spent. H2 still requires stability under
  noise/dropout/speaker change and transfer to an unconstrained natural stream.
- a fresh condition-transfer protocol then tested an unseen synthetic F0 of
  `200 Hz`, same-condition additive noise at `10 dB` SNR, and grammar-only
  `10%` dropout after real room transfer;
- every held-out tone and vowel condition beat its best causal baseline by at
  least `0.316`. Vowel F0 shift and noise each retained four states; the pooled
  vowel improvement was `0.561`;
- the vowel dropout condition improved `0.579` over baseline but collapsed all
  `u` observations into the `i`-dominated state, leaving occupancy `3`. Because
  four states per condition were frozen as mandatory, robustness-v3 fails;
- the supported evidence is predictive advantage under all three controlled
  perturbations and representation stability under F0/noise. Dropout
  representation stability, recorded-speaker transfer, and natural-stream
  transfer remain open.
- a follow-up development-only audit selected median aggregation of 60 ms
  spectral subwindows as a candidate dropout-aware representation. It retained
  four states through `20%` dropout on development folds;
- before hardware, however, a fresh synthetic condition matrix failed: pooled
  vowel improvement was `0.193`, and the `15%`/`20%` conditions improved only
  `0.105`, below the frozen `0.20`/`0.15` gates. Real I/O was prohibited and no
  new live evidence was consumed;
- short-window aggregation/dropout-percentage tuning is therefore retired for
  this family. A future dropout design needs explicit missing-observation
  handling or another genuinely different mechanism.
- that different mechanism is now implemented for the bounded self-generated
  protocol: the efference path records intentionally erased source samples, and
  perception averages only 30 ms spectral frames fully outside those erasures;
- on the already-spent dropout-v4 synthetic matrix, it retained all four states
  at 10/15/20% dropout and improved vowels by `+0.579/+0.526/+0.526`; the full-
  spectrum and interpolation controls collapsed to at most two states;
- this advances a design candidate only. The mask is known because the source
  is self-generated; it is not arbitrary-world dropout detection. Fresh
  synthetic falsification and a separately preregistered live run are required
  before the dropout-stability limitation changes.
- a six-seed prospective synthetic matrix was then frozen before execution. It
  passed every gate: tones improved `+0.614` pooled, vowels `+0.596`, all
  10/15/20% conditions retained four states, and minimum trial improvement was
  `+0.579`;
- synthetic promotion remains prohibited. The result permits one separately
  preregistered real room-transfer run; it does not itself change H2 status.
- that real run then completed 12 synchronized playback/captures under the
  frozen geometry. Every artifact hash, alignment, stream position, and gate
  verified;
- vowels retained four states at 10/15/20% dropout and improved `+0.614` pooled
  with `+0.579` minimum trial improvement. Tones retained four states and
  improved `+0.474` pooled with `+0.368` minimum;
- together with the prior 10 dB noise/F0 result and the replicated Wellington
  and MEN natural transfer, this satisfies H2's stated bounded predictive-
  advantage, bottom-up/top-down, noise, dropout, and speaker-change evidence;
- the scope remains important: dropout timing came from the bounded self-
  generated efference path at one geometry. Arbitrary-world missingness,
  unknown dropout timing, semantics, and universal language transfer are not
  implied by H2's bounded pass.
- the calibrated codebook was then audited on eight already-spent continuous
  Beijing speech clips with four participant-held-out folds. The learner saw no
  IDS/ADS labels; 18 fixed frame/horizon/representation candidates were compared
  with persistence, development-unigram, and causal moving-state baselines;
- no candidate cleared the natural-stream design gate. The best (`200 ms`
  frames, `800 ms` horizon) won only `2/8` clips, improved mean accuracy by
  `0.0036`, and regressed the worst clip by `0.0989`;
- no fresh participant or fieldsite was consumed. Frame-size/fixed-horizon
  tuning of this four-state natural representation is retired. Natural H2 needs
  a different observation target or explicit novelty/missing-event mechanism.
- a genuinely different causal observation target was then defined from only
  current and past frames: quiet, onset/recovery, stable active, and spectral
  change. Thresholds come from each clip's first two seconds; B/D condition
  labels enter neither learning nor scoring;
- on the same eight spent Beijing clips, a fixed grid selected `50 ms` frames,
  quiet quantile `0.40`, and novelty quantile `0.75`. That representation won
  all `8/8` participant-held-out clips with mean improvement `+0.0952` and was
  frozen before Wellington audio was fetched;
- trained only on Beijing participants, the 16-scalar local transition learner
  then beat the strongest persistence, development-unigram, or causal-moving-
  state baseline on all eight fresh participant-disjoint Wellington clips.
  Mean held-out accuracy improvement was `+0.0751`, the worst clip remained
  positive at `+0.0107`, and every clip occupied at least three states;
- this passes the preregistered natural cross-fieldsite event-dynamics gate. It
  materially supports H2's core bounded local-prediction claim on natural audio,
  but it is not semantic prediction, IDS/ADS discrimination, arbitrary-language
  generality, or evidence that the separate controlled-dropout failure is fixed.
- post-confirmation ablations on the now-spent Wellington clips kept the same
  four-state target. The full mechanism beat removing current bottom-up event
  context and ignoring learned top-down transition predictions on all `8/8`
  clips; hiding the quiet/saliency event reduced mean accuracy from `0.7600` to
  `0.6688` and lost on `7/8` clips;
- those ablations support the component data flow but cannot promote the result
  again, and their event-context proxy is not evidence for H3's bounded event
  workspace.
- an independently preregistered Beijing-to-ARA replication retained the same
  code, training clips, gate, and four numerically lowest paired ARA participant
  IDs. It failed before scoring because `ARA01B` was exactly the 2-second
  calibration length and `ARA03B` was `1.632` seconds;
- no participant was substituted and calibration was not shortened after audio
  access. This is retained as protocol-eligibility failure: it supplies neither
  replication nor contradictory predictive-accuracy evidence. The next
  independent protocol must freeze a deterministic minimum-duration replacement
  rule before fetching candidate audio.
- the replacement replication froze that eligibility rule at `2.10 s`, selected
  MEN by deterministic metadata/evidence order, and fetched paired participants
  numerically until four qualified. MEN01–04 all qualified, so no duration
  rejection or discretionary substitution occurred;
- using the unchanged Beijing-only learner, all `8/8` MEN clips beat the
  strongest causal baseline. Mean improvement was `+0.1106`, worst improvement
  was `+0.0654`, every clip occupied at least three states, and state remained
  bounded at 16 scalars;
- together with Wellington (`8/8`, mean `+0.0751`), this independently
  replicates natural speaker/fieldsite transfer for the causal event target.
  The remaining H2 limitation is the separately demonstrated controlled-
  dropout representation collapse, not natural-transfer absence.

Promotion criterion:

- statistically or practically meaningful held-out error reduction across more
  than one recording or synthetic family, without unbounded state growth.

The promotion criterion is met across controlled tones/vowels and two natural
fieldsite transfers without unbounded state growth. Component ablations are
positive, and the later mask-aware real run repairs the earlier controlled-
dropout collapse for known self-generated erasures. H2 is therefore supported
within its measured boundary. H3's workspace claim remains separate.

### Stage 4 — Workspace binding

Status: supported within the measured structured-tone, fixed-geometry boundary.

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

Current spent real-capture design result:

- two Corollary self-only/mixed capture families selected thresholds; a third
  paired family plus a world-only control was held out. All 19 underlying real
  PCM hashes are retained;
- flat admission scored F1 `0.727` because it preserved all world audio but also
  admitted every self-only event;
- identity-only binding exposed the required collision failure: it suppressed
  self but reduced legitimate world/overlap recall to `0.25`;
- a three-slot workspace combining matching efference identity with independent
  bottom-up world-speech evidence reached F1 `1.0`, world recall `1.0`, and self-
  suppression specificity `1.0` on the held-out family;
- capacity `1/2/3/4` produced F1 `0.800/0.889/1.000/1.000`, demonstrating a
  bounded capacity requirement and plateau rather than unbounded growth;
- this is design evidence only. The speech/self features were produced upstream
  by Corollary and are not a recomputed Cora signal residual. A fresh Cora task
  must freeze the metric and compare flat, fixed-window, and scheduled binding
  under synchronized self-only, world-only, and overlap playback.

Current structured Cora protocol:

- a 15-trial device-agnostic protocol now separates development and held-out
  self/world frequency bands and includes self-only, world-only, and equal-gain
  overlap conditions;
- flat, fixed-self-window, and scheduled identity-plus-world-evidence strategies
  are scored across binding capacities `1/2/3/4`;
- the first synthetic run retained a real design failure: a symmetric midpoint
  world-evidence threshold made equal-gain overlap numerically borderline and
  reduced scheduled world recall to `0.667`;
- the corrected development-only rule places the threshold at 10% of the clean
  self→world evidence gap, reflecting the safety priority to preserve world
  overlap. Scheduled F1 then reached `1.0` versus flat `0.8` and fixed-window
  `0.667`; capacity `1/2/3/4` reached `0.857/0.923/1.0/1.0`;
- this is synthetic plumbing only. The external-world route is user-frozen at
  output device 2 (`External Headphones`), 180 degrees, and 50 cm until
  explicitly changed. Live I/O remains deliberately unwired until the Mac mini
  self-source geometry and level are also confirmed and frozen;
- the device-agnostic live boundary is nevertheless implemented and tested:
  self/world tracks stay separate, overlap sources start concurrently, one
  source owns the alignment chirp, and explicitly aligned two-channel captures
  feed the same frozen metric;
- a bounded acquisition runner now retains separate self/world/capture WAVs and
  hashes for all 15 trials, verifies stream position and chirp alignment, and
  fails closed on a source-start error. Its injected-I/O path is capped at 25
  seconds of playback and 45 seconds of capture; real-hardware claims are still
  rejected before any callback runs;
- the concrete CLI route now exists but is fail-closed before output creation or
  callbacks. It requires an explicit bounded-playback flag and a committed
  preregistration matching the exact runner, CLI, I/O, protocol, and standing-
  geometry hashes. The final preregistration freezes the self source centered
  directly below the array at a 4 cm center-to-center distance, -90 degrees
  elevation, and 100% system volume. This preregistration is committed before
  the first bounded hardware execution;
- attempt 1 completed all 15 synchronized trials. At capacity 3, scheduled
  identity plus world evidence reached F1 `1.0`, world recall `1.0`, and self-
  suppression specificity `1.0`, versus flat F1 `0.8` and fixed-window F1
  `0.667`/world recall `0.5`. Capacity `1/2/3/4` reached
  `0.857/0.923/1.0/1.0`, satisfying the frozen classification gate;
- a post-run PCM audit found `2,287` clipped samples across all nine self-
  bearing captures and no clipping in the six world-only captures. Because
  clipping can distort spectral evidence, attempt 1 is retained as a protocol
  result and quality failure, not H3 promotion. Attempt 2 freezes only the self
  digital playback gain from `0 dB` to `-12 dB` and adds a zero-clipping gate;
  physical geometry, system/hardware levels, trials, threshold rule, and all
  classification criteria remain unchanged;
- attempt 2 at `-12 dB` retained identical classifier metrics and reduced
  clipping by `94.58%`, from `2,287` samples in nine captures to `124` samples
  in seven captures, but still failed the zero-clipping gate. Attempt 3 is
  preregistered at `-18 dB`; all other variables and criteria remain frozen;
- attempt 3 completed all 15 real synchronized trials, retained the same
  scheduled/flat/fixed-window and capacity results, and contained zero clipped
  samples. It passed every frozen classification and capture-quality criterion.
  H3 is therefore supported within this single-room structured-tone boundary;
  natural speech binding, arbitrary spectral overlap, geometry transfer, and
  measured neural theta/gamma dynamics remain outside the claim.

### Stage 5 — Perception-production closure

Status: mechanism demonstrated; the first preregistered five-vowel,
five-unseen-singer VocalSet confirmation failed.

Required evidence:

- a five-by-five vowel confusion matrix from synthesized and recorded vowels;
- at least 90% identity preservation on clean synthesized vowels;
- formant error reported separately for F1, F2, and F3;
- held-out voices or formant targets not used to tune the estimator;
- motor plans learned or adapted from error before claiming motor learning.

Current recorded-human result:

- clean synthetic identity remained `1.0`, with F1/F2/F3 MAE near
  `24/41/49 Hz`;
- the untouched five-singer VocalSet straight-long-tone set scored `9/25`
  (`0.36`) against the frozen `0.80` gate;
- vowel recall was `a/e/i/o/u = 0.4/0.8/0.0/0.4/0.2`; singer accuracy ranged
  from `0.0` to `0.6`;
- recorded F1/F2/F3 MAE was approximately `528/647/440 Hz` relative to Cora's
  fixed adult reference targets;
- all 25 external hashes and the five-by-five confusion matrix are retained,
  but third-party audio remains outside Git under CC BY 4.0;
- these five singers are spent and cannot tune the next estimator;
- a disjoint five-singer development set then tested a bounded retry ladder:
  speaker normalization reached at most `0.28`, an LPC-order sweep `0.40`, a
  learned continuous spectral-to-formant map `0.52`, and a spectral-centroid
  classifier that emits canonical motor coordinates `0.56`. All used
  leave-one-singer-out scoring and remained below the frozen `0.80` gate;
- a separately frozen ordinary-read-speech test then used 500 aligned CMU
  Arctic vowel tokens from five development speakers. The fixed shared LPC
  estimator peaked at `0.572` (120 ms, order 20), with recall
  `a/e/i/o/u = 0.86/0.95/0.62/0.13/0.30`. Accuracy ranged from `0.33` to
  `0.77` by speaker. Ordinary speech therefore did not rescue the mechanism,
  and the five CMU Arctic confirmation speakers also remain unfetched;
- the reserved five-singer confirmation set remains unfetched and untouched.
  It must not be consumed until the corpus/domain/mechanism direction is chosen
  and a development candidate reaches a credible preregistration point.

H4 remains open. Both sung and ordinary read speech now falsify the current
shared fixed-formant route at development scale. More tuning toward a learned
phone classifier—or changing the frozen five-vowel mapping after seeing these
errors—risks replacing the narrow perception-production closure claim with
general supervised identity classification. No motor-learning claim is made.

### Stage 6 — Real-time efficiency

Status: offline/live resources, callback ADC-frame timing, and same-host
estimated CPU energy measured on real hardware. Quiet and structured-salient
timing gates passed. H5 is supported within the tested host/software boundary;
natural speech timing, XMOS internal delay, and embodied energy remain open.

The native CoreAudio path has a measured sub-40 ms *host capture-call* p95 at a
25 ms frame size. PortAudio reports an additional nominal input latency near
95 ms, yielding an estimated ADC-to-workspace p95 near 127 ms. Callback-quantum
jitter still fails the 30 ms host-call deadline on some frames. Three frozen
10-second native runs completed `1,200/1,200` frames with zero PortAudio overflow
flags, steady-state RTF `0.998` or lower, process CPU below `3.8%`, and peak RSS
below `63 MiB`. However, their operational deadline-miss rates were
`18.5%/14.0%/11.75%`, all above the `5%` gate. The live workload was a quiet
path with no emitted events.

The 1,000-frame offline path reached p95 `0.0039 ms` and RTF `0.000167`. A
same-input, same-instrumentation conventional fixed-threshold/prototype baseline
reached p95 `0.000834 ms` and was about `4.7x` faster, as expected for a simpler
resource floor. This is measured efficiency evidence, not yet a hard real-time
guarantee, device-timestamped measurement, or energy result.

A frozen single-variable retry moved capture from `48 kHz` to the XVF3800's
advertised native `16 kHz`. Reported input latency fell from `95.3 ms` to
`42.0 ms`, and estimated ADC-to-workspace p95 fell from about `126 ms` to
`73 ms`. The three operational miss rates remained `16.25%/14.25%/18.5%`, so
the deadline failure did not improve. The remaining timing problem is bounded
to the PortAudio/CoreAudio callback-quantum-to-blocking-read interface rather
than pipeline compute or host sample-rate conversion.

A preregistered callback/reblocking path then preserved PortAudio
`inputBufferAdcTime` for every exact 25 ms frame and compared it with stream
time at the workspace decision. Three eligible 10-second quiet-path runs
completed `1,200/1,200` timestamped frames with zero overflow, callback-status,
timestamp-discontinuity, fallback, or 30 ms device-deadline misses. ADC frame
end-to-workspace p95 was `10.19–10.21 ms`; onset-to-workspace p95, including
25 ms frame accumulation, was `35.19–35.21 ms`. CPU remained below `4.2%` and
peak RSS below `62.3 MiB`. The quiet-path device-timestamp gate therefore
passes. One originally scheduled third run stalled during teardown without
emitting a report; it is retained as incomplete, and its single frozen
replacement used non-draining PortAudio shutdown.

This first closed the device-timestamped quiet-path latency requirement on this
hardware; salient timing still required a separate loaded-path gate.

A follow-on structured-salient gate stratified the same device timestamps by
whether the pipeline actually emitted a workspace event. The initial `+12 dB`
and first `-6 dB` runs passed timing but failed zero-clipping with `5,830` and
`107` clipped samples. A final preregistered `-12 dB` attempt retained
`48/47/47` salient frames across three runs and had zero clipped samples across
`960,000` captured samples. Salient ADC frame-end-to-workspace p95 was
`9.94–10.20 ms`; quiet p95 was `10.10–10.29 ms`. There were zero device
deadline misses, overflows, callback-status events, timestamp discontinuities,
or fallbacks. CPU remained below `4.15%` and peak RSS below `65.2 MiB`.

The structured-salient timing gate therefore passes within this fixed
pitch-contour, room, geometry, input, and host boundary. Natural speech timing,
geometry transfer, raw CoreAudio timestamps, XMOS internal DSP delay, and
energy remain open.

A follow-on natural-speech gate played one hash-pinned, already-spent Music Lab
IDS utterance through the same world speaker. Both the initial `-12 dB` and
single permitted `-18 dB` three-run attempts passed every timing, salience,
stream-integrity, CPU, RSS, and real-time-factor criterion. Final salient ADC
frame-end-to-workspace p95 was `10.40–10.56 ms` with zero device misses, but
one run contained `108` full-scale captured samples while the other two were
clean. The lower gain did not monotonically remove clipping, so the frozen stop
rule is active. This is positive descriptive workload timing and a capture-
quality failure, not closure of the natural-speech boundary.

The final Stage 6 comparison used three preregistered 10-second trials each of
Cora, a simpler fixed-threshold/prototype baseline, and idle, sampled by macOS
`powermetrics` on the same M4 Pro. All nine trials and `20` central power samples
per trial passed their integrity gates. Cora and the conventional baseline drew
similar sustained estimated CPU power (`13.66 W` versus `13.53 W`), but the
baseline processed `2.44x` as many frames per second. Cora therefore used
`27.36 J` per million frames (`27.36 µJ/frame`) versus `11.08 J` for the
baseline, a `2.47x` energy-per-frame cost. This is a measured tradeoff, not an
efficiency win. One idle trial contained unrelated power spikes; they are
retained, and total energy per frame—not idle-subtracted power—is primary.

Together, the named-host latency, RTF, CPU, RSS, overflow/deadline, conventional
baseline, and estimated energy evidence supports H5 within the sustained
synthetic/structured host-software boundary. It does not measure wall-plug or
acoustic energy, XVF3800 consumption, paced live-runtime energy, natural-speech
timing, XMOS internal DSP delay, or cross-device efficiency.

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
