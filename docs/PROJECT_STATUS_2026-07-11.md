# Project status — 2026-07-11

This document consolidates the current evidence against [VISION.md](../VISION.md).
It does not supersede preregistrations or promote a failed gate. When a summary
and a retained artifact disagree, the retained artifact and its frozen protocol
are authoritative.

For acronyms, corpus codes, metrics, and common report fields, see the
[data dictionary](DATA_DICTIONARY.md).
The final requirement-by-requirement checkpoint is retained in the
[VISION evidence audit](VISION_EVIDENCE_AUDIT_2026-07-11.md).

## Executive outcome

- Stages 0 and 1 are implemented and supported by reproducible real-audio
  evaluation.
- Broad cross-cultural Stage 2 register detection is falsified within tested
  scope. The vocal front gate and relational cohort effect remain narrow
  evidence; active register work is stopped.
- H2, H3, and H5 are supported only inside their measured boundaries.
- The tested bounded-local H4 representation family is formally falsified
  within scope. Active H4 work is stopped; the passed plant is infrastructure
  only, not closure evidence.
- No new reserved Stage 2 or H4 confirmation data was consumed after its
  applicable development stop rule fired. Wellington had already been consumed
  for a separate H2 target and is not fieldsite-pristine for Stage 2.

## Evidence matrix

| Stage | Current verdict | Strongest supporting evidence | Boundary or contradiction |
|---|---|---|---|
| 0 — Reproducible substrate | Implemented | Deterministic unit/integration suite; explicit synthetic/live contracts; fail-loud real I/O; hash-pinned external-data runners | Passing plumbing is not empirical promotion |
| 1 — Real-audio evaluation | Passed in bounded corpora | [prior-art replay](evidence/2026-07-10-prior-art-replay.json), Music Lab PCM/float WAV evaluation, participant/fieldsite manifests | Broader devices, rooms, ages, and spontaneous conditions remain open |
| 2 — Environmental vocal gating | Broad register detection falsified | [vocal front-gate pass](evidence/2026-07-10-yin-vocal-gate-v2-result.json) and cohort alignment's low FPR remain narrow evidence | [Final disposition](evidence/2026-07-11-stage2-broad-register-disposition.json) stops active register research after absolute, adaptive, temporal, predictive, cohort, and local nonlinear failures; USD/TOR remain untouched |
| 3 — Predictive hierarchy | H2 supported within tested boundaries | Natural transfer replicated on Wellington and MEN; controlled real known-efference dropout retained four states and improved tone/vowel prediction | Unknown dropout timing, arbitrary missing world input, semantics, and room transfer remain outside the claim |
| 4 — Workspace binding | H3 supported within structured-tone and exact-efference speech boundaries | [structured attempt 3](evidence/2026-07-10-structured-workspace-real-attempt-3-result.json) and the [fresh spectral-template v3 run](evidence/2026-07-11-natural-speech-workspace-spectral-v3-real-result.json) both passed scheduled F1/recall/specificity and zero-clipping gates | A [no-replay same-source extension](evidence/2026-07-11-h3-same-source-forward-model-real-result.json) failed at F1/recall/specificity `0.727/0.667/0.667`; the supported natural result still requires a prior replay of the exact utterance and does not establish generalized self voice, semantics, transfer, or neural dynamics |
| 5 — Synthetic plant infrastructure | Plant supported; bounded-local inverse falsified | Option F passed the deterministic five-gesture plant; Options G and H each preserved `20/20` identities | [Final disposition](evidence/2026-07-11-h4-bounded-local-family-disposition.json) retains the plant as infrastructure and stops inverse-closure work; global/nonlocal or categorical alternatives require a future VISION revision |
| 6 — Real-time efficiency | H5 supported within measured host boundary | Quiet and structured callback gates passed; [lower-volume natural-speech gate](evidence/2026-07-11-stage6-natural-speech-hardware-volume-diagnostic-result.json) passed at 10.34–10.42 ms salient p95; [six-fieldsite timing panel](evidence/2026-07-11-stage6-diverse-natural-speech-result.json) retained 10.17–10.38 ms salient p95 with clean playback intervals; [energy result](evidence/2026-07-11-stage6-energy-result.json) is measured | Two diverse-panel runs clipped only in trailing quiet and failed whole-capture integrity; energy is a 2.47× baseline cost; XMOS internal, wall-plug, acoustic, and cross-device energy remain open |

## H3 same-source generalization failure

The supported exact-efference result stores a compact sensory consequence of
the exact commanded utterance. The next candidate removed that replay
requirement: a 13-scalar forward model learned one centered log-spectral
Mac-mini-speaker-to-XVF3800 path from nine prior commands and applied it to each
new commanded spectrum. Its leave-one-command-out spent-development audit
passed every frozen gate and selected threshold `0.300564`.

The separately preregistered fresh run used three H3-unused self commands and
three H3-unused world sources, with no new-utterance calibration trials. All
nine captures completed with verified alignment, stream position, source
hashes, and zero clipping. The classifier nevertheless failed:

- self-only distances were `0.1403/0.1448/0.3706`;
- overlap distances were `0.2336/0.2218/0.5614`;
- capacity-3 scheduled F1/recall/specificity were
  `0.727/0.667/0.667`, versus flat F1 `0.8` and fixed-window recall `0.5`.

Each overlap was farther from predicted self than its paired self probe, but
between-command offsets destroyed the global separation: minimum overlap minus
maximum self was `-0.1488`. The retained
[post-hoc analysis](evidence/2026-07-11-h3-same-source-forward-model-failure-analysis.json)
does not select a new threshold. Same-data normalization or threshold tuning is
blocked. H3 remains supported only at its earlier exact-efference boundary.

## Stage 2 final disposition

The authorized predictive redesign used every ordered 50 ms prosodic state and
four class-conditional first-order linear-Gaussian dynamics. On 200 already-
spent clips across 18 leave-one-fieldsite-out folds:

- zero-threshold F1 was `0.484`, FPR `0.077`;
- frozen ceiling-constrained selection reached F1 `0.545`, recall `0.466`, FPR
  `0.099`, and song FPR `0.071`;
- minimum fieldsite F1 remained `0.0` in ACO and ARA;
- the 31 false negatives split between speech/song (`16`) and IDS/ADS (`15`),
  so no single gate explains the failure;
- no WEL, USD, or TOR data was newly accessed by this workflow. USD and TOR
  remain pristine; WEL was already spent at fieldsite level by H2.

The [post-hoc failure analysis](evidence/2026-07-11-predictive-register-dynamics-failure-analysis.json)
shows useful score separation but poor cross-field transport. The frozen stop
rule prohibits parameter tuning. A further attempt requires an explicit choice
among accepting the broad Stage 2 falsification, changing the claim to cohort-
adaptive operation, or authorizing a more complex nonlinear sequence model.
The [cross-workflow provenance correction](evidence/2026-07-11-stage2-reserved-fieldsite-provenance-correction.json)
records why Wellington is protected but not fieldsite-pristine.
The [fieldsite-name correction](evidence/2026-07-11-fieldsite-name-correction.json)
records that official corpus code `MEN` means Mentawai Islanders and `MES`
means Colombian Mestizos; earlier prose used incorrect expansions, but no data,
labels, predictions, or metrics changed.

The explicitly authorized cohort-adaptive revision then aligned each eligible
held-out fieldsite's unlabeled transition covariance to training-cohort
geometry, separately for known speech and song batches. ARA was preregistered
ineligible because it contained only one speaker. Across the remaining 198
clips and 17 fieldsites:

- selected F1 improved descriptively to `0.681`, with precision `0.865`, recall
  `0.561`, FPR `0.035`, and zero song false positives;
- ACO remained at F1 `0.0`; MBE, PNG, QUE, and VAN were also below the frozen
  per-fieldsite minimum;
- the 25 false negatives again crossed both score boundaries: 15 failed the
  speech score, 11 failed the IDS score, and one failed both;
- the result therefore failed pooled F1 and minimum-fieldsite gates despite
  excellent false-positive control and `197/198` vocal-front retention.

This is useful evidence that relational cohort context helps, but it does not
rescue reliable cross-cultural register detection. Its stop rule now blocks
same-family tuning and USD/TOR confirmation.

The subsequently authorized nonlinear candidate learned 32 temporal
prototypes from label-blind four-frame contexts with winner-local updates and
used Gaussian radial-basis activations in predictive readouts. Codebook health
was strong in every fold: utilization was `1.0` and quantization error fell.
Despite that, selected F1 was `0.535`, recall `0.466`, FPR `0.113`, and song FPR
`0.083`; ACO and ARA remained at F1 `0.0`. The same-data linear predictive
candidate was slightly better at `0.545`. The nonlinear mechanism is therefore
retained as another confirmation-readiness failure, with USD/TOR untouched.
The [same-data failure comparison](evidence/2026-07-11-nonlinear-predictive-codebook-failure-analysis.json)
shows that nonlinear and linear speech/IDS score rankings correlated about
`0.89`, and their selected decisions agreed on `178/200` clips. The codebook
rescued four linear false negatives but lost four linear true positives and
added two net false positives. It therefore did not reveal a complementary
cross-fieldsite invariant hidden by the linear readout.

The user selected the conservative final disposition: reliable broad cross-
cultural IDS/ADS register detection is falsified for the tested mechanism
families. Active Stage 2 work is stopped. The vocal front gate remains a narrow
environmental-control result, and the cohort result remains descriptive
relational evidence; neither is promoted to general register detection. Deep or
global sequence learning, a narrowed false-positive-control claim, same-family
tuning, and USD/TOR access remain unauthorized.

## H4 continuous inverse-synthesis outcome

The evidence-backed mechanism fork and approved continuous inverse-synthesis
alternative are documented in [H4_MECHANISM_DECISION.md](H4_MECHANISM_DECISION.md).
The candidate was preregistered, implemented at `f174243`, and evaluated once
on already-spent development material.

The current shared fixed-formant route is contradicted in both recorded domains:

- VocalSet held-out confirmation: `9/25 = 0.36`;
- disjoint VocalSet development ladder: maximum `0.56`;
- CMU Arctic ordinary read-speech development: `286/500 = 0.572`, with `/o/`
  recall `0.13` and `/u/` recall `0.30`.

The [continuous result](evidence/2026-07-11-h4-continuous-inverse-synthesis-development-result.json)
then scored `17/20 = 0.85` on clean synthetic trials, `11/25 = 0.44` on
VocalSet, and `287/500 = 0.574` on CMU Arctic. The respective gates were
`0.90`, `0.80`, and `0.80`. Adaptation strictly lowered spectral reconstruction
loss on all `525` recorded examples, but final identity mostly stayed unchanged:
it rescued/lost `1/2` VocalSet decisions and `17/9` CMU decisions. This is the
central H4 result: acoustic reconstruction improvement is not evidence of a
useful shared perception-production identity by itself.
The [post-hoc transition analysis](evidence/2026-07-11-h4-continuous-inverse-synthesis-failure-analysis.json)
records these derived rescue/loss counts without a candidate rerun.

Consuming either reserved confirmation set is not credible. The stop rule
blocks same-data tuning. A supervised phone classifier would materially revise
the analysis-by-synthesis claim; another H4 attempt requires a genuinely new
mechanism and an explicit governance decision.

## H4 geometric articulatory-synthesis outcome

Option D was frozen in a synthetic-only
[preregistration](evidence/2026-07-11-h4-option-d-synthetic-preregistration.json),
implemented at `62c9535`, and evaluated exactly once. Its ten-section physical
plant and 64-prototype winner-local predictor used label-free Halton babbling;
five vowel names entered only post-hoc evaluation.

The forward learner passed Gate 1 on 256 disjoint points. Median sensory
distance was `0.0871` (gate `≤0.20`), and strict win fractions against neutral,
nearest-prototype, and persistence were `0.996/0.953/0.996` (gates `≥0.75`).
This establishes useful local approximation of the frozen synthetic surface.

It did not establish closure. The selected `/u/` plant point was nearest to
`/e/`, failing Gate 0. Gate 2 reached identity `0.80` and combined
acoustic/equivalence agreement `0.60`, below `0.95/0.90`; F1/F2/F3 MAE was
`7/12/56 Hz`, so F3 exceeded `30 Hz`. All `/u/` trials were reperceived as
`/e/`. Every bounds, loss, label-isolation, and search-cost check passed, with a
maximum of 17 forward predictions against the limit of 23. No recorded audio
or protected data was accessed. The frozen stop rule prohibits tuning or a
same-point retry.

## H4 factorized plant outcome

The next plant separated five functional controls and used a frozen design-only
search before opening disjoint Halton indices `8193–9216`. On those untouched
points, P2 expressivity passed every gate: all five target identities and local
neighborhoods were valid, minimum pairwise sensory distance was `0.4017`, and
F1/F2/F3 MAE was `15/95/33 Hz`. The selected `/u/` was
`330/895/2175 Hz`, resolving Option D's `/u/`→`/e/` collapse.

P0 nevertheless required three separated analytic formants for every arbitrary
confirmation state. Nine states did not satisfy that rule, yielding
`1015/1024`; all geometry, finite-transfer, waveform, and determinism checks
passed. The overall plant result is therefore retained as failed, and the
preregistered stop rule blocked babbling and inversion. This is positive
evidence for the factorization's vowel-region expressivity, not permission to
waive the domain-wide gate after inspection.

The post-hoc decomposition found each missing third resonance immediately above
3.5 kHz. A disjoint Option F candidate therefore expanded the structural band
to 5 kHz without changing the plant. It passed all `1024/1024` states, all five
target identities and neighborhoods, waveform integrity, and pairwise
separation; F1/F2/F3 MAE was `15/56/100 Hz`. The result advances the synthetic
plant/readout to learner design, not H4 promotion.

Option G then learned 128 winner-local affine regions from 2,048 label-free
babbling points and evaluated on disjoint forward and inverse sets. It retained
all five post-hoc vowel identities and stayed inside its 23-prediction budget,
but the continuous representation did not close: median held-out sensory error
was `0.264`, combined acoustic/equivalence agreement `0.40`, and F1/F2/F3
reconstruction MAE `16/54/142 Hz`. The frozen failure blocks capacity or search
tuning on these points.

Option H then replaced hard winners with a 64-neighbor Gaussian-weighted local
affine fit over 2,048 disjoint babbling exemplars. It did not improve the
boundary: median error was `0.293`, strict wins over nearest-exemplar `0.479`,
combined agreement `0.40`, and F1/F2/F3 MAE `15/90/163 Hz`. Identity remained
`20/20`, demonstrating again that categorical preservation is weaker than the
continuous H4 closure claim. Another local kernel, density, or search retry is
not credible. A global nonlinear encoder or relaxed claim requires revising
VISION.md.

## Protected data

| Program | Development/spent | Protected and untouched |
|---|---|---|
| Stage 2 Music Lab register | 200 hash-pinned clips across 18 development fieldsites; WEL01–04 speech clips separately spent by H2 | no further `WEL` use; pristine register candidates: `USD`, `TOR` |
| VocalSet H4 retry | `female4`, `female5`, `male6`, `male7`, `male8` | `female6`, `female7`, `female9`, `male9`, `male11` |
| CMU Arctic H4 retry | `bdl`, `slt`, `clb`, `jmk`, `ksp` | confirmation: `rms`, `awb`, `axb`, `eey`, `ljm`; reserve: `aew`, `ahw`, `aup`, `fem`, `gka`, `lnh`, `rxr`, `slp` |

Third-party audio remains outside Git. Checked-in artifacts retain protocols,
selection rules, hashes, aggregate results, and stop decisions.

## Current hardware operating point

- XVF3800 input `:0`, callback capture at 16 kHz, two fused host channels.
- External-world speaker: device 2, 180°, 50 cm, approximately 50% hardware
  knob position. This replaces the prior maximum-volume standing setting.
- Self speaker: Mac mini Speakers centered beneath the XVF3800, 4 cm center-to-
  center; no self output was used in the final natural-speech timing gate.

The lower-volume natural-speech gate completed 1,200 frames with zero clipping,
overflows, device misses, callback-status events, timestamp discontinuities, or
fallbacks. The earlier maximum-volume clipping failure remains part of the
evidence and bounds this operating point.

The later six-fieldsite breadth panel completed 4,800 frames with zero timing or
stream-integrity failures and zero clipping during active playback. KRA and MES
nevertheless contributed `71` full-scale samples during trailing quiet, so the
strict whole-capture gate failed. Four of six runs were completely clean; the
telemetry does not identify the source of the two quiet-interval transients.
The retained [post-hoc localization](evidence/2026-07-11-stage6-diverse-natural-speech-failure-analysis.json)
shows smooth, low-frequency, channel-0-dominant input envelopes seconds after
playback, with no callback or timestamp fault. This rejects a software-stream
corruption explanation but cannot distinguish an unobserved room sound from an
analog/device-front-end transient.

## Verification state

At the 2026-07-12 operational-closure checkpoint the full test suite reports
`484 passed, 3 skipped`. The
skips are intentional hardware/environment-dependent tests, not hidden failures.
The bounded readiness command also validates every required retained stage
artifact against its frozen SHA-256 pin and exits nonzero on a missing,
malformed, replaced, or failed evidence gate.
The read-only GitHub Actions workflow runs that gate before the hardware-safe
suite on every push and pull request; a regression test enforces the order,
permissions, timeout, and absence of live-I/O flags.
The working branch is `codex/vision-evidence`; the latest research checkpoints
include:

- `337f423` — lower-volume natural-speech timing pass;
- `e3b1f68` — predictive register dynamics preregistration;
- `65c5d2a` — predictive dynamics implementation;
- `87636b5` — predictive dynamics falsification;
- `fb46101` — post-hoc failure decomposition;
- `b867722` — cohort-adaptive preregistration;
- `a17fa59` — cohort-adaptive implementation;
- `ec7b4f1` — diverse natural-speech timing preregistration;
- `9bf73c8` — six-stimulus audit implementation;
- `15cf53b` — nonlinear predictive codebook preregistration;
- `b2370ea` — nonlinear predictive codebook implementation.
- `f174243` — frozen continuous inverse-synthesis implementation.
- `937e9cb` — fresh H3 same-source live runner.
- `43ecf5f` — hash-bound no-replay H3 real preregistration.
- `0c08f9a` — exact Option D synthetic preregistration.
- `62c9535` — frozen Option D implementation before candidate metrics.
- `68d7e42` — frozen Option E plant-confirmation preregistration.
- `6ce2dc1` — frozen Option E confirmation implementation before metrics.
- `b476ac7` — frozen Option F expanded-band preregistration.
- `3e97516` — frozen Option F implementation before metrics.
- `e1bcd09` — frozen Option G local-closure preregistration.
- `d6dbc03` — frozen Option G implementation before metrics.
- `22bc25f` — frozen Option H smooth-local preregistration.
- `f9c2136` — frozen Option H implementation before metrics.

## Final governance state

The Stage 2 and H4 decisions are complete. Broad cross-cultural register
detection and the bounded-local H4 representation family are falsified within
their tested scopes; active work on both is stopped. No pristine USD/TOR,
protected H4 confirmation data, or additional WEL material should be fetched.
Deep/global replacements, narrowed claims, and same-family retries require a
future explicit VISION revision.
