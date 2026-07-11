# Project status — 2026-07-11

This document consolidates the current evidence against [VISION.md](../VISION.md).
It does not supersede preregistrations or promote a failed gate. When a summary
and a retained artifact disagree, the retained artifact and its frozen protocol
are authoritative.

For acronyms, corpus codes, metrics, and common report fields, see the
[data dictionary](DATA_DICTIONARY.md).

## Executive outcome

- Stages 0 and 1 are implemented and supported by reproducible real-audio
  evaluation.
- Stage 2 is mixed and not promoted. The vocal front gate has bounded positive
  evidence, but four families of cross-cultural register models failed to
  generalize reliably.
- H2, H3, and H5 are supported only inside their measured boundaries.
- H4 remains open as a broad hypothesis, but both tested mechanism families are
  falsified at their frozen gates. The fixed-formant route failed on recorded
  voices, and the approved continuous inverse-synthesis candidate reduced
  reconstruction loss without reaching reliable vowel identity.
- No new reserved Stage 2 or H4 confirmation data was consumed after its
  applicable development stop rule fired. Wellington had already been consumed
  for a separate H2 target and is not fieldsite-pristine for Stage 2.

## Evidence matrix

| Stage | Current verdict | Strongest supporting evidence | Boundary or contradiction |
|---|---|---|---|
| 0 — Reproducible substrate | Implemented | Deterministic unit/integration suite; explicit synthetic/live contracts; fail-loud real I/O; hash-pinned external-data runners | Passing plumbing is not empirical promotion |
| 1 — Real-audio evaluation | Passed in bounded corpora | [prior-art replay](evidence/2026-07-10-prior-art-replay.json), Music Lab PCM/float WAV evaluation, participant/fieldsite manifests | Broader devices, rooms, ages, and spontaneous conditions remain open |
| 2 — Parentese-relevant saliency | Mixed; not promoted | [narrow Beijing confirmation](evidence/2026-07-10-ids-confirmation.json), [Mentawai/Toposa mode-scoped pass](evidence/2026-07-10-mode-scoped-register-result.json), [vocal front-gate pass](evidence/2026-07-10-yin-vocal-gate-v2-result.json) | [final paired-summary failure](evidence/2026-07-10-expanded-register-v6-result.json), [six-phase failure](evidence/2026-07-11-temporal-register-development-result.json), [predictive-dynamics failure](evidence/2026-07-11-predictive-register-dynamics-result.json), [cohort-adaptive failure](evidence/2026-07-11-cohort-adaptive-register-result.json), and [nonlinear-codebook failure](evidence/2026-07-11-nonlinear-predictive-codebook-result.json) block general promotion |
| 3 — Predictive hierarchy | H2 supported within tested boundaries | Natural transfer replicated on Wellington and MEN; controlled real known-efference dropout retained four states and improved tone/vowel prediction | Unknown dropout timing, arbitrary missing world input, semantics, and room transfer remain outside the claim |
| 4 — Workspace binding | H3 supported within structured-tone and exact-efference speech boundaries | [structured attempt 3](evidence/2026-07-10-structured-workspace-real-attempt-3-result.json) and the [fresh spectral-template v3 run](evidence/2026-07-11-natural-speech-workspace-spectral-v3-real-result.json) both passed scheduled F1/recall/specificity and zero-clipping gates | Natural result requires one prior replay of the exact utterance, three fresh pairs, fixed geometry/levels, and one room; not generalized self voice, semantics, transfer, or neural dynamics |
| 5 — Perception-production closure | H4 open; fixed and continuous candidates falsified | The continuous candidate strictly reduced reconstruction loss on every recorded example while respecting all bounds and its 23-synthesis budget | [continuous inverse-synthesis result](evidence/2026-07-11-h4-continuous-inverse-synthesis-development-result.json) scored synthetic `0.85`, VocalSet `0.44`, and CMU Arctic `0.574`, failing `0.90/0.80/0.80`; protected speakers remain untouched |
| 6 — Real-time efficiency | H5 supported within measured host boundary | Quiet and structured callback gates passed; [lower-volume natural-speech gate](evidence/2026-07-11-stage6-natural-speech-hardware-volume-diagnostic-result.json) passed at 10.34–10.42 ms salient p95; [six-fieldsite timing panel](evidence/2026-07-11-stage6-diverse-natural-speech-result.json) retained 10.17–10.38 ms salient p95 with clean playback intervals; [energy result](evidence/2026-07-11-stage6-energy-result.json) is measured | Two diverse-panel runs clipped only in trailing quiet and failed whole-capture integrity; energy is a 2.47× baseline cost; XMOS internal, wall-plug, acoustic, and cross-device energy remain open |

## Stage 2 disposition pending

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

At this checkpoint the full test suite reports `407 passed, 3 skipped`. The
skips are intentional hardware/environment-dependent tests, not hidden failures.
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

## Decisions required before further Stage 2 or H4 empirical work

1. Decide whether Stage 2 should now record broad cross-cultural register
   detection as falsified or authorize a deep/global mechanism outside the
   current local-learning contract.
2. Decide whether H4 should stop active work after two falsified mechanism
   families or authorize a genuinely different representation. Same-data
   tuning of the continuous candidate is not an option.

Until those decisions are made, no pristine USD/TOR or additional WEL
confirmation data should be fetched and no same-family threshold,
representation, or hyperparameter retry should be run.
