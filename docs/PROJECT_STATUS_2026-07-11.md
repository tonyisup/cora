# Project status — 2026-07-11

This document consolidates the current evidence against [VISION.md](../VISION.md).
It does not supersede preregistrations or promote a failed gate. When a summary
and a retained artifact disagree, the retained artifact and its frozen protocol
are authoritative.

## Executive outcome

- Stages 0 and 1 are implemented and supported by reproducible real-audio
  evaluation.
- Stage 2 is mixed and not promoted. The vocal front gate has bounded positive
  evidence, but four families of cross-cultural register models failed to
  generalize reliably.
- H2, H3, and H5 are supported only inside their measured boundaries.
- H4 remains open. The current fixed-formant perception-production route failed
  on held-out recorded singers and on two disjoint development domains.
- No new reserved Stage 2 or H4 confirmation data was consumed after its
  applicable development stop rule fired. Wellington had already been consumed
  for a separate H2 target and is not fieldsite-pristine for Stage 2.

## Evidence matrix

| Stage | Current verdict | Strongest supporting evidence | Boundary or contradiction |
|---|---|---|---|
| 0 — Reproducible substrate | Implemented | Deterministic unit/integration suite; explicit synthetic/live contracts; fail-loud real I/O; hash-pinned external-data runners | Passing plumbing is not empirical promotion |
| 1 — Real-audio evaluation | Passed in bounded corpora | [prior-art replay](evidence/2026-07-10-prior-art-replay.json), Music Lab PCM/float WAV evaluation, participant/fieldsite manifests | Broader devices, rooms, ages, and spontaneous conditions remain open |
| 2 — Parentese-relevant saliency | Mixed; not promoted | [narrow Beijing confirmation](evidence/2026-07-10-ids-confirmation.json), [Menominee/Toposa mode-scoped pass](evidence/2026-07-10-mode-scoped-register-result.json), [vocal front-gate pass](evidence/2026-07-10-yin-vocal-gate-v2-result.json) | [final paired-summary failure](evidence/2026-07-10-expanded-register-v6-result.json), [six-phase failure](evidence/2026-07-11-temporal-register-development-result.json), [predictive-dynamics failure](evidence/2026-07-11-predictive-register-dynamics-result.json), and [cohort-adaptive failure](evidence/2026-07-11-cohort-adaptive-register-result.json) block general promotion |
| 3 — Predictive hierarchy | H2 supported within tested boundaries | Natural transfer replicated on Wellington and MEN; controlled real known-efference dropout retained four states and improved tone/vowel prediction | Unknown dropout timing, arbitrary missing world input, semantics, and room transfer remain outside the claim |
| 4 — Workspace binding | H3 supported within structured boundary | [real synchronized attempt 3](evidence/2026-07-10-structured-workspace-real-attempt-3-result.json) passed scheduled F1/recall/specificity and zero-clipping gates | Structured tones, fixed geometry, one room; not natural speech or measured neural theta/gamma dynamics |
| 5 — Perception-production closure | H4 open; current route falsified | Clean synthesis remained perfect | [held-out VocalSet result](evidence/2026-07-11-vocalset-recorded-vowel-result.json) scored 0.36; sung development peaked at 0.56; [CMU Arctic development](evidence/2026-07-11-cmu-arctic-development-result.json) peaked at 0.572 |
| 6 — Real-time efficiency | H5 supported within measured host boundary | Quiet and structured callback gates passed; [lower-volume natural-speech gate](evidence/2026-07-11-stage6-natural-speech-hardware-volume-diagnostic-result.json) passed at 10.34–10.42 ms salient p95 with zero clipping; [energy result](evidence/2026-07-11-stage6-energy-result.json) is measured | Energy is a 2.47× cost versus the conventional baseline, not a win; natural-speech evidence is one utterance at an uncalibrated knob setting; XMOS internal, wall-plug, acoustic, and cross-device energy remain open |

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

## H4 disposition pending

The current shared fixed-formant route is contradicted in both recorded domains:

- VocalSet held-out confirmation: `9/25 = 0.36`;
- disjoint VocalSet development ladder: maximum `0.56`;
- CMU Arctic ordinary read-speech development: `286/500 = 0.572`, with `/o/`
  recall `0.13` and `/u/` recall `0.30`.

Consuming either reserved confirmation set would not be credible. Continuing
with a supervised phone classifier or changing the frozen vowel mapping after
seeing these errors would materially revise the analysis-by-synthesis claim.

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

## Verification state

At this checkpoint the full test suite reports `375 passed, 3 skipped`. The
skips are intentional hardware/environment-dependent tests, not hidden failures.
The working branch is `codex/vision-evidence`; the latest research checkpoints
include:

- `337f423` — lower-volume natural-speech timing pass;
- `e3b1f68` — predictive register dynamics preregistration;
- `65c5d2a` — predictive dynamics implementation;
- `87636b5` — predictive dynamics falsification;
- `fb46101` — post-hoc failure decomposition;
- `b867722` — cohort-adaptive preregistration;
- `a17fa59` — cohort-adaptive implementation.

## Decisions required before further empirical work

1. Decide whether Stage 2 should now record broad cross-cultural register
   detection as falsified or authorize a materially different mechanism.
2. Decide whether H4 should record the current shared fixed-formant route as
   falsified or revise the perception-production hypothesis.

Until those decisions are made, no pristine USD/TOR or additional WEL
confirmation data should be fetched and no same-family threshold,
representation, or hyperparameter retry should be run.
