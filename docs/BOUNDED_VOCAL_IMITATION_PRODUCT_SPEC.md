# Bounded vocal-imitation product specification

Status: primary product and engineering direction, authorized 2026-07-15.

## 1. Product purpose

Cora-Infans’ primary product direction is an **authorized, bounded vocal-imitation session** for a known project owner. The system receives a project-owner vocal target through the declared microphone path, selects and recalls a response from a fixed finite motor-prototype repertoire without consuming the target's display label, and produces an auditable session report.

The product is grounded in the retained V4 project-owner result, but V4 is immutable evidence—not a reusable training set, a default model, or an open-ended product session.

## 2. Product claim

A successful product session may claim only that, under its recorded authorization, device, room, repertoire, and protocol:

- an authorized target was captured through the declared sensory path;
- a finite, disclosed repertoire supported label-blind target-specific association and recall;
- associations froze before held evaluation;
- each held selection beat the frozen rest and off-target associations under preregistered contrastive gates;
- the final report identifies source, response, scoring, and provenance artifacts; and
- each result remains replayable while its encrypted audio is retained, and its
  integrity remains attestable after deletion from externally anchored hashes.

A pass does **not** claim phoneme recognition, arbitrary speech imitation, speaker generalization, linguistic understanding, infant learning, anatomical fidelity, continuous acoustic-to-articulatory inversion, or H4 perception-production closure.

## 3. Inherited non-negotiable boundaries

- Stage 2 broad register-detection research remains stopped.
- H4 continuous closure remains falsified for the tested bounded-local families; this product direction does not reopen it.
- The V4 result and prior stopped candidates remain retained evidence. Future sessions must use a separately versioned protocol and fresh recordings.
- Protected Stage 2 fieldsites, VocalSet speakers, CMU Arctic confirmation speakers, and CMU Arctic reserve speakers remain inaccessible.
- Raw owner recordings and generated WAVs remain outside Git. Git may retain only manifests, hashes, aggregate metrics, protocol versions, and non-identifying reports.
- Microphone or speaker use requires an explicit authorization flag and must fail loudly if device, capture-quality, provenance, or operating-point requirements are missing.

## 4. Product-session flow

A future product session has five distinct states:

1. **Prepare** — validate the versioned protocol, explicit authorization, device route, operating-point metadata, and empty/new session directory.
2. **Develop** — capture only the preregistered development targets and derive target-specific associations from the fixed repertoire.
3. **Freeze** — write and hash the associations, acoustic centroids, label-blind selection rule, contrastive gates, and repertoire identity. No held evaluation capture may be opened before this succeeds.
4. **Evaluate** — for each held trial, capture the target, select one frozen association without label metadata, commit the emitted response and self prediction before playback, capture the response, and score the completed pair. Evaluation may not train, select a new prototype, or replace an association.
5. **Present** — emit a replayable report/dashboard that labels its evidence class and contains no silent synthetic fallback.

A missing prepare gate before I/O produces `blocked_pre_io`; a preregistered
machine-integrity fault after I/O produces `technical_invalid`; and any other
missed quality, provenance, association-freeze, label-blind selection,
contrastive, prediction-commitment, artifact-attestation, or score gate produces
`behavioral_failed`. None may be reworded as qualitative success.

The initial offline-only report contract is frozen in
[`2026-07-15-bounded-vocal-imitation-product-session-preregistration.json`](evidence/2026-07-15-bounded-vocal-imitation-product-session-preregistration.json).
It authorizes metadata validation only; it does not authorize live I/O, scoring,
association selection, or a new recording.

## 5. Required session artifacts

Each evaluated trial must identify the following artifacts, whether by local path plus SHA-256 or by a retained private manifest:

```text
source_human.wav        # owner-source provenance; may alias target_captured for live owner capture
target_captured.wav     # external target through Cora's microphone path
agent_emitted.wav       # selected finite-repertoire response
response_captured.wav   # agent response through Cora's microphone path
self_prediction.wav     # predicted microphone consequence, committed before response capture
report.json             # immutable metric, provenance, and verdict record
```

Audio artifacts must carry their SHA-256 in the report. `report.json` names its
own path but cannot honestly contain its final self-hash; a sibling
`session-manifest.json` attests the completed report hash. A real completed
session must then anchor the session-manifest hash in a public-safe
`session-attestation.json` backed by a Git commit or owner signature. A local
manifest that can be rewritten together with the report is integrity metadata,
not an immutable attestation.

A synthetic/replay fixture may substitute deterministic synthetic artifacts, but must use `evidence_class: synthetic_fixture`. It may never be displayed as live or real-authorized evidence.

## 6. Score separation

Product reports must keep two score families separate:

- **Imitation fidelity** compares the microphone-captured external target with the microphone-captured agent response. It is the primary user-facing outcome.
- **Self-model fidelity** compares the predicted microphone consequence of the selected agent response with the actual captured agent response. It measures self/efference-model calibration, not imitation success.

Both reports may include active vocal-event duration difference, RMS-envelope
distance, voiced pitch-contour distance, and mel/cepstral distance. Absolute
capture-window onset/offset is excluded because target and response recordings
have independent time origins. Raw waveform MSE is diagnostic only because
phase and timing shifts make it unsuitable as the principal perceptual score.

### Offline replay scorers

The retained `offline-replay-v1` contract is historical and remains described
by its frozen preregistration. It compared whole-file duration and absolute
capture-window boundaries, so it must not gate fixed-window real sessions.

The currently implemented `offline-replay-v2` scorer is deterministic and
reads pre-existing WAV files only. It detects each active vocal interval,
removes independent capture-window offsets, and emits the following
unaggregated component distances in **both** score families:

- `active_duration_difference_ms` — absolute difference between detected vocal-event durations after resampling to 16 kHz;
- `rms_envelope_distance` — RMS distance between time-normalized active-interval 20 ms energy envelopes;
- `pitch_contour_distance_hz` — mean absolute difference between time-normalized voiced autocorrelation pitch contours; and
- `cepstral_distance` — Euclidean distance between active-interval mean log-spectrum cepstral vectors.

These are comparison signals, not product thresholds, perceptual quality guarantees, phoneme/word recognition, authorization evidence, or a substitute for a separately frozen real-session protocol. The scorer requires an explicit association-freeze attestation, writes `report.json` and a sibling `session-manifest.json`, and performs no capture, playback, device discovery, association selection, training, or V4 mutation.

The historical V1 scorer preregistration is
[`2026-07-15-bounded-vocal-imitation-offline-replay-scorer-preregistration.json`](evidence/2026-07-15-bounded-vocal-imitation-offline-replay-scorer-preregistration.json).
The corrected V2 scorer preregistration is
[`2026-07-15-bounded-vocal-imitation-offline-replay-scorer-v2-preregistration.json`](evidence/2026-07-15-bounded-vocal-imitation-offline-replay-scorer-v2-preregistration.json).

The precise thresholds, aggregation rule, repeat count, and scoring implementation for a new product-session version must be frozen before its first capture. V4’s scientific thresholds are historical facts; they are not automatically product thresholds.

### Product-level repeatability cohort

A first product-level repeatability claim uses one preregistered cohort of five
ordered session slots under an unchanged protocol, repertoire, scorer, and
threshold set. All five slots must be resolved and reported; at least three
must be fully `passed`. Passes need not be consecutive, and a later non-pass
does not erase an earlier pass. No session outside the five frozen slots may be
substituted or added to reach the threshold.

A preflight that blocks before live I/O is reported as `blocked_pre_io` and does
not consume its session slot. Once a session accepts its first audio sample, the
slot is consumed. A machine-detected transport or hardware integrity fault is
`technical_invalid`: it cannot pass and counts against the five-slot cohort,
but it is not evidence of behavioral failure. Technical invalidity must be
derived from preregistered machine telemetry and committed before waveform
scores are opened; it cannot be assigned manually after seeing a poor result. A
technically valid session that misses a quality, selection, contrastive,
commitment, or score gate is `behavioral_failed`.

The cohort report keeps behavioral repeatability (`passed` divided by
technically valid started sessions) separate from operational reliability
(technically valid divided by all started session slots), and reports all
pre-I/O blocks separately. One passed session remains a measured pilot. A
failed five-slot cohort is retained evidence; collecting more sessions under
the same version cannot rescue the first product-level claim.

## 7. Evidence classes and presentation

Every product-facing artifact must visibly state one of:

- `retained_v4_evidence` — the completed, immutable 2026-07-13 V4 result;
- `synthetic_fixture` — deterministic presentation or test material only; or
- `real_authorized_session` — a separately preregistered and explicitly authorized product session.

The hero demo may summarize retained V4 evidence and render synthetic/replay fixtures. It must not independently capture audio, train associations, or imply that its replay path is live hardware evidence.

### Product-session evidence presentation

`--hero-demo --source product-session --session-dir DIR --mode replay` is a presentation-only path for a **completed** product-session directory. Before exposing any audio control, it validates `report.json`, verifies its final hash against `session-manifest.json`, and rechecks every audio artifact hash recorded in the report. It preserves the session’s evidence class and presents imitation fidelity and self-model fidelity separately with unaggregated component metrics and provenance rows.

This source is replay-only. It rejects live mode, performs no recording, playback, device discovery, association selection, or training, and does not create a new product-session claim. The presenter copies only the four display artifacts (captured target, agent-emitted response, captured response, and self prediction) after hash verification; source-human audio remains provenance-only. Its frozen presentation boundary is recorded in [`2026-07-15-bounded-vocal-imitation-product-session-evidence-presentation-preregistration.json`](evidence/2026-07-15-bounded-vocal-imitation-product-session-evidence-presentation-preregistration.json).

## 8. Path to the next real product session

Before any new project-owner recording, create and approve a separate versioned session protocol that freezes:

- owner authorization and retention/deletion decision;
- device route, physical operating point, capture prompt, and take counts;
- development and held-evaluation roles;
- fixed repertoire, label-blind association lookup, rejection/ambiguity rules,
  contrastive gates, and association-freeze point;
- metric version, thresholds, aggregation, and stop rules;
- artifact directory/manifest schema; and
- prediction precommitment, external manifest attestation, deletion enforcement,
  failure-status taxonomy, and a bounded repeatability cohort across fresh
  sessions.

That protocol is an operational product experiment. It is not permission to reuse V4 inputs, tune V4 after the fact, access protected data, or broaden Stage 2/H4 claims.

The owner-approved, non-executable V1 protocol design is available in
[`BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V1.md`](BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V1.md),
with its matching machine-readable record in
[`2026-07-15-bounded-vocal-imitation-session-v1-preregistration.json`](evidence/2026-07-15-bounded-vocal-imitation-session-v1-preregistration.json).
The hash-pinned owner approval is retained in
[`2026-07-16-bounded-vocal-imitation-session-v1-owner-approval.json`](evidence/2026-07-16-bounded-vocal-imitation-session-v1-owner-approval.json).
Its status is `owner_approved_not_executable`: protocol and retention approval
do not permit recording, playback, device discovery, or a new product claim.
Execution still requires the separate threshold calibration, reviewed runtime
support, five-slot cohort registration, preflight, and fresh per-session
live-I/O authorization.
