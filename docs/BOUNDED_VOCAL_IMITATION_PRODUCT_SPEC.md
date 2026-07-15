# Bounded vocal-imitation product specification

Status: primary product and engineering direction, authorized 2026-07-15.

## 1. Product purpose

Cora-Infans’ primary product direction is an **authorized, bounded vocal-imitation session** for a known project owner. The system receives a project-owner vocal target through the declared microphone path, selects and recalls a response from a fixed finite motor-prototype repertoire, and produces an auditable session report.

The product is grounded in the retained V4 project-owner result, but V4 is immutable evidence—not a reusable training set, a default model, or an open-ended product session.

## 2. Product claim

A successful product session may claim only that, under its recorded authorization, device, room, repertoire, and protocol:

- an authorized target was captured through the declared sensory path;
- a finite, disclosed repertoire supported target-specific association and recall;
- associations froze before held evaluation;
- the final report identifies source, response, scoring, and provenance artifacts; and
- each result remains reproducible from its retained manifest and hashes.

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
2. **Develop** — capture only the preregistered development takes and derive target-specific associations from the fixed repertoire.
3. **Freeze** — write and hash the association artifact. No held evaluation capture may be opened before this succeeds.
4. **Evaluate** — score held target material only against the already-frozen association. Evaluation may not select or replace a prototype.
5. **Present** — emit a replayable report/dashboard that labels its evidence class and contains no silent synthetic fallback.

A missed quality, provenance, association-freeze, or score gate produces `blocked` or `failed`. It must not be reworded as qualitative success.

The initial offline-only report contract is frozen in
[`2026-07-15-bounded-vocal-imitation-product-session-preregistration.json`](evidence/2026-07-15-bounded-vocal-imitation-product-session-preregistration.json).
It authorizes metadata validation only; it does not authorize live I/O, scoring,
association selection, or a new recording.

## 5. Required session artifacts

Each evaluated trial must identify the following artifacts, whether by local path plus SHA-256 or by a retained private manifest:

```text
source_human.wav        # owner-source provenance when applicable
target_captured.wav     # external target through Cora's microphone path
agent_emitted.wav       # selected finite-repertoire response
response_captured.wav   # agent response through Cora's microphone path
self_prediction.wav     # predicted microphone consequence, if available
report.json             # immutable metric, provenance, and verdict record
```

Audio artifacts must carry their SHA-256 in the report. `report.json` names its own path but cannot honestly contain its final self-hash; a sibling immutable session manifest must attest the completed report hash when one is needed.

A synthetic/replay fixture may substitute deterministic synthetic artifacts, but must use `evidence_class: synthetic_fixture`. It may never be displayed as live or real-authorized evidence.

## 6. Score separation

Product reports must keep two score families separate:

- **Imitation fidelity** compares the microphone-captured external target with the microphone-captured agent response. It is the primary user-facing outcome.
- **Self-model fidelity** compares the predicted microphone consequence of the selected agent response with the actual captured agent response. It measures self/efference-model calibration, not imitation success.

Both reports may include onset/offset alignment, duration difference, RMS-envelope distance, voiced pitch-contour distance, and mel/cepstral distance. Raw waveform MSE is diagnostic only because phase and timing shifts make it unsuitable as the principal perceptual score.

### Offline replay scorer v1

The currently implemented `offline-replay-v1` scorer is deterministic and reads pre-existing WAV files only. It emits the following unaggregated component distances in **both** score families:

- `onset_offset_alignment_ms` — mean absolute onset/offset discrepancy measured on 20 ms RMS-envelope frames;
- `duration_difference_ms` — absolute duration difference after resampling to 16 kHz;
- `rms_envelope_distance` — RMS distance between time-normalized 20 ms energy envelopes;
- `pitch_contour_distance_hz` — mean absolute difference between time-normalized voiced autocorrelation pitch contours; and
- `cepstral_distance` — Euclidean distance between mean log-spectrum cepstral vectors.

These are comparison signals, not product thresholds, perceptual quality guarantees, phoneme/word recognition, authorization evidence, or a substitute for a separately frozen real-session protocol. The scorer requires an explicit association-freeze attestation, writes `report.json` and a sibling `session-manifest.json`, and performs no capture, playback, device discovery, association selection, training, or V4 mutation.

The frozen scorer preregistration is [`2026-07-15-bounded-vocal-imitation-offline-replay-scorer-preregistration.json`](evidence/2026-07-15-bounded-vocal-imitation-offline-replay-scorer-preregistration.json).

The precise thresholds, aggregation rule, repeat count, and scoring implementation for a new product-session version must be frozen before its first capture. V4’s scientific thresholds are historical facts; they are not automatically product thresholds.

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
- fixed repertoire and association-freeze point;
- metric version, thresholds, aggregation, and stop rules;
- artifact directory/manifest schema; and
- repeatability requirements across fresh sessions.

That protocol is an operational product experiment. It is not permission to reuse V4 inputs, tune V4 after the fact, access protected data, or broaden Stage 2/H4 claims.

A proposed, non-authorizing V1 draft is available in
[`BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V1.md`](BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V1.md),
with its matching machine-readable record in
[`2026-07-15-bounded-vocal-imitation-session-v1-preregistration.json`](evidence/2026-07-15-bounded-vocal-imitation-session-v1-preregistration.json).
It remains `draft_not_authorized`: it cannot permit recording, playback, device
discovery, or a new product claim until the owner approves it and the required
runtime support exists.
