# Bounded vocal-imitation product-session protocol V1

Status: OWNER-APPROVED DESIGN — NOT EXECUTABLE; RECORDING AND LIVE I/O REMAIN UNAUTHORIZED.

## 1. Purpose and authority

This is the owner-approved protocol design for a **future** bounded
project-owner vocal-imitation product session. Owner approval was recorded on
2026-07-16. It freezes the intended operational values, but it is not a session
authorization record, not an instruction to open a microphone or speaker, and
not a continuation or modification of the completed V4 experiment.

Owner approval of the design and retention decision is complete. Execution
still requires all of the following later events:

1. a separate hashed calibration record freezes and passes the proposed scorer
   and selection thresholds without using V4 or future session audio;
2. a later implementation task supports this protocol's label-blind held-target
   selection, contrastive gates, two held takes per target, prediction
   commitment, retention enforcement, cohort ledger, and required
   manifest/report fields, followed by implementation review and tests;
3. the five ordered session-slot identifiers are registered before the first
   live I/O;
4. a later preflight verifies the named physical route without silently falling
   back to another device; and
5. the owner gives a fresh, one-session live-I/O authorization at execution
   time.

Until then, no capture, playback, device discovery, repertoire selection, or
association training is authorized. V4 is immutable retained evidence. Stage 2
and H4 remain stopped; this protocol does not reopen either program.

## 2. Narrow proposed claim and non-claims

If executed and all gates pass, a future result may claim only repeatable,
label-blind target-specific association and recall for one explicitly
authorized owner, three declared vowel targets, one declared room/device
operating point, and one fixed finite repertoire.

It must not claim a perceptual-quality guarantee, phoneme recognition,
generalized imitation, speaker identity, arbitrary speech reproduction,
linguistic understanding, infant learning, anatomical recovery, or continuous
perception-production/H4 closure. A passing score is not a perceptual-quality
guarantee.

## 3. Owner authorization, privacy, and retention

Before any live call, the owner must approve a session authorization record that
contains no credential or raw-audio content and states:

- the protocol version, session identifier, date, target set, and stated narrow
  claim;
- that the owner is the sole speaker and may stop the session at any time;
- that raw owner and generated WAV artifacts remain outside Git;
- an owner-controlled, encrypted local retention location recorded in reports
  only as a non-sensitive storage alias, with a per-session encryption key and
  either backup exclusion or an explicit inventory of backup copies; and
- that raw and generated WAV artifacts, temporary PCM, copied presentation
  WAVs, backup copies, and the per-session key are deleted **30 calendar days**
  after a completed session report, unless the owner explicitly extends
  retention before that deadline.

Deletion uses cryptographic erasure of the per-session key plus best-effort
removal of encrypted files and inventoried copies. A public-safe deletion
attestation records deadline, completion time, method, scope, and result. The
retained private manifest may keep source paths and SHA-256 values. Git may
retain only this protocol, the authorization-free preregistration, manifest
schema, externally anchored hashes, aggregate metrics, non-identifying reports,
and deletion attestation. Deletion does not erase a failure record: its
non-audio report and hashes remain retained, but it can no longer be replayed
or rescored.

## 4. Proposed physical operating point

This operating point is a proposed, fail-closed route—not a claim that the
hardware is currently available or calibrated.

| Surface | Proposed value | Required recording/report field |
| --- | --- | --- |
| Host | Mac mini | host model/OS/build identifier |
| Input | XVF3800 selector `:0`, callback capture, 16 kHz, two fused host channels | device selector, backend, sample rate, channel count |
| Target capture | no simultaneous stimulus playback | input metadata and capture-quality result |
| Response playback | Mac mini Speakers centered beneath the XVF3800, 4 cm center-to-center | output device identity, geometry, system volume, digital peak |
| Response level | system volume 100%; output PCM peak no greater than 0.10 full scale | measured configured values, not inferred defaults |
| Start cue | 100 ms, 660 Hz, amplitude 0.08, then 250 ms settle; cue/settle excluded from retained target audio | cue hash, route, and exclusion confirmation |
| Retained capture window | 5.0 seconds per take | duration, frame count, start/end timestamps |

A new latency/route preflight is required before every response-capture block
and after any interruption that may change speaker, microphone, room, mute, or
volume state. If the preflight fails, the session is `blocked`; no substitute
route or synthetic fallback is permitted.

## 5. Frozen target set and capture schedule

The proposed target set is `/a/`, `/i/`, and `/u/`. The owner produces a
comfortable, steady vowel for approximately two seconds after the cue. Target
names organize capture and display only; selection code must not consume them
as learner features.

Each fresh session has twelve target captures and six response captures. A
development take captures only the target. A held trial captures a target and,
after label-blind selection and prediction commitment, the paired response:

| Phase | Target captures per target | Response captures per target | Total microphone captures | Permission |
| --- | ---: | ---: | ---: | --- |
| Development | 2 | 0 | 6 | allowed after authorization and preflight |
| Held evaluation | 2 | 2 | 12 | prohibited until development freeze verifies |

Each microphone capture is a 5.0-second retained window. Captures are ordered by
a pre-registered schedule and never selected after listening to results. Before
opening I/O, the runner appends an attempt record with session, trial, role, and
monotonic attempt number. Exactly one `technical_abort` is allowed per planned
trial only when no audio sample was accepted and no partial WAV was created. If
any sample was accepted, a partial artifact exists, or a quality/selection/score
gate fails, the attempt consumes the trial and no retry is permitted. Attempt
records are append-only; deleting a partial file cannot convert a consumed
attempt into a technical abort.

At the session level, a preflight refusal before live I/O is
`blocked_pre_io` and does not consume the preregistered cohort slot. Once the
first audio sample is accepted, that slot is consumed. A machine-detected
buffer underflow/overflow, callback discontinuity, route loss, truncated or
corrupt capture, or equivalent transport-integrity fault is
`technical_invalid`. It ends the session and prevents a pass, but it must not be
reported as evidence that the imitation behavior failed. The runner derives and
commits this status from preregistered machine telemetry before opening any
waveform score; an operator cannot assign it after seeing a poor behavioral
result. A technically valid session that misses any quality, selection,
contrastive, commitment, or score gate is `behavioral_failed`.

## 6. Fixed repertoire and association freeze

The proposed repertoire is the existing finite 96-entry Halton schedule (start
`20011`, stride `43`, last `24096`) with its exact repertoire digest recorded
before development capture. This is a disclosed fixed bank, not a new
continuous controller or a new H4 mechanism.

The target representation is exactly the label-blind 30-dimensional auditory
path frozen in Section 3 of `VOCAL_IMITATION_PROTOCOL.md`: decoded/downmixed
16 kHz audio, highest-RMS 480 ms voiced window, averaged Hann-window power,
30 frozen log-spaced Gaussian bands, log energy, mean removal, and unit
normalization. The implementation and its digest must be frozen before capture.

After exactly six quality-passing development captures:

1. derive one acoustic centroid and one repertoire association per target from
   development material only;
2. write a hash-pinned `frozen-vocal-associations.json` containing the
   development capture hashes, centroid vector hashes, repertoire digest,
   selected prototype IDs, representation/scoring versions, selection and
   contrastive thresholds, and protocol version;
3. require three distinct selected prototype IDs; then
4. verify that no centroid or association can be selected or replaced
   afterward; then
5. permit the six held target captures.

For each held target, selection code receives only the target WAV and the three
frozen opaque centroids. It computes the frozen auditory vector and selects the
unique nearest centroid in deterministic opaque-ID order. Selection fails if
the nearest distance exceeds `0.30`, if the runner-up margin is below `0.05`,
or if any distance tie is within `1e-12`. The runner writes and hashes
`selection-commitment.json` before the scheduled display label becomes
available to verdict code. Only then may the verdict compare the committed
opaque association with the scheduled target.

The held target must also be at least `15%` closer to its selected prototype
than to the frozen rest prototype and at least `0.05` closer than to every
off-target frozen prototype in the same 30-dimensional space. These are
contrastive gates, not user-facing waveform-fidelity scores.

Any missing/changed development artifact, repertoire mismatch, or failed
freeze blocks held capture. Held material may select and score only among the
already-frozen associations.

## 7. Capture-quality gates

Every target and response capture must satisfy the current bounded quality
checks after decode/downmix/resample:

- finite samples;
- no full-scale sample;
- selected voiced-window RMS at least `0.01`;
- selected voiced-window peak at least `0.03` and below `0.98`;
- autocorrelation pitch in `70–350 Hz`; and
- normalized periodicity at least `0.30`.

For response capture additionally require verified full-span overlap: capture
ready → 250 ms pre-roll → playback of the exact emitted WAV → capture remains
active through playback completion. Playback onset must be anchored in capture
stream-frame coordinates. A wall-clock-only anchor is insufficient for a
product claim.

Before response capture opens, the runner must write and hash the exact
`agent_emitted.wav` and a `self_prediction.wav` generated only from the frozen
self model and emitted response. `prediction-commitment.json` records both
hashes, the self-model/version digest, inputs, and completion time. Neither file
may depend on `response_captured.wav`; any post-response creation or mutation is
a failed trial.

## 8. Proposed scoring gates

Use `offline-replay-v2` exactly as documented in the product specification.
The following values are **proposed engineering acceptance limits**, frozen
before capture if this draft is approved:

| Component | Proposed maximum |
| --- | ---: |
| Active vocal-event duration difference | 150 ms |
| RMS-envelope distance | 0.05 |
| Voiced pitch-contour distance | 30 Hz |
| Cepstral distance | 0.50 |

Every component must pass for **each** held take. An unvoiced/invalid component
is a failure, not a zero, a skipped cell, or a substitute aggregate. The report
shows held waveform scores separately from development association metrics and
the required rest/off-target contrastive gates. No opaque weighted composite
may override a failed component.

These values remain provisional until a separate, hashed calibration record
shows deterministic pass/fail behavior on untouched synthetic timing, energy,
pitch, spectral, silence-offset, and invalid-voicing fixtures. That calibration
set may not contain V4 or future session audio. The limits are not validated
perceptual thresholds and must not be marketed as such. They cannot be changed
after any capture in a session. Any future threshold revision requires a new
protocol version, fresh sessions, and a new approval.

## 9. Session and claim gates

A single session is `passed` only if all twelve target captures and six response
captures, authorization, route/preflight, quality, freeze, three-distinct-
prototype, label-blind selection, correct-association, contrastive,
prediction-commitment, artifact-integrity, externally anchored manifest, and
per-held-take component gates pass. Any missing or failed row is classified
before I/O as `blocked_pre_io`, after I/O from preregistered machine-integrity
telemetry as `technical_invalid`, or otherwise as `behavioral_failed`; there is
no qualitative-success fallback.

A first product-level repeatability claim uses one bounded cohort whose
machine-readable preregistration freezes, before the first live I/O:

- one cohort identifier and exactly five ordered session-slot identifiers;
- this exact protocol, repertoire, representation, scorer, threshold, route,
  and capture-window version; and
- the rule that every started slot is retained and no sixth or replacement
  session may contribute to the claim.

All five slots must be resolved and reported separately before the aggregate
verdict. At least **three of the five** must be fully `passed`. Passes need not
be consecutive, and a later `technical_invalid` or `behavioral_failed` session
does not erase an earlier pass. A slot blocked before live I/O may resume after
preflight succeeds because no session attempt began. Once any audio sample is
accepted, the slot is consumed even if the owner stops or a hardware fault
invalidates the session.

The cohort report must show both:

- **behavioral repeatability** — `passed` sessions divided by technically valid
  started sessions; and
- **operational reliability** — technically valid sessions divided by all
  started session slots, with `technical_invalid` reasons and pre-I/O blocks
  listed separately.

One passed session is a measured pilot only. Fewer than three passes after all
five slots resolve is a failed cohort. Extra sessions under this protocol
version cannot rescue it; any successor product-level attempt requires a new
protocol version and a new preregistered cohort.

No representation, repertoire, device route, capture window, threshold,
aggregation rule, or stop rule may change after the first session result. A
change creates a new protocol candidate, not a retry.

## 10. Required private artifacts and public-safe records

Each held-trial directory must contain or identify by relative path and
SHA-256:

```text
source_human.wav
target_captured.wav
agent_emitted.wav
response_captured.wav
self_prediction.wav
report.json
selection-commitment.json
prediction-commitment.json
```

`target_captured.wav` is the canonical imitation target. `source_human.wav` is
source provenance/context, not a substitute canonical target. For direct live
owner capture it aliases the same path and hash as `target_captured.wav`; it is
not a second sensory route or microphone capture. The session root contains
`frozen-vocal-associations.json`, `session-manifest.json`,
`session-attestation.json`, the authorization record, append-only attempt log,
and deletion attestation. `report.json` must not attempt to self-hash;
`session-manifest.json` attests its completed hash, and the public-safe
`session-attestation.json` anchors the manifest hash through a Git commit or
owner signature. The report keeps imitation fidelity (captured target ↔
captured response) and self-model fidelity (precommitted self prediction ↔
captured response) as separate score families.

## 11. Required implementation and approval gates

This protocol must not be run until a later implementation task adds support
for the frozen representation and label-blind association lookup, two held
takes per target, distinct-prototype and contrastive gates, response
playback/capture full-span evidence, pre-response prediction commitment,
route/preflight persistence, the proposed manifest and external-attestation
fields, append-only no-retry state machine, retention/deletion enforcement, and
the five-slot cohort ledger, session-status taxonomy, separate behavioral and
operational denominators, and all scoring/claim gates above. The corrected
offline scorer exists, but no live session orchestrator implements this state
machine. That implementation needs its own tests and review; protocol approval
does not authorize live execution by itself.

The owner approved this protocol design and its retention decision on
2026-07-16. After calibration, implementation, review, cohort registration, and
preflight, the owner must still issue one fresh authorization for each session.
Only that per-session authorization can permit real microphone or speaker use.
