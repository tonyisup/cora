# Bounded vocal-imitation product-session protocol V1

Status: DRAFT — NOT AUTHORIZED FOR RECORDING OR LIVE I/O.

## 1. Purpose and authority

This is a proposed protocol for a **future** bounded project-owner
vocal-imitation product session. It defines proposed operational values before
any new owner recording. It is not an authorization record, not an instruction
to open a microphone or speaker, and not a continuation or modification of the
completed V4 experiment.

Execution requires all of the following later events:

1. the project owner explicitly approves this exact protocol version and its
   retention decision for one named session;
2. a later implementation task supports this protocol's two held takes per
   target and writes its required manifest/report fields;
3. a later preflight verifies the named physical route without silently falling
   back to another device; and
4. the owner gives a fresh, one-session live-I/O authorization at execution
   time.

Until then, no capture, playback, device discovery, repertoire selection, or
association training is authorized. V4 is immutable retained evidence. Stage 2
and H4 remain stopped; this protocol does not reopen either program.

## 2. Narrow proposed claim and non-claims

If executed and all gates pass, a future result may claim only repeatable,
target-specific association and recall for one explicitly authorized owner,
three declared vowel targets, one declared room/device operating point, and one
fixed finite repertoire.

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
  only as a non-sensitive storage alias; and
- that raw and generated WAV artifacts, temporary PCM, and copied presentation
  WAVs are securely deleted **30 calendar days** after a completed session
  report, unless the owner explicitly extends retention before that deadline.

The retained private manifest may keep source paths and SHA-256 values. Git may
retain only this protocol, the authorization-free preregistration, manifest
schema, hashes, aggregate metrics, and non-identifying reports. Deletion does
not erase a failure record: its non-audio report and hashes remain retained.

## 4. Proposed physical operating point

This operating point is a proposed, fail-closed route—not a claim that the
hardware is currently available or calibrated.

| Surface | Proposed value | Required recording/report field |
| --- | --- | --- |
| Host | Mac mini | host model/OS/build identifier |
| Input | XVF3800 selector `:0`, callback capture, 16 kHz, two fused host channels | device selector, backend, sample rate, channel count |
| Target capture | no simultaneous stimulus playback | input metadata and capture-quality result |
| Response playback | Mac mini Speakers centered beneath the XVF3800, 4 cm center-to-center | output device identity, geometry, system volume, digital peak |
| Response level | system volume 20%; output PCM peak no greater than 0.10 full scale | measured configured values, not inferred defaults |
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

Each fresh session has exactly twelve planned captures:

| Phase | Takes per target | Total | Permission |
| --- | ---: | ---: | --- |
| Development | 2 | 6 | allowed after authorization and preflight |
| Held evaluation | 2 | 6 | prohibited until development freeze verifies |

Each take is a 5.0-second retained window. Captures are ordered by a
pre-registered schedule and never selected after listening to results. A
technical failure before a WAV artifact is created may be logged once as
`technical_abort`; it does not consume a take. No retry is permitted after a
quality or score failure. A failed planned take ends that target's session
result and is retained as failure evidence.

## 6. Fixed repertoire and association freeze

The proposed repertoire is the existing finite 96-entry Halton schedule (start
`20011`, stride `43`, last `24096`) with its exact repertoire digest recorded
before development capture. This is a disclosed fixed bank, not a new
continuous controller or a new H4 mechanism.

After exactly six quality-passing development captures:

1. derive one association per target from development material only;
2. write a hash-pinned `frozen-vocal-associations.json` containing the
   development capture hashes, repertoire digest, selected prototype IDs,
   scoring version, and protocol version;
3. verify that no association can be selected or replaced afterward; then
4. permit the six held captures.

Any missing/changed development artifact, repertoire mismatch, or failed
freeze blocks held capture. Held material may be scored only against the
already-frozen association.

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

## 8. Proposed scoring gates

Use `offline-replay-v1` exactly as documented in the product specification.
The following values are **proposed engineering acceptance limits**, frozen
before capture if this draft is approved:

| Component | Proposed maximum |
| --- | ---: |
| Onset/offset alignment | 80 ms |
| Duration difference | 150 ms |
| RMS-envelope distance | 0.05 |
| Voiced pitch-contour distance | 30 Hz |
| Cepstral distance | 0.50 |

Every component must pass for **each** held take. An unvoiced/invalid component
is a failure, not a zero, a skipped cell, or a substitute aggregate. The report
also shows each development and held score, plus the rest-prototype comparison,
but no opaque weighted composite may override a failed component.

These values are calibrated only against deterministic scorer behavior (for
example, synthetic ±20/±40 Hz and duration/energy perturbations). They are not
validated perceptual thresholds and must not be marketed as such. They cannot
be changed after any capture in a session. Any future threshold revision
requires a new protocol version, fresh sessions, and a new approval.

## 9. Session and claim gates

A single session is `passed` only if all twelve planned captures, authorization,
route/preflight, quality, freeze, artifact integrity, and per-held-take
component gates pass. Any missing or failed row yields `blocked` or `failed`;
there is no qualitative-success fallback.

A first product-level claim additionally requires **three fresh passed
sessions**, each with a new authorization record, fresh manifest, fresh
captures, and the unchanged protocol/repertoire/scoring/threshold values. The
three sessions must be reported separately before any aggregate statement. A
pass on one session is a measured pilot result only, not a repeatability claim.

No representation, repertoire, device route, capture window, threshold,
aggregation rule, or stop rule may change after the first session result. A
change creates a new protocol candidate, not a retry.

## 10. Required private artifacts and public-safe records

Each take directory must contain or identify by relative path and SHA-256:

```text
source_human.wav
target_captured.wav
agent_emitted.wav
response_captured.wav
self_prediction.wav
report.json
session-manifest.json
frozen-vocal-associations.json
```

`target_captured.wav` is the canonical imitation target. `source_human.wav` is
source provenance/context, not a substitute canonical target. `report.json`
must not attempt to self-hash; `session-manifest.json` attests its completed
hash. The report keeps imitation fidelity (captured target ↔ captured response)
and self-model fidelity (self prediction ↔ captured response) as separate score
families.

## 11. Required implementation and approval gates

This protocol must not be run until a later implementation task adds support
for two held takes per target, response playback/capture full-span evidence,
route/preflight persistence, the proposed manifest fields, the no-retry state
machine, and all scoring/claim gates above. That implementation needs its own
tests and review; this draft does not authorize code changes by itself.

After implementation, the owner must review this document and its matching
preregistration, explicitly approve the proposed values or amend them in a new
version, and then issue one fresh session authorization. Only that later
approval can permit real microphone or speaker use.
