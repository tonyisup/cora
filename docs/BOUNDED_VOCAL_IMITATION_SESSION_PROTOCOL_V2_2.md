# Bounded vocal-imitation product-session protocol V2.2 candidate

Status: OFFLINE IMPLEMENTED CANDIDATE — NOT OWNER-APPROVED, NOT EXECUTABLE,
AND NO LIVE I/O AUTHORIZED.

## 1. Purpose and predecessor disposition

V2.2 is a fresh successor to the closed V2.1 cohort. V2.1 accepted six clean
development captures and froze three distinct associations, then correctly
stopped before held I/O because no self model had been specified, produced,
or hash-pinned before the required prediction commitment. Its one consumed
failure and four unstarted slots cannot contribute to V2.2.

V2.2 fixes that precondition by scheduling one bounded, machine-generated
self-path calibration capture before any owner target. The resulting causal FIR
model must freeze before development opens. V2.2 also supplies one versioned,
durable 19-capture state machine and role-specific completion artifacts so the
entire held-response path can be dry-run before live execution.

No V2.1 audio, association, self model, slot, result, or authorization may be
reused. Behavioral thresholds, scoring thresholds, repertoire, target schedule,
physical geometry, playback helper, privacy, and retention remain unchanged.

## 2. Authority and execution gates

This candidate permits no device mutation, playback, microphone capture,
cohort registration, session, or claim. Execution requires, in order:

1. deterministic offline calibration and complete state-machine tests;
2. full-suite regression tests and hash-pinned V2.2 artifacts;
3. owner approval of the exact V2.2 protocol and unchanged 30-day retention;
4. confirmation that the existing cue-route evidence remains valid, or a
   separately authorized cue diagnostic if any invalidation condition occurred;
5. registration of a fresh five-slot V2.2 cohort using storage-safe identifiers;
6. fresh per-session authorization that explicitly includes the machine
   self-calibration playback/capture and the owner target captures; and
7. a passing read-only preflight immediately before that session.

V1, V2, or V2.1 live-I/O authority does not transfer. The passed V2.1 cue result
may be carried forward only as route evidence when the exact native binary,
speaker UID, driver, display route, room, speaker placement, system volume, and
mute state are unchanged. Carry-forward authorizes no replay.

## 3. Stable device identity and operating point

Direct CoreAudio property enumeration must create no AudioQueue and send zero
audio frames. Each route resolves exactly one frozen UID and cross-checks its
exact name; numeric indices are informational.

| Role | CoreAudio UID | Required name |
| --- | --- | --- |
| Self | `BuiltInSpeakerDevice` | `Mac mini Speakers` |
| External world | `BuiltInHeadphoneOutputDevice` | `External Headphones` |

The self source remains centered 4 cm directly below the XVF3800 at 100%
system volume and commanded PCM peak no greater than 0.10. The external-world
source remains 180 degrees and 50 cm from the array through External Headphones
at approximately 50% hardware volume. Input remains the uniquely resolved
XVF3800 callback stream at 16 kHz with two fused host channels.

## 4. Pre-development self calibration

The first scheduled microphone capture is `calibration-01-self-chirp`; it is
not owner vocal data. After 250 ms callback pre-roll, the self speaker emits an
exact 120 ms linear chirp from 250 to 3,500 Hz at peak 0.08 and 16 kHz. Capture
continues for exactly 1,000 ms and must span confirmed playback completion.

The calibration must satisfy:

- the approved UID-bound callback-before-stop player and binary digest;
- exactly 16,000 captured frames and two host channels;
- no overflow, callback status, timestamp discontinuity, or clipping;
- capture peak below 0.98;
- recovered chirp correlation at least 0.10 within a 500 ms causal window
  around the stream-frame playback anchor;
- recovered chirp-window RMS at least 0.001; and
- a finite 128-sample causal FIR.

The frozen `self-model.json` records stimulus, capture, onset, impulse, model,
and implementation digests and declares that no owner vocal or post-response
data contributed. It must be written and hashed before development capture 1.
A missing or changed model blocks. A machine calibration failure after accepted
samples is `technical_invalid`, consumes the slot, and permits no replacement.

## 5. Frozen 19-capture schedule and durable transitions

The ordered schedule is:

1. one machine self-calibration capture;
2. six development targets `/a/, /i/, /u/, /a/, /i/, /u/`; and
3. six held pairs in `/a/, /i/, /u/, /a/, /i/, /u/` order, each target then
   response.

This is 19 microphone captures: one machine calibration, twelve owner targets,
and six self responses. Owner target and response windows remain exactly five
seconds. The slot becomes consumed when the calibration accepts its first audio
sample, earlier than V2.1’s first owner target.

Before every I/O operation, the append-only ledger writes an `opened_before_io`
record. Calibration, development target, held target, and held response each
have distinct role-specific completion files. A completion requires accepted
audio, technical validity, and its role gate. A failed completion prevents the
next transition. Exactly one technical abort remains possible only with zero
accepted samples and no partial WAV; no accepted capture or failed gate may be
retried.

Session and cohort identifiers must match `[a-z0-9][a-z0-9-]{7,79}` so the
approved encrypted storage accepts the exact registered identifier.

## 6. Unchanged association and held-selection contract

After six quality-passing development captures, V2.2 freezes the same
96-entry Halton repertoire, label-blind 30-band auditory representation, three
opaque centroids, and three distinct selected prototype IDs. Held capture is
blocked until the V2.2-branded association file and evaluation-key hash agree.

Every target retains the finite/no-clipping, RMS 0.01, peak 0.03–0.98, pitch
70–350 Hz, and periodicity 0.30 quality gates. Held selection retains nearest
distance at most 0.30, runner-up margin at least 0.05, tie tolerance `1e-12`,
selected improvement over rest at least 15%, and margin over every off-target
prototype at least 0.05. Scheduled labels remain unavailable until selection
commitment.

## 7. Prediction, response playback, and scoring

Before each held response opens, the runner must:

1. verify the pre-development self-model digest;
2. write the exact selected `agent_emitted.wav`;
3. generate `self_prediction.wav` only by applying that frozen model to the
   emitted waveform;
4. hash both WAVs and the complete self model in a V2.2 prediction commitment;
5. verify selection, contrastive, and scheduled-label verdict gates; and
6. prove neither WAV depends on response capture.

Response callback capture begins at least 250 ms before playback and remains
active through UID-bound callback-before-stop completion. Onset and completion
are anchored in callback stream-frame coordinates.

Scoring remains `offline-replay-v2`. Every held response independently requires
active-duration difference at most 150 ms, RMS-envelope distance at most 0.05,
voiced pitch-contour distance at most 30 Hz, and cepstral distance at most 0.50
in both imitation and self-model families. No composite overrides a failure.

## 8. Offline end-to-end proof required before approval

The exact candidate must complete all 19 state transitions using deterministic
fixtures without opening a device. The proof must include calibration chirp
recovery, self-model freeze before development, six development completions,
three distinct associations, six held label-blind selections, six correct
post-commitment verdicts, six prediction commitments, six distinct held target
and response completion files, and six fully evaluated score reports. A test
that stops before any held response is insufficient.

## 9. Privacy, retention, status, and repeatability

Retention remains 30 calendar days with an AES-256 per-session sparse bundle,
per-session key, verified backup exclusion or inventory, raw audio outside Git,
private manifest, external manifest anchor, and cryptographic erasure. The
owner remains the sole speaker and may stop at any time.

A new V2.2 cohort must register five fresh ordered slots. All five must resolve;
at least three must fully pass; passes need not be consecutive. Status meanings
remain `blocked_pre_io`, `technical_invalid`, `behavioral_failed`, and `passed`.
Behavioral repeatability and operational reliability remain separate. No V2.1
audio or slot contributes.

## 10. Claim boundary

A future passing V2.2 cohort could support only repeatable, label-blind,
target-specific association and recall for the authorized owner at one frozen
room/device operating point and finite repertoire, with a preregistered local
self-path model. It cannot support phoneme recognition, arbitrary speech
reproduction, perceptual quality, speaker identity, linguistic understanding,
infant learning, generalized imitation, H4 closure, or production availability.
