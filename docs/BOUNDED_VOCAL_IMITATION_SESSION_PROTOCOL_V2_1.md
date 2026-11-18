# Bounded vocal-imitation product-session protocol V2.1 candidate

Status: OFFLINE REVISION — NOT OWNER-APPROVED, NOT EXECUTABLE, AND NO LIVE I/O
AUTHORIZED.

## 1. Purpose and lineage

V2.1 is a separately versioned successor to the rejected V2 native player. The
authorized V2 cue diagnostic produced neither a returned-buffer callback nor an
audible cue. No microphone was opened, no cohort was registered, and no session
was executed. V1 and V2 records remain historical evidence and cannot contribute
audio, slots, passes, or authorization to V2.1.

V2.1 changes one ordering rule: after enqueue, prime, and start, the player must
observe the returned-buffer callback **before** it requests non-immediate stop.
It then verifies that the queue is no longer running and disposes
non-immediately. CoreAudio UID binding and every behavioral, scoring, retention,
and repeatability rule remain unchanged from approved V2.

## 2. Authority gates

This candidate authorizes no playback, microphone capture, device mutation,
cohort, session, or claim. Progress requires, in order:

1. offline build, static analysis, and contract tests;
2. hash-pinned V2.1 source, binary, protocol, operating point, and evidence;
3. owner approval of those exact artifacts and the unchanged 30-day retention;
4. separate authorization for one exact V2.1 cue diagnostic;
5. passing machine completion evidence and immediate owner audibility
   confirmation while the microphone remains closed;
6. registration of a new five-slot V2.1 cohort; and
7. fresh one-session live-I/O authorization before each session.

Approval or cue authorization granted to V1 or V2 does not transfer.

## 3. Frozen device identity and geometry

Numeric device positions are informational only. Direct read-only CoreAudio
property enumeration must resolve exactly one frozen UID and cross-check its
exact name. Enumeration may not create or start an AudioQueue and must report
zero audio frames sent.

| Role | Frozen CoreAudio UID | Required name |
| --- | --- | --- |
| Self | `BuiltInSpeakerDevice` | `Mac mini Speakers` |
| External world | `BuiltInHeadphoneOutputDevice` | `External Headphones` |

The self source remains centered 4 cm directly below the XVF3800, at 100%
system volume and PCM peak no greater than 0.10. The external-world source
remains 180 degrees and 50 cm from the array through External Headphones at
approximately 50% hardware volume. Missing, duplicate, renamed, muted, or
misconfigured devices block before live I/O.

## 4. Callback-before-stop native playback

V2.1 uses `cora-audioqueue-player-v2`. For every finite-rate linear-PCM WAV it
must:

1. create an AudioQueue and set its current device to the frozen UID;
2. read back and verify the current-device UID;
3. allocate and enqueue the full buffer before starting;
4. prime and start the queue;
5. wait for at least one returned-buffer callback, with a ten-second timeout;
6. require callback elapsed time to cover at least 90% of WAV duration;
7. only after that callback, request `AudioQueueStop(queue, false)`;
8. wait until `kAudioQueueProperty_IsRunning` becomes false;
9. require total elapsed time to cover at least 90% of WAV duration; and
10. dispose with `AudioQueueDispose(queue, false)`.

Exit zero without every field is failure. There is no fallback. Error cleanup
may dispose immediately only after the candidate has already failed.

## 5. Exact-cue validation

The cue remains exactly 100 ms at 660 Hz and amplitude 0.08, followed by a
250 ms settle excluded from retained target audio. After exact-artifact owner
approval, a separately authorized diagnostic may play that cue once through
the self UID with the microphone closed. Machine evidence must show UID
verification, callback-before-stop ordering, stopped queue, duration coverage,
and non-immediate successful disposal. The owner must immediately confirm the
cue was audible. Any failure blocks without consuming a cohort slot and grants
no retry.

The confirmation does not authorize a session. It must be repeated after any
output-device, driver, display, mute, volume, room, or speaker change.

## 6. Unchanged behavioral and scoring contract

V2.1 preserves:

- targets `/a/`, `/i/`, and `/u/`;
- development and held schedules `/a/, /i/, /u/, /a/, /i/, /u/`;
- twelve five-second target captures and six five-second response captures;
- callback XVF3800 capture at 16 kHz with two fused host channels;
- the 96-entry fixed Halton repertoire;
- the label-blind 30-dimensional auditory representation and three frozen
  prototype associations;
- nearest-centroid maximum 0.30, runner-up margin minimum 0.05, and tie
  tolerance `1e-12`;
- selected-prototype improvement over rest of at least 15% and margin over
  every off-target prototype of at least 0.05; and
- no retry after any accepted audio sample or failed gate.

Capture-quality gates remain finite samples, no full-scale samples, voiced RMS
at least 0.01, peak at least 0.03 and below 0.98, pitch 70–350 Hz, and
periodicity at least 0.30.

Scoring remains `offline-replay-v2`; every held take independently requires
active-duration difference at most 150 ms, RMS-envelope distance at most 0.05,
voiced pitch-contour distance at most 30 Hz, and cepstral distance at most 0.50.
No composite may override a component failure.

## 7. Response capture integrity

Before each response, exact emitted and self-prediction WAVs are written and
hash-committed. Callback capture begins at least 250 ms before playback and
continues through confirmed completion. Playback onset, callback completion,
and queue-stop completion are anchored in callback stream-frame coordinates. A
post-start UID, callback, stop, timeout, route, or capture fault is
`technical_invalid` and consumes the slot.

## 8. Privacy and retention

The unchanged retention decision is 30 calendar days with AES-256 per-session
sparse bundles, per-session keys, verified backup exclusion or inventory, raw
audio outside Git, a private manifest, an external manifest anchor, and
cryptographic erasure. The owner is the sole speaker and may stop at any time.

## 9. Repeatability and status

A new V2.1 cohort must register exactly five new ordered slots. All five must
resolve; at least three must fully pass; passes need not be consecutive.
`blocked_pre_io` consumes no slot. `technical_invalid`, `behavioral_failed`,
and `passed` consume a started slot. Behavioral repeatability and operational
reliability remain separately reported. No V1 or V2 slot or audio contributes.

## 10. Claim boundary

A future passing V2.1 cohort could support only repeatable, label-blind,
target-specific association and recall for the authorized owner at one frozen
room/device operating point and finite repertoire. It cannot support phoneme
recognition, arbitrary speech reproduction, perceptual quality, speaker
identity, linguistic understanding, infant learning, generalized imitation,
H4 closure, or production availability.
