# Bounded vocal-imitation product-session protocol V2 candidate

Status: IMPLEMENTED CANDIDATE — NOT OWNER-APPROVED, NOT EXECUTABLE, AND NO LIVE
I/O AUTHORIZED.

## 1. Purpose and relationship to V1

V2 is a clean successor candidate to the closed V1 cohort. V1 remains immutable
historical evidence. V2 does not reinterpret V1 audio, replace a V1 slot, or
rescue the V1 product threshold.

V1 exposed two playback-boundary defects:

1. output indices changed when an unrelated CoreAudio endpoint appeared, while
   preflight checked the frozen name and number independently rather than
   binding them; and
2. ffmpeg's AudioToolbox output muxer could accept a 100 ms cue and return zero
   while immediately flushing and disposing its queue, without proving the
   enqueued buffers reached audible output.

V2 changes only output identity and playback-completion semantics. Target set,
capture schedule, representation, repertoire, selection gates, score limits,
retention, status taxonomy, no-retry rule, and three-of-five cohort rule remain
unchanged.

## 2. Authority and execution gates

This candidate permits no playback, microphone capture, device mutation,
session execution, or product claim. Execution requires, in order:

1. offline implementation review and passing tests;
2. hash-pinned native-player build and deterministic contract evidence;
3. owner approval of this exact V2 protocol and retention decision;
4. a separately authorized exact-cue diagnostic through the UID-bound player;
5. immediate owner confirmation that the cue was audible while the microphone
   remained closed;
6. registration of a new five-slot V2 cohort; and
7. fresh one-session authorization immediately before each V2 session.

## 3. Stable output identity

V2 never treats an AudioToolbox array position as device identity. Zero-frame
enumeration yields the current index, human-readable name, and CoreAudio UID.
The runner resolves the frozen UID, requires exactly one match, requires the
name to match, and uses the resulting index only as an informational transport
detail.

| Role | Frozen CoreAudio UID | Required name |
| --- | --- | --- |
| Self | `BuiltInSpeakerDevice` | `Mac mini Speakers` |
| External world | `BuiltInHeadphoneOutputDevice` | `External Headphones` |

Adding, removing, or reordering unrelated devices must not change the selected
UID. A missing, duplicate, or renamed UID blocks before live I/O. Physical
geometry and level remain self at 4 cm directly below the XVF3800 with 100%
system volume and PCM peak no greater than 0.10; world remains 180 degrees and
50 cm through External Headphones.

## 4. Drain-confirmed native playback

V2 uses `cora-audioqueue-player-v1`, not ffmpeg's AudioToolbox output muxer, for
all cues and generated responses. For each WAV, the player must:

1. decode a finite-rate linear-PCM WAV;
2. create an AudioQueue and bind `kAudioQueueProperty_CurrentDevice` to the
   frozen UID;
3. read back and verify the current-device UID;
4. allocate and enqueue the full buffer before starting;
5. prime and start the queue;
6. request `AudioQueueStop(queue, false)` so stopping is non-immediate;
7. wait until `kAudioQueueProperty_IsRunning` becomes false;
8. observe at least one returned-buffer callback;
9. require playback elapsed time to cover at least 90% of WAV duration; and
10. dispose successfully with `AudioQueueDispose(queue, false)`.

Exit code zero without UID verification, a buffer callback, stopped-queue
evidence, and elapsed-duration evidence is a technical failure. The player has
a ten-second hard drain timeout and no fallback.

## 5. Cue and audibility preflight

The target start cue remains exactly 100 ms at 660 Hz and amplitude 0.08,
followed by 250 ms settle; cue and settle are excluded from retained target
audio. Before registering or opening a V2 cohort session, a separately
authorized diagnostic plays the exact cue once through the V2 player with the
microphone closed. The owner must immediately confirm audibility. Both machine
drain evidence and owner confirmation are required. A miss blocks without
consuming any session slot.

The confirmation authorizes no later session. Device enumeration and volume
checks repeat before every session, and a fresh cue confirmation is required
after any output-device, driver, display, mute, volume, room, or speaker change.

## 6. Unchanged behavioral session

Each fresh session retains the V1 schedule:

- targets `/a/`, `/i/`, `/u/`;
- development order `/a/, /i/, /u/, /a/, /i/, /u/`;
- held order `/a/, /i/, /u/, /a/, /i/, /u/`;
- twelve five-second target captures and six five-second response captures;
- callback XVF3800 input at 16 kHz and two fused host channels;
- the existing 96-entry fixed Halton repertoire;
- the existing label-blind 30-dimensional auditory representation;
- three distinct frozen prototype associations;
- nearest-centroid maximum 0.30, runner-up margin minimum 0.05, and tie
  tolerance `1e-12`;
- selected-prototype improvement over rest of at least 15% and margin over
  every off-target prototype of at least 0.05; and
- no retry after any accepted audio sample or failed gate.

Capture-quality gates remain finite samples, no full-scale samples, voiced
window RMS at least 0.01, peak at least 0.03 and below 0.98, pitch 70–350 Hz,
and periodicity at least 0.30.

## 7. Response playback and scoring

Before each response, the exact emitted WAV and self-prediction WAV are written
and hash-committed. The UID-bound player must return drain evidence while the
callback capture remains active from at least 250 ms before playback through
confirmed playback completion. Playback onset and drain completion are anchored
in callback stream-frame coordinates.

Scoring remains `offline-replay-v2`; each held take must independently pass:

| Component | Maximum |
| --- | ---: |
| Active vocal-event duration difference | 150 ms |
| RMS-envelope distance | 0.05 |
| Voiced pitch-contour distance | 30 Hz |
| Cepstral distance | 0.50 |

No composite score may override a failed component. These remain engineering
limits, not perceptual-quality guarantees.

## 8. Integrity, privacy, and retention

V2 retains the V1 AES-256 per-session sparse-bundle storage, verified backup
exclusion or backup inventory, raw audio outside Git, private manifest,
external manifest anchor, 30-calendar-day retention, and cryptographic erasure.
The owner remains the sole speaker and may stop at any time.

V2 reports UID, resolved informational index, name, player source/binary hashes,
callback count, queue-drain state, elapsed playback time, system volume, and
full-span capture coordinates for every playback.

## 9. Session and cohort status

Status meanings remain `blocked_pre_io`, `technical_invalid`,
`behavioral_failed`, and `passed`. Device/UID mismatch or failed cue audibility
before capture is `blocked_pre_io` and consumes no slot. A post-start playback
UID, callback, drain, timeout, or capture-transport failure is
`technical_invalid` and consumes the slot.

A new V2 cohort must register exactly five new ordered slot identifiers before
its first live I/O. All five must resolve; at least three must fully pass;
passes need not be consecutive. Behavioral repeatability and operational
reliability remain separate. No V1 slot or audio may contribute.

## 10. Claim boundary

A future passing V2 cohort may support only repeatable, label-blind,
target-specific association and recall for the authorized owner at one frozen
room/device operating point and finite repertoire. It cannot support phoneme
recognition, arbitrary speech reproduction, perceptual quality, speaker
identity, linguistic understanding, infant learning, generalized imitation,
H4 closure, or production availability.
