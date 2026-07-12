# Project-owner vocal-imitation protocol

Status: v1 frozen before the first project-owner recording.

This protocol advances the bounded synthetic speech-learning demo to a new,
project-owner-recorded target. It is a demo-layer experiment, not a retry of
the stopped H4 inverse-closure families and not evidence about infant learning,
phoneme recognition, or general human speech.

## 1. Authorized data and physical boundary

- The only recorded input is a new vowel produced by the project owner for this
  protocol. External corpora and every protected path named in `VISION.md` are
  prohibited.
- Capture requires the literal `--authorize-project-owner-recording` flag.
- Input is the XVF3800 callback path at 16 kHz and its two fused host channels.
  No four-microphone geometry is inferred.
- There is no playback during teacher capture. External-speaker geometry is
  therefore irrelevant to this stage.
- Each target has exactly two development takes and one confirmation take.
  Confirmation audio is decoded only after the motor association is frozen.

## 2. Spoken prompt

For each of `/a/`, `/i/`, and `/u/`, produce a comfortable, steady vowel for
roughly one second. Do not imitate a particular pitch. Each capture lasts
1.8 seconds, allowing quiet lead-in and tail.

The vowel name is capture organization and display metadata. The learner gets
only an opaque hash and a 30-dimensional auditory vector.

## 3. Frozen preprocessing

1. Verify the manifest hash and require the WAV to live beneath the session's
   `captures/` directory.
2. Decode uncompressed WAV, mix the two fused channels to mono, and linearly
   resample to 16 kHz when needed.
3. Select the highest-RMS 480 ms window on an 80 ms stride.
4. Require finite samples, no sample at or above full scale, selected-window RMS
   at least `0.01`, selected-window peak at least `0.03` and below `0.98`, an
   autocorrelation pitch between `70–350 Hz`, and normalized pitch periodicity
   at least `0.30`. This prevents steady fan/background energy from satisfying
   the capture gate without a voiced target.
5. Compute averaged 64 ms Hann-window power spectra at 16 ms hops. At the
   plant's frozen 30 log-spaced frequencies, integrate Gaussian bands whose
   width is `max(80 Hz, 10% of center frequency)`, take log energy, remove the
   mean, and unit-normalize.
6. Unit-normalize the arithmetic mean of the two development vectors. This is
   the sole learning target.

## 4. Bounded learning and confirmation

- Reuse the 96 motors, Halton schedule, local frontier, 12 attempts, three-entry
  association capacity, and final-three recall contract in `DEMO_SPEC.md`.
- Recompute each babbled consequence through the same waveform feature path at
  frozen `110 Hz`; human and self targets are therefore compared in one
  operational auditory representation.
- Development selection must remain label-blind.
- After selection and recall finish, decode the confirmation take and score the
  already-frozen learned prototype. Confirmation may never select or replace a
  motor.

## 5. Frozen gates and stop rule

Every target must satisfy:

- all three capture-quality gates pass and all hashes match;
- the two development takes are at most `0.30` apart;
- development distance improves at least `20%` from rest and finishes at most
  `0.45` from the learned prototype;
- confirmation distance improves at least `15%` relative to the same rest
  prototype and finishes at most `0.45` from the already-learned prototype;
- the final three development attempts recall the same prototype;
- confirmation is accessed after learning and does not change the association;
- memory remains within three targets.

The aggregate demo passes only if `/a/`, `/i/`, and `/u/` pass. A missed gate is
retained as a result; thresholds and preprocessing may not be tuned on those
takes. Any successor requires a new version and fresh recordings.

## 6. Claim boundary

A pass demonstrates bounded acoustic target association and deterministic
recall for three new vowels from one known project owner in one capture setup.
It does not establish exact articulatory recovery, continuous inverse closure,
speaker identity, linguistic categories, generalization to new people, rooms,
registers, languages, or cultures, or autonomous language acquisition.
