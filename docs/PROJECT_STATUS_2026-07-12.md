# Cora-Infans project status — 2026-07-12

## Operational closure

The bounded VISION is now implemented as an end-to-end engineering contract:

- every retained stage artifact is verified against a frozen SHA-256 pin;
- Stage 4 has an executable exact-replay validator binding command, prior
  replay, efference identity, sensory consequence, capacity, and clipping;
- Stage 6 rejects reports that omit or drift from the complete host, input,
  output-route, geometry, level, backend, frame, stimulus, or operator-confirmed
  operating point;
- a read-only preflight combines current host/device discovery with retained
  evidence and the standing physical manifest;
- a single explicitly authorized command performs one bounded ten-second
  Stage 6 acceptance run and retains raw plus normalized reports;
- CI uses immutable action commit pins, Python `3.13.14`, an exact dependency
  constraint set, read-only permissions, readiness before tests, and no live-I/O
  flags.

The operational procedure and stop rules are documented in
[OPERATIONAL_RUNBOOK.md](OPERATIONAL_RUNBOOK.md).

## Verification

- Clean Python 3.13 constrained installation: passed.
- Hash-pinned operational readiness: passed.
- Full hardware-safe suite: `481 passed, 3 skipped`.
- After the XVF3800 was reconnected, read-only preflight passed the host, input,
  format, output-route, contract, and evidence checks.
- Operational acceptance attempt 1 completed 400 frames with clean stream
  integrity but emitted zero salient frames. The operator heard the utterance
  only barely; active and quiet peaks were nearly identical (`0.0521` and
  `0.0502`). The attempt is retained as a failed, ineligible acoustic-level run.
- The zero-salience normalizer defect was corrected offline without replay. Its
  corrected verdict remains failure because salient latency was unavailable.
- Preflight now requires a fresh confirmation that the external mechanical knob
  is at approximately `50%`; accepting any lower level was too permissive.
- Corrected acceptance attempt 2 passed at the restored frozen level: `208`
  salient frames, `10.224 ms` salient ADC-frame-end p95, `277` active playback
  frames, and zero clipping, overflows, deadline misses, callback faults,
  timestamp discontinuities, or fallbacks.
- The retained attempt-2 result is SHA-256 pinned by operational readiness.
- No protected corpus path was accessed.

## Operational disposition

The bounded VISION is operationally accepted at the named physical point. Future
sessions begin with readiness and read-only preflight:

```bash
python3 -m cora_infans --operational-preflight \
  --manifest config/operational-point.json
```

The stimulus, `-18 dB` digital gain, saliency threshold, geometry, frame count,
and all criteria remain frozen. The accepted run does not reopen Stage 2 or H4,
and it does not represent retained Stage 4 evidence as freshly rerun.

## Bounded speech-learning demo

The offline v3 demo passes synthetic-mirror and cross-configuration imitation
for `/a/`, `/i/`, and `/u/`. It babbles a deterministic 96-action repertoire,
uses label-blind local frontier search, stores three target-specific prototype
associations, and recalls each learned action for the final three attempts.

Cross-configuration sensory distance improves by `72.0%`, `73.8%`, and `62.4%`
for `/a/`, `/i/`, and `/u/`, ending at `0.144`, `0.177`, and `0.156`. All are
inside the frozen `0.20` gate. Target and learned presentation WAVs plus complete
learning curves are retained outside Git and SHA-256 identified by the
[v3 result](evidence/2026-07-12-speech-learning-demo-v3-result.json).

V1 and v2 failures remain retained. This demo is finite prototype association
over synthetic output, not continuous inverse closure, human speech learning,
phoneme recognition, or new H4 evidence.

## Project-owner vocal-imitation bridge

- A separately frozen protocol now connects new project-owner recordings to
  the finite 96-prototype learner through a common 30-band waveform feature
  representation.
- Capture is fail-loud, authorization-gated, hash-identified, fixed to the
  XVF3800 16 kHz callback and its two fused host channels, and retains exactly
  two development plus one confirmation take for each of `/a/`, `/i/`, `/u/`.
- All associations freeze before confirmation decoding. Confirmation cannot
  select or replace a motor, protected corpora are inaccessible through this
  session-local manifest, and a missed gate stops this v1 recording set.
- The synthetic-injected protocol fixture passes all capture, learning, recall,
  confirmation, provenance, and dashboard contracts. The first human recording
  set remains to be captured and is not yet claimed as passing. The distinction
  is retained in the
  [pre-human calibration](evidence/2026-07-12-vocal-imitation-v1-prehuman-calibration.json).
- The first live v1 `/a/` take missed the spoken interval and failed unchanged
  voice-presence gates. V1 is retained and stopped. V2 increases only the
  acquisition window from 1.8 to 4.0 seconds, requires a fresh session, and
  leaves every analysis, learning, and confirmation gate unchanged.
  The fresh 4.0-second synthetic fixture passes and is retained separately as
  [v2 pre-human calibration](evidence/2026-07-12-vocal-imitation-v2-prehuman-calibration.json).
- V2 later retained four valid development takes but missed the `/u/` interval
  during another delayed tool start. V2 is stopped without reusing those takes.
  V3 adds a bounded physical start cue on Mac mini Speakers before discarding
  cue/settle samples and opening the unchanged 4.0-second retained window.
- The V3 hardware rehearsal emitted the bounded cue on Mac mini Speakers,
  returned a clean callback, observed a post-cue voiced interval, and discarded
  all audio without consuming a protocol take.
