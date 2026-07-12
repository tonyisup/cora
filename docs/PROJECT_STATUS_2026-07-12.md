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
- Full hardware-safe suite: `460 passed, 3 skipped`.
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
- No protected corpus path was accessed.

## Remaining physical step

Set the external speaker mechanical knob to approximately `50%`, set the fresh
confirmation in `config/operational-point.json` to `true`, and rerun:

```bash
python3 -m cora_infans --operational-preflight \
  --manifest config/operational-point.json
```

Only after a passing preflight may one corrected bounded acceptance run execute.
The stimulus, `-18 dB` digital gain, saliency threshold, geometry, frame count,
and all criteria remain frozen. That run rechecks the current Stage 6 physical
path; it does not reopen Stage 2 or H4, and it does not represent retained Stage
4 evidence as freshly rerun.
