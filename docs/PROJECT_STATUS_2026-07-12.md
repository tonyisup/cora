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
- Full hardware-safe suite: `457 passed, 3 skipped`.
- Current read-only host preflight: blocked correctly because CoreAudio exposes
  External Headphones and Mac mini Speakers but does not currently enumerate
  the XVF3800 input.
- No playback or capture was attempted after that failed preflight.
- No protected corpus path was accessed.

## Remaining physical step

Reconnect or restore the XVF3800 so it appears in CoreAudio, then rerun:

```bash
python3 -m cora_infans --operational-preflight \
  --manifest config/operational-point.json
```

Only after a passing preflight may the bounded acceptance command in the
runbook execute. That run rechecks the current Stage 6 physical path; it does
not reopen Stage 2 or H4, and it does not represent retained Stage 4 evidence as
freshly rerun.
