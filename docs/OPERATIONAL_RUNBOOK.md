# Cora-Infans operational runbook

This runbook operates the bounded system described by [VISION.md](../VISION.md).
It is not an authorization to reopen stopped Stage 2 or Stage 5 research, touch
protected data, or make broader claims.

## 1. Frozen physical point

Before using any live command, confirm all of the following:

- XVF3800 input selector `:0`, callback capture, `16 kHz`, two fused host
  channels;
- Mac mini Speakers centered directly beneath the XVF3800, `4 cm`
  center-to-center, system volume `100%`;
- External Headphones output device `2`, source at `180°` and `50 cm`, hardware
  knob at approximately `50%` for bounded acceptance and never higher;
- no intentional change to the room, routes, or protocol-specific digital gain.

The machine-readable standing attestation is
[`config/operational-point.json`](../config/operational-point.json). If any
physical fact changes, edit that manifest before running preflight. A mismatched
manifest must fail; do not modify the contract to make a changed setup pass.
The repository manifest deliberately stores the 50%-knob confirmation as
`false`; set it to `true` only after physically checking the knob for the current
session.

## 2. Evidence and software readiness

From the repository root:

```bash
python3 -m cora_infans --operational-readiness
```

This verifies the packaged contract and every retained evidence SHA-256 pin. It
does not inspect current devices or execute the test suite.

Then run the read-only physical preflight:

```bash
python3 -m cora_infans --operational-preflight \
  --manifest config/operational-point.json
```

Preflight checks the host family, AudioToolbox-capable `ffmpeg`, XVF3800
enumeration, two-channel `16 kHz` input support, retained evidence, and all
operator-confirmed geometry/level fields. It does not open an audio stream or
play sound. Exit status `1` means the setup is not ready; status `2` means the
manifest or invocation is invalid.

## 3. Hardware-safe tests

```bash
python3 -m pytest tests -q
```

The suite uses fakes for hardware paths. CI runs pinned Python and dependencies,
then readiness and this suite, with read-only repository permissions.

## 4. Bounded live acceptance

Only continue if preflight exits `0`. The acceptance stimulus is the already-
spent `ACO02B.wav` whose SHA-256 is frozen in the operational contract. No
protected corpus path is authorized.

```bash
python3 -m cora_infans --run-operational-acceptance \
  --manifest config/operational-point.json \
  --stimulus-wav /private/tmp/cora-ids-spent-200/ACO02B.wav \
  --output-dir /private/tmp/cora-operational-acceptance \
  --authorize-live-io
```

The command performs exactly one ten-second external-world run: 400 callback
frames of 25 ms, digital playback gain `-18 dB`, no repeat. It writes:

- `stage6-live-raw.json` — the complete listener telemetry;
- `operational-acceptance.json` — normalized physical metadata, raw-report
  SHA-256, all Stage 6 criteria, and the bounded verdict.

The command refuses before playback if preflight or stimulus hashing fails.
Exit status `1` means the run completed but missed an operational criterion;
status `2` means it was blocked or invalid. Preserve both outputs for any
completed failure. Never tune a threshold, gain, geometry, or representation to
rescue that result.

This run rechecks Stage 6 only. Stage 4 exact-efference support remains the
hash-pinned retained exact-replay result; it is not silently represented as
freshly re-executed.

## 5. Failure handling

- Missing XVF3800: reconnect or restore the device, then rerun read-only
  preflight. Do not use another microphone.
- Wrong geometry or level: restore the frozen point or document a VISION
  revision; do not edit the report.
- Clipping, overflow, callback status, timestamp discontinuity, fallback, or
  latency above `10.5 ms`: retain the failed report and stop.
- Hash mismatch: stop and identify the file/provenance problem. Do not replace a
  pin after observing metrics.
- Protected-data request: stop. It requires a separate VISION revision and
  preregistration.
