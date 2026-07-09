# Live SPEAK State and Mic Suppression Design

> **For Hermes:** This is a design boundary document, not an implementation plan to execute live playback immediately.

**Goal:** Define the safety architecture required before cora-infans may add any live speaker output or simultaneous speaker/mic validation.

**Status:** Drafted after M8-infans-E. No playback implementation exists in this milestone.

---

## Why this boundary exists

cora-infans now has two simulation-only gates:

1. `--promote-self-forward-model`
   - validates deterministic efference → predicted self → residual suppression,
   - includes learning, robustness, world-preservation, and wrong-efference checks.

2. `--promote-speaking-readiness`
   - validates conversation policy and self-forward-model promotion together,
   - authorizes only `simulated_speak_authorized=true`,
   - keeps `live_playback_allowed=false`.

Those gates prove the offline logic is coherent. They do not prove live playback safety. Live playback adds OS buffering, roundtrip latency, room coloration, acoustic feedback risk, and self-listening contamination. This design doc is the required bridge before code crosses that boundary.

---

## Non-negotiable constraints

- No automatic playback during normal `--listen-live` capture.
- No live speaker output unless an explicit SPEAK-state command opts in.
- No simultaneous mic/speaker experiment unless frames are tagged and suppression windows are active.
- No fake 4-mic claims: the current XVF3800 host path exposes 2 fused channels.
- No biological overclaim: live validation may demonstrate an engineered corollary-discharge loop, not biological equivalence.
- No persistent firmware changes.
- No new heavyweight dependencies in the first live slice.

---

## Required preconditions before live SPEAK mode

A live SPEAK command may run only if all are true:

1. `run_forward_model_promotion_gate(...)` verdict is `promote_simulation_only`.
2. `run_speaking_policy_promotion_gate(...)` verdict is `promote_simulation_only`.
3. `run_speaking_readiness_gate(...)` verdict is `promote_simulation_only`.
4. The user explicitly invokes a live command; no existing command silently changes behavior.
5. The command records:
   - output device,
   - input device,
   - sample rate,
   - generated efference WAV path,
   - captured WAV path,
   - measured/aligned latency estimate,
   - suppression/tagging window.

---

## Proposed SPEAK state

Introduce a small explicit state machine, probably in a new module:

- `IDLE`
  - no queued utterance,
  - mic frames processed normally.

- `READY_TO_SPEAK`
  - speaking readiness gate passed,
  - utterance/efference prepared,
  - no playback yet.

- `SPEAKING`
  - speaker output active,
  - mic frames within the expected self-audio window are tagged as self-contaminated,
  - signal-level suppression may run against retained efference prediction.

- `POST_SPEAK_SUPPRESSION`
  - playback ended,
  - continue tagging for a short decay window to catch room echo / buffering tail.

- `RECOVERING`
  - normal listening resumes only after suppression window closes.

This should be represented as structured report state first, not hidden flags in `LiveListener`.

---

## Mic frame tagging policy

Every captured frame during live SPEAK mode must include explicit tags:

- `speak_state`: one of the states above,
- `speaker_output_active`: bool,
- `efference_id`: stable ID for the generated utterance/WAV,
- `expected_self_audio`: bool,
- `self_suppression_applied`: bool,
- `latency_aligned_ms`: measured/estimated alignment,
- `contamination_policy`: `tag_only` or `subtract_predicted_self`,
- `usable_for_world_learning`: bool.

Default first live slice should use `tag_only`, not subtraction, until latency is measured.

---

## Efference PCM retention

Before playback starts, the runtime must retain the exact generated output as efference:

- generated PCM array,
- generated WAV path,
- sample rate,
- duration,
- vowel/family/utterance metadata,
- forward-model parameters used for prediction.

Never reconstruct efference from captured mic audio. The efference source of truth is the generated output sent to the speaker.

---

## Latency measurement strategy

First live validation should estimate latency conservatively:

1. Generate an efference WAV with a short non-speech calibration transient followed by the test vocalization.
2. Play the WAV through the explicit output device.
3. Capture mic audio through the explicit input device.
4. Cross-correlate the calibration transient against capture to estimate roundtrip latency.
5. Use that estimate only for tagging windows in the first slice.
6. Do not claim accurate self-subtraction until the alignment is stable across repeated trials.

Acceptance target for initial latency smoke:

- latency estimate finite,
- repeated estimates within a documented tolerance,
- capture frames tagged with the latency used,
- no untagged frames during expected self-audio window.

---

## Feedback safety policy

First live SPEAK validation must be quiet and bounded:

- short generated output only,
- low amplitude default,
- fixed maximum duration,
- no looped playback,
- playback process must be waited/terminated,
- mic capture must end even if playback errors,
- report must include any playback/capture errors.

If playback starts but capture fails, the command must fail loudly and mark the trial unusable.

---

## Proposed first live command

Do not overload `--listen-live`.

Add a future explicit command such as:

```bash
PYTHONPATH=src python3 -m cora_infans --speak-live-smoke \
  --vowel a \
  --out-dir /tmp/cora_speak_smoke \
  --input-device :0 \
  --output-device-index 2 \
  --duration-ms 300 \
  --max-capture-ms 1500
```

The command name should make risk and scope obvious. It should not run unless the user explicitly asks for live SPEAK validation.

---

## First implementation slice after this doc

When implementation is approved, the first slice should still avoid policy-driven free speaking. It should be a bounded smoke harness:

1. Write RED tests for report shape with injectable fake playback/capture functions.
2. Implement a pure helper that builds the planned output/capture paths and tagging windows.
3. Implement fake playback/capture unit tests; no hardware in unit tests.
4. Add CLI with explicit command name.
5. Run unit tests.
6. Only then run one real hardware smoke if the user explicitly asks.

---

## Acceptance criteria before real live playback is considered safe enough to try

- `--promote-self-forward-model` passes.
- `--promote-speaking-readiness` passes.
- The live SPEAK design doc exists and is current.
- Unit tests cover SPEAK state transitions and frame tagging.
- Fake playback/capture smoke test produces generated and captured artifact paths.
- Real hardware command is explicit and bounded.
- Live report says whether it is `tag_only` or `subtract_predicted_self`.
- First live run defaults to `tag_only`.

---

## Out of scope for the first live slice

- Free-form conversational TTS.
- Learned speaking policy.
- PPO/distilled M8 policy.
- Continuous voice interaction.
- Automatic playback inside `--listen-live`.
- Claiming room-calibrated signal-level suppression.
- Importing og-cora's full live stack.

---

## Real-I/O implementation plan (the `--real-io` slice)

> **Status:** implemented (2026-07-09). Gated, bounded, `tag_only`. Default
> `--speak-live-smoke` (no flag) stays blocked/fake-I/O. `--real-io` runs only
> when `run_speaking_readiness_gate()` is promoted; otherwise it aborts before
> any I/O. Red→green TDD; unit tests use injected fakes (no hardware in CI).
> Items 1–5 of "First implementation slice" above are already implemented as M8-G
> (fake-I/O smoke helper) and M8-H (safe-by-default CLI). This section specifies the
> next, gated step: wiring the *real* CoreAudio output + XVF3800 capture behind an
> explicit flag, reusing existing primitives rather than inventing new ones.

### Reused existing primitives (no new dependencies)

- Capture: `LiveMicArray` (`src/cora_infans/io/live_mic.py`) — ffmpeg avfoundation
  device `:0`, 2 fused channels, fails loud via `LiveMicError`, retains `last_pcm`.
- Playback: `ParenteseStimulus.play(path)` (`src/cora_infans/io/live_listener.py`) —
  ffmpeg `audiotoolbox` sink to `STIMULUS_OUTPUT_DEVICE_INDEX = 2`, non-blocking
  `Popen`, raises `LiveMicError` on failure. We will add a thin `play_wav_to_device`
  wrapper that takes an arbitrary WAV (the generated efference) plus a device index,
  so we do not depend on the parentese synth.
- Latency: `estimate_playback_latency_ms` (`src/cora_infans/live_speak_smoke.py`) —
  deterministic numpy XCORR stub, already implemented, not yet called by any live path.
- Gates: `run_speaking_readiness_gate(...)` — already implemented; returns
  `simulated_speak_authorized` and `live_playback_allowed: false`.

### CLI shape

Extend the existing `--speak-live-smoke` command with a real-I/O mode:

```bash
PYTHONPATH=src python3 -m cora_infans --speak-live-smoke \
  --vowel a \
  --out-dir /tmp/cora_speak_smoke \
  --real-io \
  --input-device :0 \
  --output-device-index 2 \
  --duration-ms 300 \
  --max-capture-ms 1500
```

Behavior:

- Without `--real-io`: unchanged. Efference WAV only, `blocked_missing_injected_io`,
  no hardware touched.
- With `--real-io`:
  1. Run `run_speaking_readiness_gate()`. If verdict != `promote_simulation_only`
     (or any required sub-gate failed), abort before any I/O with
     `verdict=blocked_gates_not_promoted`, `real_hardware=false`. No speakers/mic.
  2. Synthesize efference WAV (as today), retain PCM + `efference_id`.
  3. Start capture (`LiveMicArray.capture_frame` or a bounded capture-to-WAV) on the
     input device **before/around** playback.
  4. Play the efference WAV to the output device via the `play_wav_to_device` wrapper.
  5. Stop capture after `max_capture_ms`; write `captured_wav`.
  6. Call `estimate_playback_latency_ms(efference_pcm=..., captured_pcm=..., sample_rate=...)`
     and record `latency_aligned_ms`.
  7. Emit `frame_tags` (READY → SPEAKING → POST_SPEAK_SUPPRESSION → RECOVERING) with
     `contamination_policy=tag_only`, `real_hardware=true`, and
     `live_playback_allowed` taken from the gate result (still `false` in slice 1;
     the field documents that free/looped playback is not authorized).
  8. Any playback/capture error sets `verdict=failed`, includes the error text, and
     marks the trial `usable=False`. Playback Popen is awaited/terminated; capture
     ends even on playback error.

### Safety gates baked in

- Default stays safe (no `--real-io` ⇒ no hardware).
- `--real-io` refuses to run unless the readiness gate promoted.
- Bounded: fixed `duration_ms`, fixed `max_capture_ms`, no loop, low amplitude default
  (reuse the volume ceiling already in `ParenteseStimulus.play`, or clamp amplitude).
- No simultaneous free speaking; this is a bounded smoke, not conversational TTS.
- Honest channel reporting: capture stays 2 fused channels, no 4-mic fabrication.
- `contamination_policy=tag_only` in the first real slice; no signal-level subtraction
  claim until latency is repeatable across trials.

### TDD sequence (what I will do once approved)

1. RED: extend `tests/test_live_speak_smoke.py` with a test that passes
   `--real-io` to the CLI but expects it to **refuse before I/O** when the gate is
   mocked to fail (we can't easily mock the gate from CLI; instead test the helper
   `run_real_io_speak_trial(...)` directly: if `readiness_gate` returns non-promote,
   it raises/`verdict=blocked_gates_not_promoted` and never calls playback/capture fns).
2. GREEN: implement `run_real_io_speak_trial(...)` in `live_speak_smoke.py` that takes
   `playback_fn`/`capture_fn` callables (so unit tests inject fakes) plus a
   `readiness_gate` callable; wires the real functions only when invoked from CLI.
3. Add `play_wav_to_device(path, device_index, sample_rate)` thin wrapper reusing
   ffmpeg audiotoolbox (mirror `ParenteseStimulus.play`).
4. Wire `--real-io` + `--input-device` + `--output-device-index` into
   `speak_live_smoke_cli` and `main` dispatch.
5. Unit tests with injected fake real-I/O fns assert report shape, `real_hardware=true`,
   `latency_aligned_ms` populated, tags present, gate-refusal path.
6. Full suite green (no real hardware exercised in CI).
7. Only after approval + green tests, run ONE real hardware smoke on request.

### Acceptance before the first real run

- Unit tests cover: gate-refusal (no I/O), happy real-I/O path with injected fakes,
  latency estimate finite, frame tags present, error path marks trial unusable.
- CLI `--real-io` without gate promotion fails loudly and touches no hardware.
- Report records input/output device, sample rate, efference + captured WAV paths,
  latency estimate, tagging window, `contamination_policy`, `real_hardware`,
  `live_playback_allowed`.

---

## Strategic recommendation

Proceed in this order:

1. Keep current M8 gates local and simulation-only.
2. Implement a bounded `--speak-live-smoke` harness with fake I/O tests first.
3. Run real hardware only after fake tests and explicit user consent.
4. Start with `tag_only` frame labeling.
5. Promote to signal-level subtraction only after latency is repeatable.

This preserves the project's honesty rule: demonstrate the mechanism step by step, do not fake biological depth, and do not let live playback contaminate the evidence chain.
