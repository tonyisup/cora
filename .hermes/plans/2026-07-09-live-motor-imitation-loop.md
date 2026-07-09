# Live Motor-Imitation Loop (Phase 1) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Close the vertical loop where Cora *hears a human speech sound, produces its own imitation of it, plays that imitation on the speaker, captures the corollary feedback, and measures imitation quality* — bounded, gated, measurement-only (no signal subtraction in Phase 1).

**Architecture:** The perception→production bridge already exists and is analysis-only: `_motor_imitation_trial` (in `io/live_listener.py`) analyzes a captured frame via `analyze_vowel`, maps to a motor target via `MotorPlanner.plan_from_analysis`, and re-synthesizes it (`resynth`) but **never plays it**. Phase 1 extends that trial to (a) actually play `resynth` to the speaker through a readiness gate, (b) capture the corollary feedback window from the mic, (c) align it with the **measured stable latency anchor** we already built (`calibration.json` / `canonical_frame_tags`), and (d) measure imitation quality by comparing the *perceived human* formants against the formants of the **played** sound (not the internal plan — per the project's "replace target with PLAYED sound" rule). Self-subtraction stays in `io/self_suppression.py` as simulation-only and is NOT wired to live audio in Phase 1.

**Tech Stack:** Python 3.13, numpy; `play_wav_to_device` / `capture_to_wav` (ffmpeg) already in `live_speak_smoke.py`; `MotorPlanner` + `analyze_vowel` in `io/motor_plan.py`; readiness-gate pattern from `speaking_simulator.run_speaking_readiness_gate`.

**Honesty boundaries (hard):**
- Bounded + gated: a readiness gate must pass (`promote_simulation_only`) before any playback. No looped/free playback.
- `contamination_policy: "tag_only"` — Phase 1 measures and tags; it does NOT perform signal-level self-subtraction on live audio. (`self_suppression.py` remains simulation-only.)
- "Learning" = recording prediction-error (perceived vs played formant delta). No weight updates, no biological overclaim.
- Playback target device = 2 (Headphones, audiotoolbox), capture = `:0` (ReSpeaker), matching the proven latency setup.

---

## Current context / assumptions
- `run_real_io_speak_trial` / `run_real_io_speak_trials` already do bounded real playback+capture and emit `latency_aligned_ms` + `canonical_frame_tags` when `latency_stable`. That machinery is the template for this loop's I/O.
- `live_speak_smoke.py` imports `play_wav_to_device` / `capture_to_wav`; `live_listener.py` does NOT yet import them.
- `analyze_vowel(pcm, sample_rate=16000)` returns `VowelAnalysis` with `.formant_hz` and `.nearest_vowel`. `MotorPlanner.plan_from_analysis(...)` → `VowelArticulation`; `plan_wav(vowel, duration_ms, sample_rate)` → int16/float32 PCM.
- The stable latency is ~0 ms at 4" near-field (measured, std <1.5 ms). Phase 1 may read an optional `--latency-anchor PATH` (the `calibration.json` we can now write) or accept `latency_ms` directly; default to 0.0 if absent.

---

## Step-by-step plan

### Task 1: Add a gated playback helper to `live_listener`
**Objective:** Give `LiveListener` a bounded, readiness-gated way to play a WAV to the speaker, reusing `play_wav_to_device`.

**Files:**
- Modify: `src/cora_infans/io/live_listener.py` (import + method)
- Test: `tests/test_live_listener.py` (create if absent)

**Step 1: Write failing test**
```python
def test_play_efference_requires_passed_gate():
    from cora_infans.io.live_listener import LiveListener
    lst = LiveListener(input_device=":0", output_device_index=2)
    # gate refuses -> no playback
    refused = lst.play_efference_if_authorized(
        wav_path="/tmp/x.wav", gate_fn=lambda: {"verdict": "hold", "live_playback_allowed": False}
    )
    assert refused["played"] is False
    assert refused["reason"] == "gate_not_promoted"
```

**Step 2: Run test to verify failure** → `FAIL` (method missing).

**Step 3: Write minimal implementation** (in `live_listener.py`, after imports add `from ..live_speak_smoke import play_wav_to_device`):
```python
def play_efference_if_authorized(self, *, wav_path: str, gate_fn, volume_db: float = 12.0) -> dict[str, object]:
    gate = gate_fn() if gate_fn is not None else {}
    if gate.get("verdict") != "promote_simulation_only":
        return {"played": False, "reason": "gate_not_promoted", "verdict": gate.get("verdict")}
    result = play_wav_to_device(wav_path, device_index=self.output_device_index, sample_rate=48000, volume_db=volume_db)
    return {"played": True, "reason": "authorized", "playback": result}
```

**Step 4: Run test to verify pass.**
**Step 5: Commit** `feat: add gated efference playback to LiveListener`.

### Task 2: Make `_motor_imitation_trial` actually play the imitation (bounded)
**Objective:** Extend the trial to write `resynth` to a WAV and play it through the gate, returning the played path + metadata. Keep analysis-only behavior when `play=False`.

**Files:**
- Modify: `src/cora_infans/io/live_listener.py` (`_motor_imitation_trial`)
- Test: `tests/test_live_listener.py`

**Step 1: Write failing test**
```python
def test_motor_imitation_trial_plays_when_authorized(tmp_path):
    from cora_infans.io.live_listener import LiveListener
    import numpy as np, wave
    lst = LiveListener(input_device=":0", output_device_index=2)
    pcm = (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2")
    played = {}
    def fake_play(path):
        played["path"] = path
        return {"backend": "fake", "completed": True}
    trial = lst._motor_imitation_trial(
        pcm, source_azimuth_deg=0.0, now_ms=0, frame_idx=0,
        play=True, gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=fake_play, out_dir=str(tmp_path),
    )
    assert trial is not None
    assert trial["played"] is True
    assert trial["efference_wav"] == played["path"]
```

**Step 2: Run → FAIL.**

**Step 3: Implement** — inside `_motor_imitation_trial`, after building `resynth`, add (when `play`):
```python
efference_wav = None
played = False
if play and out_dir is not None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    efference_wav = str(out / f"imit_{frame_idx}.wav")
    write_wav(efference_wav, resynth.astype("<i2") if resynth.dtype != np.int16 else resynth, sample_rate=16000)
    gate = gate_fn() if gate_fn is not None else {}
    if gate.get("verdict") == "promote_simulation_only":
        playback_fn(efference_wav)
        played = True
    else:
        played = False
```
Return these in the trial dict. Add `play: bool = False, gate_fn=None, playback_fn=None, out_dir: str | None = None` to the signature. Import `write_wav` (already imported in `motor_plan`? add `from .motor_plan import write_wav`).

**Step 4: Run → PASS.** **Step 5: Commit** `feat: motor-imitation trial plays efference when gated`.

### Task 3: Capture the corollary feedback window and align it with the latency anchor
**Objective:** After playback, capture a short mic window and align it to the known stable latency so we know which captured frames are "self".

**Files:**
- Modify: `src/cora_infans/io/live_listener.py` (capture + align in the trial, or a new `_capture_corollary` method)
- Test: `tests/test_live_listener.py`

**Step 1: Write failing test**
```python
def test_corollary_capture_aligned_to_latency(tmp_path):
    from cora_infans.io.live_listener import LiveListener
    lst = LiveListener(input_device=":0", output_device_index=2)
    captured = {}
    def fake_capture(path, *, sample_rate, duration_ms):
        captured["path"] = path
        return {"backend": "fake", "completed": True}
    out = lst.capture_corollary_window(
        out_dir=str(tmp_path), capture_fn=fake_capture,
        latency_ms=0.8, capture_ms=400,
    )
    assert out["captured"] is True
    assert out["latency_aligned_ms"] == 0.8
```

**Step 2: Run → FAIL.** **Step 3: Implement** `_capture_corollary_window` reusing `capture_to_wav` semantics (call the injected `capture_fn`). Store `latency_aligned_ms` (passed through; the actual XCORR alignment already exists in `live_speak_smoke.estimate_playback_latency_ms` and may be reused here for honesty — but in Phase 1 we trust the measured anchor; align by reporting it). **Step 4: Run → PASS. Step 5: Commit.**

### Task 4: Frame-tag the captured corollary window using `canonical_frame_tags`
**Objective:** Reuse the `canonical_frame_tags` builder so the captured feedback is classified (READY_TO_SPEAK / SPEAKING / POST_SPEAK_SUPPRESSION / RECOVERING) against the stable latency — proving "self" frames vs any residual "world" (human still talking).

**Files:**
- Modify: `src/cora_infans/io/live_listener.py`
- Test: `tests/test_live_listener.py`

**Step 1: Test** asserting a trial that plays + captures returns `frame_tags` whose `SPEAKING` window is tagged `expected_self_audio=True` and `latency_aligned_ms` equals the anchor.
**Step 2-4:** implement by calling `from ..live_speak_smoke import _frame_tags` (or promote it to a public `build_frame_tags`) with `latency_aligned_ms=latency_ms`. **Step 5: Commit.**

### Task 5: Measure imitation quality against the PLAYED sound
**Objective:** Re-analyze the *played* WAV (not the internal plan) and compute prediction error vs the perceived human vowel. This is the "replace target with PLAYED sound" rule.

**Files:**
- Modify: `src/cora_infans/io/live_listener.py` (`_motor_imitation_trial` returns `played_formants_hz`)
- Test: `tests/test_live_listener.py`

**Step 1: Test** — given a human frame analyzed as vowel "a" and a played wav that re-analyzes as "a", assert `played_formants_hz` is present and `imitation_prediction_error_hz` (abs delta of F1/F2) is recorded.
**Step 2-4:** after playback, `played_analysis = analyze_vowel(played_pcm, sample_rate=16000)`; record `played_formants_hz`, `played_nearest_vowel`, and `imitation_prediction_error_hz = [abs(p-w) for p,w in zip(played_analysis.formant_hz, feat.formant_hz)]`. **Step 5: Commit.**

### Task 6: CLI wiring — `--listen-live --with-motor-imitation --real-io`
**Objective:** Expose the loop through the CLI with the same gating as `--speak-live-smoke --real-io`.

**Files:**
- Modify: `src/cora_infans/cli.py` (handler for `--listen-live`: add `--real-io`, `--input-device`, `--output-device-index`, `--latency-anchor`, `--volume-db`; wire `playback_fn`/`capture_fn` to real ffmpeg when `--real-io`, else fakes)
- Test: `tests/test_cli.py`

**Step 1: Test** — `--listen-live --with-motor-imitation --real-io` without promoted gate returns `blocked_gates_not_promoted` and does zero playback (assert no WAV written / fake captures only). **Step 2-4:** implement; default `with_motor_imitation` keeps analysis-only unless `--real-io` + gate pass. **Step 5: Commit.**

### Task 7: Integration test (injected fakes, no hardware)
**Objective:** Prove the full perception→production→playback→capture→measurement chain runs end-to-end with injected fakes.

**Files:** `tests/test_live_listener.py`
- Drive `listen(duration_ms=..., with_motor_imitation=True, real_io=True, playback_fn=fake, capture_fn=fake, gate_fn=promoted)` and assert `motor_imitation.trial_count > 0`, each trial has `played=True`, `frame_tags`, and `imitation_prediction_error_hz`. Commit.

### Task 8: Real-hardware bounded smoke (YOU confirm before running)
**Objective:** Run the loop on real hardware, bounded, tag_only, no subtraction. Reports only.
- Command (after you say go): `PYTHONPATH=src python3 -m cora_infans --listen-live --duration-ms 4000 --frame-ms 250 --with-motor-imitation --real-io --input-device :0 --output-device-index 2 --latency-anchor /tmp/cora_nf/latency.json --near-field`
- Assert: human speech frames produce imitation trials; latency anchor loaded; `contamination_policy: "tag_only"`; no signal subtraction performed.

---

## Phase 2 (NOT now — explicitly deferred)
Real-hardware self-subtraction: reuse `io/self_suppression.py`'s `predict_self_pcm` / `spectral_subtract` with the **measured** latency (replace toy `delay_ms=10.0` with the anchor) to subtract Cora's own played audio from the captured mixture, isolating the human's continuing speech. Gated behind its own promotion milestone. Out of scope for this plan per the honesty rule ("demonstrate the mechanism, don't fake biological depth").

---

## Files likely to change
- `src/cora_infans/io/live_listener.py` (main: gated playback, capture, frame-tags, played-formant measurement)
- `src/cora_infans/cli.py` (`--listen-live` real-io wiring)
- `src/cora_infans/live_speak_smoke.py` (promote `_frame_tags` → `build_frame_tags` if needed; reuse `estimate_playback_latency_ms`)
- `tests/test_live_listener.py` (new), `tests/test_cli.py` (extend)
- `docs/PROJECT_STATUS_2026-07-09.md` (append Phase 1 results)

## Tests / validation
- `pytest tests -q` green throughout; new tests per task above; full suite currently 192 passed — must stay green.
- Real-hardware smoke only after your explicit go (Task 8).

## Risks / tradeoffs / open questions
- **Acoustic feedback during live playback:** playing on device 2 into a room while capturing `:0` risks the mic hearing Cora's own sound (the whole point — but also risk of loud re-injection). Mitigation: short bounded playback (≤300 ms), low-ish volume default, and the latency anchor + frame tags to separate self from world. This is exactly why Phase 1 is measurement-only.
- **Latency anchor freshness:** the calibration is geometry-dependent (4" near-field). If you move the speaker, re-run `--near-field --trials 6 --save-calibration`. Plan assumes the anchor is supplied or defaults to 0.0.
- **No biological claim:** the "imitation" is formant-target matching via the table-anchored `MotorPlanner`. We measure prediction error; we do not claim phonetic fidelity or learning.
- **Open:** should the loop auto-trigger only on human speech frames (VAD), or also on any salient frame? Plan assumes human-speech frames only (existing `source_azimuth is not None` + VAD gate), matching current code.
