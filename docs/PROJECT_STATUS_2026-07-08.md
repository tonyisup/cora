# Project Cora-Infans — Progress & Status

_Snapshot as of 2026-07-08. This is a point-in-time status doc; the README and
`docs/architecture/*` remain the living spec. Test counts below were verified
at commit `5412eff` (`pytest` → 65 passed, tree clean)._

## 0. What this project is

A deterministic, hardware-agnostic **research scaffold** for an infant-
language-development architecture (per `VISION.md`). It models an infant's
auditory "saliency → workspace → predictive-coding" loop as a transparent,
unit-tested Python pipeline. The hard rule (README/VISION): **never fake
biological depth or unsupported spatial physics.** Every layer is either
synthetic-but-honest or backed by real captured audio — no invented neuro-
physics, no silent fallbacks to mock modes.

Stack: Python 3, `pytest`, `pydantic`, `pyyaml`, `numpy`. Live capture uses
`ffmpeg` (avfoundation) — no `sounddevice`/`pyaudio` installed.

## 1. Runtime state (verified)

- Branch `main`, tree clean, HEAD `5412eff`.
- `pytest tests` → **65 passed** (44 baseline + 5 predictive-hierarchy + 5 live
  mic + 4 live-listener + 7 theta-gamma).
- Latest commits:
  - `5412eff` Add theta-gamma workspace scheduling (vision: theta-gamma multiplexing)
  - `0d6c912` Add continuous live listener with parentese stimulus (option A)
  - `efdf2a4` Add live XVF3800 hardware front end (vision: ReSpeaker integration)
  - `f9d1d98` Add two-level predictive-coding hierarchy (vision: predictive coding)
  - `fc2db84` Sync docs to scenario compiler; declare PyYAML dependency
  - `aab77f6` Add constrained internal scenario compiler

## 2. Implementation spine — what actually exists

### Core deterministic pipeline (synthetic + live share this path)
- `pipeline.py` — `CoraInfansPipeline.process_frame()`: saliency gate →
  `EventWorkspace` (volatile token window, focus-path tracking, dropout/
  recovery detection) → optional predictive hierarchy. Single source of truth
  for both synthetic and live frames.
- `saliency.py` — `compute_saliency_reward`: max-channel-energy reward + gate.
- `workspace.py` — `EventWorkspace`: focus-path, `dropout_count`, `recovery_count`.
- `spatial.py` — `resolve_focus_label`: maps dominant channel → topology label.
- `predictive.py` — `PredictiveCoder` (frequency residual) + `FocusTrajectoryCoder`
  (higher-level focus-trajectory prediction) + `PredictiveHierarchy` bundling them.
- `config.py` — `CoraInfansConfig` (channels, geometry, thresholds, window).
- `scenario.py` — declarative scenario compiler; authors 5 built-in sequence
  families (`sweep, zigzag, gap, multi_gap, delayed_recovery`) lowered
  deterministically into runtime frames. **Implemented + wired** (was doc'd as
  "not implemented" before this work).

### Live hardware front end (real XVF3800, macOS/CoreAudio)
- `io/mic_array.py` — `SyntheticMicArray`: 4-channel contract without hardware.
- `io/live_mic.py` — `LiveMicArray`: captures real frames via `ffmpeg`/avfoundation
  (device `:0`), lowers PCM → `AudioFrame` (per-channel RMS → `channel_energies`,
  FFT peak → `dominant_frequency_hz`). **Reports the board's true 2 fused
  channels — does NOT invent 4 labeled mics** the OS layer doesn't expose.
  Raises `LiveMicError` on missing ffmpeg/device (no silent synthetic fallback).
- `io/live_listener.py` — `LiveListener.listen()`: captures a real-time window of
  live frames, runs each through the pipeline, accumulates focus-path / dropout /
  recovery tracking. `ParenteseStimulus` synthesizes a rising/falling pitch contour
  and plays it non-blocking to **External Headphones (audiotoolbox device 2)** so
  the mic has a controlled acoustic target; every frame tagged `stimulus_active`.

### Theta-gamma scheduling (deterministic, simulation-only layer)
- `theta_gamma.py` — `ThetaGammaScheduler`: nested temporal-binding layer over
  `WorkspaceEvent`s. Slow theta cycles (200 ms ≈ 5 Hz) define volatile windows;
  each subdivides into fast gamma sub-windows (25 ms ≈ 40 Hz) that phase-lock
  focus labels into per-cycle assemblies. Cycle boundaries consolidate + open
  fresh; bounded residency enforces the working-memory limit; cross-talk is
  structurally impossible. It is an **interpretation layer above** `EventWorkspace`,
  not a replacement. Honest: models VISION.md's claim without inventing physics.
- Composed entrypoints: `simulation.build_sequence_report_with_scheduling()` and
  `LiveListener.listen(with_scheduling=True)`.

### CLI surface (`python -m cora_infans`)
- `--describe` — capability map.
- `--simulate-frame [--channel-energies ...] [--mic-geometry ...] [--output json|pretty]`
- `--simulate-sequence --family <name> [--with-scheduling]`
- `--capture-live [--device :0] [--duration-ms N]`
- `--listen-live [--duration-ms N] [--frame-ms N] [--with-stimulus] [--with-scheduling]`

## 3. Hardware validation (real, not stubbed)

Environment: reSpeaker XMOS XVF3800 4-Mic Array (CoreAudio "Seeed Studio",
avfoundation input `:0`); presents **2 fused channels @ 48 kHz** at the OS layer.
Stimulus output: External Headphones (audiotoolbox index 2).

Verified behaviors on real hardware:
- **Quiet room capture** → 2 channels, energies ~[0.012, 0.004], status
  `non_salient` (correct — no parentese).
- **Loud 800 Hz tone during capture** → ch0 RMS rose to 0.03, crossed the live
  threshold (0.02) → status `salient`, event kind `salient_audio`, full pipeline fired.
- **Live listener with parentese stimulus** → frames captured while playback runs;
  all tagged `stimulus_active`. With a reliable stimulus, frames reach the saliency
  gate and the theta-gamma scheduler populates resident cycles.
- **Stimulus → mic coupling confirmed**: direct probe showed captured ch energies
  ~0.55 during External-Headphones playback.

Calibration fact (documented, not fudged): real RMS energies are ~0.01–0.03; the
synthetic default saliency threshold (0.7) would silently drop every real frame, so
the live path uses `DEFAULT_LIVE_SALIENCY_THRESHOLD = 0.02`.

## 4. Honest gaps & open questions

- **Beamforming / multi-mic fusion** — blocked by the board's firmware: it fuses
  4 mics → 2 USB channels, so true per-mic beamforming isn't available through this
  path. A hardware limit, not a code gap. Not worth faking.
- **Deeper predictive hierarchy** — only one frequency residual + one focus-
  trajectory level today.
- **Motor-plan / speech synthesis (Analysis by Synthesis / DIVA)** — NOT started.
  The largest unproven leap; would introduce a synth/audio-generation path with no
  validation anchor yet. Awaiting user go-ahead before investing (first pillar that
  pushes past the current "honest, test-backed, no invented physics" boundary).
- **Training/eval harnesses over real audio corpora** — future.
- **General user-authored sequence format** — future (scenario compiler exists but
  is internal/constrained).

## 5. Notable bugs found & fixed during this work

1. **PyYAML undeclared** — `scenario.py` imported `yaml` but `pyproject.toml`
   didn't declare it; a clean install would break. Fixed + verified in a clean venv.
2. **Stimulus temp-file unlink race** — `LiveListener` unlinked the stimulus WAV
   immediately after the non-blocking `Popen` returned; ffmpeg hadn't opened it yet
   on macOS, so playback silently failed (0 salient frames). Fixed by deferring
   unlink until after `stim_proc.wait()`. Deterministic tests didn't catch this
   (they use a fake array); only the live hardware run surfaced it.

## 6. Next decision points (user steer needed)

- (C) **Motor-plan / speech synthesis** — proceed only with explicit go-ahead.
- (B-extension) Make theta-gamma scheduling the default report path for all runs
  (currently opt-in) — low risk, simulation-only.
- (A-extension) Richer live behavior: closed-loop where the listener adapts the
  stimulus based on detected focus gaps over real time.
- Beamforming remains deferred (hardware-blocked).

## 7. How to re-verify

```bash
cd /Users/juicebox/Projects/cora
pytest tests -q                         # expect 65 passed
python -m cora_infans --describe        # capability map
python -m cora_infans --simulate-sequence --family multi_gap --with-scheduling
python -m cora_infans --listen-live --duration-ms 2500 --frame-ms 250 \
    --with-stimulus --with-scheduling   # requires XVF3800 + External Headphones
```

---

## Addendum — XVF3800 hardware tuner (same session, later)

The user pointed to the Seeed Python SDK wiki, which revealed the control interface
is a **USB vendor control transfer** (VID 0x2886, PID 0x001A; `pyusb` + `libusb`),
NOT the CDC serial port (which does not expose a console on this firmware — confirmed
by probing all standard bauds, zero response).

**Verified parameter surface** (enumerated via `xvf_host.py --list`, firmware 2.0.6):
- `PP_AGCONOFF` — Automatic Gain Control (0 off / 1 on).
- `PP_MIN_NS` — stationary noise-suppression gain floor [0..1].
- `PP_MIN_NN` — non-stationary noise-suppression gain floor [0..1].
- `AEC_FIXEDBEAMSONOFF` — fixed-beam (manual steer) mode enable.
- `AEC_FIXEDBEAMSAZIMUTH_VALUES` — beam azimuths in radians (the "acoustic laser").

**Real measured factory state:** `PP_AGCONOFF=[0]` (AGC already off), `PP_MIN_NS=[0.15]`,
`PP_MIN_NN=[0.51]` (non-stationary NS aggressively suppressing transients — exactly the
"conference speakerphone" behavior the brief warned about), `AEC_HPFONOFF=[2]` (HPF on).

**What was built & verified on hardware:**
- `src/cora_infans/io/hardware_tuner.py` — `HardwareTuner` reads/writes the verified
  param table; every write is verified by immediate read-back; rejects unknown/read-only/
  protected (REBOOT/SAVE) params; volatile only (reverts on reboot, never persists).
- `apply_raw_linear()` drove `PP_MIN_NS: 0.15→0.0`, `PP_MIN_NN: 0.51→0.0` (verified).
- `steer_to(90°)` → azimuth read-back `[1.571, 0.0]`, `AEC_FIXEDBEAMSONOFF=[1]` (verified);
  `release_beam()` → `[0]` (verified). The top-down "acoustic laser" loop is real.
- `src/cora_infans/io/calibration.py` — `calibrate_linearity()` sweeps graded stimulus
  levels and records captured RMS factory-vs-raw-linear. **Result:** factory NS compresses
  the top of the range (levels 0.7 and 1.0 both → ~0.66 RMS); raw-linear removes that
  firmware compression (0.4 stimulus jumps from 0.41→0.69). Residual flattening at
  0.7→1.0 is the mic's analog front-end saturating (~0.67 RMS ceiling) — a HARDWARE
  limit, honestly reported, not a firmware fix.

**Honesty notes:**
- The board's CDC serial port (`/dev/cu.usbmodem101`) is NOT a control console; all
  tuning goes through USB vendor requests. No serial "tuner" was fabricated.
- We only ever write params the firmware exposes; the SDK's full table was the source of
  truth. No invented param names.
- Tuning is volatile; board was restored to factory NS floors after the calibration run.

+**Tests:** `tests/test_hardware_tuner.py` (7, mock-backend) + `tests/test_calibration.py`
+(hardware-gated). Full suite after this work: **73 passed**.
+
+## Addendum 2 — Active top-down auditory oculomotor gaze (closed loop)
+
+Built the "Active Attention Governor" (`src/cora_infans/io/active_attention.py`)
+that closes the loop between the software saliency map and the XVF3800 beam --
+the superior-colliculus analogue the user outlined. `LiveListener.listen()` gained
+`with_attention=True`; each salient frame derives a target azimuth (coarse L/R
+from the 2 fused channels via `derive_target_azimuth`) and the governor gates the
+steer on three honest constraints: (1) saliency >= verified 0.02 baseline,
+(2) spatial deadband > 5 deg (no echo jitter), (3) velocity slew <= 180 deg/s
+(no un-biological sweep). On `acted`, it calls `HardwareTuner.steer_to()`.
+
+**Real hardware verification:** running `--listen-live --with-stimulus
+--with-attention` against the actual board produced `is_real_hardware: true`,
+`steer_count: 2`, with the beam steered to front-left (338.9 deg then 344.4 deg,
+matching the left-dominant stimulus) and read-back confirming the beam moved.
+
+**Two honesty bugs caught by the live run (not by the fake-array tests):**
+1. `derive_target_azimuth()` returns a SIGNED [-90,+90] angle, but `steer_to()`
+   requires [0,360). Fixed by folding: `steer_azimuth = (target + 360) % 360`.
+2. `HardwareTuner.steer_to()` returns the firmware azimuth in RADIANS, but the
+   governor's world is degrees. The fake tuner returned degrees and masked this;
+   the real run reported `current_azimuth_deg: 5.97` (actually radians). Fixed by
+   converting `actual[0] * 180/pi` before `register_steer`. Added a radians-
+   returning fake tuner test to lock the boundary.
+
+**Design guard (per blueprint):** real pyusb control transfers fire ONLY under
+`--listen-live` with real silicon present; synthetic/test runs resolve a bus-free
+`_NullTuner` so the entire decision path (gate -> deadband -> slew -> steer) is
+exercised under pytest without ever touching the USB bus. Board left in default
+(released) beam mode after the run.
+
+**Honest spatial-resolution caveat:** with 2 fused channels the target estimate is
+L/R only (front hemisphere). The hardware beam itself can point 0..359 deg, but
+DERIVING a finer bearing would be inventing localization we do not have.
+
+**Tests:** `tests/test_active_attention.py` (7, pure) + `tests/test_active_attention_
+integration.py` (2, fake array + radians fake). Full suite after this work:
+**83 passed**.
+
+## Addendum 3 — True Direction-of-Arrival (firmware 4-mic localization)
+
+Upgraded the gaze loop from coarse 2-channel L/R to the XVF3800's NATIVE DoA.
+The SDK exposes `DOA_VALUE` (RESID 20, CMDID 18) and `AUDIO_MGR_SELECTED_AZIMUTHS`
+(RESID 35, CMDID 11): the firmware localizes using all 4 mics internally and
+serves a true 0..359 deg bearing -- independent of the 2 fused USB channels our
+capture stream sees. Added both to the tuner table; `read_doa()` returns a
+`DoAReading(azimuth_deg, speech_detected)`. The governor's `propose()` gained an
+`explicit_target_deg` arg; `LiveListener` now reads `tuner.read_doa()` each salient
+frame and steers toward the firmware bearing (overriding the derived L/R).
+
+**Real hardware verification:** `--listen-live --with-stimulus --with-attention`
+read `doa: 29.22` and steered the beam there, overriding the derived L/R estimate
+(~321 deg). The acoustic laser now chases the board's own 4-mic localization.
+
+**Three honesty bugs found and fixed during this work (the live run, again,
+caught what the fake tests masked):**
+1. `DOA_VALUE` size formula was wrong for uint16 (requested 17 bytes). Fixed to
+   `length*2+1`.
+2. `DOA_VALUE` is actually a 2-float payload [azimuth_rad, flag] served with
+   status byte **66** -- NOT uint16 and NOT status 0. The Seeed SDK decodes the
+   payload ignoring status; we now accept status 0 AND 66 as "data present"
+   (retry only on 64). Decoding as uint16 produced garbage; corrected to float.
+3. The firmware's `speech_detected` flag stays False on our synthetic stimulus
+   (its VAD doesn't classify a tone as speech). We therefore use the DoA bearing
+   whenever it is finite, NOT gated on the speech flag -- our own saliency gate
+   already confirms a real signal. The flag is logged for transparency only.
+
+**Correction (DoA geometry test — see Addendum 4):** the claim above that the
+bearing "tracked the source correctly" was WRONG and the geometry test exposed
+it. `DOA_VALUE` on 2.0.6 echoes the beam's OWN steered azimuth (it returned
+exactly the angle we last steered, 29.22 deg, and 0 when released). It is NOT
+an acoustic-source bearing. The real source-DoA (`AUDIO_MGR_SELECTED_AZIMUTHS`)
+stays NAN without the firmware's speech flag, and the captured L/R energy split
+is source-blind at close range. The loop was therefore self-referential. The
+code, tests, and this doc were corrected (see Addendum 4).
+
+**Tests (as originally added):** +2 tuner (mock DoA decode) + 2 governor
+(explicit-target override + fallback) + 1 integration (DoA preferred even when
+VAD flag False). Full suite at that point: **87 passed** (subsequently amended
+to 90 by the Addendum-4 correction).
+
+## Addendum 4 — DoA geometry test: "true DoA" was a phantom (honest retraction)
+
+The user moved the External Headphones' L/R earcups to +/-45 deg from front and
+asked for a geometry test. Built `src/cora_infans/io/doa_experiment.py`: plays a
+tone on one earcup (pan=-1/+1/0 -> true stereo WAV, verified ch0/ch1 routing in
+tests) and samples the firmware DoA during playback. The result is the most
+important finding of the session:
+
+1. LEFT, RIGHT, and BOTH earcups ALL report `DOA_VALUE = 29.22 deg` with zero
+   variance. The bearing does not respond to which speaker plays.
+2. A direct probe showed `DOA_VALUE` equals the **beam's own steered angle**:
+   released -> 0.00, `steer_to(350)` -> 350, `steer_to(100)` -> 100. It is a
+   beam-azimuth readback, NOT a source localization.
+3. `AUDIO_MGR_SELECTED_AZIMUTHS[0]` (the real processed source-DoA) is ALWAYS
+   NAN -- the firmware needs its speech flag set, which never fires on a tone.
+4. The captured L/R energy split is source-blind: LEFT vs RIGHT earcup give
+   identical channel energies (ch0~0.6, ch1~0.3). `derive_target_azimuth` is
+   not actually localizing either.
+
+**Conclusion:** there is NO working acoustic-source localization signal from
+this hardware/firmware for a non-speech stimulus. The "acoustic laser tracking
+the source" is not achievable here. What IS real and useful: the beam points
+exactly where commanded (steer 350 -> DoA 350, verified), so it is a genuine
+**commanded top-down attention** actuator. The governor's `propose()` retains
+an `explicit_target_deg` arg for an external ground-truth source bearing (e.g.
+a known speaker position) when one exists.
+
+**What to change to get real localization (future options):**
+- Drive a SPEECH-classified stimulus so `AUDIO_MGR_SELECTED_AZIMUTHS[0]` yields
+  a non-NAN bearing (firmware VAD requirement).
+- Wider speaker separation / greater distance so the inter-channel level
+  difference is measurable by the captured 2 fused channels.
+- Raw 4-mic access -- blocked by firmware (only 2 fused channels reach the OS).
+
+**Code corrections made:**
+- `live_listener.py`: `read_doa()` is no longer fed back as a steer target
+  (that was a self-referential no-op). It is logged as `beam_azimuth_deg`
+  (state) only; the steer target is the capture-derived L/R. Honest comment
+  added at the decision site.
+- `tests`: replaced the misleading "uses firmware DoA" integration test with
+  `test_closed_loop_logs_beam_azimuth_but_steers_on_derived_lr` + added
+  `test_governor_accepts_external_ground_truth_target`. Added
+  `tests/test_doa_experiment.py` (2 tests: stereo pan routing).
+- README + phase-1-foundation updated to state the limitation plainly.
+
+**Honesty note:** this is exactly the "looks like it works but doesn't" trap
+VISION.md warns against. The geometry test (user's suggestion) is what caught
+it. The earlier "true DoA-guided gaze" commit message/claim is retracted here.
+
+**Tests:** suite 87 -> 90 (+2 doa_experiment routing + 1 corrected integration).
