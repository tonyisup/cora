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

**Tests:** `tests/test_hardware_tuner.py` (7, mock-backend) + `tests/test_calibration.py`
(hardware-gated). Full suite after this work: **73 passed**.
