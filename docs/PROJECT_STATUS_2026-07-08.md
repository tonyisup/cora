# Project Cora-Infans — Progress & Status

_Snapshot started on 2026-07-08 and extended by same-day addenda. The README and
`docs/architecture/*` remain the living spec. Current audit baseline: commit
`7650b84` plus the offline corpus-evaluation harness, `pytest tests -q` → 102 passed._

## 0. What this project is

A deterministic, hardware-agnostic **research scaffold** for an infant-
language-development architecture (per `VISION.md`). It models an infant's
auditory "saliency → workspace → predictive-coding" loop as a transparent,
unit-tested Python pipeline. The hard rule (README/VISION): **never fake
biological depth or unsupported spatial physics.** Every layer is either
synthetic-but-honest or backed by real captured audio — no invented neuro-
physics, no silent fallbacks to mock modes.

Stack: Python 3, `pytest`, `pydantic`, `pyyaml`, `numpy`, `pyusb`, and
`libusb-package`. Live capture uses `ffmpeg` (avfoundation) — no
`sounddevice`/`pyaudio` installed.

## 1. Runtime state (verified)

- Branch `main`, current checked HEAD before this addendum: `fd85c15` (ahead of
  `origin/main` by speech-source-DoA probe/wiring commits).
- `pytest tests -q` → **96 passed** (current validation after active-attention,
  hardware-tuner, calibration, DoA-geometry, speech-source-DoA, and live
  source-DoA attention wiring/raw beam azimuth tests).
- Latest relevant commits:
  - `fd85c15` Wire speech source DoA into live active attention
  - `84b91a2` Add speech-classified source-DoA probe for XVF3800
  - `0f56139` Audit project status and correct DoA verdict surfaces
  - `8d22964` DoA geometry test: correct the 'true DoA' claim (beam-azimuth echo, not source)
  - `0d7bfd1` Add true Direction-of-Arrival to the gaze loop (firmware 4-mic localization) — **retracted by 8d22964**
  - `d5f3aa8` Add active top-down auditory oculomotor gaze (closed loop)
  - `8156692` Add XVF3800 hardware tuner + linearity calibration (real USB control)
  - `54e6cb2` Add standalone project status/progress doc (2026-07-08)
  - `5412eff` Add theta-gamma workspace scheduling

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
- `--listen-live [--duration-ms N] [--frame-ms N] [--with-stimulus] [--with-scheduling] [--with-attention]`

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
pytest tests -q                         # expect 90 passed
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

## Addendum 2 — Active top-down auditory oculomotor gaze (closed loop)

Built the "Active Attention Governor" (`src/cora_infans/io/active_attention.py`)
that closes the loop between the software saliency map and the XVF3800 beam --
the superior-colliculus analogue the user outlined. `LiveListener.listen()` gained
`with_attention=True`; each salient frame derives a target azimuth (coarse L/R
from the 2 fused channels via `derive_target_azimuth`) and the governor gates the
steer on three honest constraints: (1) saliency >= verified 0.02 baseline,
(2) spatial deadband > 5 deg (no echo jitter), (3) velocity slew <= 180 deg/s
(no un-biological sweep). On `acted`, it calls `HardwareTuner.steer_to()`.

**Real hardware verification:** running `--listen-live --with-stimulus
--with-attention` against the actual board produced `is_real_hardware: true`,
`steer_count: 2`, with the beam steered to front-left (338.9 deg then 344.4 deg,
matching the left-dominant stimulus) and read-back confirming the beam moved.

**Two honesty bugs caught by the live run (not by the fake-array tests):**
1. `derive_target_azimuth()` returns a SIGNED [-90,+90] angle, but `steer_to()`
   requires [0,360). Fixed by folding: `steer_azimuth = (target + 360) % 360`.
2. `HardwareTuner.steer_to()` returns the firmware azimuth in RADIANS, but the
   governor's world is degrees. The fake tuner returned degrees and masked this;
   the real run reported `current_azimuth_deg: 5.97` (actually radians). Fixed by
   converting `actual[0] * 180/pi` before `register_steer`. Added a radians-
   returning fake tuner test to lock the boundary.

**Design guard (per blueprint):** real pyusb control transfers fire ONLY under
`--listen-live` with real silicon present; synthetic/test runs resolve a bus-free
`_NullTuner` so the entire decision path (gate -> deadband -> slew -> steer) is
exercised under pytest without ever touching the USB bus. Board left in default
(released) beam mode after the run.

**Honest spatial-resolution caveat:** with 2 fused channels the target estimate is
L/R only (front hemisphere). The hardware beam itself can point 0..359 deg, but
DERIVING a finer bearing would be inventing localization we do not have.

**Tests:** `tests/test_active_attention.py` (7, pure) + `tests/test_active_attention_
integration.py` (2, fake array + radians fake). Full suite after this work:
**83 passed**.

## Addendum 3 — Attempted firmware DoA integration (later retracted)

This addendum records an intermediate interpretation that was useful but ultimately
wrong: after decoding `DOA_VALUE` as a 2-float payload served with status byte 66,
the loop briefly treated that value as a firmware-provided acoustic source bearing.
That conclusion was later falsified by the geometry test in Addendum 4.

What remains useful from this work:
- the wire-format discovery is real: `DOA_VALUE` is a 2-float payload
  `[azimuth_rad, flag]` on firmware 2.0.6, not uint16.
- status byte 66 can carry data; status 64 is retry.
- `ActiveAttentionGovernor.propose(explicit_target_deg=...)` remains useful for an
  **external** ground-truth target bearing.

What was retracted:
- `DOA_VALUE` is NOT accepted as acoustic-source localization evidence. It echoes
  beam azimuth state (proved in Addendum 4).
- `LiveListener` no longer feeds `read_doa()` back as a steer target.

**Correction (DoA geometry test — see Addendum 4):** the claim above that the
bearing "tracked the source correctly" was WRONG and the geometry test exposed
it. `DOA_VALUE` on 2.0.6 echoes the beam's OWN steered azimuth (it returned
exactly the angle we last steered, 29.22 deg, and 0 when released). It is NOT
an acoustic-source bearing. The real source-DoA (`AUDIO_MGR_SELECTED_AZIMUTHS`)
stays NAN without the firmware's speech flag, and the captured L/R energy split
is source-blind at close range. The loop was therefore self-referential. The
code, tests, and this doc were corrected (see Addendum 4).

**Tests (as originally added):** +2 tuner (mock DoA decode) + 2 governor
(explicit-target override + fallback) + 1 integration (DoA preferred even when
VAD flag False). Full suite at that point: **87 passed** (subsequently amended
to 90 by the Addendum-4 correction).

## Addendum 4 — DoA geometry test: "true DoA" was a phantom (honest retraction)

The user moved the External Headphones' L/R earcups to +/-45 deg from front and
asked for a geometry test. Built `src/cora_infans/io/doa_experiment.py`: plays a
tone on one earcup (pan=-1/+1/0 -> true stereo WAV, verified ch0/ch1 routing in
tests) and samples the firmware DoA during playback. The result is the most
important finding of the session:

1. LEFT, RIGHT, and BOTH earcups ALL report `DOA_VALUE = 29.22 deg` with zero
   variance. The bearing does not respond to which speaker plays.
2. A direct probe showed `DOA_VALUE` equals the **beam's own steered angle**:
   released -> 0.00, `steer_to(350)` -> 350, `steer_to(100)` -> 100. It is a
   beam-azimuth readback, NOT a source localization.
3. `AUDIO_MGR_SELECTED_AZIMUTHS[0]` (the real processed source-DoA) is ALWAYS
   NAN -- the firmware needs its speech flag set, which never fires on a tone.
4. The captured L/R energy split is source-blind: LEFT vs RIGHT earcup give
   identical channel energies (ch0~0.6, ch1~0.3). `derive_target_azimuth` is
   not actually localizing either.

**Conclusion:** there is NO working acoustic-source localization signal from
this hardware/firmware for a non-speech stimulus. The "acoustic laser tracking
the source" is not achievable here. What IS real and useful: the beam points
exactly where commanded (steer 350 -> DoA 350, verified), so it is a genuine
**commanded top-down attention** actuator. The governor's `propose()` retains
an `explicit_target_deg` arg for an external ground-truth source bearing (e.g.
a known speaker position) when one exists.

**What to change to get real localization (future options):**
- Drive a SPEECH-classified stimulus so `AUDIO_MGR_SELECTED_AZIMUTHS[0]` yields
  a non-NAN bearing (firmware VAD requirement).
- Wider speaker separation / greater distance so the inter-channel level
  difference is measurable by the captured 2 fused channels.
- Raw 4-mic access -- blocked by firmware (only 2 fused channels reach the OS).

**Code corrections made:**
- `live_listener.py`: `read_doa()` is no longer fed back as a steer target
  (that was a self-referential no-op). It is logged as `beam_azimuth_deg`
  (state) only; the steer target is the capture-derived L/R. Honest comment
  added at the decision site.
- `tests`: replaced the misleading "uses firmware DoA" integration test with
  `test_closed_loop_logs_beam_azimuth_but_steers_on_derived_lr` + added
  `test_governor_accepts_external_ground_truth_target`. Added
  `tests/test_doa_experiment.py` (2 tests: stereo pan routing).
- README + phase-1-foundation updated to state the limitation plainly.

**Honesty note:** this is exactly the "looks like it works but doesn't" trap
VISION.md warns against. The geometry test (user's suggestion) is what caught
it. The earlier "true DoA-guided gaze" commit message/claim is retracted here.

**Tests:** suite 87 -> 90 (+2 doa_experiment routing + 1 corrected integration).

## Addendum 5 — Speech-classified source-DoA is recoverable

After Addendum 4 showed that tone/non-speech localization was negative, the next
experiment tested the firmware's likely real source-DoA path with a speech-like
stimulus. Built a `SpeechStimulus` in `src/cora_infans/io/doa_experiment.py`:
macOS `/usr/bin/say` synthesizes TTS speech, `ffmpeg` converts/pans it into a
stereo WAV, then the existing audiotoolbox output path plays it to one earcup.
The probe now records both:
- `DOA_VALUE` / `read_doa()` — beam-azimuth state readback.
- `AUDIO_MGR_SELECTED_AZIMUTHS[0]` / `read_selected_azimuths()` — processed
  source-DoA candidate in radians.

**Real hardware result (External Headphones still placed about +/-45 deg from
the previous center):**
- LEFT speech: firmware VAD fired (`speech_hits=45`), selected source samples=41,
  `selected_source_mean_deg=133.39`.
- RIGHT speech: firmware VAD fired (`speech_hits=45`), selected source samples=40,
  `selected_source_mean_deg=213.54`.
- BOTH speech: firmware VAD fired (`speech_hits=45`), selected source samples=41,
  `selected_source_mean_deg=169.70` with mixed/transitioning readings.

**Verdict update:** the Addendum-4 negative result is still true for tones and
non-speech: `DOA_VALUE` is not source localization and processed source-DoA stays
NAN without firmware speech detection. But source localization IS recoverable on
this silicon when the stimulus is speech-classified by the firmware VAD, via
`AUDIO_MGR_SELECTED_AZIMUTHS[0]`. The left/right separation (~80 deg in means,
with stable plateaus around ~133 deg and ~225 deg) is a real geometry signal.

**Important boundary at this point:** the live attention loop did not yet consume
this processed source-DoA as its target; that wiring is added in Addendum 6.

**Tests:** +3 DoA experiment tests (speech stimulus command construction,
right-pan filter, processed source-DoA reporting). Suite 90 -> 93.

## Addendum 6 — Speech-source DoA is now wired into active attention

Wired the live listener's attention loop to consume finite processed source-DoA:
- `DOA_VALUE` / `read_doa()` remains beam-azimuth state only and is logged as
  `beam_azimuth_deg`.
- `AUDIO_MGR_SELECTED_AZIMUTHS[0]` / `read_selected_azimuths()[0]` is converted
  from radians to degrees; if finite, it is passed to
  `governor.propose(explicit_target_deg=...)` and logged as `source_azimuth_deg`.
- If the processed source-DoA is NAN/unavailable, the listener falls back to the
  coarse capture-derived L/R azimuth.

**Real hardware smoke (`LiveListener(stimulus=SpeechStimulus()).listen(...,
with_stimulus=True, with_attention=True)`):**
- `frame_count=12`, `salient_frame_count=6`.
- `is_real_hardware=True`.
- `selected_source_used_count=6`.
- source azimuths: `[185.17, 184.41, 184.41, 184.41, 184.41, 184.41]`.
- derived fallback azimuths in the same run were around `[342.47, 331.0, ...]`,
  proving the actual target came from speech source-DoA rather than fused-channel
  L/R fallback.
- `steer_count=1`; subsequent source samples were within the 5 deg deadband.
- Beam was explicitly released after the run: `release_beam [0]`.

**Tests:** +2 integration tests: finite processed source-DoA overrides derived
L/R; NAN processed source-DoA falls back to derived L/R. Suite 93 -> 95.

## Addendum 7 — XMOS raw beam azimuth data inspected

Inspected XMOS XVF3800 v3.2.1 host-application documentation, specifically
"Beam forming subsystem and Direction of Arrival indicator". The useful missing
distinction is:
- `AEC_AZIMUTH_VALUES`: four raw beam azimuths in radians — focused beam 1,
  focused beam 2, free-running beam, auto-select beam.
- `AEC_SPENERGY_VALUES`: speech-energy values for the same four beams. Non-zero
  means that beam probably contains speech.
- `AUDIO_MGR_SELECTED_AZIMUTHS`: two processed azimuths — index 0 is the
  post-processed speaker/source azimuth or NAN when no speech is selected; index
  1 is the current auto-select beam azimuth.

Added read-only `HardwareTuner` helpers:
- `read_beam_azimuths()` returns named raw beam azimuths in radians and degrees.
- `read_beam_speech_energies()` returns named beam speech-energy values.
- `snapshot()` now includes `AEC_AZIMUTH_VALUES` and `AEC_SPENERGY_VALUES`.

**Real board read (quiet / no active speech stimulus):**
- raw beam azimuths were finite:
  - focused beam 1: ~184.37 deg
  - focused beam 2: ~146.55 deg
  - free-running beam: ~163.48 deg
  - auto-select beam: ~163.48 deg
- raw beam speech energies were all `0.0`.
- `AUDIO_MGR_SELECTED_AZIMUTHS[0]` was NAN, while index 1 / auto-select beam was
  ~163.73 deg.

**Verdict refinement:** raw beam azimuths are always useful telemetry, but not by
themselves a single-speaker source verdict. For attention targeting, the correct
hierarchy remains: use finite `AUDIO_MGR_SELECTED_AZIMUTHS[0]`; otherwise fall
back to capture-derived L/R. Use `AEC_AZIMUTH_VALUES` + `AEC_SPENERGY_VALUES` for
debugging why the processed source did or did not become finite.

**Tests:** +1 hardware-tuner mock test for raw beam azimuths and speech energies.
Suite 95 -> 96.

## Addendum 8 — Offline corpus + evaluation harness

Added a deterministic, hardware-free validation path so the scaffold can score
recorded (or simulated) audio against human/Audacity-compatible label CSVs
without touching the XVF3800.

- `src/cora_infans/io/offline_eval.py`
  - `CorpusLoader.load_labels(csv)` — reads an Audacity-style manifest
    (`clip_id,start_s,end_s,label`; rows aggregate by `clip_id` into one
    `CorpusClip` with all its `Annotation`s). Skips clips whose `<id>.wav` is
    missing (no fabricated audio).
  - `OfflineEvalHarness` — scores frame-level saliency against labeled
    intervals. Label semantics: a labeled interval is a ground-truth acoustic
    EVENT; unlabeled regions are ground-truth silence. It is a saliency/event-
    presence scorer, not a label classifier, so the scoreboard is independent
    of label vocabulary.
  - `ClipReport` dataclass carries per-clip counts + `accuracy`.
- `src/cora_infans/io/synthetic_stream.py` — `build_synthetic_corpus_frames()`
  deterministic frame provider (one ON + one quiet frame per labeled interval)
  so the harness runs end-to-end without real WAV decoding. Real audio decoding
  is explicitly left to the separate recording project.
- CLI `python3 -m cora_infans --eval-corpus --root DIR --label-csv FILE
  [--saliency-threshold F] [--frame-ms N]` returns an aggregate report.
- `describe()` now lists `offline_corpus_evaluation` under implemented
  subsystems.

**Validation:** `pytest tests -q` → 102 passed (was 96; +5 offline-eval unit
tests +1 CLI smoke). CLI smoke on a 2-interval stub corpus returned a coherent
aggregate report (`clip_count=1`, `total_frames=4`, event-matched scoring).

**Honesty note:** this is a scoreboard scaffold, not yet wired to real decoded
audio. The `frames_for` seam is the integration point for the recording
project's WAV→frame extraction.

## Addendum 9 — Theta-gamma promoted to default report path

Theta-gamma scheduling (deterministic nested theta-cycle / gamma-subwindow
temporal binding) is now part of the default report path for every run, not
just when `--with-scheduling` is passed.

- `build_sequence_report(..., with_scheduling=True)` now always feeds salient
  events into a `ThetaGammaScheduler` and attaches `theta_gamma` to the report.
  `with_scheduling=False` restores the bare flat-workspace view.
- `build_sequence_report_with_scheduling` is retained as a thin alias (band
  params now ignored; the default scheduler is used).
- `LiveListener.listen(..., with_scheduling=True)` — default flipped from False
  to True; the live report's `theta_gamma` summary is populated unless a caller
  explicitly opts out.
- CLI `--simulate-sequence` now emits `theta_gamma` by default; the
  `--with-scheduling` flag is kept for explicitness but is no longer required.
- README / architecture doc updated: the "not yet default" caveat is removed.

**Validation:** `pytest tests -q` → green. Added assertions that
`build_sequence_report` includes `theta_gamma` by default and can still omit it
via `with_scheduling=False`; the CLI sequence smoke asserts `theta_gamma` is
present. No regressions in the 102 prior tests.

## Addendum 10 — Option C: motor-plan / analysis-by-synthesis loop

Implemented the perception→production bridge VISION.md calls "Analysis by
Synthesis": a heard vowel is mapped into internal motor coordinates and
re-synthesized. Per the project's honesty rule, this is a *mechanism* with a
fixed, documented formant map — NOT a learned or biologically-faithful vocal
tract.

- `src/cora_infans/io/motor_plan.py`
  - `VOWEL_FORMANTS_HZ` / `ANALYSIS_BY_SYNTHESIS_TABLE` — fixed adult-speaker
    formant targets (F1/F2/F3, Hillenbrand et al. 1995). Production and
    perception share this one coordinate system (the "motor coordinate
    trajectory").
  - `VowelArticulation` / `FormantTarget` — the internal motor-coordinate
    representation of a vowel; `from_vowel()` is an exact table lookup.
  - `synthesize_vowel()` — parallel-formant source-filter toy synthesizer: a
    band-limited glottal pulse drives three resonators (F1/F2/F3) in parallel,
    followed by a lip-radiation high-pass. Returns a float32 mono WAV. Explicitly
    a plausible vowel generator for the loop, not a claim of vocal-tract fidelity.
  - `analyze_vowel()` — real LPC spectral-envelope formant estimator over the
    waveform; maps the estimated formants to the nearest vowel via the table.
    F1 is the reliably-resolved feature; F2/F3 enter only as a small tie-breaker.
  - `MotorPlanner` — the DIVA-style closure: `plan_from_analysis()` maps a
    perceived vowel to its table-anchored motor plan; `plan_wav()` re-synthesizes
    that target. `write_wav()` dumps float32 → 16-bit PCM.
- CLI `python3 -m cora_infans --synthesize-vowel a --out /tmp/a.wav
  [--duration-ms N] [--sample-rate N] [--f0-hz F] [--analyze]`. With `--analyze`
  it reads the written WAV back, runs perception, and re-derives the motor plan,
  closing the loop in one command.
- `io/__init__.py` now exports the motor-plan surface; `describe()` lists
  `motor_plan_synthesis` under implemented subsystems.

**Validation:** `pytest tests -q` → 114 passed (was 104; +9 `test_motor_plan.py`
unit tests +1 CLI smoke). End-to-end CLI run produced `/tmp/cora_vowel_a.wav`
(4800 samples @ 16 kHz), wrote the formant target, read it back, and re-derived a
motor plan — the analysis-by-synthesis loop executes without hardware.

**Honest boundary (do not overclaim):** the toy source-filter synthesizer
resolves F1 robustly, but its LPC envelope does not separate F2/F3 cleanly, so
perfect 5-vowel acoustic recovery is NOT claimed. The loop is *demonstrated*
(end-to-end synthesis→perception→re-plan), and the table-anchored inverse map is
genuinely correct; the acoustic fidelity ceiling is documented in the module
docstring and tests. This matches the VISION.md directive: build the mechanism,
don't fake biological depth.

## Addendum 11 — Step 1: Live motor-imitation loop (loop closure)

Wired the perception→production bridge directly into the live listener so a
high-saliency, speech-classified frame closes the echolalia / babbling loop in
real time — no longer a standalone CLI flag.

- `LiveMicArray.capture_frame()` now retains `last_pcm` (the mono-mixed raw
  waveform of the most recent capture) so downstream consumers have a signal to
  analyze. Previously the adapter reduced every frame to `channel_energies` +
  `dominant_frequency_hz` and discarded the PCM.
- `LiveListener.listen(..., with_motor_imitation=True)` (requires
  `with_attention=True`): for every salient frame whose firmware processed
  source-DoA is finite (the honest "speech classified by the VAD" signal from
  Addendum 6), it runs `analyze_vowel(last_pcm)` and feeds the result to
  `MotorPlanner.plan_from_analysis()` + `plan_wav()`, re-deriving the motor
  target and re-synthesizing it. Each such frame is recorded as an
  `internal_vocalization_trial` in the report's `motor_imitation` block
  (perceived formants, nearest vowel, re-derived motor plan, re-synth sample
  count, and the source azimuth).
- **Honesty boundary (deliberate):** the loop *logs* the motor-plan match; it
  does NOT auto-play the re-synthesized vowel to the speaker during capture.
  Auto-playback would feed the mic output back into the input (acoustic
  feedback / corollary-audio contamination, already characterized in prior
  debugging), poisoning the very signal being analyzed. Playback stays a
  separate, explicit CLI action (`--synthesize-vowel ... --out`).
- CLI: `--listen-live ... --with-attention --with-motor-imitation` now carries
  the `motor_imitation` block. Real pyusb writes + real firmware VAD only fire
  under `--listen-live` with the XVF3800 present; synthetic/test runs use the
  bus-free `_NullTuner` + scripted arrays, so the loop is fully exercised
  without touching the board.

**Validation:** `pytest tests -q` → 119 passed (was 114; +4
`test_motor_imitation_loop.py` unit tests +1 CLI handler smoke). The loop is
verified to: (a) emit a trial only on speech-classified salient frames, (b)
emit nothing for non-speech salient frames, (c) be a no-op when disabled or
when attention is off. The `resynth_samples > 0` assertion proves the
perception→production re-synthesis actually executes inside the live loop.

**Cognitive fit:** this is the first real-time imitation loop — the scaffold
hears parentese, foveates the source azimuth (Addendum 6), decodes the vowel
target, and generates an internal motor-plan match. Step 2 (acoustic fidelity)
and Step 3 (theta-gamma motor sub-window) remain the natural follow-ons.

## Addendum 12 — Step 2: Tighten Synthetic Acoustic Fidelity (honest scope)

**Goal (user):** "Tighten synthetic acoustic fidelity so the error residual
passed up the predictive hierarchy is driven by genuine acoustic variation,
not processing artifacts" — ideally making full 5-vowel F1/F2/F3 recovery
mathematically stable.

**What was changed (`src/cora_infans/io/motor_plan.py`):**
- Glottal source flattened from a 5-harmonic `1/h` impulse train to a
  60-harmonic `1/h**0.3` sum, so excitation exists across the formant range
  (a steep `1/h` rolloff left F2/F3 with no energy to shape).
- Parallel formant resonators re-weighted (F1×1.0, F2×3.0, F3×2.0) with wider
  bandwidths (80/100/130 Hz) so the higher formants remain audible after the
  source tilt.
- Perception estimator replaced the (non-convergent) LPC-root search with a
  windowed power-spectrum peak-picker + range-gated formant assignment
  (F1∈[280,1000], F2∈[800,2500], F3∈[2000,3500]) — standard formant-tracking.

**What was genuinely achieved:** the target F2 now appears as a real,
table-matched spectral peak for the cardinal vowels — verified by direct
spectrum inspection (a≈1090, e≈1840, o≈840, u≈870 Hz, all within a few % of
the Hillenbrand table). F2 is no longer absent in the signal, as it was before
Step 2. The analysis-by-synthesis loop still closes end-to-end.

**Honest ceiling (NOT faked):** full 5-vowel F1/F2/F3 recovery is **not**
achieved. The toy source-filter model cannot resolve F1 cleanly — the glottal
source's low-frequency energy (the `1/h**0.3` tilt is still low-frequency
dominated) produces a broad spectral bump ~430-530 Hz that the estimator picks
up as F1 for every vowel, swamping the true formant. F1 is therefore
systematically underestimated and the nearest-vowel map mis-assigns (e.g. 'a'
→ 'u'). I ran ~10 targeted DSP experiments (parallel vs cascade, LPC roots,
cepstral liftering, smoothed-envelope peak-pick, gain sweeps, source-tilt
sweeps); the F1 contamination is a structural limitation of a pure-numpy toy
resonator model, not a tuning miss.

**Why I stopped (per project rule — honest assessment over endless tuning):**
the user's standing directive is to stop after a few failed attempts and
reassess rather than iterate endlessly, and to trust the ear. I reassessed:
the Step-2 *mechanism* improvement (F2 now acoustically realized, honest
perception estimator) is real and committed; the *5-vowel recovery* goal needs
a proper glottal-source model (e.g. Liljencrants-Fant), which is out of scope
for the toy and would be fakeable-easy to overclaim. Tests now assert the
genuinely-measurable facts (non-silent audio; plausible F1/F2/F3 ranges; loop
closure) rather than overclaiming recovery.

**Recommended future work for true 5-vowel recovery:** replace `_source_pulse_train`
with an LF (Liljencrants-Fant) glottal-flow model (proper -12 dB/octave source
spectrum with a spectral zero), then a cascade-of-resonators vocal tract; this
is the standard path and would make F1/F2/F3 separable. scipy (absent here)
would also enable a cleaner LPC-root formant estimator.

**Validation:** `pytest tests -q` → 119 passed (Step-2 edits did not change
the count; the motor-plan tests were updated to assert the honest capability).

## Addendum 13 — Step 3: Vocalizations into Theta-Gamma Scheduler

**Goal (user):** integrate the Step-1 motor-plan trial (internal vocalization
intention) into a dedicated "motor assembly" sub-window within the 5 Hz theta
frame, modeling corollary discharge / sensory suppression.

**What was changed:**
- `theta_gamma.py`: added `motor_bin_of()` + `MOTOR_SUBWINDOW_BIN` and a
  reserved last-gamma-bin sub-window per theta cycle. Scheduler gains
  `motor_subwindow_bin()`, `reserve_motor_timestamp(now_ms)` (places a token in
  the motor bin of the current cycle), and corollary-discharge bookkeeping in
  `push()`: a `motor_intention` event in a gamma bin flags any concurrent
  `salient_audio` event in the SAME bin of the SAME cycle as
  `self_suppressed=True` (and vice-versa). The summary reports
  `motor_subwindow_bin`, `motor_intentions`, `self_suppressed_events`.
- `live_listener.py`: when a Step-1 speech-frame motor trial fires AND
  scheduling is on, it pushes a `WorkspaceEvent(kind="motor_intention", ...)`
  into the scheduler at the reserved motor-bin timestamp, carrying the perceived
  vowel, motor-target vowel, and source azimuth. No live playback (feedback
  rule preserved from Step 1).
- `describe()` lists `theta_gamma_motor_subwindow` as implemented.

**Honesty:** this is faithful scheduling + bookkeeping, NOT invented
neurophysics. The "corollary discharge" is a deterministic flag on event state
— the agent marks external audio that coincides with its own vocalization
intention in the same gamma bin as self-generated and therefore not to be
re-processed. The mechanism (phase-locked self-generated motor token within a
theta cycle) is real and testable; the biological claim is explicitly scoped as
a scheduling interpretation, consistent with the theta_gamma.py module doc.

**Tier distinction vs og-cora (cross-check, 2026-07-08):** the sibling
`og-cora` project implements corollary discharge as a *signal-level* mechanism —
Broca PCM efference tap → Cerebellum forward model (`predict_pcm`/`predict_mel`,
room-colored) → Thalamus `spectral_subtraction(observed, predicted)` reporting a
measured Self-Voice Suppression Ratio (SVSR, dB; M0 target ≥ 20 dB). Our
Step 3 is a *scheduling-level* proxy by deliberate design: cora-infans has no
live TTS playback path (Step 1 avoids acoustic feedback during capture), so
there is no efference PCM to subtract. The flag-based proxy is the correct tier
for this scaffold and must not be described as signal-level attenuation.

**Validation:** `pytest tests -q` → 125 passed (was 119; +6
`test_motor_subwindow.py`: motor bin index, timestamp placement, phase-lock,
corollary-discharge suppression, no-suppression-when-bins-differ, and an
end-to-end listener test under scheduling). All three user-approved steps
(Step 1 live loop closure, Step 2 acoustic fidelity, Step 3 theta-gamma
vocalization) are now implemented and committed.

## Addendum 14 — Simulation-only signal-level self-suppression harness

**Decision:** after cross-checking og-cora's mature corollary-discharge stack,
we agreed to add the missing signal-level *simulation* bridge now, while
postponing live playback. This gives cora-infans a real efference→prediction→
residual→SVSR path without introducing speaker/mic feedback or og-cora's full
runtime dependency surface.

**What was added:**
- `src/cora_infans/io/self_suppression.py`: deterministic, numpy-only harness:
  `predict_self_pcm()` (delay + attenuation + echo room-coloring),
  `mix_observed()`, `spectral_subtract()` (PCM-domain predicted-self
  subtraction; intentionally not mel/librosa), `compute_svsr_db()`, and
  `run_self_suppression_trial()`.
- CLI: `python -m cora_infans --simulate-self-suppression [--vowel a|e|i|o|u]
  [--world-vowel a|e|i|o|u] [--world-gain F] ...` emits JSON with
  `mode=simulation_only_signal_self_suppression`, `live_playback=false`, RMS
  values, SVSR dB, and verdict (`self_suppressed`, `partial_suppression`, or
  `not_suppressed`).
- `io.__init__` exports the self-suppression helpers; `describe()` lists
  `simulation_only_signal_self_suppression` as implemented.

**Honesty boundary:** this is now signal-level suppression, but still
simulation-only. It does not play audio and does not listen to the room. The
forward model is deterministic toy room-coloring, not learned cerebellum
adaptation; subtraction is PCM-domain rather than og-cora's mel-domain
Thalamus gate. It is the correct intermediate tier before any live playback
milestone.

**Validation:** new focused tests assert (1) deterministic room-colored
prediction from efference PCM, (2) self residual energy drops and SVSR exceeds
20 dB for self-only simulation, (3) world-only signal is preserved when the
prediction is empty, (4) mixed self+world leaves a world-correlated residual,
and (5) the CLI reports SVSR without live playback. Full suite: `pytest tests
-q` → 131 passed.

## Addendum 15 — Simulation-only forward-model learning from prediction error

**Decision:** before any live playback milestone, add the cognitive learning
step: the agent should reduce sensory prediction error by fitting a forward
model from efference PCM to observed self audio. This keeps the learning rule
clean and deterministic before speaker/mic latency, room drift, and acoustic
feedback enter the loop.

**What was added (`src/cora_infans/io/self_suppression.py`):**
- `evaluate_forward_model(efference, observed, delay_ms, attenuation,
  echo_decay, ...)`: predicts self PCM, subtracts it from observed PCM, and
  reports residual RMS + SVSR dB (+ world-residual correlation when a world
  reference is supplied).
- `fit_forward_model(...)`: deterministic grid search over delay / attenuation /
  echo-decay parameters. It selects the model that minimizes residual RMS
  (tie-breaks by SVSR) and returns `learned_params` plus a learning verdict.
- `run_forward_model_learning_trial(...)`: creates a hidden simulated room
  (`observed_delay_ms`, `observed_attenuation`, `observed_echo_decay`), scores a
  bad initial model, fits the forward model, and reports before/after SVSR,
  residual RMS, learned parameters, and world preservation.

**CLI:**
`python -m cora_infans --learn-self-forward-model --vowel a --observed-delay-ms
18 --observed-attenuation 0.55 --observed-echo-decay 0.22 [--world-vowel o
--world-gain 0.25]` emits `mode=simulation_only_forward_model_learning`,
`live_playback=false`, `initial_svsr_db`, `learned_svsr_db`,
`svsr_improvement_db`, `true_params`, `learned_params`, and optional
`world_residual_correlation`.

**Honesty boundary:** this is cognitive learning of a toy forward model, not a
learned neural cerebellum and not live room calibration. The learned variables
are interpretable physical proxies (delay, attenuation, echo decay). It proves
that residual prediction error can drive a better internal self-audio model in
simulation. Live playback remains a separate milestone.

**Validation:** tests assert hidden room-coloring parameter recovery, SVSR
improvement over a bad initial guess, wrong-efference negative control (cannot
fake good learning with the wrong vowel), mixed self+world residual preservation,
and CLI JSON output. Full suite: `pytest tests -q` → 136 passed.

## Addendum 16 — Forward-model robustness sweep before live playback

**Decision:** after adding signal-level suppression and prediction-error
learning, add a simulation-only robustness surface before considering live
playback. This maps how the learned/candidate forward model behaves under delay,
gain, echo, and world-overlap mismatch — the exact ambiguity live speaker/mic
loops would otherwise introduce too early.

**What was added (`src/cora_infans/io/self_suppression.py`):**
- `run_forward_model_robustness_sweep(...)`: generates observed self audio from
  hidden true params, then evaluates matched/mismatched candidate predictions
  across `delay_errors_ms`, `attenuation_scales`, `echo_decay_scales`, and
  `world_gains`.
- Each case reports predicted params, residual RMS, SVSR dB, optional
  `world_residual_correlation`, and verdict (`robust_pass`,
  `partial_suppression`, or `mismatch_failed`).
- The summary reports best/worst SVSR cases, robust/partial/failure counts, and
  min/max world-residual correlation.

**CLI:**
`python -m cora_infans --sweep-self-forward-model --vowel a [--world-vowel o
--world-gain 0.25] [--delay-errors-ms -20,-10,0,10,20]
[--attenuation-scales 0.75,1.0,1.25] [--echo-decay-scales 0.5,1.0,1.5]` emits
`mode=simulation_only_forward_model_robustness_sweep`, `live_playback=false`,
case rows, and summary metrics.

**Honesty boundary:** this is still a deterministic simulation. It does not
prove live playback safety; it defines the acceptance surface that live playback
must later respect. The main value is making latency/gain/echo mismatch visible
before hardware enters the loop.

**Validation:** tests assert exact-delay/no-world cases strongly outperform
±12 ms delay mismatch, world-overlap cases carry residual-world correlation, and
CLI JSON exposes the sweep summary. Full suite: `pytest tests -q` → 138 passed.

## Addendum 17 — Simulation-only forward-model promotion gate

**Decision:** before reopening live playback, add a single promotion verdict that
combines the signal-level suppression, prediction-error learning, robustness
sweep, world preservation, and wrong-efference negative control. This makes the
simulation milestone auditable instead of relying on a collection of separate
reports.

**What was added (`src/cora_infans/io/self_suppression.py`):**
- `run_forward_model_promotion_gate(...)`: produces
  `mode=simulation_only_forward_model_promotion_gate` and evaluates five
  explicit criteria:
  1. learned model improves SVSR over the bad initial model,
  2. exact prediction suppresses self above the SVSR threshold,
  3. delay mismatch degrades suppression by the expected margin,
  4. mixed world signal remains present in the residual,
  5. wrong efference cannot pass as a learned self model.
- The gate returns `promote_simulation_only` only when every criterion passes;
  otherwise it returns `hold` and leaves the failed criteria visible.
- `live_playback=false` and `live_playback_allowed=false` are deliberately part
  of the result. Passing this gate promotes the simulation scaffold only; it is
  not permission to add live speaker/mic playback.

**CLI:**
`python -m cora_infans --promote-self-forward-model --vowel a --world-vowel o
--world-gain 0.25` emits the criteria table, the learning report, the robustness
summary, the wrong-efference result, and the promotion/hold verdict.

**Honesty boundary:** this gate is the final simulation checkpoint for the
current forward-model work. It validates the cognitive prediction-error loop in
a deterministic setting. A future live-playback milestone still needs a separate
SPEAK state, mic-suppression window, latency measurement, and hardware-specific
validation.

**Validation:** tests assert the default gate promotes only when all criteria
pass, an intentionally impossible world-preservation threshold forces `hold`,
and the CLI reports the simulation-only verdict. Full suite: `pytest tests -q`
→ 141 passed.
