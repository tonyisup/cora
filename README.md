# Project Cora-Infans

Project Cora-Infans is an audio-centric research scaffold for infant-inspired auditory and cognitive learning.

Current status
- Vision document exists in `VISION.md`
- Minimal Python package scaffold exists
- Deterministic audio-first runtime slices are implemented and tested:
  - synthetic 4-channel frame -> acoustic saliency gate -> predictive residual update -> bounded workspace event record
  - multi-frame focus-path tracking over label-only microphone topology
  - interrupted tracking and recovery across non-salient dropouts
- CLI surfaces exist for architecture summary, single-frame simulation, and built-in sequence simulation families

Implemented today
- `python3 -m pytest tests -q` validation suite
- `python3 -m cora_infans --describe`
- `python3 -m cora_infans --simulate-frame`
- `python3 -m cora_infans --simulate-sequence`
- `python3 -m cora_infans --simulate-sequence --family sweep`
- `python3 -m cora_infans --simulate-sequence --family zigzag`
- `python3 -m cora_infans --simulate-sequence --family gap`
- `python3 -m cora_infans --simulate-sequence --family multi_gap`
- `python3 -m cora_infans --capture-live [--device :0] [--duration-ms N]`
- `python3 -m cora_infans --simulate-sequence --family sweep --with-scheduling`
- `python3 -m cora_infans --snapshot-firmware`
- `python3 -m cora_infans --tune-raw-linear` (volatile: AGC off + NS floors 0)
- `python3 -m cora_infans --listen-live --duration-ms N --frame-ms N --with-stimulus --with-attention --with-scheduling`
- parameterized single-frame simulation CLI with structured report output
- reusable simulation helpers for future training/eval scripts
- configurable 4-channel topology labels
- temporal focus-path and focus-transition reporting
- selectable built-in deterministic sequence families
- sequence-level salient vs non-salient frame summary metadata
- sequence-level dropout and recovery summary metadata
- declarative scenario compiler (`src/cora_infans/scenario.py`) that authors the built-in sequence families and lowers them deterministically into the runtime frame path
- two-level predictive-coding hierarchy (`PredictiveHierarchy`: frequency residual + focus-trajectory prediction), wired into the pipeline and simulation defaults
- live XVF3800 hardware front end (`src/cora_infans/io/live_mic.py`): captures real frames via ffmpeg/avfoundation, lowers to AudioFrame, runs through the same pipeline; no synthetic fallback
- continuous live listener (`src/cora_infans/io/live_listener.py`): accumulates focus-path / dropout / recovery tracking over a real-time window; optional parentese stimulus plays to External Headphones (audiotoolbox device 2) so the mic has a controlled acoustic target
- theta-gamma workspace scheduling (`src/cora_infans/theta_gamma.py`): deterministic nested theta-cycle / gamma-subwindow temporal-binding layer over the event stream (5 Hz theta windows, 40 Hz gamma bins, bounded residency); models VISION.md's multiplexing claim without inventing neurophysics
- XVF3800 hardware tuner (`src/cora_infans/io/hardware_tuner.py`): real on-chip DSP control over USB vendor control transfers (pyusb/libusb) — asserts AGC off, drives noise-suppression floors to 0 for linear capture, and enables manual beam steering (the "acoustic laser") by writing fixed-beam azimuths; every write verified by read-back; volatile only (no reboot/save)
- linearity calibration probe (`src/cora_infans/io/calibration.py`): measures captured RMS across graded stimulus levels factory-vs-raw-linear to verify the tuning yields a more linear channel-energy <-> acoustic-pressure relationship
- active auditory oculomotor gaze (`src/cora_infans/io/active_attention.py` + `LiveListener.with_attention`): the "Active Attention Governor" closes the loop between the saliency map and the hardware beam. Each salient frame derives a target azimuth (coarse L/R from the 2 fused channels), then the governor gates the steer on (1) saliency > 0.02, (2) spatial deadband > 5 deg, (3) velocity slew <= 180 deg/s, and calls `HardwareTuner.steer_to()`. Real pyusb writes fire only under `--listen-live` with real silicon; synthetic/test runs use a bus-free `_NullTuner` so the loop is fully exercised without touching the board.
  - **Honest limitation (verified by the DoA geometry test):** on firmware 2.0.6, the firmware's `DOA_VALUE` echoes the beam's OWN steered azimuth (not the acoustic source bearing), and its real source-DoA (`AUDIO_MGR_SELECTED_AZIMUTHS`) stays NAN without speech detection. The captured L/R energy split is also source-blind at close range. So the beam is **commanded top-down attention** (it points where told), NOT automatic source-following. `read_doa()` is logged as beam-azimuth state only; the steer target is the capture-derived L/R. A caller with an external ground-truth source bearing passes it via `governor.propose(explicit_target_deg=...)`. See `doa_experiment.py` and the status doc Addendum 4.

Not implemented yet
- live ReSpeaker XMOS XVF3800 hardware integration (real capture, saliency gate, stimulus loop, AND on-chip DSP tuning all landed)
- real multi-mic fusion / beamforming beyond the single fixed-beam steer (board fuses 4 mics -> 2 USB channels; manual steer of the one fixed beam is supported, true per-mic array beamforming is not)
- deeper predictive-coding hierarchy beyond one frequency residual + one focus-trajectory level
- theta-gamma workspace scheduling (deterministic scheduler implemented; not yet the default report path for all runs)
- motor-plan / speech synthesis loop
- training and evaluation harnesses over real audio corpora
- general user-authored sequence format

Why this slice first
- it gives the repo a truthful executable baseline
- it keeps the hardware boundary explicit
- it proves input -> reward -> prediction -> workspace plumbing without faking biological depth
- it establishes a reusable substrate for future experiment scripts
- it begins to model isolation and tracking over time without inventing unsupported spatial physics

Quick start
- Install the package in editable mode:
  `python3 -m pip install -e .`
- Run tests:
  `python3 -m pytest tests -q`
- Preferred installed command (when Python's scripts directory is on your PATH):
  `cora-infans --describe`
- Module fallback that always works:
  `python3 -m cora_infans --describe`
- Run the default deterministic single-frame simulation:
  `python3 -m cora_infans --simulate-frame`
- Run a custom frame simulation:
  `python3 -m cora_infans --simulate-frame --timestamp-ms 250 --dominant-frequency-hz 350.0 --channel-energies 0.1,0.2,0.8,0.3 --initial-prediction-hz 300.0 --mic-geometry front_left,front_right,rear_right,rear_left`
- Pretty-print the frame report:
  `python3 -m cora_infans --simulate-frame --output pretty`
- Force a non-salient frame outcome:
  `python3 -m cora_infans --simulate-frame --saliency-threshold 0.95`
- Run the default temporal focus sequence demo:
  `python3 -m cora_infans --simulate-sequence`
- Run the explicit sweep family:
  `python3 -m cora_infans --simulate-sequence --family sweep`
- Run the zigzag family:
  `python3 -m cora_infans --simulate-sequence --family zigzag`
- Run the gap/recovery family:
  `python3 -m cora_infans --simulate-sequence --family gap`
- Run the repeated-gap recovery family:
  `python3 -m cora_infans --simulate-sequence --family multi_gap`
- Run the delayed-recovery family:
  `python3 -m cora_infans --simulate-sequence --family delayed_recovery`

Sequence family note
- Sequence families are constrained built-in demos.
- They exist to compare tracking behavior without inventing a general user-authored sequence DSL yet.
- Current families:
  - `sweep`: front_left -> front_right -> rear_right
  - `zigzag`: front_left -> rear_right -> front_right -> rear_left
  - `gap`: front_left -> non_salient_gap -> rear_right
  - `multi_gap`: front_left -> non_salient_gap -> front_right -> non_salient_gap -> rear_right
  - `delayed_recovery`: front_left -> non_salient_gap -> non_salient_gap -> non_salient_gap -> rear_right

Packaging note
- Editable install is verified.
- The console script entrypoint is `cora-infans`.
- If `cora-infans` is not found on your shell PATH, use `python3 -m cora_infans ...` or add Python's scripts directory to PATH.

Example frame report fields
- `status`: `salient` or `non_salient`
- `focus_path`: active focus labels inside the current workspace window
- `resolved_config`: runtime config actually used
- `frame`: captured/simulated input frame
- `event`: emitted workspace event or `null`
- `workspace_snapshot`: current bounded workspace contents

Example sequence report fields
- `family`: built-in sequence family used for the run
- `frame_count`: number of simulated frames
- `salient_frame_count`: number of frames that produced salient events
- `non_salient_frame_count`: number of frames rejected by the saliency gate
- `dropout_count`: number of non-salient frames in the sequence
- `recovery_count`: number of salient frames that immediately follow a non-salient frame
- `emitted_event_count`: number of salient events admitted to the workspace
- `focus_path`: ordered focus labels over the active window
- `focus_transitions`: label-to-label transition strings
- `frames`: input sequence frames, including non-salient gaps
- `events`: emitted salient events only
- `workspace_snapshot`: workspace state after the sequence completes

Current code structure
- `src/cora_infans/config.py` — runtime config, including channel topology labels
- `src/cora_infans/io/mic_array.py` — synthetic 4-channel input adapter
- `src/cora_infans/io/live_mic.py` — live ReSpeaker XVF3800 adapter (ffmpeg/avfoundation), lowers real PCM into the AudioFrame contract; fails loudly if the device/ffmpeg is missing (no synthetic fallback)
- `src/cora_infans/predictive.py` — minimal predictive residual update
- `src/cora_infans/saliency.py` — deterministic acoustic saliency reward helper
- `src/cora_infans/spatial.py` — label-based focus resolution over configured topology
- `src/cora_infans/workspace.py` — bounded event workspace plus focus-path tracking
- `src/cora_infans/pipeline.py` — saliency gate + reward + predictive/workspace wiring
- `src/cora_infans/simulation.py` — reusable frame/sequence simulation helpers and report builders
- `src/cora_infans/scenario.py` — declarative scenario compiler (Pydantic schema + YAML loader + deterministic lowering into the runtime frame path); authors the built-in sequence families
- `src/cora_infans/cli.py` — CLI parsing and presentation layer

Architecture note
- `docs/architecture/phase-1-foundation.md`
- `docs/PROJECT_STATUS_2026-07-08.md` — point-in-time progress/status snapshot (verified test counts + hardware validation)

Implementation plan
- `docs/plans/2026-07-07-cora-infans-kickoff.md`
