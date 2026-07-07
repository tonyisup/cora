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
- `python3 -m cora_infans --simulate-sequence --family delayed_recovery`
- parameterized single-frame simulation CLI with structured report output
- reusable simulation helpers for future training/eval scripts
- configurable 4-channel topology labels
- temporal focus-path and focus-transition reporting
- selectable built-in deterministic sequence families
- sequence-level salient vs non-salient frame summary metadata
- sequence-level dropout and recovery summary metadata

Not implemented yet
- live ReSpeaker XMOS XVF3800 hardware integration
- real multi-mic fusion / beamforming
- deeper predictive-coding hierarchy beyond one residual update stream
- theta-gamma workspace scheduling
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
- `src/cora_infans/predictive.py` — minimal predictive residual update
- `src/cora_infans/saliency.py` — deterministic acoustic saliency reward helper
- `src/cora_infans/spatial.py` — label-based focus resolution over configured topology
- `src/cora_infans/workspace.py` — bounded event workspace plus focus-path tracking
- `src/cora_infans/pipeline.py` — saliency gate + reward + predictive/workspace wiring
- `src/cora_infans/simulation.py` — reusable frame/sequence simulation helpers and report builders
- `src/cora_infans/cli.py` — CLI parsing and presentation layer

Architecture note
- `docs/architecture/phase-1-foundation.md`

Implementation plan
- `docs/plans/2026-07-07-cora-infans-kickoff.md`
