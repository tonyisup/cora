# Phase 1 Foundation

This document describes the current runnable substrate for Project Cora-Infans.

Goal
- keep the first implementation honest, deterministic, and hardware-agnostic
- prove the minimal loop before claiming richer infant-like cognition
- add temporal tracking behavior only where it can be defended with synthetic, test-backed evidence

Current execution paths
1. `cli.py` parses user-facing simulation options into typed `SimulationParams`
2. `simulation.py` turns those params into deterministic frame or sequence reports
3. `io/mic_array.py` produces an `AudioFrame` representing a synthetic 4-channel capture
4. `pipeline.py` applies saliency gating, computes saliency reward, and updates predictive state
5. `workspace.py` stores emitted events inside a bounded workspace window and exposes focus-path tracking
6. final reports return structured runtime state for either a single frame or a temporal sequence

Current module responsibilities
- `src/cora_infans/config.py`
  - owns inspectable runtime defaults
  - current public fields: `mic_channel_count`, `mic_channel_geometry`, `saliency_threshold`, `workspace_window_ms`
  - validates geometry length against configured channel count

- `src/cora_infans/io/mic_array.py`
  - represents the 4-mic array contract without requiring real hardware
  - validates channel count before constructing an `AudioFrame`

- `src/cora_infans/theta_gamma.py` — theta-gamma workspace scheduler
  - deterministic nested temporal-binding layer over the event stream: slow
    theta cycles (default 200 ms ~ 5 Hz) define volatile workspace windows;
    each is subdivided into fast gamma sub-windows (default 25 ms ~ 40 Hz)
    that phase-lock active focus labels into per-cycle assemblies
  - at a theta-cycle boundary the active cycle consolidates (frozen) and a
    fresh volatile cycle opens; only the current cycle is mutable
  - a bounded capacity keeps only the most recent N cycles resident (working
    memory limit); cross-talk is structurally impossible (a cycle binds only
    events whose timestamp falls inside it)
  - exposes a structured `summary()` (resident cycles, per-cycle gamma bins,
    bound focus labels, dominant label, binding-stability fraction) for
    analysis; it is an interpretation layer above the flat EventWorkspace, not
    a replacement for it
- `src/cora_infans/io/live_listener.py` — continuous live listener + parentese stimulus
  - `LiveListener.listen()` captures a time window of real frames, runs each through
    the pipeline, and accumulates focus_path / dropout / recovery tracking (the
    same sequence-report shape as the built-in families, fed by live audio)
  - `ParenteseStimulus` synthesizes a simple rising/falling pitch contour and plays
    it to the External Headphones output device (audiotoolbox index 2) so the mic
    has a repeatable acoustic target; playback is non-blocking so capture overlaps it
  - every captured frame is tagged `stimulus_active` so stimulated vs quiet-room input
    is separable downstream; no frame is ever fabricated
- `src/cora_infans/io/live_mic.py`
  - the real hardware front end for the reSpeaker XMOS XVF3800
  - pulls a short capture through ffmpeg (avfoundation device `:0`) and lowers
    real PCM into the `AudioFrame` contract (per-channel RMS -> channel_energies,
    FFT peak -> dominant_frequency_hz)
  - reports the device's true channel count (2 fused channels at the macOS layer,
    not 4 labeled mics); never fabricates phantom channels
- `src/cora_infans/io/hardware_tuner.py` — XVF3800 on-chip DSP control
  - `HardwareTuner` talks to the board over **USB vendor control transfers**
    (VID 0x2886, PID 0x001A; pyusb + libusb), NOT the CDC serial port (which is not a
    console on this firmware). The verified param table was enumerated from Seeed's
    `xvf_host.py --list`.
  - `apply_raw_linear()` asserts AGC off and drives noise-suppression floors
    (`PP_MIN_NS`, `PP_MIN_NN`) to 0 so channel energies stay linear with acoustic
    pressure; `steer_to(deg)` enables fixed-beam mode and points the beam at a top-down
    selected azimuth (the "acoustic laser").
  - every write is verified by immediate read-back; unknown/read-only/protected
    (REBOOT/SAVE_CONFIGURATION) params are rejected; all writes are VOLATILE (revert on
    reboot) so tuning experiments cannot persistently mutate or brick the board
- `src/cora_infans/io/calibration.py` — linearity calibration probe
  - `calibrate_linearity()` plays graded stimulus tones (via External Headphones) and
    records captured RMS factory-vs-raw-linear to verify the tuning yields a more linear
    channel-energy <-> acoustic-pressure relationship (real hardware measurement)
  - `LiveMicArray` raises `LiveMicError` if ffmpeg is missing or the device cannot be
    opened -- no silent synthetic fallback
  - `build_live_frame_report` is the single integration point with the pipeline,
    so a live capture and a synthetic frame exercise identical runtime code

- `src/cora_infans/saliency.py`
  - owns the deterministic acoustic saliency reward primitive
  - current behavior: peak channel energy minus mean channel energy, floored at zero

- `src/cora_infans/spatial.py`
  - resolves focus labels from configured channel topology
  - current representation is label-only topology, not physical angles or coordinates

- `src/cora_infans/io/active_attention.py` — Active Attention Governor (closed loop)
  - the superior-colliculus analogue: mediates between the software saliency
    map and the hardware beam. PURE logic (no USB import), fully unit-testable.
  - `derive_target_azimuth()` maps the 2 fused channels to a coarse front
    hemisphere angle (left negative, right positive, +/-90 deg) -- honest about
    the 2-channel resolution; it does NOT claim 360 deg localization.
  - `ActiveAttentionGovernor.propose()` gates a steer on three constraints:
    (1) saliency >= 0.02 gate, (2) spatial deadband > 5 deg (no jitter),
    (3) velocity slew <= 180 deg/s (no un-biological sweep). Only then does it
    return `acted=True`; the owner issues the bus write and calls
    `register_steer()` to advance state and accumulate saliency reward.
  - `LiveListener.listen(with_attention=True)` drives the loop: each salient
    frame -> derive target -> governor.propose -> `HardwareTuner.steer_to()`
    (real pyusb only when silicon is present) -> `register_steer`. A bus-free
    `_NullTuner` is used in synthetic/test runs so the loop is exercised under
    pytest without touching the XVF3800.
  - honest unit handling: `HardwareTuner.steer_to()` returns the firmware's
    azimuth in RADIANS; the listener converts to degrees before recording.
  - **Honest localization verdict (verified by DoA geometry tests, doa_experiment.py):**
    on firmware 2.0.6 the firmware's `DOA_VALUE` echoes the beam's OWN steered
    azimuth -- it is NOT an acoustic source bearing. The real source-DoA path is
    `AUDIO_MGR_SELECTED_AZIMUTHS[0]`: it stays NAN for tone/non-speech stimuli,
    but becomes finite when speech fires the firmware VAD. Current listener code
    logs `read_doa()` as beam-azimuth *state* only and still steers on the
    capture-derived L/R; future source-following should consume the processed
    source-DoA when finite, or accept an external ground-truth bearing via
    `governor.propose(explicit_target_deg=...)`.
  - `src/cora_infans/io/doa_experiment.py` — research measurement harness: plays
    a tone or macOS TTS speech stimulus on one earcup of the External Headphones
    (pan=-1/+1/0) and samples firmware DoA/source-DoA registers during playback.
    Used to discover the beam-echo limitation and the speech-VAD source-DoA path;
    repeatable for future geometry checks.
  - owns the minimal predictive-coding primitives
  - `PredictiveCoder`: residual against prior frequency prediction plus updated prediction state (lower level)
  - `FocusTrajectoryCoder`: predicts the next focus label from the recent trajectory; reports a hit/miss residual (higher level)
  - `PredictiveHierarchy`: two-level local error minimizer bundled for the pipeline

- `src/cora_infans/workspace.py`
  - owns bounded short-term event storage
  - evicts events that fall outside the configured time window
  - exposes `focus_path()` over active workspace events

- `src/cora_infans/pipeline.py`
  - owns the actual first-loop runtime behavior
  - responsibilities:
    - compute saliency from channel energies
    - reject non-salient frames
    - compute saliency reward
    - resolve focus label from configured topology
    - enrich salient events with predictive state when a coder is present
    - push emitted events into the workspace

- `src/cora_infans/simulation.py`
  - owns reusable simulation orchestration for scripts and tests
  - converts `SimulationParams` into normalized frame reports
  - converts ordered `SimulationParams` sequences into focus-transition reports
  - exposes built-in deterministic sequence families for constrained temporal demos
  - computes sequence-level summary metadata for salient vs non-salient frames
  - computes sequence-level dropout and recovery counters from the actual frame pattern
  - this is the preferred entrypoint for future non-CLI experiments

- `src/cora_infans/scenario.py`
  - owns the declarative scenario compiler (internal, not a public CLI surface)
  - defines a Pydantic v2 `ScenarioDefinition` / `ScenarioSegment` schema with hard biological invariants
  - loads authored YAML into that schema via `load_scenario_definition_yaml`
  - exposes built-in family definitions through `builtin_scenario_definition`
  - lowers a scenario into `list[SimulationParams]` via `lower_scenario_definition`, which is the only thing the runtime executes
  - enforces v0.1 constraints: segment duration must divide cleanly by `frame_step_ms`, envelopes are compiler-owned (flat / linear fade), silence segments lower to uniform low-energy fields
  - `simulation.py` re-expresses every built-in `--family` through this lowering path, so families are now data, not separate hardcoded branches

- `src/cora_infans/cli.py`
  - owns CLI parsing, validation, and output formatting only
  - should stay thin and delegate execution to `simulation.py`
  - installed console-script entrypoint: `cora-infans`
  - current commands:
    - `--describe`
    - `--simulate-frame`
    - `--simulate-sequence`
  - current sequence family selector:
    - `--family sweep|zigzag|gap|multi_gap|delayed_recovery`

What is proven today
- deterministic salient path
- deterministic non-salient path
- parameter overrides for timestamp, dominant frequency, channel energies, initial prediction, channel topology labels, and saliency threshold
- clean CLI error handling for malformed inputs and wrong channel count
- structured JSON and pretty JSON output modes
- editable install works and exposes a `cora-infans` console script entrypoint
- label-only focus resolution over configurable 4-channel topology
- temporal focus-path tracking over deterministic multi-frame sequences
- predictive residual state persists across frames inside a sequence report
- multiple built-in sequence families can be compared through a stable CLI/runtime surface
- interrupted tracking can be modeled through a non-salient dropout followed by resumed focus
- repeated-dropout recovery can be modeled across more than one interruption in the same sequence
- delayed recovery can be modeled across several consecutive non-salient frames before attention resumes
- the built-in sequence families are authored through a declarative scenario compiler and deterministically lowered into the runtime frame path (no separate hardcoded family branches)
- sequence reports explicitly distinguish total frames, salient frames, and non-salient frames
- sequence reports now expose dropout and recovery counters for direct family-to-family comparison

What is explicitly not proven yet
- automatic acoustic-source-following in the live listener (speech-classified
  source-DoA is now recoverable, but not yet wired as the attention target)
- beamforming or physical spatial localization beyond commanded fixed-beam steer
- raw 4-mic access (the OS-visible stream is 2 fused channels)
- stacked predictive layers
- theta-gamma scheduling as the default report path for every run (it is
  implemented and opt-in, not yet default everywhere)
- motor-plan synthesis
- training over real recorded data
- arbitrary user-authored temporal trajectories

Why label-only topology right now
- the vision requires a 4-mic auditory front end, but does not pin down the physical geometry details tightly enough to justify fake directional physics
- label-only topology lets the project model isolate/track behavior without pretending it knows real-world azimuths
- this keeps the current substrate honest while leaving room for future hardware-grounded spatial reasoning

Why built-in sequence families right now
- they let the project compare different temporal tracking patterns without committing to a general sequence input format too early
- they preserve deterministic, test-backed behavior at the CLI and runtime layers
- they keep the user-facing surface narrow while the underlying report contract matures

Current built-in sequence families
- `sweep`: front_left -> front_right -> rear_right
- `zigzag`: front_left -> rear_right -> front_right -> rear_left
- `gap`: front_left -> non_salient_gap -> rear_right
- `multi_gap`: front_left -> non_salient_gap -> front_right -> non_salient_gap -> rear_right
- `delayed_recovery`: front_left -> non_salient_gap -> non_salient_gap -> non_salient_gap -> rear_right

Why the gap-family variants matter
- they test interrupted tracking rather than only continuous salient motion
- they prove the sequence report can include non-salient frames while emitting only salient events
- they create a stable regression surface for dropout and recovery behavior without requiring live noise injection or hardware capture
- `multi_gap` specifically checks whether recovery remains stable across repeated interruptions, not just a single gap
- `delayed_recovery` checks whether summary metrics remain honest when recovery is postponed across multiple consecutive dropouts

Why sequence-level frame metadata matters
- it makes dropout behavior auditable without inspecting the full frame/event arrays by hand
- it distinguishes total temporal input from attended input at a glance
- it creates a clean summary surface for future sequence-family comparisons and evaluation reports

Why dropout and recovery counters matter
- they summarize interruption structure without forcing downstream tools to diff raw frame arrays
- `dropout_count` measures how many frames were rejected by the saliency gate
- `recovery_count` measures how many times attention resumes immediately after a dropout
- together they let repeated-gap and delayed-recovery families be compared on resilience, not just on final focus path

Near-term extension path
1. keep simulation logic reusable rather than expanding CLI complexity
2. add richer report metadata only when it supports a real experiment decision
3. expose additional built-in deterministic sequence families before inventing a general user-authored sequence format
4. add live hardware adapters behind the same `AudioFrame` contract
5. add deeper predictive/state modules only after they have deterministic tests and a real report surface
