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

- `src/cora_infans/saliency.py`
  - owns the deterministic acoustic saliency reward primitive
  - current behavior: peak channel energy minus mean channel energy, floored at zero

- `src/cora_infans/spatial.py`
  - resolves focus labels from configured channel topology
  - current representation is label-only topology, not physical angles or coordinates

- `predictive.py`
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
- live microphone capture
- beamforming or physical spatial localization
- actual ReSpeaker channel semantics
- stacked predictive layers
- oscillatory workspace scheduling
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
