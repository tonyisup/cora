# Scenario Spec v0.1

This document defines the first declarative scenario format for Project Cora-Infans.

Status
- internal architecture spec + implemented in `src/cora_infans/scenario.py`
- backs the built-in `--family` sequence families through `builtin_scenario_definition` + `lower_scenario_definition`
- not a public CLI/product surface (no YAML ingestion CLI; YAML loading is internal/test-only)

Goal
- introduce a minimal declarative experiment format without turning the project into a generic simulation engine
- keep the substrate biologically grounded in acoustic conditions that feed the Acoustic Saliency Reward Loop
- preserve deterministic lowering into the existing `SimulationParams` / sequence-report runtime

Non-goals
- no open-ended timeline DSL
- no arbitrary trajectory language
- no loops, branching, callbacks, or formulas authored in YAML
- no raw per-frame channel-energy escape hatch in v0.1
- no direct control of workspace state or predictive state beyond existing config inputs

Why this exists
The current built-in sequence families are disciplined and useful, but they will become an ad hoc catalog if every new temporal experiment requires another hard-coded family. The scenario spec is the narrowest acceptable step beyond that catalog.

The format must describe biologically meaningful acoustic episodes, not abstract scene choreography.

Design boundary
Allowed
- foreground acoustic episodes
- voiced target properties
- segment duration
- target channel focus
- background noise floor
- simple opaque amplitude envelopes
- deterministic lowering into fixed-size runtime frames

Forbidden
- arbitrary timestamp authoring per frame
- free-form trajectory primitives
- generic motion semantics
- loops, conditionals, or repetition syntax
- raw channel-energy arrays in authored YAML
- custom functions or formula hooks
- multiple simultaneous foreground targets in one segment
- direct motor-plan or oscillator definitions

Authoritative representation
The authoritative in-memory representation should be a Pydantic V2 schema.
Human-authored YAML is allowed only as a serialization of that schema.

Why Pydantic first
- hard validation
- explicit invariants
- narrow vocabulary
- deterministic lowering boundary

## Authoritative Pydantic V2 draft

```python
from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChannelLabel(str, Enum):
    front_left = "front_left"
    front_right = "front_right"
    rear_right = "rear_right"
    rear_left = "rear_left"


class ForegroundKind(str, Enum):
    voiced_target = "voiced_target"
    silence = "silence"


class EnvelopeKind(str, Enum):
    flat = "flat"
    fade_in = "fade_in"
    fade_out = "fade_out"


class ScenarioSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_ms: int = Field(gt=0, le=2000)
    foreground_kind: ForegroundKind

    dominant_frequency_hz: float | None = Field(default=None, gt=50, le=4000)
    vowel_label: Literal["a", "e", "i", "o", "u"] | None = None

    target_channel: ChannelLabel | None = None
    target_energy: float = Field(default=0.85, ge=0.0, le=1.0)
    distractor_energy: float = Field(default=0.2, ge=0.0, le=1.0)
    background_noise_floor: float = Field(default=0.2, ge=0.0, le=1.0)

    envelope: EnvelopeKind = EnvelopeKind.flat

    @model_validator(mode="after")
    def validate_biological_constraints(self) -> "ScenarioSegment":
        if self.foreground_kind == ForegroundKind.voiced_target:
            if self.dominant_frequency_hz is None:
                raise ValueError("voiced_target requires dominant_frequency_hz")
            if self.target_channel is None:
                raise ValueError("voiced_target requires target_channel")
        if self.foreground_kind == ForegroundKind.silence:
            if self.target_channel is not None:
                raise ValueError("silence cannot define target_channel")
            if self.dominant_frequency_hz is not None:
                raise ValueError("silence cannot define dominant_frequency_hz")
        return self


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["0.1"]
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=400)

    frame_step_ms: int = Field(default=50, ge=25, le=250)
    mic_channel_geometry: tuple[ChannelLabel, ChannelLabel, ChannelLabel, ChannelLabel] = (
        ChannelLabel.front_left,
        ChannelLabel.front_right,
        ChannelLabel.rear_right,
        ChannelLabel.rear_left,
    )
    initial_prediction_hz: float = Field(default=1000.0, gt=50, le=4000)
    saliency_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    segments: list[ScenarioSegment] = Field(min_length=1, max_length=32)

    @field_validator("mic_channel_geometry")
    @classmethod
    def unique_geometry(cls, value):
        if len(set(value)) != 4:
            raise ValueError("mic_channel_geometry must contain 4 unique channel labels")
        return value
```

## Lowering model

A scenario lowers into `list[SimulationParams]`.
That lowered sequence remains the only thing the current runtime executes.

Lowering stages
1. validate YAML into `ScenarioDefinition`
2. expand each segment into one or more fixed-size frames using `frame_step_ms`
3. convert each expanded frame into `SimulationParams`
4. run the existing sequence-report path unchanged

Key rule
The authored format describes segments, not frames.
Frame-level details are compiler output, not authored surface area.

## Segment semantics

### `foreground_kind: voiced_target`
Represents a salience-bearing voiced acoustic target.

Required
- `dominant_frequency_hz`
- `target_channel`

Meaning
- one channel receives the segment's target emphasis
- remaining channels receive distractor/background values
- the resulting frame may become salient or non-salient depending on threshold and envelope state

### `foreground_kind: silence`
Represents a non-salient acoustic gap or dropout episode.

Required
- no target channel
- no dominant frequency

Meaning
- all channels are lowered to a uniform low-energy field derived from `background_noise_floor`
- this segment is intended to compile into non-salient frames under normal threshold settings

## Envelope lowering rule

This is intentionally simple and intentionally opaque to the YAML author.
The author selects only `flat`, `fade_in`, or `fade_out`.
The compiler owns the actual per-frame energy schedule.

Rule for v0.1
- frame count for a segment is determined by segment expansion under `frame_step_ms`
- if a segment expands to exactly 1 frame, every envelope kind resolves to `[1.0]`
- the compiler applies a simple linear scale across the segment's generated frames
- no custom envelope formulas are authorable

Examples
- any envelope over 1 frame -> `[1.0]`
- `flat` over 4 frames -> `[1.0, 1.0, 1.0, 1.0]`
- `fade_in` over 4 frames -> `[0.0, 0.33, 0.66, 1.0]`
- `fade_out` over 4 frames -> `[1.0, 0.66, 0.33, 0.0]`

Interpretation
- the scale multiplies target-bearing energy terms during lowering
- silence segments may ignore envelope scaling if the compiled result is already uniform low-energy silence
- this keeps authored YAML simple and keeps deterministic temporal shaping inside the compiler

Rounding note
The exact floating-point values may be represented with standard Python float precision. The important invariant is monotonic linear interpolation over the generated frame count.

Single-frame note
Envelopes require at least 2 frames to express a slope. If an author wants an actual gradient, the segment duration must be at least `2 * frame_step_ms`.

## Duration expansion

v0.1 rule
- segments expand into fixed-size frame steps using `frame_step_ms`
- for the first implementation, durations should be chosen to divide cleanly by `frame_step_ms`
- if they do not divide cleanly, the compiler should reject the scenario rather than inventing hidden remainder behavior

Reason
This is stricter than necessary, but it protects the substrate from accidental timing ambiguity in the first version.

## YAML mock: `multi_gap`

```yaml
version: "0.1"
name: multi_gap
description: Repeated dropout recovery across two interruptions.
frame_step_ms: 50
mic_channel_geometry:
  - front_left
  - front_right
  - rear_right
  - rear_left
initial_prediction_hz: 1000.0
saliency_threshold: 0.7

segments:
  - foreground_kind: voiced_target
    duration_ms: 50
    dominant_frequency_hz: 1200.0
    target_channel: front_left
    target_energy: 0.9
    distractor_energy: 0.2
    background_noise_floor: 0.1
    envelope: flat

  - foreground_kind: silence
    duration_ms: 50
    background_noise_floor: 0.2
    envelope: flat

  - foreground_kind: voiced_target
    duration_ms: 50
    dominant_frequency_hz: 1200.0
    target_channel: front_right
    target_energy: 0.85
    distractor_energy: 0.2
    background_noise_floor: 0.1
    envelope: flat

  - foreground_kind: silence
    duration_ms: 50
    background_noise_floor: 0.2
    envelope: flat

  - foreground_kind: voiced_target
    duration_ms: 50
    dominant_frequency_hz: 1200.0
    target_channel: rear_right
    target_energy: 0.85
    distractor_energy: 0.2
    background_noise_floor: 0.1
    envelope: flat
```

## YAML mock: `delayed_recovery`

```yaml
version: "0.1"
name: delayed_recovery
description: Delayed recovery after several consecutive non-salient gaps.
frame_step_ms: 50
mic_channel_geometry:
  - front_left
  - front_right
  - rear_right
  - rear_left
initial_prediction_hz: 1000.0
saliency_threshold: 0.7

segments:
  - foreground_kind: voiced_target
    duration_ms: 50
    dominant_frequency_hz: 1200.0
    target_channel: front_left
    target_energy: 0.9
    distractor_energy: 0.2
    background_noise_floor: 0.1
    envelope: flat

  - foreground_kind: silence
    duration_ms: 150
    background_noise_floor: 0.2
    envelope: flat

  - foreground_kind: voiced_target
    duration_ms: 50
    dominant_frequency_hz: 1200.0
    target_channel: rear_right
    target_energy: 0.85
    distractor_energy: 0.2
    background_noise_floor: 0.1
    envelope: flat
```

## Why this is acceptable under the vision

This spec remains aligned with `VISION.md` because it models:
- acoustic salience-bearing targets
- temporal interruption and resumed tracking
- deterministic sensory gating inputs
- a narrow, biologically motivated front end for the existing predictive/workspace loop

It does not yet attempt to model:
- beamforming physics
- Theta-Gamma workspace scheduling
- motor synthesis
- semantic attractor dynamics
- arbitrary environment simulation

## Explicit v0.1 restrictions

The following must be rejected in the first implementation:
- frame-level authored timestamps
- free-form per-channel energy arrays in YAML
- repeat/loop syntax
- branching logic
- more than one simultaneous voiced target in a segment
- arbitrary geometry labels outside the 4-channel topology enum
- hidden compiler heuristics for odd durations

## Minimal future-safe extensions

Allowed candidates for later consideration
- `vowel_label`
- constrained `f0_contour` enum such as `flat | rising | falling`
- constrained masking noise enum
- constrained parentese emphasis flag

Not allowed by default
- raw frame authoring
- formulas
- procedural scenario generation hooks
- user-facing public CLI ingestion

## Recommended implementation order

1. add the Pydantic V2 schema in Python
2. add YAML loader into that schema
3. add deterministic lowering from scenario -> `list[SimulationParams]`
4. re-express existing built-in families through the lowering path
5. keep the feature internal until the boundary proves stable

## Acceptance criteria for implementation

When implementation begins, it should not be accepted unless:
- invalid scenarios are rejected with clear validation errors
- duration/frame-step divisibility is enforced in v0.1
- `multi_gap` and `delayed_recovery` can be represented through the scenario schema without raw frame escape hatches
- existing sequence-report metrics remain derivable from lowered scenarios
- no public-facing generic scenario CLI is introduced by default
