from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from .simulation import SimulationParams


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
    def unique_geometry(
        cls,
        value: tuple[ChannelLabel, ChannelLabel, ChannelLabel, ChannelLabel],
    ) -> tuple[ChannelLabel, ChannelLabel, ChannelLabel, ChannelLabel]:
        if len(set(value)) != 4:
            raise ValueError("mic_channel_geometry must contain 4 unique channel labels")
        return value


_BUILTIN_SCENARIO_DATA: dict[str, dict[str, object]] = {
    "sweep": {
        "version": "0.1",
        "name": "sweep",
        "description": "Deterministic salient sweep across the front-left to rear-right path.",
        "frame_step_ms": 50,
        "mic_channel_geometry": ["front_left", "front_right", "rear_right", "rear_left"],
        "segments": [
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "front_left",
                "target_energy": 0.9,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "front_right",
                "target_energy": 0.8,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "rear_right",
                "target_energy": 0.85,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
        ],
    },
    "zigzag": {
        "version": "0.1",
        "name": "zigzag",
        "description": "Alternating salient path across the 4-channel auditory field.",
        "frame_step_ms": 50,
        "mic_channel_geometry": ["front_left", "front_right", "rear_right", "rear_left"],
        "segments": [
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "front_left",
                "target_energy": 0.9,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "rear_right",
                "target_energy": 0.85,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "front_right",
                "target_energy": 0.8,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "rear_left",
                "target_energy": 0.9,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
        ],
    },
    "gap": {
        "version": "0.1",
        "name": "gap",
        "description": "Single non-salient dropout between two salient targets.",
        "frame_step_ms": 50,
        "mic_channel_geometry": ["front_left", "front_right", "rear_right", "rear_left"],
        "segments": [
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "front_left",
                "target_energy": 0.9,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "silence",
                "duration_ms": 50,
                "background_noise_floor": 0.2,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "rear_right",
                "target_energy": 0.85,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
        ],
    },
    "multi_gap": {
        "version": "0.1",
        "name": "multi_gap",
        "description": "Repeated dropout recovery across two interruptions.",
        "frame_step_ms": 50,
        "mic_channel_geometry": ["front_left", "front_right", "rear_right", "rear_left"],
        "segments": [
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "front_left",
                "target_energy": 0.9,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "silence",
                "duration_ms": 50,
                "background_noise_floor": 0.2,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "front_right",
                "target_energy": 0.85,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "silence",
                "duration_ms": 50,
                "background_noise_floor": 0.2,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "rear_right",
                "target_energy": 0.85,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
        ],
    },
    "delayed_recovery": {
        "version": "0.1",
        "name": "delayed_recovery",
        "description": "Delayed recovery after several consecutive non-salient gaps.",
        "frame_step_ms": 50,
        "mic_channel_geometry": ["front_left", "front_right", "rear_right", "rear_left"],
        "segments": [
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "front_left",
                "target_energy": 0.9,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
            {
                "foreground_kind": "silence",
                "duration_ms": 150,
                "background_noise_floor": 0.2,
            },
            {
                "foreground_kind": "voiced_target",
                "duration_ms": 50,
                "dominant_frequency_hz": 1200.0,
                "target_channel": "rear_right",
                "target_energy": 0.85,
                "distractor_energy": 0.2,
                "background_noise_floor": 0.1,
            },
        ],
    },
}


def load_scenario_definition_yaml(raw_yaml: str) -> ScenarioDefinition:
    payload = yaml.safe_load(raw_yaml)
    if not isinstance(payload, dict):
        raise ValueError("scenario YAML must define a mapping at the document root")
    return ScenarioDefinition.model_validate(payload)


def builtin_scenario_definition(family: str) -> ScenarioDefinition:
    try:
        payload = _BUILTIN_SCENARIO_DATA[family]
    except KeyError as exc:
        raise ValueError(f"unknown sequence family: {family}") from exc
    return ScenarioDefinition.model_validate(payload)


def _linear_envelope_scales(envelope: EnvelopeKind, frame_count: int) -> list[float]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [1.0]
    if envelope is EnvelopeKind.flat:
        return [1.0] * frame_count
    if envelope is EnvelopeKind.fade_in:
        return [index / (frame_count - 1) for index in range(frame_count)]
    return [1.0 - (index / (frame_count - 1)) for index in range(frame_count)]


def _target_channel_energies(
    segment: ScenarioSegment,
    geometry: tuple[ChannelLabel, ChannelLabel, ChannelLabel, ChannelLabel],
    scale: float,
) -> list[float]:
    target_label = segment.target_channel
    assert target_label is not None
    target_index = geometry.index(target_label)
    if target_label is ChannelLabel.front_left:
        base_energies = [segment.target_energy, segment.distractor_energy, segment.background_noise_floor, segment.background_noise_floor]
    elif target_label is ChannelLabel.front_right:
        base_energies = [segment.distractor_energy, segment.target_energy, segment.distractor_energy, segment.background_noise_floor]
    elif target_label is ChannelLabel.rear_right:
        base_energies = [segment.distractor_energy, segment.distractor_energy, segment.target_energy, segment.background_noise_floor]
    else:
        base_energies = [segment.background_noise_floor, segment.distractor_energy, segment.background_noise_floor, segment.target_energy]

    energies: list[float] = []
    for index, energy in enumerate(base_energies):
        if index == target_index:
            energies.append(energy * scale)
        elif energy == segment.distractor_energy:
            energies.append(energy * scale)
        else:
            energies.append(energy)
    return energies


def lower_scenario_definition(scenario: ScenarioDefinition) -> list[SimulationParams]:
    from .simulation import SimulationParams

    lowered: list[SimulationParams] = []
    timestamp_ms = 0
    for segment in scenario.segments:
        if segment.duration_ms % scenario.frame_step_ms != 0:
            raise ValueError("duration_ms must divide cleanly by frame_step_ms")
        frame_count = segment.duration_ms // scenario.frame_step_ms
        scales = _linear_envelope_scales(segment.envelope, frame_count)
        for scale in scales:
            timestamp_ms += scenario.frame_step_ms
            if segment.foreground_kind is ForegroundKind.silence:
                channel_energies = [segment.background_noise_floor] * 4
                dominant_frequency_hz = scenario.initial_prediction_hz
            else:
                channel_energies = _target_channel_energies(
                    segment,
                    scenario.mic_channel_geometry,
                    scale,
                )
                dominant_frequency_hz = segment.dominant_frequency_hz
                assert dominant_frequency_hz is not None
            lowered.append(
                SimulationParams(
                    timestamp_ms=timestamp_ms,
                    dominant_frequency_hz=dominant_frequency_hz,
                    channel_energies=channel_energies,
                    initial_prediction_hz=scenario.initial_prediction_hz,
                    mic_channel_geometry=tuple(label.value for label in scenario.mic_channel_geometry),
                    saliency_threshold=scenario.saliency_threshold,
                )
            )
    return lowered
