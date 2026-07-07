import textwrap

import pytest

from cora_infans.scenario import (
    ChannelLabel,
    EnvelopeKind,
    ForegroundKind,
    ScenarioDefinition,
    ScenarioSegment,
    builtin_scenario_definition,
    lower_scenario_definition,
    load_scenario_definition_yaml,
)


def test_scenario_segment_rejects_voiced_target_without_required_fields() -> None:
    with pytest.raises(ValueError, match="voiced_target requires dominant_frequency_hz"):
        ScenarioSegment(
            duration_ms=50,
            foreground_kind=ForegroundKind.voiced_target,
            target_channel=ChannelLabel.front_left,
        )

    with pytest.raises(ValueError, match="voiced_target requires target_channel"):
        ScenarioSegment(
            duration_ms=50,
            foreground_kind=ForegroundKind.voiced_target,
            dominant_frequency_hz=1200.0,
        )


def test_load_scenario_definition_yaml_parses_delayed_recovery_shape() -> None:
    scenario = load_scenario_definition_yaml(
        textwrap.dedent(
            """
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
            """
        )
    )

    assert scenario.name == "delayed_recovery"
    assert scenario.frame_step_ms == 50
    assert len(scenario.segments) == 3
    assert scenario.segments[1].foreground_kind is ForegroundKind.silence


def test_lower_scenario_definition_uses_unit_scale_for_single_frame_fade_out() -> None:
    scenario = ScenarioDefinition(
        version="0.1",
        name="single_frame_fade_out",
        description="A one-frame voiced target with fade_out should stay at full scale.",
        frame_step_ms=50,
        mic_channel_geometry=(
            ChannelLabel.front_left,
            ChannelLabel.front_right,
            ChannelLabel.rear_right,
            ChannelLabel.rear_left,
        ),
        segments=[
            ScenarioSegment(
                duration_ms=50,
                foreground_kind=ForegroundKind.voiced_target,
                dominant_frequency_hz=1200.0,
                target_channel=ChannelLabel.front_left,
                target_energy=0.9,
                distractor_energy=0.2,
                background_noise_floor=0.1,
                envelope=EnvelopeKind.fade_out,
            )
        ],
    )

    lowered = lower_scenario_definition(scenario)

    assert len(lowered) == 1
    assert lowered[0].timestamp_ms == 50
    assert lowered[0].channel_energies == [0.9, 0.2, 0.1, 0.1]


def test_lower_scenario_definition_rejects_duration_not_divisible_by_frame_step() -> None:
    scenario = ScenarioDefinition(
        version="0.1",
        name="bad_duration",
        description="Durations must divide cleanly by the frame step in v0.1.",
        frame_step_ms=50,
        segments=[
            ScenarioSegment(
                duration_ms=75,
                foreground_kind=ForegroundKind.silence,
                background_noise_floor=0.2,
            )
        ],
    )

    with pytest.raises(ValueError, match="duration_ms must divide cleanly by frame_step_ms"):
        lower_scenario_definition(scenario)


def test_builtin_scenario_definition_for_multi_gap_lowers_to_existing_sequence() -> None:
    scenario = builtin_scenario_definition("multi_gap")

    lowered = lower_scenario_definition(scenario)

    assert [frame.timestamp_ms for frame in lowered] == [50, 100, 150, 200, 250]
    assert [frame.channel_energies for frame in lowered] == [
        [0.9, 0.2, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2],
        [0.2, 0.85, 0.2, 0.1],
        [0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.85, 0.1],
    ]
