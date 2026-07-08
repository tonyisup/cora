from cora_infans.simulation import (
    SimulationParams,
    build_builtin_sequence_report,
    build_sequence_report,
    build_simulation_report,
)



def test_build_simulation_report_returns_salient_status_and_resolved_config() -> None:
    report = build_simulation_report(SimulationParams())

    assert report["status"] == "salient"
    assert report["focus_path"] == ["channel_1"]
    assert report["resolved_config"]["mic_channel_count"] == 4
    assert report["resolved_config"]["saliency_threshold"] == 0.7
    assert report["event"] is not None
    assert report["event"]["workspace_state"]["focus_label"] == "channel_1"
    assert report["event"]["workspace_state"]["saliency_reward"] == 0.5
    assert report["event"]["workspace_state"]["predictive_residual_hz"] == 200.0



def test_build_simulation_report_returns_non_salient_when_threshold_blocks_event() -> None:
    report = build_simulation_report(
        SimulationParams(saliency_threshold=0.95)
    )

    assert report["status"] == "non_salient"
    assert report["event"] is None
    assert report["workspace_snapshot"] == []



def test_build_simulation_report_applies_parameter_overrides() -> None:
    report = build_simulation_report(
        SimulationParams(
            timestamp_ms=250,
            dominant_frequency_hz=350.0,
            channel_energies=[0.1, 0.2, 0.8, 0.3],
            initial_prediction_hz=300.0,
        )
    )

    assert report["frame"]["timestamp_ms"] == 250
    assert report["event"]["workspace_state"]["focus_channel"] == 2
    assert report["event"]["workspace_state"]["updated_prediction_hz"] == 325.0



def test_build_sequence_report_tracks_focus_transitions_over_time() -> None:
    report = build_sequence_report(
        [
            SimulationParams(
                timestamp_ms=100,
                channel_energies=[0.9, 0.2, 0.1, 0.1],
                mic_channel_geometry=(
                    "front_left",
                    "front_right",
                    "rear_right",
                    "rear_left",
                ),
            ),
            SimulationParams(
                timestamp_ms=200,
                channel_energies=[0.2, 0.8, 0.2, 0.1],
                mic_channel_geometry=(
                    "front_left",
                    "front_right",
                    "rear_right",
                    "rear_left",
                ),
            ),
            SimulationParams(
                timestamp_ms=300,
                channel_energies=[0.2, 0.2, 0.85, 0.1],
                mic_channel_geometry=(
                    "front_left",
                    "front_right",
                    "rear_right",
                    "rear_left",
                ),
            ),
        ]
    )

    assert report["frame_count"] == 3
    assert report["emitted_event_count"] == 3
    assert report["focus_path"] == ["front_left", "front_right", "rear_right"]
    assert report["focus_transitions"] == [
        "front_left->front_right",
        "front_right->rear_right",
    ]
    assert report["workspace_snapshot"][-1]["workspace_state"]["focus_label"] == "rear_right"



def test_build_builtin_sequence_report_supports_zigzag_family() -> None:
    report = build_builtin_sequence_report("zigzag")

    assert report["family"] == "zigzag"
    assert report["frame_count"] == 4
    assert report["focus_path"] == [
        "front_left",
        "rear_right",
        "front_right",
        "rear_left",
    ]
    assert report["focus_transitions"] == [
        "front_left->rear_right",
        "rear_right->front_right",
        "front_right->rear_left",
    ]



def test_build_builtin_sequence_report_supports_gap_family() -> None:
    report = build_builtin_sequence_report("gap")

    assert report["family"] == "gap"
    assert report["frame_count"] == 3
    assert report["salient_frame_count"] == 2
    assert report["non_salient_frame_count"] == 1
    assert report["emitted_event_count"] == 2
    assert report["focus_path"] == ["front_left", "rear_right"]
    assert report["focus_transitions"] == ["front_left->rear_right"]
    assert len(report["frames"]) == 3
    assert len(report["events"]) == 2



def test_build_builtin_sequence_report_supports_multi_gap_family() -> None:
    report = build_builtin_sequence_report("multi_gap")

    assert report["family"] == "multi_gap"
    assert report["frame_count"] == 5
    assert report["salient_frame_count"] == 3
    assert report["non_salient_frame_count"] == 2
    assert report["dropout_count"] == 2
    assert report["recovery_count"] == 2
    assert report["emitted_event_count"] == 3
    assert report["focus_path"] == ["front_left", "front_right", "rear_right"]
    assert report["focus_transitions"] == [
        "front_left->front_right",
        "front_right->rear_right",
    ]



def test_build_builtin_sequence_report_supports_delayed_recovery_family() -> None:
    report = build_builtin_sequence_report("delayed_recovery")

    assert report["family"] == "delayed_recovery"
    assert report["frame_count"] == 5
    assert report["salient_frame_count"] == 2
    assert report["non_salient_frame_count"] == 3
    assert report["dropout_count"] == 3
    assert report["recovery_count"] == 1
    assert report["emitted_event_count"] == 2
    assert report["focus_path"] == ["front_left", "rear_right"]
    assert report["focus_transitions"] == ["front_left->rear_right"]


def test_build_sequence_report_includes_theta_gamma_by_default() -> None:
    # Scheduling is now part of the default report path.
    report = build_sequence_report(
        [
            SimulationParams(
                timestamp_ms=100,
                channel_energies=[0.9, 0.2, 0.1, 0.1],
                mic_channel_geometry=("front_left", "front_right", "rear_right", "rear_left"),
            ),
            SimulationParams(
                timestamp_ms=200,
                channel_energies=[0.2, 0.8, 0.2, 0.1],
                mic_channel_geometry=("front_left", "front_right", "rear_right", "rear_left"),
            ),
        ]
    )
    assert "theta_gamma" in report
    assert report["theta_gamma"]["theta_ms"] == 200
    assert report["theta_gamma"]["gamma_ms"] == 25
    assert report["theta_gamma"]["resident_cycle_count"] >= 1


def test_build_sequence_report_can_omit_scheduling() -> None:
    report = build_sequence_report(
        [
            SimulationParams(
                timestamp_ms=100,
                channel_energies=[0.9, 0.2, 0.1, 0.1],
                mic_channel_geometry=("front_left", "front_right", "rear_right", "rear_left"),
            ),
        ],
        with_scheduling=False,
    )
    assert "theta_gamma" not in report
