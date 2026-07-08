import json
import os
import subprocess
import sys



def test_describe_command_prints_json_summary() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, "-m", "cora_infans", "--describe"],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["project"] == "cora-infans"
    assert payload["implemented_slice"] == "saliency_workspace_live_attention"
    assert "predictive_hierarchy" in payload["implemented_subsystems"]
    assert "commanded_active_attention" in payload["implemented_subsystems"]
    assert "offline_corpus_evaluation" in payload["implemented_subsystems"]
    assert "analysis_by_synthesis" in payload["planned_subsystems"]
    assert "non_speech_source_localization" in payload["planned_subsystems"]
    assert (
        payload["verdicts"]["source_localization"]
        == "speech_stimulus_recoverable_non_speech_negative"
    )



def test_simulate_frame_command_prints_event_and_workspace_snapshot() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, "-m", "cora_infans", "--simulate-frame"],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["event"]["kind"] == "salient_audio"
    assert payload["event"]["workspace_state"]["focus_channel"] == 1
    assert payload["event"]["workspace_state"]["predictive_residual_hz"] == 200.0
    assert payload["event"]["workspace_state"]["updated_prediction_hz"] == 1100.0
    assert len(payload["workspace_snapshot"]) == 1
    assert payload["workspace_snapshot"][0] == payload["event"]



def test_simulate_frame_command_accepts_parameter_overrides() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-frame",
            "--timestamp-ms",
            "250",
            "--dominant-frequency-hz",
            "350.0",
            "--channel-energies",
            "0.1,0.2,0.8,0.3",
            "--initial-prediction-hz",
            "300.0",
            "--mic-geometry",
            "front_left,front_right,rear_right,rear_left",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["frame"]["timestamp_ms"] == 250
    assert payload["resolved_config"]["mic_channel_geometry"] == [
        "front_left",
        "front_right",
        "rear_right",
        "rear_left",
    ]
    assert payload["event"]["workspace_state"]["focus_channel"] == 2
    assert payload["event"]["workspace_state"]["focus_label"] == "rear_right"
    assert payload["event"]["workspace_state"]["dominant_frequency_hz"] == 350.0
    assert payload["event"]["workspace_state"]["predictive_residual_hz"] == 50.0
    assert payload["event"]["workspace_state"]["updated_prediction_hz"] == 325.0



def test_simulate_frame_command_rejects_wrong_channel_count() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-frame",
            "--channel-energies",
            "0.1,0.2,0.8",
        ],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "expected 4 channels, got 3" in result.stderr



def test_simulate_frame_command_rejects_non_numeric_frequency() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-frame",
            "--dominant-frequency-hz",
            "nope",
        ],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "invalid dominant frequency" in result.stderr



def test_simulate_frame_command_can_emit_no_event_when_threshold_is_higher_than_signal() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-frame",
            "--saliency-threshold",
            "0.95",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["status"] == "non_salient"
    assert payload["resolved_config"]["saliency_threshold"] == 0.95
    assert payload["event"] is None
    assert payload["workspace_snapshot"] == []



def test_simulate_frame_command_surfaces_status_and_resolved_config_for_salient_case() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, "-m", "cora_infans", "--simulate-frame"],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["status"] == "salient"
    assert payload["resolved_config"]["mic_channel_count"] == 4
    assert payload["resolved_config"]["saliency_threshold"] == 0.7
    assert payload["resolved_config"]["workspace_window_ms"] == 250



def test_simulate_frame_command_pretty_output_prints_human_readable_json() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-frame",
            "--output",
            "pretty",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    assert '  "status": "salient"' in result.stdout
    assert '  "resolved_config": {' in result.stdout
    assert "\n" in result.stdout

    payload = json.loads(result.stdout)
    assert payload["status"] == "salient"



def test_simulate_sequence_command_prints_focus_transition_report() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, "-m", "cora_infans", "--simulate-sequence"],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["family"] == "sweep"
    assert payload["frame_count"] == 3
    assert payload["emitted_event_count"] == 3
    assert payload["focus_path"] == ["front_left", "front_right", "rear_right"]
    assert payload["focus_transitions"] == [
        "front_left->front_right",
        "front_right->rear_right",
    ]
    # theta-gamma is now part of the default sequence report path.
    assert "theta_gamma" in payload
    assert payload["theta_gamma"]["theta_ms"] == 200



def test_simulate_sequence_command_accepts_zigzag_family_override() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-sequence",
            "--family",
            "zigzag",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["family"] == "zigzag"
    assert payload["frame_count"] == 4
    assert payload["focus_path"] == [
        "front_left",
        "rear_right",
        "front_right",
        "rear_left",
    ]



def test_simulate_sequence_command_accepts_gap_family_override() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-sequence",
            "--family",
            "gap",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["family"] == "gap"
    assert payload["frame_count"] == 3
    assert payload["salient_frame_count"] == 2
    assert payload["non_salient_frame_count"] == 1
    assert payload["emitted_event_count"] == 2
    assert payload["focus_path"] == ["front_left", "rear_right"]
    assert payload["focus_transitions"] == ["front_left->rear_right"]



def test_simulate_sequence_command_accepts_multi_gap_family_override() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-sequence",
            "--family",
            "multi_gap",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["family"] == "multi_gap"
    assert payload["frame_count"] == 5
    assert payload["salient_frame_count"] == 3
    assert payload["non_salient_frame_count"] == 2
    assert payload["dropout_count"] == 2
    assert payload["recovery_count"] == 2
    assert payload["emitted_event_count"] == 3
    assert payload["focus_path"] == ["front_left", "front_right", "rear_right"]
    assert payload["focus_transitions"] == [
        "front_left->front_right",
        "front_right->rear_right",
    ]



def test_simulate_sequence_command_accepts_delayed_recovery_family_override() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-sequence",
            "--family",
            "delayed_recovery",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["family"] == "delayed_recovery"
    assert payload["frame_count"] == 5
    assert payload["salient_frame_count"] == 2
    assert payload["non_salient_frame_count"] == 3
    assert payload["dropout_count"] == 3
    assert payload["recovery_count"] == 1
    assert payload["emitted_event_count"] == 2
    assert payload["focus_path"] == ["front_left", "rear_right"]
    assert payload["focus_transitions"] == ["front_left->rear_right"]


def test_eval_corpus_command_runs_offline_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    (tmp_path / "clip01.wav").write_bytes(b"RIFF")
    label_csv = tmp_path / "labels.csv"
    label_csv.write_text(
        "clip_id,start_s,end_s,label\n"
        "clip01,0.0,1.0,salient_speech\n"
        "clip01,1.0,2.0,non_speech\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--eval-corpus",
            "--root",
            str(tmp_path),
            "--label-csv",
            str(label_csv),
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["clip_loaded"] == 1
    assert payload["clip_count"] == 1
    assert payload["total_frames"] == 4  # two intervals * (on + quiet) frames
    assert payload["per_clip"][0]["clip_id"] == "clip01"
