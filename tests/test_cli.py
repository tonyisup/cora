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
    assert "motor_plan_synthesis" in payload["implemented_subsystems"]
    assert "simulation_only_speaking_policy" in payload["implemented_subsystems"]
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


def test_self_suppression_command_reports_svsr_without_live_playback() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--simulate-self-suppression",
            "--vowel",
            "a",
            "--world-vowel",
            "o",
            "--world-gain",
            "0.25",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "simulation_only_signal_self_suppression"
    assert payload["live_playback"] is False
    assert payload["vowel"] == "a"
    assert payload["world_vowel"] == "o"
    assert payload["svsr_db"] > 3.0
    assert payload["residual_rms"] < payload["observed_rms"]
    assert payload["verdict"] in ("self_suppressed", "partial_suppression")


def test_learn_self_forward_model_command_reports_prediction_error_improvement() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--learn-self-forward-model",
            "--vowel",
            "a",
            "--observed-delay-ms",
            "18",
            "--observed-attenuation",
            "0.55",
            "--observed-echo-decay",
            "0.22",
            "--world-vowel",
            "o",
            "--world-gain",
            "0.25",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "simulation_only_forward_model_learning"
    assert payload["live_playback"] is False
    assert payload["learned_svsr_db"] > payload["initial_svsr_db"]
    assert payload["svsr_improvement_db"] > 3.0
    assert payload["learned_params"]["delay_ms"] == 18.0
    assert payload["world_residual_correlation"] > 0.2


def test_sweep_self_forward_model_command_reports_robustness_surface() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--sweep-self-forward-model",
            "--vowel",
            "a",
            "--world-vowel",
            "o",
            "--world-gain",
            "0.25",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "simulation_only_forward_model_robustness_sweep"
    assert payload["live_playback"] is False
    assert payload["case_count"] > 1
    assert payload["summary"]["best_svsr_db"] > payload["summary"]["worst_svsr_db"]
    assert payload["summary"]["max_world_residual_correlation"] > 0.2


def test_promote_self_forward_model_command_reports_simulation_only_verdict() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--promote-self-forward-model",
            "--vowel",
            "a",
            "--world-vowel",
            "o",
            "--world-gain",
            "0.25",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "simulation_only_forward_model_promotion_gate"
    assert payload["verdict"] == "promote_simulation_only"
    assert payload["live_playback"] is False
    assert payload["live_playback_allowed"] is False
    assert all(item["passed"] for item in payload["criteria"].values())



def test_synthesize_vowel_command_writes_wav_and_closes_loop(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    out = tmp_path / "v.wav"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--synthesize-vowel",
            "a",
            "--out",
            str(out),
            "--duration-ms",
            "200",
            "--analyze",
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["vowel"] == "a"
    assert payload["out"] == str(out)
    assert out.exists()
    # --analyze closes the loop: perception + motor-plan re-derivation present.
    assert "perceived_formants_hz" in payload
    assert payload["perceived_nearest_vowel"] in ("a", "e", "i", "o", "u")
    assert payload["motor_plan_from_perception"]["vowel"] == payload["perceived_nearest_vowel"]


def test_listen_live_cli_handler_wires_motor_imitation() -> None:
    # Exercises the --listen-live handler's --with-motor-imitation flag without
    # real hardware by substituting a fake array + bus-free tuner in-process.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener, _NullTuner
    from cora_infans.io.live_mic import LiveMicArray
    from cora_infans.pipeline import AudioFrame
    from cora_infans import cli as cli_mod

    class _FakeArray:
        channel_count = 2
        last_pcm = None

        def __init__(self, frames):
            self._frames = frames
            self._i = 0

        def capture_frame(self, duration_ms=None, dominant_frequency_hz=None):
            item = self._frames[self._i % len(self._frames)]
            self._i += 1
            self.last_pcm = item["pcm"]
            return item["frame"]

    from cora_infans.io.motor_plan import synthesize_vowel

    def _frame(ch0, ch1):
        return {
            "pcm": synthesize_vowel("a", duration_ms=240, sample_rate=16000),
            "frame": AudioFrame(
                timestamp_ms=0, channel_energies=[ch0, ch1], dominant_frequency_hz=300.0
            ),
        }

    fake = _FakeArray([_frame(0.01, 0.01), _frame(0.20, 0.25)])
    tuner = _NullTuner()
    tuner.read_selected_azimuths = lambda: [1.0, 0.0]  # finite speech azimuth
    real_array_cls = cli_mod.LiveMicArray
    real_resolve = LiveListener._resolve_tuner
    try:
        # Route the handler's LiveMicArray(...) construction to the fake array.
        cli_mod.LiveMicArray = lambda device=None: fake  # type: ignore[assignment]
        LiveListener._resolve_tuner = lambda self: tuner  # type: ignore[method-assign]
        report = cli_mod.listen_live(
            ["--listen-live", "--with-attention", "--with-motor-imitation", "--duration-ms", "500", "--frame-ms", "250"]
        )
    finally:
        cli_mod.LiveMicArray = real_array_cls  # type: ignore[assignment]
        LiveListener._resolve_tuner = real_resolve  # type: ignore[method-assign]

    assert "motor_imitation" in report
    assert report["motor_imitation"]["enabled"] is True
    assert report["motor_imitation"]["trial_count"] >= 1


