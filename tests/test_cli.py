import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


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
    assert payload["implemented_slice"] == "bounded_operational_auditory_system"
    assert payload["operational_contract"] == {
        "schema_version": 1,
        "status": "bounded_operational_contract",
        "authority": "VISION.md",
    }
    assert "machine_enforced_operational_contract" in payload["implemented_subsystems"]
    assert "retained_evidence_readiness_aggregator" in payload["implemented_subsystems"]
    assert "self_describing_operating_point_validator" in payload["implemented_subsystems"]
    assert "read_only_operational_preflight" in payload["implemented_subsystems"]
    assert "bounded_live_operational_acceptance" in payload["implemented_subsystems"]
    assert "bounded_offline_speech_learning_demo" in payload["implemented_subsystems"]
    assert "predictive_hierarchy" in payload["implemented_subsystems"]
    assert "commanded_active_attention" in payload["implemented_subsystems"]
    assert "offline_corpus_evaluation" in payload["implemented_subsystems"]
    assert "prior_art_capture_replay" in payload["implemented_subsystems"]
    assert "participant_disjoint_parentese_confirmation" in payload["implemented_subsystems"]
    assert "cross_fieldsite_parentese_falsification" in payload["implemented_subsystems"]
    assert "speaker_and_mode_adaptive_register_gate" in payload["implemented_subsystems"]
    assert "long_window_vocal_activity_gate" in payload["implemented_subsystems"]
    assert "joint_register_pipeline_falsification" in payload["implemented_subsystems"]
    assert "expanded_register_refit_falsification" in payload["implemented_subsystems"]
    assert "safe_online_predictor_regret_gate" in payload["implemented_subsystems"]
    assert "structured_prediction_fake_harness" in payload["implemented_subsystems"]
    assert "structured_prediction_real_runner" in payload["implemented_subsystems"]
    assert "motor_plan_synthesis" in payload["implemented_subsystems"]
    assert "simulation_only_forward_model_robustness_sweep" in payload["implemented_subsystems"]
    assert "simulation_only_forward_model_promotion_gate" in payload["implemented_subsystems"]
    assert "simulation_only_speaking_policy" in payload["implemented_subsystems"]
    assert "simulation_only_speaking_policy_promotion_gate" in payload["implemented_subsystems"]
    assert "simulation_only_speaking_readiness_gate" in payload["implemented_subsystems"]
    assert payload["planned_subsystems"] == []
    assert payload["stopped_programs"] == [
        "stage2_register_detection",
        "stage5_local_inverse_closure",
    ]
    assert "cold_start_parentese_detection" in payload["unsupported_expansions"]
    assert "generalized_no_replay_self_voice" in payload["unsupported_expansions"]
    assert "wall_plug_energy" in payload["unsupported_expansions"]
    assert (
        payload["verdicts"]["source_localization"]
        == "speech_stimulus_recoverable_non_speech_negative"
    )


def test_operational_contract_cli_and_stage6_validation(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    contract = subprocess.run(
        [sys.executable, "-m", "cora_infans", "--show-operational-contract"],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )
    payload = json.loads(contract.stdout)
    assert payload["stage6"]["latency_max_ms"] == 10.5
    assert payload["stage4"]["exact_commanded_utterance_replay_required"] is True

    report = tmp_path / "run.json"
    report.write_text(
        json.dumps(
            {
                "operating_point": {
                    "schema_version": payload["stage6"][
                        "operating_point_schema_version"
                    ],
                    "host": payload["stage6"]["host"],
                    "input": payload["stage6"]["input"],
                    "self_source": payload["stage6"]["self_source"],
                    "world_source": payload["stage6"]["world_source"],
                    "operator_attestations": {
                        "self_source_geometry_confirmed": True,
                        "self_source_system_volume_confirmed": True,
                        "world_source_geometry_confirmed": True,
                        (
                            "world_source_hardware_volume_at_approximately_50_percent"
                        ): True,
                    },
                },
                "real_capture": True,
                "capture_backend": payload["stage6"]["capture_backend"],
                "frame_ms": payload["stage6"]["acceptance_profile"]["frame_ms"],
                "frame_count": payload["stage6"]["acceptance_profile"][
                    "target_frame_count"
                ],
                "salient_frame_count": payload["stage6"]["acceptance_profile"][
                    "minimum_salient_frame_count"
                ],
                "active_playback_frame_count": 250,
                "software_playback_gain_db": payload["stage6"][
                    "acceptance_profile"
                ]["software_playback_gain_db"],
                "stimulus_sha256": payload["stage6"]["acceptance_profile"][
                    "stimulus_sha256"
                ],
                "salient_adc_frame_end_to_workspace_p95_ms": 10.4,
                "device_deadline_miss_count": 0,
                "overflow_count": 0,
                "callback_status_count": 0,
                "timestamp_discontinuity_count": 0,
                "fallback_frame_count": 0,
                "active_playback_clipped_sample_count": 0,
            }
        )
    )
    validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--validate-stage6-operating-point",
            "--report",
            str(report),
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )
    assert json.loads(validation.stdout)["operating_point_passed"] is True


def test_operational_preflight_cli_uses_manifest(monkeypatch, tmp_path) -> None:
    import cora_infans.cli as cli

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"operating_point": {"schema_version": 1}}))
    captured = {}
    monkeypatch.setattr(
        cli,
        "build_operational_preflight",
        lambda payload: captured.update(payload) or {"preflight_passed": True},
    )

    report = cli.operational_preflight_cli(
        ["--operational-preflight", "--manifest", str(manifest)]
    )

    assert report["preflight_passed"] is True
    assert captured["operating_point"] == {"schema_version": 1}


def test_operational_acceptance_cli_requires_authorization() -> None:
    import cora_infans.cli as cli

    with pytest.raises(ValueError, match="authorize-live-io"):
        cli.run_operational_acceptance_cli(
            [
                "--run-operational-acceptance",
                "--manifest",
                "manifest.json",
                "--stimulus-wav",
                "stimulus.wav",
                "--output-dir",
                "out",
            ]
        )


def test_speech_learning_demo_cli_runs_offline_and_retains_artifacts(
    tmp_path,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--demo-speech-learning",
            "--scenario",
            "cross-configuration",
            "--vowel",
            "a",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    report = json.loads(result.stdout)
    assert report["demo_passed"] is True
    assert report["vowels"] == ["a"]
    assert report["sessions"][0]["learning"]["learner_input"][
        "vowel_label_provided"
    ] is False
    assert (tmp_path / "a-target.wav").is_file()
    assert (tmp_path / "a-learned.wav").is_file()
    assert (tmp_path / "speech-learning-session.json").is_file()
    assert (tmp_path / "speech-learning-demo.html").is_file()



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
    import wave

    import numpy as np

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    sample_rate = 8000
    tone = 0.4 * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_rate) / sample_rate)
    samples = np.concatenate([tone, np.zeros(sample_rate)])
    with wave.open(str(tmp_path / "clip01.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((samples * 32767.0).astype("<i2").tobytes())
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
    assert payload["total_frames"] == 8
    assert payload["evidence_kind"] == "real_audio"
    assert payload["input_mode"] == "wav"
    assert payload["per_clip"][0]["sample_rate"] == sample_rate
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


def test_promote_speaking_policy_command_reports_simulation_only_verdict() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, "-m", "cora_infans", "--promote-speaking-policy"],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "simulation_only_speaking_policy_promotion_gate"
    assert payload["verdict"] == "promote_simulation_only"
    assert payload["live_playback"] is False
    assert payload["live_playback_allowed"] is False
    assert all(criterion["passed"] for criterion in payload["criteria"].values())


def test_promote_speaking_readiness_command_requires_both_simulation_gates() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, "-m", "cora_infans", "--promote-speaking-readiness"],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "simulation_only_speaking_readiness_gate"
    assert payload["verdict"] == "promote_simulation_only"
    assert payload["simulated_speak_authorized"] is True
    assert payload["live_playback"] is False
    assert payload["live_playback_allowed"] is False
    assert payload["criteria"]["speaking_policy_promoted"]["passed"] is True
    assert payload["criteria"]["self_forward_model_promoted"]["passed"] is True


def test_speak_live_smoke_cli_is_blocked_without_fake_io() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    with tempfile.TemporaryDirectory() as out_dir:
        result = subprocess.run(
            [sys.executable, "-m", "cora_infans", "--speak-live-smoke", "--out-dir", out_dir],
            capture_output=True,
            check=True,
            text=True,
            env=env,
        )
        payload = json.loads(result.stdout)
        assert payload["mode"] == "bounded_live_speak_smoke"
        assert payload["verdict"] == "blocked_missing_injected_io"
        assert payload["real_hardware"] is False
        assert payload["live_playback"] is False
        assert payload["live_playback_allowed"] is False
        assert Path(payload["artifacts"]["efference_wav"]).exists()
        assert payload["artifacts"]["captured_wav"] is None


def test_speak_live_smoke_cli_runs_fake_io_when_requested() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    with tempfile.TemporaryDirectory() as out_dir:
        result = subprocess.run(
            [sys.executable, "-m", "cora_infans", "--speak-live-smoke", "--out-dir", out_dir, "--fake-io"],
            capture_output=True,
            check=True,
            text=True,
            env=env,
        )
        payload = json.loads(result.stdout)
        assert payload["verdict"] == "fake_io_smoke_complete"
        assert payload["real_hardware"] is False
        assert payload["live_playback"] is False
        assert payload["live_playback_allowed"] is False
        assert Path(payload["artifacts"]["efference_wav"]).exists()
        assert Path(payload["artifacts"]["captured_wav"]).exists()


def test_listen_live_rejects_anchor_without_synchronized_provenance(tmp_path) -> None:
    import pytest

    from cora_infans.cli import listen_live

    anchor = tmp_path / "old-anchor.json"
    anchor.write_text('{"latency_ms": -0.62}', encoding="utf-8")

    with pytest.raises(ValueError, match="synchronized-capture provenance"):
        listen_live(["--listen-live", "--latency-anchor", str(anchor)])


def test_listen_live_calibrates_real_imitation_session_without_anchor(monkeypatch) -> None:
    import cora_infans.cli as cli
    import cora_infans.live_speak_smoke as smoke

    observed = {}
    playback_calls = []

    class _FakeListener:
        def __init__(self, *args, **kwargs):
            pass

        def listen(self, **kwargs):
            observed["listen"] = kwargs
            return {"motor_imitation": None}

    def _fake_calibration(**kwargs):
        observed["calibration"] = kwargs
        kwargs["playback_fn"]("probe.wav")
        return {
            "latency_stable": True,
            "synchronized_capture": True,
            "latency_aligned_ms": 1.25,
        }

    monkeypatch.setattr(cli, "LiveListener", _FakeListener)
    monkeypatch.setattr(cli, "LiveMicArray", lambda device: object())
    monkeypatch.setattr(smoke, "run_real_io_speak_trials", _fake_calibration)
    monkeypatch.setattr(
        smoke,
        "play_wav_to_device",
        lambda path, **kwargs: playback_calls.append(kwargs) or {"completed": True},
    )

    cli.listen_live(
        [
            "--listen-live",
            "--with-motor-imitation",
            "--real-io",
            "--duration-ms",
            "1",
        ]
    )

    assert observed["calibration"]["trials"] == 6
    assert observed["calibration"]["stimulus"].kind == "chirp"
    assert observed["calibration"]["max_capture_ms"] == 3000
    assert observed["calibration"]["causal_window_ms"] == 500.0
    assert playback_calls[0]["device_index"] == 2
    assert observed["listen"]["latency_anchor_ms"] == 1.25
    assert observed["listen"]["self_subtraction_stable_calibration"] is True


def test_listen_live_cli_real_io_path_invokes_playback_fn() -> None:
    # Regression: the --real-io playback/capture lambdas must accept the kwargs
    # play_efference_if_authorized forwards (device_index/sample_rate/volume_db).
    # A prior bug had the lambda reject them -> TypeError on hardware. We inject
    # no-op fakes (no real audio) but drive the full CLI wiring through.
    import cora_infans.cli as cli
    import cora_infans.io.live_listener as ll
    import cora_infans.live_speak_smoke as sm
    import numpy as np
    from cora_infans.pipeline import AudioFrame

    pcm = (np.sin(np.arange(3200) * 0.05) * 30000).astype("<i2")

    class _FakeArray:
        channel_count = 2
        last_pcm = pcm  # type: ignore[assignment]

        def capture_frame(self, *, duration_ms=None, dominant_frequency_hz=None):
            return AudioFrame(
                timestamp_ms=0, channel_energies=[0.05, 0.01], dominant_frequency_hz=1000.0
            )

    class _FakeTuner:
        is_real_hardware = False

        def read_selected_azimuths(self):
            return [0.0, 90.0]

        def read_doa(self):
            class _D:
                azimuth_deg = 0.0
                speech_detected = True

            return _D()

    played = []
    captured = []
    orig_array = cli.LiveMicArray
    orig_gate = sm.run_speaking_readiness_gate
    orig_play = sm.play_wav_to_device
    orig_cap = sm.capture_to_wav
    orig_hw = ll.HardwareTuner
    orig_null = ll._NullTuner
    cli.LiveMicArray = lambda device: _FakeArray()  # type: ignore[assignment]
    # Force the listener to use our fake tuner (finite source azimuth) instead
    # of probing real hardware, so the motor-imitation branch fires.
    ll.HardwareTuner = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[assignment]
        __import__("cora_infans.io.hardware_tuner", fromlist=["HardwareTunerError"]).HardwareTunerError("forced fake")
    )
    ll._NullTuner = _FakeTuner  # type: ignore[assignment]
    sm.run_speaking_readiness_gate = lambda: {"verdict": "promote_simulation_only"}  # type: ignore[assignment]
    sm.play_wav_to_device = lambda path, **kw: played.append(kw) or {"backend": "fake"}  # type: ignore[assignment]
    def _fake_ready_capture(path, **kw):
        captured.append(kw)
        kw["ready_event"].set()
        time.sleep(0.15)
        return {"backend": "fake"}

    sm.capture_to_wav = _fake_ready_capture  # type: ignore[assignment]
    try:
        report = cli.listen_live(
            [
                "--listen-live",
                "--with-motor-imitation",
                "--with-attention",
                "--real-io",
                "--duration-ms",
                "200",
                "--frame-ms",
                "200",
                "--latency-anchor",
                "/tmp/none.json",  # missing file is fine: anchor stays None
            ]
        )
    except FileNotFoundError:
        # latency-anchor file absent -> the CLI raises; acceptable, we only care
        # the playback lambda shape is correct up to that point. Provide one.
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write('{"latency_ms": 0.0, "synchronized_capture": true}')
            anchor = fh.name
        report = cli.listen_live(
            [
                "--listen-live",
                "--with-motor-imitation",
                "--with-attention",
                "--real-io",
                "--duration-ms",
                "200",
                "--frame-ms",
                "200",
                "--latency-anchor",
                anchor,
            ]
        )
    finally:
        cli.LiveMicArray = orig_array  # type: ignore[assignment]
        sm.run_speaking_readiness_gate = orig_gate  # type: ignore[assignment]
        sm.play_wav_to_device = orig_play  # type: ignore[assignment]
        sm.capture_to_wav = orig_cap  # type: ignore[assignment]
        ll.HardwareTuner = orig_hw  # type: ignore[assignment]
        ll._NullTuner = orig_null  # type: ignore[assignment]

    # The playback lambda was invoked with the kwargs play_efference_if_authorized
    # forwards (the exact shape that previously raised TypeError).
    assert played, "playback lambda was never invoked"
    assert "device_index" in played[0]
    assert "volume_db" in played[0]
    mi = report["motor_imitation"]
    assert mi is not None and mi["trial_count"] >= 1
    assert mi["trials"][0]["played"] is True


def test_listen_live_saliency_threshold_flag_parses() -> None:
    # The near-field self-subtraction run needs a lower saliency gate than the
    # default 0.7 so quiet/near-field human speech can trigger motor imitation.
    # Assert --saliency-threshold is parsed and forwarded to LiveListener.
    import cora_infans.cli as cli

    captured = {}

    class _FakeListener:
        def __init__(self, array=None, **kw):
            captured.update(kw)

        def listen(self, **kw):
            captured["listen"] = kw
            return {"motor_imitation": None, "frame_count": 0, "salient_frame_count": 0}

    orig = cli.LiveListener
    cli.LiveListener = _FakeListener  # type: ignore[assignment]
    try:
        cli.listen_live(
            [
                "--listen-live",
                "--saliency-threshold",
                "0.2",
                "--with-motor-imitation",
                "--duration-ms",
                "200",
                "--frame-ms",
                "200",
            ]
        )
    finally:
        cli.LiveListener = orig  # type: ignore[assignment]
    assert captured.get("saliency_threshold") == 0.2


def test_listen_live_sample_rate_flag_reaches_mic_array(monkeypatch) -> None:
    import cora_infans.cli as cli

    captured = {}

    class _FakeArray:
        channel_count = 2

        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeListener:
        def __init__(self, array=None, **kwargs):
            pass

        def listen(self, **kwargs):
            return {"frame_count": 0}

    monkeypatch.setattr(cli, "LiveMicArray", _FakeArray)
    monkeypatch.setattr(cli, "LiveListener", _FakeListener)

    cli.listen_live(
        [
            "--listen-live",
            "--duration-ms",
            "25",
            "--frame-ms",
            "25",
            "--sample-rate",
            "16000",
            "--capture-backend",
            "native",
        ]
    )

    assert captured == {
        "device": ":0",
        "sample_rate": 16000,
        "stream_backend": "native",
    }


def test_listen_live_stimulus_volume_flag_reaches_stimulus(monkeypatch) -> None:
    import cora_infans.cli as cli

    captured = {}

    class _FakeListener:
        def __init__(self, array=None, **kwargs):
            captured.update(kwargs)

        def listen(self, **kwargs):
            return {"frame_count": 0}

    monkeypatch.setattr(cli, "LiveListener", _FakeListener)
    monkeypatch.setattr(cli, "LiveMicArray", lambda **kwargs: object())

    cli.listen_live(
        [
            "--listen-live",
            "--with-stimulus",
            "--stimulus-volume-db",
            "-6",
            "--duration-ms",
            "25",
        ]
    )

    assert captured["stimulus"].volume_db == -6.0
    assert captured["stimulus"].output_device_index == 2


def test_listen_live_with_self_subtraction_requires_real_io() -> None:
    # Phase 2 Task 4: --with-self-subtraction is meaningless without --real-io
    # (no capture to fit/subtract); the CLI must refuse.
    import cora_infans.cli as cli

    try:
        cli.listen_live(
            [
                "--listen-live",
                "--with-motor-imitation",
                "--with-self-subtraction",
                "--duration-ms",
                "200",
                "--frame-ms",
                "200",
            ]
        )
        raise AssertionError("expected ValueError: --with-self-subtraction requires --real-io")
    except ValueError as exc:
        assert "real-io" in str(exc)


def test_listen_live_cli_parses_with_self_subtraction_flag() -> None:
    # Phase 2 Task 4: with --real-io --with-self-subtraction, the CLI wires both
    # playback and capture fns and passes do_self_subtraction=True to the listener.
    import cora_infans.cli as cli
    import cora_infans.io.live_listener as ll
    import cora_infans.live_speak_smoke as sm
    import numpy as np
    import tempfile
    from cora_infans.pipeline import AudioFrame

    pcm = (np.sin(np.arange(3200) * 0.05) * 30000).astype("<i2")

    class _FakeArray:
        channel_count = 2
        last_pcm = pcm  # type: ignore[assignment]

        def capture_frame(self, *, duration_ms=None, dominant_frequency_hz=None):
            return AudioFrame(
                timestamp_ms=0, channel_energies=[0.05, 0.01], dominant_frequency_hz=1000.0
            )

    class _FakeTuner:
        is_real_hardware = False

        def read_selected_azimuths(self):
            return [0.0, 90.0]

        def read_doa(self):
            class _D:
                azimuth_deg = 0.0
                speech_detected = True

            return _D()

    played = []
    captured = []
    orig_array = cli.LiveMicArray
    orig_gate = sm.run_speaking_readiness_gate
    orig_play = sm.play_wav_to_device
    orig_cap = sm.capture_to_wav
    orig_hw = ll.HardwareTuner
    orig_null = ll._NullTuner
    cli.LiveMicArray = lambda device: _FakeArray()  # type: ignore[assignment]
    ll.HardwareTuner = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[assignment]
        __import__("cora_infans.io.hardware_tuner", fromlist=["HardwareTunerError"]).HardwareTunerError("forced fake")
    )
    ll._NullTuner = _FakeTuner  # type: ignore[assignment]
    sm.run_speaking_readiness_gate = lambda: {"verdict": "promote_simulation_only"}  # type: ignore[assignment]
    sm.play_wav_to_device = lambda path, **kw: played.append(kw) or {"backend": "fake"}  # type: ignore[assignment]
    # Capture fn must write a real WAV: the trial reads captured PCM back to fit
    # + subtract, so a no-op fake leaves forward_model_fit None. Write a
    # self+world mixture so the subtraction path has something to act on.
    from cora_infans.io.motor_plan import write_wav
    from cora_infans.io.self_suppression import mix_observed, predict_self_pcm

    _sr = 48000
    _n = _sr // 4
    _eff = (np.sin(np.arange(_n) * 0.05) * 0.8).astype(np.float32)
    _world = (np.sin(np.arange(_n) * 0.20) * 0.3).astype(np.float32)
    _obs = mix_observed(
        predict_self_pcm(_eff, sample_rate=_sr, delay_ms=0.0, attenuation=0.55, echo_decay=0.22),
        _world,
        world_gain=1.0,
    )

    def _fake_capture(path, **kw):
        captured.append(kw)
        write_wav(path, (np.clip(_obs, -1.0, 1.0) * 32767.0).astype("<i2"), _sr)
        kw["ready_event"].set()
        time.sleep(0.15)
        return {"backend": "fake"}

    sm.capture_to_wav = _fake_capture  # type: ignore[assignment]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write('{"latency_ms": -0.62, "synchronized_capture": true}')
        anchor = fh.name
    try:
        report = cli.listen_live(
            [
                "--listen-live",
                "--with-motor-imitation",
                "--with-attention",
                "--real-io",
                "--with-self-subtraction",
                "--duration-ms",
                "200",
                "--frame-ms",
                "200",
                "--latency-anchor",
                anchor,
            ]
        )
    finally:
        cli.LiveMicArray = orig_array  # type: ignore[assignment]
        sm.run_speaking_readiness_gate = orig_gate  # type: ignore[assignment]
        sm.play_wav_to_device = orig_play  # type: ignore[assignment]
        sm.capture_to_wav = orig_cap  # type: ignore[assignment]
        ll.HardwareTuner = orig_hw  # type: ignore[assignment]
        ll._NullTuner = orig_null  # type: ignore[assignment]

    assert played, "playback lambda was never invoked"
    assert captured, "capture lambda must be wired for self-subtraction"
    mi = report["motor_imitation"]
    assert mi is not None and mi["trial_count"] >= 1
    trial = mi["trials"][0]
    assert trial["played"] is True
    # Self-subtraction path executed: world_residual is computed per trial.
    assert trial.get("world_residual") is not None


def test_structured_workspace_real_cli_blocks_before_audio_callbacks(tmp_path) -> None:
    import cora_infans.cli as cli

    preregistration = tmp_path / "blocked-preregistration.json"
    preregistration.write_text(
        json.dumps(
            {
                "mode": "structured_workspace_binding_real_preregistration",
                "status": "blocked_missing_self_geometry",
                "bounded_playback_capture_authorized": True,
                "frozen_hashes": {},
                "hardware": {
                    "input": {"avfoundation_device": ":0", "channels": 2},
                    "self_source": {
                        "output_device_index": 3,
                        "playback_gain_db": 0.0,
                    },
                    "world_source": {
                        "output_device_index": 2,
                        "playback_gain_db": 0.0,
                    },
                },
            }
        )
    )
    output = tmp_path / "must-not-exist"
    calls = []
    original_play = cli.play_wav_to_device
    original_capture = cli.capture_to_wav
    cli.play_wav_to_device = lambda *args, **kwargs: calls.append((args, kwargs))  # type: ignore[assignment]
    cli.capture_to_wav = lambda *args, **kwargs: calls.append((args, kwargs))  # type: ignore[assignment]
    try:
        try:
            cli.structured_workspace_binding_cli(
                [
                    "--evaluate-structured-workspace-binding",
                    "--real-io",
                    "--out-dir",
                    str(output),
                    "--preregistration",
                    str(preregistration),
                    "--authorize-bounded-playback",
                ]
            )
        except ValueError as exc:
            assert "preregistration blocked" in str(exc)
        else:
            raise AssertionError("incomplete H3 preregistration must block")
    finally:
        cli.play_wav_to_device = original_play  # type: ignore[assignment]
        cli.capture_to_wav = original_capture  # type: ignore[assignment]

    assert calls == []
    assert not output.exists()
