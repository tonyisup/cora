"""Tests for self-describing operational preflight and acceptance."""

import hashlib
import json
from pathlib import Path

import pytest

from cora_infans.operational_acceptance import (
    build_operational_preflight,
    load_operating_point_manifest,
    run_operational_acceptance,
    stage6_report_from_live_listener,
)
from cora_infans.operational_contract import (
    OPERATIONAL_CONTRACT,
    evaluate_stage6_operating_point,
)


def _manifest() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    manifest = load_operating_point_manifest(root / "config/operational-point.json")
    manifest["operating_point"]["operator_attestations"][
        "world_source_hardware_volume_at_approximately_50_percent"
    ] = True
    return manifest


def _passing_probe() -> dict[str, object]:
    return {
        "platform_system": "Darwin",
        "platform_release": "test",
        "host_name": "Mac mini",
        "host_model": "Mac15,12",
        "host_chip": "Apple M4 Pro",
        "ffmpeg_path": "/opt/homebrew/bin/ffmpeg",
        "ffmpeg_has_audiotoolbox": True,
        "ffmpeg_error": None,
        "xvf3800_input": {
            "index": 0,
            "name": "reSpeaker XVF3800 4-Mic Array",
            "max_input_channels": 2,
        },
        "xvf3800_16khz_two_channel_supported": True,
        "input_probe_error": None,
        "enumerated_device_count": 4,
        "portaudio_output_device_names": ["reSpeaker XVF3800 4-Mic Array"],
        "coreaudio_output_device_names": [
            "External Headphones",
            "Mac mini Speakers",
        ],
    }


def _raw_listener_report() -> dict[str, object]:
    frames = [
        {
            "stimulus_playback_active": index < 250,
            "clipped_sample_count": 0,
        }
        for index in range(400)
    ]
    return {
        "frame_count": 400,
        "salient_frame_count": 120,
        "frames": frames,
        "stimulus": {
            "volume_db": -18.0,
            "source_wav_sha256": OPERATIONAL_CONTRACT["stage6"][
                "acceptance_profile"
            ]["stimulus_sha256"],
        },
        "runtime_performance": {
            "real_capture": True,
            "capture_backend": "sounddevice_portaudio_coreaudio_callback",
            "deadline_ms": 25.0,
            "device_timestamped_deadline_miss_count": 0,
            "input_overflow_count": 0,
            "workload_latency_ms": {
                "salient": {
                    "adc_frame_end_to_workspace_decision": {"p95": 10.4}
                }
            },
        },
        "capture_callback_status_count": 0,
        "capture_callback_timestamp_discontinuity_count": 0,
        "capture_stream_fallback_reason": None,
        "capture_quality": {"clipped_sample_count": 0},
    }


def test_preflight_validates_evidence_manifest_host_and_device() -> None:
    report = build_operational_preflight(
        _manifest(), device_probe=_passing_probe
    )

    assert report["preflight_passed"] is True
    assert all(report["checks"].values())
    assert "no audio stream is opened" in report["claim_boundary"]


def test_repository_manifest_requires_fresh_fifty_percent_confirmation() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_operating_point_manifest(root / "config/operational-point.json")

    report = build_operational_preflight(
        manifest, device_probe=_passing_probe
    )

    assert report["preflight_passed"] is False
    assert report["checks"]["world_volume_operator_confirmed"] is False


def test_preflight_uses_coreaudio_not_portaudio_for_output_routes() -> None:
    probe = _passing_probe()
    probe["portaudio_output_device_names"] = [
        "ASUS PB287Q",
        "reSpeaker XVF3800 4-Mic Array",
    ]

    report = build_operational_preflight(
        _manifest(), device_probe=lambda: probe
    )

    assert report["preflight_passed"] is True
    assert report["checks"]["external_headphones_discovered"] is True
    assert report["checks"]["mac_mini_speakers_discovered"] is True


def test_listener_report_normalizes_to_complete_operating_point() -> None:
    manifest = _manifest()
    report = stage6_report_from_live_listener(
        _raw_listener_report(), manifest["operating_point"]
    )
    evaluation = evaluate_stage6_operating_point(report)

    assert report["active_playback_frame_count"] == 250
    assert report["active_playback_clipped_sample_count"] == 0
    assert evaluation["operating_point_passed"] is True


def test_quiet_only_live_report_is_a_retained_gate_failure_not_schema_error() -> None:
    manifest = _manifest()
    raw = _raw_listener_report()
    raw["salient_frame_count"] = 0
    raw["runtime_performance"]["workload_latency_ms"] = {
        "quiet": {
            "adc_frame_end_to_workspace_decision": {"p95": 10.2}
        }
    }

    report = stage6_report_from_live_listener(
        raw, manifest["operating_point"]
    )
    evaluation = evaluate_stage6_operating_point(report)

    assert report["salient_adc_frame_end_to_workspace_p95_ms"] is None
    assert evaluation["operating_point_passed"] is False
    assert evaluation["criteria"]["minimum_salient_frames_met"] is False
    assert evaluation["criteria"]["salient_latency_available"] is False


def test_live_acceptance_writes_raw_and_normalized_reports(
    tmp_path, monkeypatch
) -> None:
    stimulus = tmp_path / "stimulus.wav"
    stimulus.write_bytes(b"already-spent-test-stimulus")
    digest = hashlib.sha256(stimulus.read_bytes()).hexdigest()
    monkeypatch.setitem(
        OPERATIONAL_CONTRACT["stage6"]["acceptance_profile"],
        "stimulus_sha256",
        digest,
    )
    raw = _raw_listener_report()
    raw["stimulus"]["source_wav_sha256"] = digest

    report = run_operational_acceptance(
        manifest=_manifest(),
        stimulus_wav=stimulus,
        output_dir=tmp_path / "out",
        live_runner=lambda _: raw,
        device_probe=_passing_probe,
    )

    assert report["operational_acceptance_passed"] is True
    assert (tmp_path / "out/stage6-live-raw.json").is_file()
    retained = json.loads(
        (tmp_path / "out/operational-acceptance.json").read_text()
    )
    assert retained["operational_acceptance_passed"] is True


def test_preflight_failure_blocks_live_runner(tmp_path) -> None:
    called = False

    def runner(_: str) -> dict[str, object]:
        nonlocal called
        called = True
        return _raw_listener_report()

    with pytest.raises(ValueError, match="preflight failed"):
        run_operational_acceptance(
            manifest=_manifest(),
            stimulus_wav=tmp_path / "missing.wav",
            output_dir=tmp_path / "out",
            live_runner=runner,
            device_probe=lambda: {
                **_passing_probe(),
                "platform_system": "Linux",
            },
        )
    assert called is False


def test_completed_live_io_retains_failure_report_when_schema_is_invalid(
    tmp_path, monkeypatch
) -> None:
    stimulus = tmp_path / "stimulus.wav"
    stimulus.write_bytes(b"already-spent-invalid-report-test")
    digest = hashlib.sha256(stimulus.read_bytes()).hexdigest()
    monkeypatch.setitem(
        OPERATIONAL_CONTRACT["stage6"]["acceptance_profile"],
        "stimulus_sha256",
        digest,
    )

    with pytest.raises(ValueError, match="raw and failure reports"):
        run_operational_acceptance(
            manifest=_manifest(),
            stimulus_wav=stimulus,
            output_dir=tmp_path / "out",
            live_runner=lambda _: {"completed": True},
            device_probe=_passing_probe,
        )

    assert (tmp_path / "out/stage6-live-raw.json").is_file()
    failure = json.loads(
        (tmp_path / "out/operational-acceptance.json").read_text()
    )
    assert failure["operational_acceptance_passed"] is False
    assert failure["error"].startswith("invalid live report:")
