from __future__ import annotations

import json

from cora_infans.drained_playback import (
    AudioToolboxOutput,
    PLAYER_VERSION,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
    WORLD_DEVICE_NAME,
    WORLD_DEVICE_UID,
)
from cora_infans.product_session_v2 import (
    PROTOCOL_VERSION,
    build_v2_cue_confirmation,
    build_v2_read_only_preflight,
    v2_playback_callback,
    v2_unchanged_behavioral_contract,
)


def _operating_point() -> dict[str, object]:
    return {
        "input": {
            "device_selector": ":0",
            "sample_rate_hz": 16000,
            "host_channels": 2,
            "capture_backend": "sounddevice_portaudio_coreaudio_callback",
        },
        "self_source": {
            "coreaudio_device_uid": SELF_DEVICE_UID,
            "device": SELF_DEVICE_NAME,
            "system_volume_percent": 100,
        },
        "world_source": {
            "coreaudio_device_uid": WORLD_DEVICE_UID,
            "device": WORLD_DEVICE_NAME,
        },
        "playback": {
            "implementation": PLAYER_VERSION,
            "device_binding": "coreaudio_uid_with_name_cross_check",
            "buffer_callback_required": True,
            "non_immediate_stop_required": True,
            "queue_not_running_before_return_required": True,
            "non_immediate_dispose_required": True,
        },
    }


def _devices() -> tuple[AudioToolboxOutput, ...]:
    return (
        AudioToolboxOutput(2, WORLD_DEVICE_NAME, WORLD_DEVICE_UID),
        AudioToolboxOutput(3, SELF_DEVICE_NAME, SELF_DEVICE_UID),
    )


def _inspection(version: str = PLAYER_VERSION) -> dict[str, object]:
    return {
        "player_version": version,
        "binary_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "audio_device_opened": False,
    }


def test_v2_preflight_binds_uids_and_treats_indices_as_informational() -> None:
    report = build_v2_read_only_preflight(
        _operating_point(),
        devices=_devices(),
        system_volume_percent=100,
        output_muted=False,
        player_inspection=_inspection(),
    )
    assert report["preflight_passed"] is True
    assert report["audio_frames_sent"] == 0
    assert report["microphone_opened"] is False
    assert report["resolved_outputs"]["self"] == {
        "index_informational_only": 3,
        "name": SELF_DEVICE_NAME,
        "uid": SELF_DEVICE_UID,
    }


def test_v2_preflight_fails_closed_on_volume_or_player_version() -> None:
    report = build_v2_read_only_preflight(
        _operating_point(),
        devices=_devices(),
        system_volume_percent=84,
        output_muted=False,
        player_inspection=_inspection("wrong"),
    )
    assert report["preflight_passed"] is False
    assert report["checks"]["system_volume_is_100_percent"] is False
    assert report["checks"]["native_player_version_matches"] is False


def test_v2_cue_confirmation_needs_machine_drain_and_owner_audibility() -> None:
    passed = build_v2_cue_confirmation(
        playback_report={"device_bound_by_uid": True, "playback_passed": True},
        owner_heard_cue=True,
        microphone_opened=False,
    )
    assert passed["cue_preflight_passed"] is True
    assert passed["session_capture_authorized_by_this_record"] is False

    for playback, heard in (
        ({"device_bound_by_uid": False, "playback_passed": True}, True),
        ({"device_bound_by_uid": True, "playback_passed": False}, True),
        ({"device_bound_by_uid": True, "playback_passed": True}, False),
    ):
        failed = build_v2_cue_confirmation(
            playback_report=playback,
            owner_heard_cue=heard,
            microphone_opened=False,
        )
        assert failed["cue_preflight_passed"] is False


def test_v2_playback_adapter_ignores_legacy_index_and_uses_uid(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_play(path, **kwargs):
        calls.append((path, kwargs))
        return {"playback_passed": True}

    monkeypatch.setattr(
        "cora_infans.product_session_v2.play_wav_uid_bound_and_drained",
        fake_play,
    )
    adapter = v2_playback_callback(
        player_path=tmp_path / "player",
        expected_player_sha256="a" * 64,
        devices=_devices(),
    )
    result = adapter("cue.wav", device_index=999, volume_db=99.0)
    assert result["playback_passed"] is True
    assert calls[0][1]["expected_uid"] == SELF_DEVICE_UID
    assert calls[0][1]["expected_name"] == SELF_DEVICE_NAME
    assert "device_index" not in calls[0][1]


def test_v2_changes_only_the_playback_and_identity_boundary() -> None:
    contract = v2_unchanged_behavioral_contract()
    assert PROTOCOL_VERSION == "bounded-product-session-v2-candidate"
    assert contract == {
        "targets": ["a", "i", "u"],
        "development_schedule": ["a", "i", "u", "a", "i", "u"],
        "held_schedule": ["a", "i", "u", "a", "i", "u"],
        "capture_duration_ms": 5000,
        "retention_days": 30,
        "scoring_version": "offline-replay-v2",
        "representation_version": "v4-label-blind-30-band-auditory",
        "behavioral_or_score_threshold_changed_from_v1": False,
    }


def test_v2_operating_point_file_contains_no_frozen_output_indices() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "config/product-session-v2-operating-point.json").read_text()
    )
    self_source = payload["operating_point"]["self_source"]
    world_source = payload["operating_point"]["world_source"]
    assert self_source["coreaudio_device_uid"] == SELF_DEVICE_UID
    assert world_source["coreaudio_device_uid"] == WORLD_DEVICE_UID
    assert "output_device_index" not in self_source
    assert "output_device_index" not in world_source
