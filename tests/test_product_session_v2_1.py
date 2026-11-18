from __future__ import annotations

from cora_infans.drained_playback import AudioToolboxOutput
from cora_infans.drained_playback_v2_1 import (
    PLAYER_VERSION,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
    WORLD_DEVICE_NAME,
    WORLD_DEVICE_UID,
)
from cora_infans.product_session_v2_1 import (
    PROTOCOL_VERSION,
    build_v2_1_cue_confirmation,
    build_v2_1_read_only_preflight,
    v2_1_unchanged_behavioral_contract,
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
            "device_enumeration": "coreaudio_property_read_only_no_audioqueue",
            "device_binding": "coreaudio_uid_with_name_cross_check",
            "enqueue_before_start_required": True,
            "buffer_callback_before_stop_required": True,
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


def test_v2_1_preflight_requires_callback_before_stop_contract() -> None:
    inspection = {
        "player_version": PLAYER_VERSION,
        "binary_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "audio_device_opened": False,
    }
    report = build_v2_1_read_only_preflight(
        _operating_point(),
        devices=_devices(),
        system_volume_percent=100,
        output_muted=False,
        player_inspection=inspection,
    )
    assert report["preflight_passed"] is True
    assert report["audio_frames_sent"] == 0
    assert report["microphone_opened"] is False
    assert report["checks"]["buffer_callback_before_stop_required"] is True
    assert report["checks"]["read_only_property_enumeration_required"] is True

    unsafe = _operating_point()
    unsafe["playback"]["buffer_callback_before_stop_required"] = False
    failed = build_v2_1_read_only_preflight(
        unsafe,
        devices=_devices(),
        system_volume_percent=100,
        output_muted=False,
        player_inspection=inspection,
    )
    assert failed["preflight_passed"] is False


def test_v2_1_cue_confirmation_requires_ordering_and_audibility() -> None:
    playback = {
        "device_bound_by_uid": True,
        "playback_passed": True,
        "checks": {
            "callback_observed_before_stop": True,
            "stop_requested_after_callback": True,
        },
    }
    passed = build_v2_1_cue_confirmation(
        playback_report=playback,
        owner_heard_cue=True,
        microphone_opened=False,
    )
    assert passed["cue_preflight_passed"] is True
    assert passed["session_capture_authorized_by_this_record"] is False

    playback["checks"]["callback_observed_before_stop"] = False
    failed = build_v2_1_cue_confirmation(
        playback_report=playback,
        owner_heard_cue=True,
        microphone_opened=False,
    )
    assert failed["cue_preflight_passed"] is False


def test_v2_1_changes_no_behavioral_or_score_threshold() -> None:
    contract = v2_1_unchanged_behavioral_contract()
    assert PROTOCOL_VERSION == "bounded-product-session-v2.1-candidate"
    assert contract["playback_ordering_changed_from_v2"] is True
    assert contract["behavioral_or_score_threshold_changed_from_v1"] is False
    assert contract["behavioral_or_score_threshold_changed_from_v2"] is False
    assert contract["retention_days"] == 30
