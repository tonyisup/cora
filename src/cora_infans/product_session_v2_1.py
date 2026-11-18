"""Product-session V2.1 contracts for callback-before-stop playback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .drained_playback import AudioToolboxOutput
from .drained_playback_v2_1 import (
    PLAYER_VERSION,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
    WORLD_DEVICE_NAME,
    WORLD_DEVICE_UID,
    play_wav_uid_bound_and_completed,
    resolve_uid_bound_output,
)
from .product_session_live import START_CUE_AMPLITUDE, START_CUE_HZ, START_CUE_MS
from .product_session_v2 import v2_unchanged_behavioral_contract


PROTOCOL_VERSION = "bounded-product-session-v2.1-candidate"


def build_v2_1_read_only_preflight(
    operating_point: Mapping[str, object],
    *,
    devices: Sequence[AudioToolboxOutput],
    system_volume_percent: int,
    output_muted: bool,
    player_inspection: Mapping[str, object],
) -> dict[str, object]:
    """Validate V2.1 identities and player contract without live I/O."""

    input_point = operating_point.get("input")
    self_point = operating_point.get("self_source")
    world_point = operating_point.get("world_source")
    playback = operating_point.get("playback")
    if not all(
        isinstance(value, Mapping)
        for value in (input_point, self_point, world_point, playback)
    ):
        raise ValueError("V2.1 operating point is incomplete")
    assert isinstance(input_point, Mapping)
    assert isinstance(self_point, Mapping)
    assert isinstance(world_point, Mapping)
    assert isinstance(playback, Mapping)

    self_output = resolve_uid_bound_output(
        devices,
        expected_uid=SELF_DEVICE_UID,
        expected_name=SELF_DEVICE_NAME,
    )
    world_output = resolve_uid_bound_output(
        devices,
        expected_uid=WORLD_DEVICE_UID,
        expected_name=WORLD_DEVICE_NAME,
    )
    checks = {
        "input_selector_is_colon_zero": input_point.get("device_selector") == ":0",
        "input_sample_rate_is_16000": input_point.get("sample_rate_hz") == 16000,
        "input_has_two_host_channels": input_point.get("host_channels") == 2,
        "input_uses_callback_backend": input_point.get("capture_backend")
        == "sounddevice_portaudio_coreaudio_callback",
        "self_uid_matches": self_point.get("coreaudio_device_uid")
        == SELF_DEVICE_UID,
        "self_name_matches": self_point.get("device") == SELF_DEVICE_NAME,
        "world_uid_matches": world_point.get("coreaudio_device_uid")
        == WORLD_DEVICE_UID,
        "world_name_matches": world_point.get("device") == WORLD_DEVICE_NAME,
        "system_volume_is_100_percent": system_volume_percent == 100
        and self_point.get("system_volume_percent") == 100,
        "output_is_not_muted": output_muted is False,
        "native_player_version_matches": player_inspection.get("player_version")
        == PLAYER_VERSION
        and playback.get("implementation") == PLAYER_VERSION,
        "native_player_binary_is_hash_measured": isinstance(
            player_inspection.get("binary_sha256"), str
        )
        and len(str(player_inspection["binary_sha256"])) == 64,
        "native_player_source_is_hash_measured": isinstance(
            player_inspection.get("source_sha256"), str
        )
        and len(str(player_inspection["source_sha256"])) == 64,
        "native_player_probe_opened_no_device": player_inspection.get(
            "audio_device_opened"
        )
        is False,
        "uid_binding_required": playback.get("device_binding")
        == "coreaudio_uid_with_name_cross_check",
        "read_only_property_enumeration_required": playback.get(
            "device_enumeration"
        )
        == "coreaudio_property_read_only_no_audioqueue",
        "enqueue_before_start_required": playback.get(
            "enqueue_before_start_required"
        )
        is True,
        "buffer_callback_before_stop_required": playback.get(
            "buffer_callback_before_stop_required"
        )
        is True,
        "non_immediate_stop_required": playback.get("non_immediate_stop_required")
        is True,
        "queue_stop_before_return_required": playback.get(
            "queue_not_running_before_return_required"
        )
        is True,
        "non_immediate_dispose_required": playback.get(
            "non_immediate_dispose_required"
        )
        is True,
    }
    return {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "preflight_kind": "read_only_coreaudio_property_v2_1_route_and_player_preflight",
        "checks": checks,
        "preflight_passed": all(checks.values()),
        "resolved_outputs": {
            "self": {
                "index_informational_only": self_output.index,
                "name": self_output.name,
                "uid": self_output.uid,
            },
            "world": {
                "index_informational_only": world_output.index,
                "name": world_output.name,
                "uid": world_output.uid,
            },
        },
        "system_volume_percent": system_volume_percent,
        "output_muted": output_muted,
        "player_inspection": dict(player_inspection),
        "audio_frames_sent": 0,
        "microphone_opened": False,
        "claim_boundary": (
            "Read-only/zero-frame V2.1 preflight. It does not prove audible "
            "playback or authorize a cue, capture, cohort, or product claim."
        ),
    }


def v2_1_playback_callback(
    *,
    player_path: str | Path,
    expected_player_sha256: str,
    devices: Sequence[AudioToolboxOutput] | None = None,
    runner: Callable[..., Any] | None = None,
) -> Callable[..., Mapping[str, object]]:
    """Adapt V2.1 playback to the existing capture callbacks."""

    def playback(path: str, **_ignored: object) -> Mapping[str, object]:
        kwargs: dict[str, object] = {}
        if runner is not None:
            kwargs["runner"] = runner
        return play_wav_uid_bound_and_completed(
            path,
            expected_uid=SELF_DEVICE_UID,
            expected_name=SELF_DEVICE_NAME,
            player_path=player_path,
            expected_player_sha256=expected_player_sha256,
            devices=devices,
            **kwargs,
        )

    return playback


def build_v2_1_cue_confirmation(
    *,
    playback_report: Mapping[str, object],
    owner_heard_cue: bool,
    microphone_opened: bool,
) -> dict[str, object]:
    """Require machine completion evidence and immediate owner audibility."""

    checks = {
        "uid_bound": playback_report.get("device_bound_by_uid") is True,
        "callback_before_stop_confirmed": (
            isinstance(playback_report.get("checks"), Mapping)
            and playback_report["checks"].get("callback_observed_before_stop") is True
            and playback_report["checks"].get("stop_requested_after_callback") is True
        ),
        "completion_confirmed": playback_report.get("playback_passed") is True,
        "owner_heard_cue": owner_heard_cue is True,
        "microphone_remained_closed": microphone_opened is False,
    }
    return {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "cue": {
            "frequency_hz": START_CUE_HZ,
            "duration_ms": START_CUE_MS,
            "amplitude": START_CUE_AMPLITUDE,
        },
        "checks": checks,
        "cue_preflight_passed": all(checks.values()),
        "session_capture_authorized_by_this_record": False,
    }


def v2_1_unchanged_behavioral_contract() -> dict[str, object]:
    contract = v2_unchanged_behavioral_contract()
    return {
        **contract,
        "playback_ordering_changed_from_v2": True,
        "behavioral_or_score_threshold_changed_from_v2": False,
    }
