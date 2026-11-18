"""Product-session V2 contracts built on UID-bound drained playback.

V2 preserves V1's target representation, repertoire, score thresholds,
retention, fixed schedule, and no-retry semantics.  It replaces only the
defective playback boundary: output devices are bound by CoreAudio UID and a
native AudioQueue helper must prove callback and queue drain before returning.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .drained_playback import (
    AudioToolboxOutput,
    PLAYER_VERSION,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
    WORLD_DEVICE_NAME,
    WORLD_DEVICE_UID,
    play_wav_uid_bound_and_drained,
    resolve_uid_bound_output,
)
from .product_session_live import (
    CAPTURE_DURATION_MS,
    DEVELOPMENT_SCHEDULE,
    HELD_SCHEDULE,
    RETENTION_DAYS,
    START_CUE_AMPLITUDE,
    START_CUE_HZ,
    START_CUE_MS,
    TARGETS,
    build_prediction_commitment,
    commit_label_blind_selection,
    freeze_development_associations,
    sha256_json,
)


PROTOCOL_VERSION = "bounded-product-session-v2-candidate"
SCORING_VERSION = "offline-replay-v2"
REPRESENTATION_VERSION = "v4-label-blind-30-band-auditory"


def build_v2_read_only_preflight(
    operating_point: Mapping[str, object],
    *,
    devices: Sequence[AudioToolboxOutput],
    system_volume_percent: int,
    output_muted: bool,
    player_inspection: Mapping[str, object],
) -> dict[str, object]:
    """Validate stable device identities and player version without live I/O."""

    input_point = operating_point.get("input")
    self_point = operating_point.get("self_source")
    world_point = operating_point.get("world_source")
    playback = operating_point.get("playback")
    if not all(
        isinstance(value, Mapping)
        for value in (input_point, self_point, world_point, playback)
    ):
        raise ValueError("V2 operating point is incomplete")
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
        "buffer_callback_required": playback.get("buffer_callback_required") is True,
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
        "preflight_kind": "zero_frame_uid_bound_route_and_player_preflight",
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
            "Read-only/zero-frame V2 preflight. It does not prove audible playback "
            "or authorize a cue, session capture, cohort, or product claim."
        ),
    }


def v2_playback_callback(
    *,
    player_path: str | Path,
    expected_player_sha256: str,
    devices: Sequence[AudioToolboxOutput] | None = None,
    runner: Callable[..., Any] | None = None,
) -> Callable[..., Mapping[str, object]]:
    """Adapt the UID-bound player to existing target/response capture callbacks."""

    def playback(path: str, **_ignored: object) -> Mapping[str, object]:
        kwargs: dict[str, object] = {}
        if runner is not None:
            kwargs["runner"] = runner
        return play_wav_uid_bound_and_drained(
            path,
            expected_uid=SELF_DEVICE_UID,
            expected_name=SELF_DEVICE_NAME,
            player_path=player_path,
            expected_player_sha256=expected_player_sha256,
            devices=devices,
            **kwargs,
        )

    return playback


def build_v2_cue_confirmation(
    *,
    playback_report: Mapping[str, object],
    owner_heard_cue: bool,
    microphone_opened: bool,
) -> dict[str, object]:
    """Require both machine drain evidence and immediate owner audibility."""

    checks = {
        "uid_bound": playback_report.get("device_bound_by_uid") is True,
        "drain_confirmed": playback_report.get("playback_passed") is True,
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


def freeze_v2_development_associations(
    captures: Mapping[str, Sequence[tuple[str, np.ndarray, int]]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Reuse unchanged V1 learning semantics while branding a new protocol."""

    frozen, evaluation_key = freeze_development_associations(captures)
    frozen["protocol_version"] = PROTOCOL_VERSION
    frozen["v2_changes_affect_association_or_scoring"] = False
    evaluation_key["protocol_version"] = PROTOCOL_VERSION
    evaluation_key["frozen_associations_sha256"] = sha256_json(frozen)
    return frozen, evaluation_key


def commit_v2_label_blind_selection(
    *,
    target_waveform: np.ndarray,
    sample_rate: int,
    frozen: Mapping[str, object],
) -> dict[str, object]:
    commitment = commit_label_blind_selection(
        target_waveform=target_waveform,
        sample_rate=sample_rate,
        frozen=frozen,
    )
    commitment["protocol_version"] = PROTOCOL_VERSION
    return commitment


def build_v2_prediction_commitment(
    *,
    agent_emitted_path: str | Path,
    self_prediction_path: str | Path,
    self_model: Mapping[str, object],
) -> dict[str, object]:
    commitment = build_prediction_commitment(
        agent_emitted_path=agent_emitted_path,
        self_prediction_path=self_prediction_path,
        self_model=self_model,
    )
    commitment["protocol_version"] = PROTOCOL_VERSION
    return commitment


def v2_unchanged_behavioral_contract() -> dict[str, object]:
    return {
        "targets": list(TARGETS),
        "development_schedule": list(DEVELOPMENT_SCHEDULE),
        "held_schedule": list(HELD_SCHEDULE),
        "capture_duration_ms": CAPTURE_DURATION_MS,
        "retention_days": RETENTION_DAYS,
        "scoring_version": SCORING_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "behavioral_or_score_threshold_changed_from_v1": False,
    }
