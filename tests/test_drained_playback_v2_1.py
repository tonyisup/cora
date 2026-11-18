from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from cora_infans.drained_playback import AudioToolboxOutput
from cora_infans.drained_playback_v2_1 import (
    PLAYER_VERSION,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
    enumerate_coreaudio_outputs,
    native_player_source,
    play_wav_uid_bound_and_completed,
    validate_completion_report,
)


def _native_report(**changes):
    report = {
        "schema_version": 2,
        "player_version": PLAYER_VERSION,
        "requested_device_uid": SELF_DEVICE_UID,
        "device_uid_verified": True,
        "audio_byte_count": 9600,
        "sample_rate_hz": 48000,
        "duration_seconds": 0.1,
        "prepared_frames": 4800,
        "buffer_callback_count": 1,
        "callback_observed_before_stop": True,
        "stop_requested_after_callback": True,
        "queue_drain_confirmed": True,
        "callback_elapsed_seconds": 0.101,
        "playback_elapsed_seconds": 0.102,
    }
    report.update(changes)
    return report


def test_v2_1_completion_requires_callback_before_stop_and_duration() -> None:
    passed = validate_completion_report(
        _native_report(), expected_uid=SELF_DEVICE_UID
    )
    assert passed["playback_passed"] is True

    for changes in (
        {"buffer_callback_count": 0},
        {"callback_observed_before_stop": False},
        {"stop_requested_after_callback": False},
        {"callback_elapsed_seconds": 0.01},
        {"queue_drain_confirmed": False},
        {"playback_elapsed_seconds": 0.01},
    ):
        failed = validate_completion_report(
            _native_report(**changes), expected_uid=SELF_DEVICE_UID
        )
        assert failed["playback_passed"] is False


def test_v2_1_enumeration_requires_property_only_zero_frame_evidence(
    tmp_path,
) -> None:
    player = tmp_path / "player"
    player.write_bytes(b"binary")
    player_sha256 = hashlib.sha256(player.read_bytes()).hexdigest()
    report = {
        "schema_version": 2,
        "player_version": PLAYER_VERSION,
        "enumeration_mode": "coreaudio_property_read_only",
        "audio_queue_created": False,
        "audio_frames_sent": 0,
        "outputs": [
            {"index": 0, "name": SELF_DEVICE_NAME, "uid": SELF_DEVICE_UID}
        ],
    }

    def runner(command, **_kwargs):
        assert command == [str(player), "--list-devices"]
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(report), stderr=""
        )

    outputs = enumerate_coreaudio_outputs(
        player_path=player,
        expected_player_sha256=player_sha256,
        runner=runner,
    )
    assert outputs == (AudioToolboxOutput(0, SELF_DEVICE_NAME, SELF_DEVICE_UID),)

    report["audio_queue_created"] = True
    with pytest.raises(RuntimeError, match="lacked read-only evidence"):
        enumerate_coreaudio_outputs(
            player_path=player,
            expected_player_sha256=player_sha256,
            runner=runner,
        )


def test_v2_1_native_source_orders_callback_before_stop() -> None:
    source = native_player_source().read_text()
    enqueue = source.index("AudioQueueEnqueueBuffer(queue")
    start = source.index('AudioQueueStart", AudioQueueStart(queue')
    callback_wait = source.index("while (callback_count < 1)")
    stop = source.index("AudioQueueStop(queue, false)")
    stopped_queue = source.index("kAudioQueueProperty_IsRunning")
    assert enqueue < start < callback_wait < stop < stopped_queue
    assert "callback_observed_before_stop\\\":true" in source
    assert "stop_requested_after_callback\\\":true" in source
    assert "AudioQueueDispose(queue, result == 0 ? false : true)" in source


def test_v2_1_wrapper_refuses_exit_zero_without_ordering_evidence(tmp_path) -> None:
    wav = tmp_path / "cue.wav"
    wav.write_bytes(b"wav")
    player = tmp_path / "player"
    player.write_bytes(b"binary")
    player_sha256 = hashlib.sha256(player.read_bytes()).hexdigest()
    devices = (AudioToolboxOutput(3, SELF_DEVICE_NAME, SELF_DEVICE_UID),)

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(_native_report()), stderr=""
        )

    result = play_wav_uid_bound_and_completed(
        wav,
        expected_uid=SELF_DEVICE_UID,
        expected_name=SELF_DEVICE_NAME,
        player_path=player,
        expected_player_sha256=player_sha256,
        devices=devices,
        runner=runner,
    )
    assert result["playback_passed"] is True

    def unsafe_runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                _native_report(callback_observed_before_stop=False)
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="lacked completion evidence"):
        play_wav_uid_bound_and_completed(
            wav,
            expected_uid=SELF_DEVICE_UID,
            expected_name=SELF_DEVICE_NAME,
            player_path=player,
            expected_player_sha256=player_sha256,
            devices=devices,
            runner=unsafe_runner,
        )
    enumerate_coreaudio_outputs,
