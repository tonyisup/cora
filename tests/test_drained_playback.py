from __future__ import annotations

import json
import hashlib
import subprocess

import pytest

from cora_infans.drained_playback import (
    AudioToolboxOutput,
    PLAYER_VERSION,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
    enumerate_audiotoolbox_outputs,
    inspect_native_player,
    native_player_source,
    parse_audiotoolbox_device_listing,
    play_wav_uid_bound_and_drained,
    resolve_uid_bound_output,
    validate_drain_report,
)


LISTING = """
[AudioToolbox @ 0x1] CoreAudio devices:
[AudioToolbox @ 0x1] [0] ASUS PB287Q, display-uid
[AudioToolbox @ 0x1] [1] reSpeaker XVF3800 4-Mic Array, xvf-uid
[AudioToolbox @ 0x1] [2] External Headphones, BuiltInHeadphoneOutputDevice
[AudioToolbox @ 0x1] [3] Mac mini Speakers, BuiltInSpeakerDevice
video:0KiB audio:0KiB subtitle:0KiB
Output file is empty, nothing was encoded
"""


def _native_report(**changes):
    report = {
        "schema_version": 1,
        "player_version": PLAYER_VERSION,
        "requested_device_uid": SELF_DEVICE_UID,
        "device_uid_verified": True,
        "audio_byte_count": 9600,
        "sample_rate_hz": 48000,
        "duration_seconds": 0.1,
        "prepared_frames": 4800,
        "buffer_callback_count": 1,
        "queue_drain_confirmed": True,
        "playback_elapsed_seconds": 0.101,
    }
    report.update(changes)
    return report


def test_device_listing_binds_stable_uid_to_current_index() -> None:
    devices = parse_audiotoolbox_device_listing(LISTING)
    selected = resolve_uid_bound_output(
        devices, expected_uid=SELF_DEVICE_UID, expected_name=SELF_DEVICE_NAME
    )
    assert selected == AudioToolboxOutput(
        index=3, name="Mac mini Speakers", uid="BuiltInSpeakerDevice"
    )


def test_uid_resolution_rejects_name_mismatch_or_duplicate_uid() -> None:
    devices = (AudioToolboxOutput(3, "External Headphones", SELF_DEVICE_UID),)
    with pytest.raises(ValueError, match="UID/name mismatch"):
        resolve_uid_bound_output(
            devices, expected_uid=SELF_DEVICE_UID, expected_name=SELF_DEVICE_NAME
        )

    with pytest.raises(ValueError, match="duplicate UIDs"):
        parse_audiotoolbox_device_listing(
            "[0] one, duplicate\n[1] two, duplicate\n"
        )


def test_zero_frame_enumeration_requires_explicit_empty_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "cora_infans.drained_playback.shutil.which", lambda _name: "/usr/bin/ffmpeg"
    )
    commands = []

    def runner(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=LISTING)

    devices = enumerate_audiotoolbox_outputs(runner=runner)
    assert len(devices) == 4
    assert "-frames:a" in commands[0]
    assert commands[0][commands[0].index("-frames:a") + 1] == "0"

    def unsafe_runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="", stderr=LISTING.replace("audio:0KiB", "audio:1KiB")
        )

    with pytest.raises(RuntimeError, match="zero-frame"):
        enumerate_audiotoolbox_outputs(runner=unsafe_runner)


def test_drain_report_requires_uid_callback_stop_and_elapsed_duration() -> None:
    passed = validate_drain_report(_native_report(), expected_uid=SELF_DEVICE_UID)
    assert passed["playback_passed"] is True

    for changes in (
        {"requested_device_uid": "wrong"},
        {"device_uid_verified": False},
        {"buffer_callback_count": 0},
        {"queue_drain_confirmed": False},
        {"playback_elapsed_seconds": 0.01},
    ):
        failed = validate_drain_report(
            _native_report(**changes), expected_uid=SELF_DEVICE_UID
        )
        assert failed["playback_passed"] is False


def test_playback_wrapper_refuses_exit_zero_without_drain_evidence(tmp_path) -> None:
    wav = tmp_path / "cue.wav"
    wav.write_bytes(b"wav")
    player = tmp_path / "player"
    player.write_bytes(b"binary")
    player_sha256 = hashlib.sha256(player.read_bytes()).hexdigest()
    devices = (AudioToolboxOutput(3, SELF_DEVICE_NAME, SELF_DEVICE_UID),)

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_native_report()),
            stderr="",
        )

    result = play_wav_uid_bound_and_drained(
        wav,
        expected_uid=SELF_DEVICE_UID,
        expected_name=SELF_DEVICE_NAME,
        player_path=player,
        expected_player_sha256=player_sha256,
        devices=devices,
        runner=runner,
    )
    assert result["playback_passed"] is True
    assert result["device_bound_by_uid"] is True
    assert result["output"]["resolved_index"] == 3

    def undrained_runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_native_report(buffer_callback_count=0)),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="lacked drain evidence"):
        play_wav_uid_bound_and_drained(
            wav,
            expected_uid=SELF_DEVICE_UID,
            expected_name=SELF_DEVICE_NAME,
            player_path=player,
            expected_player_sha256=player_sha256,
            devices=devices,
            runner=undrained_runner,
        )


def test_native_player_enqueues_before_start_and_disposes_non_immediately() -> None:
    source = native_player_source().read_text()
    assert source.index("AudioQueueEnqueueBuffer(queue") < source.index(
        'AudioQueueStart", AudioQueueStart(queue'
    )
    assert "AudioQueueStop(queue, false)" in source
    assert "AudioQueueDispose(queue, result == 0 ? false : true)" in source
    assert "kAudioQueueProperty_IsRunning" in source


def test_native_player_inspection_measures_binary_and_source_without_audio(tmp_path) -> None:
    player = tmp_path / "player"
    player.write_bytes(b"binary")

    def runner(command, **_kwargs):
        assert command == [str(player), "--version"]
        return subprocess.CompletedProcess(
            command, 0, stdout=PLAYER_VERSION + "\n", stderr=""
        )

    inspection = inspect_native_player(player, runner=runner)
    assert inspection["player_version"] == PLAYER_VERSION
    assert inspection["binary_sha256"] == hashlib.sha256(b"binary").hexdigest()
    assert len(inspection["source_sha256"]) == 64
    assert inspection["audio_device_opened"] is False
