"""V2.1 UID-bound playback with callback-before-stop completion ordering."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .drained_playback import (
    AudioToolboxOutput,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
    WORLD_DEVICE_NAME,
    WORLD_DEVICE_UID,
    resolve_uid_bound_output,
)


PLAYER_VERSION = "cora-audioqueue-player-v2"
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def native_player_source() -> Path:
    return Path(__file__).with_name("native") / "audioqueue_player_v2.c"


def build_native_player(
    output_path: str | Path,
    *,
    runner: CommandRunner = subprocess.run,
) -> Path:
    """Compile V2.1 without opening an audio device."""

    clang = shutil.which("clang")
    if clang is None:
        raise RuntimeError("clang is required to build the native AudioQueue player")
    source = native_player_source()
    if not source.is_file():
        raise RuntimeError("V2.1 native AudioQueue player source is missing")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = runner(
        [
            clang,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-framework",
            "AudioToolbox",
            "-framework",
            "CoreFoundation",
            "-framework",
            "CoreAudio",
            "-o",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"V2.1 native player build failed: {completed.stderr}")
    version = runner(
        [str(output), "--version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if version.returncode != 0 or version.stdout.strip() != PLAYER_VERSION:
        raise RuntimeError("V2.1 native player version check failed")
    return output


def inspect_native_player(
    player_path: str | Path,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, object]:
    """Measure the V2.1 binary and source without opening an audio device."""

    binary = Path(player_path)
    if not binary.is_file():
        raise ValueError("V2.1 native player binary is missing")
    version = runner(
        [str(binary), "--version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if version.returncode != 0 or version.stdout.strip() != PLAYER_VERSION:
        raise RuntimeError("V2.1 native player version probe failed")
    return {
        "player_version": PLAYER_VERSION,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(native_player_source().read_bytes()).hexdigest(),
        "audio_device_opened": False,
    }


def enumerate_coreaudio_outputs(
    *,
    player_path: str | Path,
    expected_player_sha256: str,
    runner: CommandRunner = subprocess.run,
) -> tuple[AudioToolboxOutput, ...]:
    """Read CoreAudio output properties without creating an AudioQueue."""

    binary = Path(player_path)
    if not binary.is_file():
        raise ValueError("V2.1 native player binary is missing")
    actual_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    if actual_sha256 != expected_player_sha256:
        raise ValueError("V2.1 native player binary digest mismatch")
    completed = runner(
        [str(binary), "--list-devices"],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "V2.1 CoreAudio property enumeration failed: "
            + (completed.stderr.strip() or f"exit {completed.returncode}")
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("V2.1 device enumerator returned invalid JSON") from exc
    if not isinstance(report, Mapping):
        raise RuntimeError("V2.1 device enumerator returned a non-object report")
    if (
        report.get("player_version") != PLAYER_VERSION
        or report.get("enumeration_mode") != "coreaudio_property_read_only"
        or report.get("audio_queue_created") is not False
        or report.get("audio_frames_sent") != 0
    ):
        raise RuntimeError("V2.1 device enumerator lacked read-only evidence")
    raw_outputs = report.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise RuntimeError("V2.1 device enumerator returned no outputs")
    outputs: list[AudioToolboxOutput] = []
    for raw in raw_outputs:
        if (
            not isinstance(raw, Mapping)
            or not isinstance(raw.get("index"), int)
            or isinstance(raw.get("index"), bool)
            or not isinstance(raw.get("name"), str)
            or not isinstance(raw.get("uid"), str)
        ):
            raise RuntimeError("V2.1 device enumerator returned an invalid row")
        outputs.append(
            AudioToolboxOutput(
                index=int(raw["index"]),
                name=str(raw["name"]),
                uid=str(raw["uid"]),
            )
        )
    if len({output.index for output in outputs}) != len(outputs):
        raise RuntimeError("V2.1 device enumerator returned duplicate indices")
    if len({output.uid for output in outputs}) != len(outputs):
        raise RuntimeError("V2.1 device enumerator returned duplicate UIDs")
    return tuple(outputs)


def validate_completion_report(
    report: Mapping[str, Any], *, expected_uid: str
) -> dict[str, object]:
    """Require callback-before-stop ordering plus stopped-queue evidence."""

    duration = report.get("duration_seconds")
    elapsed = report.get("playback_elapsed_seconds")
    callback_elapsed = report.get("callback_elapsed_seconds")
    numeric_duration = isinstance(duration, (int, float)) and not isinstance(
        duration, bool
    )
    checks = {
        "player_version_matches": report.get("player_version") == PLAYER_VERSION,
        "requested_device_uid_matches": report.get("requested_device_uid")
        == expected_uid,
        "device_uid_verified": report.get("device_uid_verified") is True,
        "buffer_callback_observed": isinstance(report.get("buffer_callback_count"), int)
        and not isinstance(report.get("buffer_callback_count"), bool)
        and int(report["buffer_callback_count"]) >= 1,
        "callback_observed_before_stop": report.get(
            "callback_observed_before_stop"
        )
        is True,
        "stop_requested_after_callback": report.get("stop_requested_after_callback")
        is True,
        "queue_drain_confirmed": report.get("queue_drain_confirmed") is True,
        "positive_duration": numeric_duration and float(duration) > 0.0,
        "callback_elapsed_covers_duration": isinstance(
            callback_elapsed, (int, float)
        )
        and not isinstance(callback_elapsed, bool)
        and numeric_duration
        and float(callback_elapsed) >= 0.90 * float(duration),
        "playback_elapsed_covers_duration": isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and numeric_duration
        and float(elapsed) >= 0.90 * float(duration),
    }
    return {
        "checks": checks,
        "playback_passed": all(checks.values()),
        "native_report": dict(report),
    }


def play_wav_uid_bound_and_completed(
    path: str | Path,
    *,
    expected_uid: str,
    expected_name: str,
    player_path: str | Path,
    expected_player_sha256: str,
    devices: Sequence[AudioToolboxOutput] | None = None,
    runner: CommandRunner = subprocess.run,
) -> dict[str, object]:
    """Play one WAV through V2.1 and fail closed on any completion gate."""

    wav = Path(path)
    if not wav.is_file():
        raise ValueError("playback WAV is missing")
    available = tuple(
        devices
        or enumerate_coreaudio_outputs(
            player_path=player_path,
            expected_player_sha256=expected_player_sha256,
            runner=runner,
        )
    )
    selected = resolve_uid_bound_output(
        available,
        expected_uid=expected_uid,
        expected_name=expected_name,
    )
    binary = Path(player_path)
    if not binary.is_file():
        raise ValueError("V2.1 native player binary is missing")
    actual_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    if actual_sha256 != expected_player_sha256:
        raise ValueError("V2.1 native player binary digest mismatch")
    completed = runner(
        [str(binary), str(wav), expected_uid],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "V2.1 native playback failed: "
            + (completed.stderr.strip() or f"exit {completed.returncode}")
        )
    try:
        raw_report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("V2.1 native player returned invalid JSON") from exc
    if not isinstance(raw_report, Mapping):
        raise RuntimeError("V2.1 native player returned a non-object report")
    validated = validate_completion_report(raw_report, expected_uid=expected_uid)
    if validated["playback_passed"] is not True:
        raise RuntimeError("V2.1 native playback lacked completion evidence")
    return {
        "schema_version": 2,
        "output": {
            "resolved_index": selected.index,
            "name": selected.name,
            "uid": selected.uid,
        },
        "device_bound_by_uid": True,
        "index_used_only_after_uid_resolution": True,
        "player_binary_sha256": actual_sha256,
        **validated,
    }
