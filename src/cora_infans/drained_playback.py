"""UID-bound, drain-confirmed CoreAudio playback for product-session V2.

This module deliberately does not alter the frozen V1 ffmpeg playback path.
Device enumeration emits zero audio frames. Real playback requires the native
AudioQueue helper and accepts success only after UID verification, a returned
buffer callback, and a non-immediate queue drain.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


PLAYER_VERSION = "cora-audioqueue-player-v1"
SELF_DEVICE_UID = "BuiltInSpeakerDevice"
SELF_DEVICE_NAME = "Mac mini Speakers"
WORLD_DEVICE_UID = "BuiltInHeadphoneOutputDevice"
WORLD_DEVICE_NAME = "External Headphones"
_DEVICE_ROW = re.compile(r"\[(?P<index>\d+)\]\s+(?P<name>.*?),\s+(?P<uid>\S.*)\s*$")


@dataclass(frozen=True)
class AudioToolboxOutput:
    index: int
    name: str
    uid: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def parse_audiotoolbox_device_listing(text: str) -> tuple[AudioToolboxOutput, ...]:
    """Parse ffmpeg's zero-frame AudioToolbox device rows."""

    devices: list[AudioToolboxOutput] = []
    for line in text.splitlines():
        match = _DEVICE_ROW.search(line)
        if match is None:
            continue
        devices.append(
            AudioToolboxOutput(
                index=int(match.group("index")),
                name=match.group("name").strip(),
                uid=match.group("uid").strip(),
            )
        )
    if not devices:
        raise ValueError("AudioToolbox device listing contained no output rows")
    if len({device.index for device in devices}) != len(devices):
        raise ValueError("AudioToolbox device listing contains duplicate indices")
    if len({device.uid for device in devices}) != len(devices):
        raise ValueError("AudioToolbox device listing contains duplicate UIDs")
    return tuple(sorted(devices, key=lambda device: device.index))


def enumerate_audiotoolbox_outputs(
    *, runner: CommandRunner = subprocess.run
) -> tuple[AudioToolboxOutput, ...]:
    """Enumerate output names/UIDs with an explicit zero-frame ffmpeg run."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for zero-frame device enumeration")
    result = runner(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "info",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono",
            "-frames:a",
            "0",
            "-f",
            "audiotoolbox",
            "-list_devices",
            "true",
            "-y",
            "/private/tmp/cora-audiotoolbox-zero-frame-list.wav",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"zero-frame AudioToolbox enumeration failed: {combined[-1000:]}")
    if "audio:0KiB" not in combined or "Output file is empty" not in combined:
        raise RuntimeError("device enumeration did not prove zero-frame output")
    return parse_audiotoolbox_device_listing(combined)


def resolve_uid_bound_output(
    devices: Sequence[AudioToolboxOutput], *, expected_uid: str, expected_name: str
) -> AudioToolboxOutput:
    """Resolve a stable UID and require its human-readable name to match."""

    matches = [device for device in devices if device.uid == expected_uid]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one AudioToolbox UID: {expected_uid}")
    selected = matches[0]
    if selected.name != expected_name:
        raise ValueError(
            f"AudioToolbox UID/name mismatch: {expected_uid} is {selected.name!r}"
        )
    return selected


def native_player_source() -> Path:
    return Path(__file__).with_name("native") / "audioqueue_player.c"


def build_native_player(
    output_path: str | Path,
    *,
    runner: CommandRunner = subprocess.run,
) -> Path:
    """Compile the fixed native AudioQueue helper without opening an audio device."""

    clang = shutil.which("clang")
    if clang is None:
        raise RuntimeError("clang is required to build the native AudioQueue player")
    source = native_player_source()
    if not source.is_file():
        raise RuntimeError("native AudioQueue player source is missing")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = runner(
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
            "-o",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"native AudioQueue player build failed: {result.stderr}")
    version = runner(
        [str(output), "--version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if version.returncode != 0 or version.stdout.strip() != PLAYER_VERSION:
        raise RuntimeError("native AudioQueue player version check failed")
    return output


def inspect_native_player(
    player_path: str | Path,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, object]:
    """Measure the native player version and exact binary digest."""

    binary = Path(player_path)
    if not binary.is_file():
        raise ValueError("native AudioQueue player binary is missing")
    version = runner(
        [str(binary), "--version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if version.returncode != 0:
        raise RuntimeError("native AudioQueue player version probe failed")
    return {
        "player_version": version.stdout.strip(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(native_player_source().read_bytes()).hexdigest(),
        "audio_device_opened": False,
    }


def validate_drain_report(
    report: Mapping[str, Any], *, expected_uid: str
) -> dict[str, object]:
    """Require all native player completion evidence, not just exit code zero."""

    checks = {
        "player_version_matches": report.get("player_version") == PLAYER_VERSION,
        "requested_device_uid_matches": report.get("requested_device_uid")
        == expected_uid,
        "device_uid_verified": report.get("device_uid_verified") is True,
        "buffer_callback_observed": isinstance(report.get("buffer_callback_count"), int)
        and not isinstance(report.get("buffer_callback_count"), bool)
        and int(report["buffer_callback_count"]) >= 1,
        "queue_drain_confirmed": report.get("queue_drain_confirmed") is True,
        "positive_duration": isinstance(report.get("duration_seconds"), (int, float))
        and not isinstance(report.get("duration_seconds"), bool)
        and float(report["duration_seconds"]) > 0.0,
        "elapsed_covers_duration": isinstance(
            report.get("playback_elapsed_seconds"), (int, float)
        )
        and not isinstance(report.get("playback_elapsed_seconds"), bool)
        and isinstance(report.get("duration_seconds"), (int, float))
        and float(report["playback_elapsed_seconds"])
        >= 0.90 * float(report["duration_seconds"]),
    }
    return {
        "checks": checks,
        "playback_passed": all(checks.values()),
        "native_report": dict(report),
    }


def play_wav_uid_bound_and_drained(
    path: str | Path,
    *,
    expected_uid: str,
    expected_name: str,
    player_path: str | Path,
    expected_player_sha256: str,
    devices: Sequence[AudioToolboxOutput] | None = None,
    runner: CommandRunner = subprocess.run,
) -> dict[str, object]:
    """Play one PCM WAV only after stable UID resolution and require queue drain."""

    wav = Path(path)
    if not wav.is_file():
        raise ValueError("playback WAV is missing")
    available = tuple(devices or enumerate_audiotoolbox_outputs(runner=runner))
    selected = resolve_uid_bound_output(
        available,
        expected_uid=expected_uid,
        expected_name=expected_name,
    )
    binary = Path(player_path)
    if not binary.is_file():
        raise ValueError("native AudioQueue player binary is missing")
    actual_player_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    if actual_player_sha256 != expected_player_sha256:
        raise ValueError("native AudioQueue player binary digest mismatch")
    completed = runner(
        [str(binary), str(wav), expected_uid],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "native AudioQueue playback failed: "
            + (completed.stderr.strip() or f"exit {completed.returncode}")
        )
    try:
        raw_report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("native AudioQueue player returned invalid JSON") from exc
    if not isinstance(raw_report, Mapping):
        raise RuntimeError("native AudioQueue player returned a non-object report")
    validated = validate_drain_report(raw_report, expected_uid=expected_uid)
    if validated["playback_passed"] is not True:
        raise RuntimeError("native AudioQueue playback lacked drain evidence")
    return {
        "schema_version": 1,
        "output": {
            "resolved_index": selected.index,
            "name": selected.name,
            "uid": selected.uid,
        },
        "device_bound_by_uid": True,
        "index_used_only_after_uid_resolution": True,
        "player_binary_sha256": actual_player_sha256,
        **validated,
    }
