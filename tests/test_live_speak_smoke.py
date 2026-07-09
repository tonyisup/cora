"""Fake-I/O SPEAK smoke harness before any real playback implementation."""

import wave
from pathlib import Path

import numpy as np

from cora_infans.live_speak_smoke import run_live_speak_smoke_trial


def _fake_playback(path: str) -> dict[str, object]:
    assert Path(path).exists()
    return {"backend": "fake_playback", "started": True, "completed": True}


def _fake_capture(path: str, *, sample_rate: int, duration_ms: int) -> dict[str, object]:
    samples = int(sample_rate * duration_ms / 1000)
    pcm = np.zeros(samples, dtype="<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return {"backend": "fake_capture", "frames": samples, "completed": True}


def test_live_speak_smoke_with_fake_io_writes_artifacts_and_tags_frames(tmp_path) -> None:
    report = run_live_speak_smoke_trial(
        vowel="a",
        out_dir=str(tmp_path),
        playback_fn=_fake_playback,
        capture_fn=_fake_capture,
        duration_ms=120,
        max_capture_ms=400,
    )

    assert report["mode"] == "bounded_live_speak_smoke"
    assert report["real_hardware"] is False
    assert report["live_playback_allowed"] is False
    assert report["contamination_policy"] == "tag_only"
    assert Path(report["artifacts"]["efference_wav"]).exists()
    assert Path(report["artifacts"]["captured_wav"]).exists()
    assert report["efference"]["vowel"] == "a"
    assert report["playback"]["backend"] == "fake_playback"
    assert report["capture"]["backend"] == "fake_capture"

    tags = report["frame_tags"]
    assert {tag["speak_state"] for tag in tags} >= {
        "READY_TO_SPEAK",
        "SPEAKING",
        "POST_SPEAK_SUPPRESSION",
        "RECOVERING",
    }
    assert all("efference_id" in tag for tag in tags)
    assert any(tag["expected_self_audio"] for tag in tags)
    assert all(tag["usable_for_world_learning"] is False for tag in tags if tag["expected_self_audio"])


def test_live_speak_smoke_requires_injected_io_by_default(tmp_path) -> None:
    report = run_live_speak_smoke_trial(vowel="a", out_dir=str(tmp_path), duration_ms=120)

    assert report["verdict"] == "blocked_missing_injected_io"
    assert report["real_hardware"] is False
    assert report["live_playback_allowed"] is False
    assert Path(report["artifacts"]["efference_wav"]).exists()
    assert report["artifacts"]["captured_wav"] is None
