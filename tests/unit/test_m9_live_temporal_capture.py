"""Regression tests for the M9 live temporal capture wrapper."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from m9_sound.live_capture import run_live_temporal_capture_report


def test_live_temporal_capture_report_writes_listenable_artifacts(tmp_path: Path):
    def fake_capture_audio(playback_audio: np.ndarray, *, sr: int, **_: object) -> np.ndarray:
        assert sr == 16000
        assert playback_audio.dtype == np.float32
        # Return a slightly longer capture to mimic mic tail/room decay.
        tail = np.zeros(sr // 20, dtype=np.float32)
        return np.concatenate([playback_audio, tail])

    report = run_live_temporal_capture_report(
        output_dir=tmp_path,
        family="wide_sweep",
        input_noise_rms=0.01,
        noise_seed=19,
        device_profile="flat",
        device_gain=1.0,
        capture_audio_fn=fake_capture_audio,
    )

    rendered_wav = Path(report["rendered_wav"])
    captured_wav = Path(report["captured_wav"])
    report_json = Path(report["report_json"])

    assert report["family"] == "wide_sweep"
    assert report["live_capture_samples"] > 0
    assert rendered_wav.exists()
    assert captured_wav.exists()
    assert report_json.exists()

    with wave.open(str(rendered_wav), "rb") as rendered:
        assert rendered.getframerate() == 16000
        assert rendered.getnchannels() == 1
        assert rendered.getsampwidth() == 2
        assert rendered.getnframes() == report["rendered_samples"]

    with wave.open(str(captured_wav), "rb") as captured:
        assert captured.getframerate() == 16000
        assert captured.getnchannels() == 1
        assert captured.getsampwidth() == 2
        assert captured.getnframes() == report["live_capture_samples"]

    persisted = json.loads(report_json.read_text(encoding="utf-8"))
    assert persisted["captured_wav"] == str(captured_wav)
    assert persisted["rendered_wav"] == str(rendered_wav)
    assert persisted["family"] == "wide_sweep"
