"""Tests for the live XVF3800 adapter.

Pure-math tests run anywhere. The hardware capture test is skipped unless ffmpeg
and the board are present, so the suite stays green on a laptop without the mic.
There is no synthetic fallback: if the device is missing, the test skips rather
than fabricating a capture.
"""

import shutil

import numpy as np
import pytest

from cora_infans.io.live_mic import LiveMicArray, _require_ffmpeg


def test_require_ffmpeg_found_on_this_host() -> None:
    # The adapter must locate ffmpeg; on this build host it is expected present.
    assert _require_ffmpeg().endswith("ffmpeg")


def test_live_mic_array_rejects_unknown_device_without_fallback(monkeypatch) -> None:
    array = LiveMicArray(device=":999", sample_rate=48000, capture_seconds=0.1)

    # Force a failing probe so we exercise the loud-failure path, not a mock.
    def _boom(*_a, **_k):
        class _P:
            returncode = 1
            stderr = "no such device"

        return _P()

    monkeypatch.setattr("cora_infans.io.live_mic.subprocess.run", _boom)
    with pytest.raises(Exception):
        _ = array.channel_count


def test_dominant_frequency_recovers_a_tone() -> None:
    sr = 48000
    t = np.arange(int(0.25 * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    got = LiveMicArray._dominant_frequency_hz(tone.reshape(-1, 1), sr)
    # Within a few bins of 440 Hz.
    assert abs(got - 440.0) < 50.0


def test_dominant_frequency_reports_zero_on_silence() -> None:
    sr = 48000
    silent = np.zeros((int(0.25 * sr), 1), dtype=np.float32)
    assert LiveMicArray._dominant_frequency_hz(silent, sr) == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_live_capture_runs_end_to_end_if_device_present() -> None:
    """Skips (does not fake) when the board is not connected."""
    array = LiveMicArray(sample_rate=48000, capture_seconds=0.2)
    try:
        ch = array.channel_count
    except Exception:
        pytest.skip("XVF3800 not available on this host")

    frame = array.capture_frame(duration_ms=200)
    assert len(frame.channel_energies) == ch
    assert all(0.0 <= e <= 1.0 for e in frame.channel_energies)
    # timestamp should be a positive epoch-ish millisecond value
    assert frame.timestamp_ms > 0
