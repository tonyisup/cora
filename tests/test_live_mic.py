"""Tests for the live XVF3800 adapter.

Pure-math tests run anywhere. The hardware capture test is skipped unless ffmpeg
and the board are present, so the suite stays green on a laptop without the mic.
There is no synthetic fallback: if the device is missing, the test skips rather
than fabricating a capture.
"""

import shutil
from io import BytesIO

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


def test_capture_retains_actual_pcm_sample_rate(monkeypatch) -> None:
    import wave

    def fake_run(cmd, **kwargs):
        pcm = np.zeros((4800, 2), dtype="<i2")
        with wave.open(cmd[-1], "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(pcm.tobytes())

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("cora_infans.io.live_mic.subprocess.run", fake_run)
    array = LiveMicArray(sample_rate=48000, stream_backend="ffmpeg")

    array.capture_frame(duration_ms=100)

    assert array.last_pcm_sample_rate == 48000
    assert array.last_pcm is not None
    assert len(array.last_pcm) == 4800
    assert array.last_pcm_channels is not None
    assert array.last_pcm_channels.shape == (4800, 2)


def test_streaming_reuses_one_ffmpeg_process_across_frames(monkeypatch) -> None:
    pcm = np.zeros((9600, 2), dtype="<i2").tobytes()
    processes = []

    class FakeProcess:
        def __init__(self):
            self.stdout = BytesIO(pcm)
            self.stderr = BytesIO()
            self.terminated = False

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.terminated = True

    def fake_popen(*args, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr("cora_infans.io.live_mic.subprocess.Popen", fake_popen)
    array = LiveMicArray(sample_rate=48000, stream_backend="ffmpeg")

    with array.streaming():
        first = array.capture_frame(duration_ms=100)
        second = array.capture_frame(duration_ms=100)

    assert len(processes) == 1
    assert len(first.channel_energies) == 2
    assert len(second.channel_energies) == 2
    assert processes[0].terminated is True


def test_native_stream_lowers_float_pcm_without_ffmpeg() -> None:
    samples = np.column_stack(
        [
            np.full(4800, 0.25, dtype=np.float32),
            np.full(4800, 0.5, dtype=np.float32),
        ]
    )

    class FakeNativeStream:
        latency = 0.005
        blocksize = 0

        def read(self, sample_count):
            assert sample_count == 4800
            return samples, False

    array = LiveMicArray(sample_rate=48000, stream_backend="native")
    array._native_stream = FakeNativeStream()
    array.active_stream_backend = "sounddevice_portaudio_coreaudio"

    frame = array.capture_frame(duration_ms=100)

    assert frame.channel_energies == pytest.approx([0.25, 0.5])
    assert array.last_input_overflowed is False
    assert array.last_pcm is not None
    assert float(np.mean(array.last_pcm)) == pytest.approx(0.375)


def test_callback_stream_reblocks_pcm_and_preserves_adc_timestamp() -> None:
    class FakeNativeStream:
        latency = 0.042
        blocksize = 0
        time = 100.05

    class TimeInfo:
        def __init__(self, adc_time):
            self.inputBufferAdcTime = adc_time

    array = LiveMicArray(sample_rate=16000, stream_backend="callback")
    array._native_stream = FakeNativeStream()
    array._callback_enabled = True
    array.active_stream_backend = "sounddevice_portaudio_coreaudio_callback"
    first = np.full((200, 2), 0.25, dtype=np.float32)
    second = np.full((200, 2), 0.5, dtype=np.float32)

    array._native_input_callback(first, 200, TimeInfo(100.0), None)
    array._native_input_callback(second, 200, TimeInfo(100.0125), None)
    assert array.callback_stream_frame_position() == 400
    frame = array.capture_frame(duration_ms=25)
    timing = array.device_frame_latency_at_decision()

    assert frame.channel_energies == pytest.approx([0.3952847, 0.3952847])
    assert array.last_pcm is not None and len(array.last_pcm) == 400
    assert timing is not None
    assert timing["adc_frame_onset_to_workspace_decision_ms"] == pytest.approx(50.0)
    assert timing["adc_frame_end_to_workspace_decision_ms"] == pytest.approx(25.0)
    assert array.callback_timestamp_discontinuity_count == 0
    assert array.capture_peak_absolute == 0.5
    assert array.captured_sample_count == 800
    assert array.clipped_sample_count == 0
    assert array.last_frame_peak_absolute == 0.5
    assert array.last_frame_clipped_sample_count == 0


def test_callback_stream_cursor_survives_buffer_discard() -> None:
    class TimeInfo:
        def __init__(self, adc: float) -> None:
            self.inputBufferAdcTime = adc

    array = LiveMicArray(sample_rate=16000, stream_backend="callback")
    array._callback_enabled = True
    data = np.zeros((160, 2), dtype=np.float32)
    array._native_input_callback(data, 160, TimeInfo(10.0), None)

    assert array.discard_buffered_callback_audio() == 160
    assert array._callback_available_frames == 0
    assert array.callback_stream_frame_position() == 160


def test_callback_stream_uses_non_draining_abort_on_close() -> None:
    calls = []

    class FakeNativeStream:
        def abort(self):
            calls.append("abort")

        def stop(self):
            calls.append("stop")

        def close(self):
            calls.append("close")

    array = LiveMicArray(stream_backend="callback")
    array._native_stream = FakeNativeStream()
    array._callback_enabled = True

    array.stop_stream()

    assert calls == ["abort", "close"]


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
