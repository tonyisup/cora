"""Live audio adapter for the ReSpeaker XMOS XVF3800 (macOS / CoreAudio).

This is the first real-hardware front end for Project Cora-Infans. It pulls a
short capture from the board through ffmpeg (avfoundation device `:0`) and
lowers the raw PCM into the same `AudioFrame` contract the deterministic
pipeline already consumes.

Honesty boundary (per VISION.md / README.md "do not invent unsupported
physics"):
- At the macOS CoreAudio level the XVF3800 exposes a fused multi-channel USB
  stream, not four independently labeled microphones. We map whatever channel
  count the device actually reports (observed: 2 channels) into `channel_energies`.
  We do NOT fabricate four named channels the hardware does not hand us.
- The saliency pipeline is topology-agnostic (it compares channel energies), so
  a real 2-channel live capture flows through unchanged.

Failure behavior:
- If ffmpeg is missing, or the device cannot be opened, the adapter raises a
  clear RuntimeError. There is deliberately NO silent fallback to synthetic
  audio -- a live capture that quietly substituted fake data would poison the
  research substrate.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from ..pipeline import AudioFrame

DEFAULT_DEVICE = ":0"  # avfoundation index for the reSpeaker XVF3800
DEFAULT_SAMPLE_RATE = 48000
# The XVF3800 fuses its four mics into a 2-channel USB stream at the macOS
# CoreAudio layer; requesting more channels only pads silence. Pin to 2 so we
# never report phantom channels.
CAPTURE_CHANNELS = 2
# Real captured RMS energies sit around 0.01-0.03 in a normal room. The
# synthetic defaults use 0.7; the live path uses a hardware-calibrated default
# so quiet speech is not silently dropped by a threshold tuned for fake frames.
DEFAULT_LIVE_SALIENCY_THRESHOLD = 0.02
MIN_SALIENT_ENERGY = 1e-4  # floor to avoid log/divide issues on near-silent frames


class LiveMicError(RuntimeError):
    """Raised when a live capture cannot be obtained from the device."""


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise LiveMicError(
            "ffmpeg not found on PATH; required to capture the XVF3800 on this host"
        )
    return path


class LiveMicArray:
    """Captures short frames from a real XVF3800 and emits AudioFrames."""

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        capture_seconds: float = 0.25,
        stream_backend: str = "auto",
    ) -> None:
        if stream_backend not in ("auto", "native", "callback", "ffmpeg"):
            raise ValueError("stream_backend must be auto, native, callback, or ffmpeg")
        self.device = device
        self.sample_rate = sample_rate
        self.capture_seconds = capture_seconds
        self.stream_backend = stream_backend
        self._channel_count: int | None = None
        self._stream_process: subprocess.Popen[bytes] | None = None
        self._native_stream: Any | None = None
        self.active_stream_backend: str | None = None
        self.stream_fallback_reason: str | None = None
        self.stream_metadata: dict[str, object] | None = None
        self.last_input_overflowed = False
        self.input_overflow_count = 0
        self.callback_status_count = 0
        self.callback_timestamp_discontinuity_count = 0
        self.last_device_frame_timing: dict[str, float] | None = None
        self.capture_peak_absolute = 0.0
        self.captured_sample_count = 0
        self.clipped_sample_count = 0
        self._callback_enabled = False
        self._callback_closed = False
        self._callback_condition = threading.Condition()
        self._callback_chunks: deque[tuple[np.ndarray, float]] = deque()
        self._callback_available_frames = 0
        self._callback_last_adc_end_time: float | None = None
        self.is_real_capture = True
        # Retained raw PCM of the most recent capture, so downstream consumers
        # (e.g. the motor-imitation loop's analyze_vowel) have a waveform to
        # analyze. None until the first capture.
        self.last_pcm: np.ndarray | None = None

    def start_stream(self) -> None:
        """Open one continuous raw-PCM ffmpeg session for repeated frames."""

        if self._stream_process is not None or self._native_stream is not None:
            return
        if self.stream_backend == "callback":
            try:
                self._start_callback_stream()
                return
            except Exception as exc:
                raise LiveMicError(f"callback CoreAudio stream failed: {exc}") from exc
        if self.stream_backend in ("auto", "native"):
            try:
                self._start_native_stream()
                return
            except Exception as exc:  # native is optional only in auto mode
                if self.stream_backend == "native":
                    raise LiveMicError(f"native CoreAudio stream failed: {exc}") from exc
                self.stream_fallback_reason = f"{type(exc).__name__}: {exc}"
        self._start_ffmpeg_stream()

    def _start_native_stream(self) -> None:
        import sounddevice as sd

        device_index = self._resolve_sounddevice_input(sd)
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            # Let CoreAudio choose its optimal callback quantum. Cora reblocks
            # the native stream into the requested analysis-frame size.
            blocksize=0,
            device=device_index,
            channels=CAPTURE_CHANNELS,
            dtype="float32",
            latency="low",
        )
        stream.start()
        self._native_stream = stream
        self._channel_count = CAPTURE_CHANNELS
        self.active_stream_backend = "sounddevice_portaudio_coreaudio"
        self.stream_metadata = {
            "sample_rate": self.sample_rate,
            "channel_count": CAPTURE_CHANNELS,
            "requested_latency": "low",
            "reported_input_latency_ms": float(stream.latency) * 1000.0,
            "reported_blocksize": int(stream.blocksize),
            "device_timestamp_source": None,
        }

    def _start_callback_stream(self) -> None:
        import sounddevice as sd

        device_index = self._resolve_sounddevice_input(sd)
        with self._callback_condition:
            self._callback_closed = False
            self._callback_chunks.clear()
            self._callback_available_frames = 0
            self._callback_last_adc_end_time = None
        self._callback_enabled = True
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=0,
            device=device_index,
            channels=CAPTURE_CHANNELS,
            dtype="float32",
            latency="low",
            callback=self._native_input_callback,
        )
        stream.start()
        self._native_stream = stream
        self._channel_count = CAPTURE_CHANNELS
        self.active_stream_backend = "sounddevice_portaudio_coreaudio_callback"
        self.stream_metadata = {
            "sample_rate": self.sample_rate,
            "channel_count": CAPTURE_CHANNELS,
            "requested_latency": "low",
            "reported_input_latency_ms": float(stream.latency) * 1000.0,
            "reported_blocksize": int(stream.blocksize),
            "device_timestamp_source": "PortAudioCallbackTimeInfo.inputBufferAdcTime",
        }

    def _native_input_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        if status:
            self.callback_status_count += 1
            if bool(getattr(status, "input_overflow", False)):
                self.input_overflow_count += 1
        data = np.asarray(indata, dtype=np.float32).copy()
        if data.shape != (frames, CAPTURE_CHANNELS):
            self.callback_status_count += 1
            return
        adc_start = float(time_info.inputBufferAdcTime)
        adc_end = adc_start + frames / self.sample_rate
        with self._callback_condition:
            if (
                self._callback_last_adc_end_time is not None
                and abs(adc_start - self._callback_last_adc_end_time)
                > 2.0 / self.sample_rate
            ):
                self.callback_timestamp_discontinuity_count += 1
            self._callback_last_adc_end_time = adc_end
            self._callback_chunks.append((data, adc_start))
            self._callback_available_frames += frames
            self._callback_condition.notify_all()

    def _resolve_sounddevice_input(self, sounddevice_module: Any) -> int:
        devices = sounddevice_module.query_devices()
        preferred_name = "respeaker xvf3800"
        for index, device in enumerate(devices):
            if (
                preferred_name in str(device["name"]).lower()
                and int(device["max_input_channels"]) >= CAPTURE_CHANNELS
            ):
                return index
        raise LiveMicError("sounddevice could not find the XVF3800 input")

    def _start_ffmpeg_stream(self) -> None:
        ffmpeg = _require_ffmpeg()
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-thread_queue_size",
            "512",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "avfoundation",
            "-i",
            self.device,
            "-vn",
            "-ac",
            str(CAPTURE_CHANNELS),
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-flush_packets",
            "1",
            "pipe:1",
        ]
        self._stream_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._channel_count = CAPTURE_CHANNELS
        self.active_stream_backend = "ffmpeg_avfoundation_raw_pcm"
        self.stream_metadata = {
            "sample_rate": self.sample_rate,
            "channel_count": CAPTURE_CHANNELS,
            "reported_input_latency_ms": None,
            "reported_blocksize": None,
        }

    def stop_stream(self) -> None:
        with self._callback_condition:
            self._callback_closed = True
            self._callback_condition.notify_all()
        native_stream = self._native_stream
        self._native_stream = None
        if native_stream is not None:
            try:
                if self._callback_enabled and hasattr(native_stream, "abort"):
                    native_stream.abort()
                else:
                    native_stream.stop()
            finally:
                native_stream.close()
        self._callback_enabled = False
        process = self._stream_process
        self._stream_process = None
        if process is None:
            self.active_stream_backend = None
            return
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self.active_stream_backend = None

    @contextmanager
    def streaming(self) -> Iterator["LiveMicArray"]:
        self.start_stream()
        try:
            yield self
        finally:
            self.stop_stream()

    @property
    def channel_count(self) -> int:
        if self._channel_count is None:
            # Probe the actual device channel count so we never lie about it.
            self._channel_count = self._probe_channel_count()
        return self._channel_count

    def _probe_channel_count(self) -> int:
        ffmpeg = _require_ffmpeg()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "avfoundation", "-i", self.device,
            "-t", "0.15", "-ac", str(CAPTURE_CHANNELS), f"-ar", str(self.sample_rate),
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise LiveMicError(
                f"could not open device {self.device!r}: {proc.stderr.strip() or 'ffmpeg error'}"
            )
        try:
            import wave

            with wave.open(out_path, "rb") as w:
                return w.getnchannels()
        finally:
            import os

            os.unlink(out_path)

    def capture_frame(
        self,
        *,
        duration_ms: int | None = None,
        dominant_frequency_hz: float | None = None,
    ) -> AudioFrame:
        """Grab one live frame and lower it into an AudioFrame.

        `dominant_frequency_hz` is computed from the captured signal by default;
        callers may override it only for deterministic test injection.
        """
        seconds = (duration_ms / 1000.0) if duration_ms is not None else self.capture_seconds
        if self._native_stream is not None:
            sample_count = max(1, int(round(seconds * self.sample_rate)))
            if self._callback_enabled:
                data = self._read_callback_samples(sample_count)
                frame = self._frame_from_float_data(
                    data,
                    sample_rate=self.sample_rate,
                    dominant_frequency_hz=dominant_frequency_hz,
                )
                if self.last_device_frame_timing is not None:
                    self.last_device_frame_timing["capture_return_stream_time_s"] = (
                        float(self._native_stream.time)
                    )
                return frame
            data, overflowed = self._native_stream.read(sample_count)
            self.last_input_overflowed = bool(overflowed)
            self.input_overflow_count += int(self.last_input_overflowed)
            return self._frame_from_float_data(
                np.asarray(data, dtype=np.float32),
                sample_rate=self.sample_rate,
                dominant_frequency_hz=dominant_frequency_hz,
            )
        if self._stream_process is not None:
            sample_count = max(1, int(round(seconds * self.sample_rate)))
            raw = self._read_stream_bytes(sample_count * CAPTURE_CHANNELS * 2)
            return self._frame_from_pcm_bytes(
                raw,
                channel_count=CAPTURE_CHANNELS,
                sample_rate=self.sample_rate,
                dominant_frequency_hz=dominant_frequency_hz,
            )

        ffmpeg = _require_ffmpeg()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "avfoundation", "-i", self.device,
            "-t", f"{seconds:.3f}", "-ac", str(CAPTURE_CHANNELS), f"-ar", str(self.sample_rate),
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise LiveMicError(
                f"live capture failed on {self.device!r}: {proc.stderr.strip() or 'ffmpeg error'}"
            )

        import os
        import wave

        try:
            with wave.open(out_path, "rb") as w:
                nch = w.getnchannels()
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
        finally:
            os.unlink(out_path)

        return self._frame_from_pcm_bytes(
            raw,
            channel_count=nch,
            sample_rate=sr,
            dominant_frequency_hz=dominant_frequency_hz,
        )

    def _read_callback_samples(self, sample_count: int) -> np.ndarray:
        deadline = time.monotonic() + max(2.0, sample_count / self.sample_rate * 4.0)
        pieces: list[np.ndarray] = []
        frame_adc_onset: float | None = None
        frame_adc_end: float | None = None
        with self._callback_condition:
            while self._callback_available_frames < sample_count:
                if self._callback_closed:
                    raise LiveMicError("callback stream closed before a frame was available")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise LiveMicError("timed out waiting for callback audio")
                self._callback_condition.wait(timeout=remaining)
            remaining_samples = sample_count
            while remaining_samples > 0:
                data, adc_start = self._callback_chunks.popleft()
                take = min(remaining_samples, len(data))
                if frame_adc_onset is None:
                    frame_adc_onset = adc_start
                pieces.append(data[:take])
                frame_adc_end = adc_start + take / self.sample_rate
                if take < len(data):
                    self._callback_chunks.appendleft((data[take:], frame_adc_end))
                self._callback_available_frames -= take
                remaining_samples -= take
        if frame_adc_onset is None or frame_adc_end is None:
            raise LiveMicError("callback frame had no ADC timestamp")
        self.last_device_frame_timing = {
            "adc_frame_onset_stream_time_s": frame_adc_onset,
            "adc_frame_end_stream_time_s": frame_adc_end,
        }
        return np.concatenate(pieces, axis=0)

    def device_frame_latency_at_decision(self) -> dict[str, float] | None:
        timing = self.last_device_frame_timing
        stream = self._native_stream
        if timing is None or stream is None or not self._callback_enabled:
            return None
        decision_time = float(stream.time)
        return {
            **timing,
            "workspace_decision_stream_time_s": decision_time,
            "adc_frame_onset_to_workspace_decision_ms": (
                decision_time - timing["adc_frame_onset_stream_time_s"]
            )
            * 1000.0,
            "adc_frame_end_to_workspace_decision_ms": (
                decision_time - timing["adc_frame_end_stream_time_s"]
            )
            * 1000.0,
        }

    def _read_stream_bytes(self, expected_bytes: int) -> bytes:
        process = self._stream_process
        if process is None or process.stdout is None:
            raise LiveMicError("live PCM stream is not open")
        chunks: list[bytes] = []
        remaining = expected_bytes
        while remaining > 0:
            chunk = process.stdout.read(remaining)
            if not chunk:
                stderr = (
                    process.stderr.read().decode("utf-8", errors="replace").strip()
                    if process.stderr is not None and process.poll() is not None
                    else ""
                )
                raise LiveMicError(
                    f"live PCM stream ended early: {stderr or 'ffmpeg produced no audio'}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _frame_from_pcm_bytes(
        self,
        raw: bytes,
        *,
        channel_count: int,
        sample_rate: int,
        dominant_frequency_hz: float | None,
    ) -> AudioFrame:
        data = (
            np.frombuffer(raw, dtype="<i2")
            .reshape(-1, channel_count)
            .astype(np.float32)
        )
        data /= 32768.0
        return self._frame_from_float_data(
            data,
            sample_rate=sample_rate,
            dominant_frequency_hz=dominant_frequency_hz,
        )

    def _frame_from_float_data(
        self,
        data: np.ndarray,
        *,
        sample_rate: int,
        dominant_frequency_hz: float | None,
    ) -> AudioFrame:
        if data.ndim != 2 or data.shape[1] != CAPTURE_CHANNELS:
            raise LiveMicError(
                f"expected {CAPTURE_CHANNELS}-channel PCM, got shape {data.shape}"
            )
        absolute = np.abs(data)
        self.capture_peak_absolute = max(
            self.capture_peak_absolute, float(np.max(absolute, initial=0.0))
        )
        self.captured_sample_count += int(data.size)
        self.clipped_sample_count += int(np.count_nonzero(absolute >= 1.0))

        # Retain the raw PCM (mono mix) for downstream waveform analysis.
        self.last_pcm = data.mean(axis=1).astype(np.float32)
        self.last_pcm_sample_rate = sample_rate

        channel_energies = [
            float(np.sqrt(np.mean(col**2))) for col in data.T
        ]
        dominant = (
            dominant_frequency_hz
            if dominant_frequency_hz is not None
            else self._dominant_frequency_hz(data, sample_rate)
        )

        return AudioFrame(
            timestamp_ms=int(time.monotonic_ns() // 1_000_000),
            channel_energies=channel_energies,
            dominant_frequency_hz=dominant,
        )

    @staticmethod
    def _dominant_frequency_hz(data: np.ndarray, sample_rate: int) -> float:
        mono = data.mean(axis=1)
        energy = float(np.sqrt(np.mean(mono**2)))
        if energy < MIN_SALIENT_ENERGY:
            return 0.0
        win = mono * np.hanning(len(mono))
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
        peak_idx = int(np.argmax(spec))
        return float(freqs[peak_idx])
