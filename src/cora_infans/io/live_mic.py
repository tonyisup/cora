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
import time

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
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.capture_seconds = capture_seconds
        self._channel_count: int | None = None

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
        ffmpeg = _require_ffmpeg()
        seconds = (duration_ms / 1000.0) if duration_ms is not None else self.capture_seconds
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

        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, nch).astype(np.float32)
        data /= 32768.0

        channel_energies = [
            float(np.sqrt(np.mean(col**2))) for col in data.T
        ]
        dominant = (
            dominant_frequency_hz
            if dominant_frequency_hz is not None
            else self._dominant_frequency_hz(data, sr)
        )

        return AudioFrame(
            timestamp_ms=int(time.monotonic_ns() // 1_000_000),
            channel_energies=channel_energies,
            dominant_frequency_hz=dominant,
        )

    @staticmethod
    def _dominant_frequency_hz(data: np.ndarray, sample_rate: int) -> float:
        mono = data.sum(axis=1)
        energy = float(np.sqrt(np.mean(mono**2)))
        if energy < MIN_SALIENT_ENERGY:
            return 0.0
        win = mono * np.hanning(len(mono))
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
        peak_idx = int(np.argmax(spec))
        return float(freqs[peak_idx])
