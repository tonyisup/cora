"""Continuous live listener for the XVF3800 with an optional parentese stimulus.

This is the "A" slice: a real-time listening loop that captures frames from the
board, runs each through the deterministic pipeline, and accumulates focus-path
tracking, dropout, and recovery over a time window -- the same sequence-report
shape the built-in families produce, but fed by genuine hardware audio.

Stimulus (parentese):
- A `ParenteseStimulus` generates a simple infant-directed-speech-like contour
  (slow rising/falling pitch sweep) and plays it to the External Headphones
  output device (audiotoolbox index 2) so the mic has a controlled acoustic
  target. The listener does NOT claim the "infant" perceives anything; it only
  provides a repeatable stimulus so the saliency gate has something to detect.

Honesty boundary (per VISION.md / README.md):
- Every captured frame is tagged `stimulus_active` so downstream analysis can
  separate stimulated from quiet-room input. No frame is ever fabricated.
- The listener fails loudly if it cannot open the mic; it never substitutes
  synthetic audio for a missed capture.
"""

from __future__ import annotations

import subprocess
import tempfile

import numpy as np

from ..config import CoraInfansConfig
from ..pipeline import AudioFrame, CoraInfansPipeline
from .live_mic import (
    DEFAULT_DEVICE,
    DEFAULT_LIVE_SALIENCY_THRESHOLD,
    CAPTURE_CHANNELS,
    LiveMicArray,
    LiveMicError,
    _require_ffmpeg,
)
from ..predictive import FocusTrajectoryCoder, PredictiveCoder, PredictiveHierarchy

# External Headphones output device index (audiotoolbox), per host enumeration.
STIMULUS_OUTPUT_DEVICE_INDEX = 2
DEFAULT_STIMULUS_SAMPLE_RATE = 48000


class ParenteseStimulus:
    """Synthesizes and plays a simple parentese-like pitch contour.

    Parentese is characterized by a higher, exaggerated, and slowly modulated
    pitch relative to adult speech. We model only that acoustic envelope: a
    few rising/falling pitch sweeps on a pure tone, at a gentle tempo. This is
    a research stimulus, not a speech-synthesis claim.
    """

    def __init__(
        self,
        output_device_index: int = STIMULUS_OUTPUT_DEVICE_INDEX,
        sample_rate: int = DEFAULT_STIMULUS_SAMPLE_RATE,
    ) -> None:
        self.output_device_index = output_device_index
        self.sample_rate = sample_rate

    def synthesize_wav(self, path: str, *, sweeps: int = 3, sweep_ms: int = 400) -> None:
        """Write a parentese-like stimulus (pitch sweeps) to `path`."""
        sr = self.sample_rate
        total = int(sweeps * sweep_ms / 1000.0 * sr)
        t = np.arange(total) / sr
        # Each sweep rises then falls; consecutive sweeps step the center pitch
        # up and down to mimic the exaggerated prosody of parentese.
        freq = np.zeros(total, dtype=np.float64)
        seg = int(sweep_ms / 1000.0 * sr)
        base = 300.0
        for i in range(sweeps):
            s = i * seg
            e = s + seg
            phase = np.linspace(0.0, np.pi, seg)
            center = base * (1.0 + 0.6 * np.sin(i * np.pi / max(sweeps - 1, 1)))
            amp = (np.sin(phase) * 0.5 + 0.5) * 250.0  # 0..250 Hz swing
            freq[s:e] = center + amp
        phase = np.cumsum(2 * np.pi * freq / sr)
        tone = 0.95 * np.sin(phase)
        # Soft amplitude envelope so each sweep has a gentle attack/decay.
        env = np.ones(total)
        ramp = int(0.05 * sr)
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        pcm = (tone * env * 32767.0).astype("<i2")
        import wave

        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())

    def play(self, path: str) -> subprocess.Popen[bytes]:
        """Play `path` to the stimulus output device (non-blocking).

        Returns the Popen handle so callers can overlap capture with playback.
        Raises LiveMicError if ffmpeg cannot start the playback process.
        """
        ffmpeg = _require_ffmpeg()
        sink = "/tmp/_stim_sink.wav"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", path,
            "-af", "volume=12dB",
            "-f", "audiotoolbox", "-audio_device_index", str(self.output_device_index),
            sink,
        ]
        try:
            return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise LiveMicError(f"stimulus playback failed to start: {exc}") from exc


class LiveListener:
    """Captures a time window of live frames and accumulates tracking metrics."""

    def __init__(
        self,
        array: LiveMicArray | None = None,
        *,
        saliency_threshold: float = DEFAULT_LIVE_SALIENCY_THRESHOLD,
        initial_prediction_hz: float = 1000.0,
        workspace_window_ms: int = 250,
        stimulus: ParenteseStimulus | None = None,
    ) -> None:
        self.array = array or LiveMicArray(device=DEFAULT_DEVICE)
        self.saliency_threshold = saliency_threshold
        self.initial_prediction_hz = initial_prediction_hz
        self.workspace_window_ms = workspace_window_ms
        self.stimulus = stimulus or ParenteseStimulus()
        self._pipeline: CoraInfansPipeline | None = None

    def _build_pipeline(self) -> CoraInfansPipeline:
        ch = self.array.channel_count
        geometry = tuple(f"ch{i}" for i in range(ch))
        config = CoraInfansConfig(
            mic_channel_count=ch,
            mic_channel_geometry=geometry,
            saliency_threshold=self.saliency_threshold,
            workspace_window_ms=self.workspace_window_ms,
        )
        hierarchy = PredictiveHierarchy(
            frequency_coder=PredictiveCoder(
                initial_prediction_hz=self.initial_prediction_hz, learning_rate=0.5
            ),
            focus_coder=FocusTrajectoryCoder(channels=geometry),
        )
        return CoraInfansPipeline(config=config, predictive_hierarchy=hierarchy)

    def listen(
        self,
        *,
        duration_ms: int = 2000,
        frame_ms: int = 250,
        with_stimulus: bool = False,
        with_scheduling: bool = False,
        theta_ms: int = 200,
        gamma_ms: int = 25,
        capacity_cycles: int = 4,
    ) -> dict[str, object]:
        """Capture `duration_ms` of live audio in `frame_ms` steps.

        Returns an accumulated report (salient/non-salient counts, dropout,
        recovery, focus_path, focus_transitions) plus a per-frame log tagged
        with `stimulus_active`. When `with_scheduling` is set, salient events
        are also fed into a theta-gamma scheduler and its summary is attached.
        """
        from ..theta_gamma import ThetaGammaScheduler

        pipeline = self._build_pipeline()
        self._pipeline = pipeline
        scheduler = (
            ThetaGammaScheduler(
                theta_ms=theta_ms, gamma_ms=gamma_ms, capacity_cycles=capacity_cycles
            )
            if with_scheduling
            else None
        )

        stim_proc = None
        stim_path = None
        if with_stimulus:
            stim_path = tempfile.mktemp(suffix=".wav")
            self.stimulus.synthesize_wav(stim_path)
            stim_proc = self.stimulus.play(stim_path)

        frames: list[dict[str, object]] = []
        events: list[dict[str, object]] = []
        frame_is_salient: list[bool] = []
        target_frames = max(1, duration_ms // frame_ms)
        for _ in range(target_frames):
            frame = self.array.capture_frame(duration_ms=frame_ms)
            event = pipeline.process_frame(frame)
            is_salient = event is not None
            frame_is_salient.append(is_salient)
            frame_view = asdict_public(frame)
            frame_view["stimulus_active"] = with_stimulus
            frames.append(frame_view)
            if is_salient and event is not None:
                events.append(asdict_public(event))
                if scheduler is not None:
                    scheduler.push(event)

        if stim_proc is not None:
            try:
                stim_proc.wait(timeout=max(1, int(duration_ms / 1000) + 2))
            except Exception:
                stim_proc.kill()
            # Unlink only after playback has finished reading the file.
            if stim_path is not None:
                import os

                try:
                    os.unlink(stim_path)
                except OSError:
                    pass

        salient = len(events)
        non_salient = len(frames) - salient
        dropout = sum(1 for s in frame_is_salient if not s)
        recovery = sum(
            1
            for prev, cur in zip(frame_is_salient, frame_is_salient[1:])
            if (not prev) and cur
        )
        focus_path = pipeline.workspace.focus_path()
        return {
            "source": "live_xvf3800",
            "with_stimulus": with_stimulus,
            "frame_count": len(frames),
            "salient_frame_count": salient,
            "non_salient_frame_count": non_salient,
            "dropout_count": dropout,
            "recovery_count": recovery,
            "emitted_event_count": salient,
            "focus_path": focus_path,
            "focus_transitions": [
                f"{a}->{b}" for a, b in zip(focus_path, focus_path[1:])
            ],
            "frames": frames,
            "events": events,
            "workspace_snapshot": [
                asdict_public(ev) for ev in pipeline.workspace.snapshot()
            ],
            "theta_gamma": scheduler.summary() if scheduler is not None else None,
        }


def asdict_public(obj: object) -> dict[str, object]:
    """Public dict view for AudioFrame / WorkspaceEvent dataclasses."""
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return {k: v for k, v in dataclasses.asdict(obj).items()}  # type: ignore[arg-type]
    return dict(obj)  # type: ignore[arg-type]
