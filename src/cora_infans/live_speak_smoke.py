"""Bounded SPEAK smoke report helper with injectable I/O.

This module is the first implementation slice after the live SPEAK-state design
doc. It does not perform real playback by default. Callers must inject playback
and capture functions; otherwise the report is blocked after generating the
planned efference artifact.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np

from .io.motor_plan import (
    Vowel,
    synthesize_chirp,
    synthesize_tone,
    synthesize_vowel,
    write_wav,
)
from .speaking_simulator import run_speaking_readiness_gate

PlaybackFn = Callable[[str], dict[str, object]]
CaptureFn = Callable[..., dict[str, object]]
GateFn = Callable[[], dict[str, object]]
LatencyEstimator = Callable[..., float | None]


def save_latency_calibration(
    path: str,
    *,
    latency_ms: float,
    stimulus_key: str,
    volume_db: float,
    causal_window_ms: float,
    sample_rate: int,
    trials: int,
    std_ms: float | None,
    synchronized_capture: bool,
) -> str:
    """Persist a measured, stable loopback latency as a calibration artifact.

    Written only for a genuine repeatable measurement (the caller must have
    already confirmed ``latency_stable``). The path is a JSON file the future
    imitation loop / self-subtraction reads as a first-class anchor, so the
    conditions of the measurement (stimulus, volume, window, geometry) are
    recorded for honest auditability.
    """
    if not synchronized_capture:
        raise ValueError("refusing to save calibration without synchronized capture")
    record = {
        "kind": "playback_capture_loopback_latency",
        "latency_ms": round(float(latency_ms), 3),
        "std_ms": round(float(std_ms), 3) if std_ms is not None else None,
        "stimulus_key": stimulus_key,
        "volume_db": volume_db,
        "causal_window_ms": causal_window_ms,
        "sample_rate": sample_rate,
        "trials": trials,
        "synchronized_capture": True,
        "timing_source": "raw_pcm_byte_count",
        "contamination_policy": "tag_only",
        "note": "Anchor for frame tagging / future self-subtraction. Not a biological claim.",
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return path


@dataclass(frozen=True)
class StimulusSpec:
    """A test stimulus for the live SPEAK probe (not a biological claim)."""

    kind: str  # "vowel" | "tone" | "chirp"
    vowel: str = "a"
    freq_hz: float = 1000.0
    f0_hz: float = 200.0
    f1_hz: float = 4000.0
    amplitude: float = 0.8

    @property
    def key(self) -> str:
        if self.kind == "vowel":
            return f"v{self.vowel}"
        if self.kind == "tone":
            return f"t{int(self.freq_hz)}"
        if self.kind == "chirp":
            return f"c{int(self.f0_hz)}-{int(self.f1_hz)}"
        return "unknown"


def _stimulus_for(*, stimulus: "StimulusSpec | None", vowel: Vowel) -> "StimulusSpec":
    if stimulus is None:
        return StimulusSpec(kind="vowel", vowel=vowel)
    return stimulus


def _default_latency_estimator(
    *, efference_pcm: object, captured_pcm: object, sample_rate: int, causal_window_ms: float | None = 20.0
) -> float | None:
    return estimate_playback_latency_ms(
        efference_pcm=efference_pcm,
        captured_pcm=captured_pcm,
        sample_rate=sample_rate,
        causal_window_ms=causal_window_ms,
    )

_DEFAULT_INPUT_DEVICE = ":0"  # avfoundation index for the reSpeaker XVF3800
_DEFAULT_OUTPUT_DEVICE_INDEX = 2  # External Headphones (audiotoolbox)


def estimate_playback_latency_ms(
    *,
    efference_pcm: object,
    captured_pcm: object,
    sample_rate: int,
    expected_onset_ms: float = 0.0,
    causal_window_ms: float | None = None,
    min_peak_correlation: float = 0.1,
) -> float | None:
    """Deterministic cross-correlation latency estimate between efference and capture.

    Recovers the sample shift that best aligns ``captured_pcm`` onto ``efference_pcm``
    (normalized cross-correlation) and converts it to ms.

    Causal windowing (the M3/Corollary technique): instead of a free lag search, the
    search is constrained to ``±causal_window_ms`` around the *expected* onset
    (``expected_onset_ms``, which is 0 when playback starts at capture start). This
    rejects room-reverb false peaks outside the plausible playback→capture path and
    is what makes the latency estimate stable. When ``causal_window_ms`` is None the
    legacy free search (±500 samples) is used, for backward compatibility.
    """
    import numpy as np

    ref = np.asarray(efference_pcm, dtype=np.float64)
    cap = np.asarray(captured_pcm, dtype=np.float64)
    if ref.size == 0 or cap.size == 0:
        return None
    # A real capture is longer than the probe. Search the probe across that
    # complete timeline instead of truncating the capture to probe length (which
    # discarded delayed playback whenever device startup exceeded the probe).
    if cap.size >= ref.size and causal_window_ms is not None:
        ref_centered = ref - ref.mean()
        ref_norm = float(np.linalg.norm(ref_centered))
        if ref_norm <= 1e-12:
            return None
        center = int(round(expected_onset_ms / 1000.0 * sample_rate))
        half = int(round(causal_window_ms / 1000.0 * sample_rate))
        min_lag = max(0, center - half)
        max_lag = min(cap.size - ref.size, center + half)
        if max_lag <= min_lag:
            return None
        correlations = np.correlate(cap, ref_centered, mode="valid")
        sums = np.concatenate(([0.0], np.cumsum(cap)))
        sums_sq = np.concatenate(([0.0], np.cumsum(cap * cap)))
        count = float(ref.size)
        window_sums = sums[ref.size:] - sums[:-ref.size]
        window_sums_sq = sums_sq[ref.size:] - sums_sq[:-ref.size]
        window_norms = np.sqrt(np.maximum(window_sums_sq - window_sums**2 / count, 0.0))
        scores = correlations[min_lag : max_lag + 1] / np.maximum(
            ref_norm * window_norms[min_lag : max_lag + 1], 1e-12
        )
        best_offset = int(np.argmax(scores))
        best_lag = min_lag + best_offset
        if float(scores[best_offset]) < min_peak_correlation or best_lag in (min_lag, max_lag):
            return None
        return round(best_lag / float(sample_rate) * 1000.0, 3)
    n = min(ref.size, cap.size)
    ref = ref[:n]
    cap = cap[:n]
    # Center both signals so the correlation measures alignment, not DC offset.
    ref = ref - ref.mean()
    cap = cap - cap.mean()
    norm_ref = float(np.sqrt(np.dot(ref, ref)))
    norm_cap = float(np.sqrt(np.dot(cap, cap)))
    if norm_ref <= 1e-12 or norm_cap <= 1e-12:
        return None
    best_lag = 0
    best_corr = float("-inf")
    if causal_window_ms is None:
        min_lag = -min(n, 500)
        max_lag = min(n, 500)
    else:
        # Restrict the search to the causal window around the expected onset.
        center = int(round(expected_onset_ms / 1000.0 * sample_rate))
        half = int(round(causal_window_ms / 1000.0 * sample_rate))
        min_lag = max(-n + 1, center - half)
        max_lag = min(n - 1, center + half)
    for lag in range(min_lag, max_lag + 1):
        if lag < 0:
            a = ref[-lag:]
            b = cap[: n + lag]
        else:
            a = ref[: n - lag]
            b = cap[lag:]
        if a.size == 0:
            continue
        corr = float(np.dot(a, b)) / (norm_ref * norm_cap)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_corr < min_peak_correlation or best_lag in (min_lag, max_lag):
        return None
    return round(best_lag / float(sample_rate) * 1000.0, 3)


def _efference_id(*, stimulus_key: str, sample_rate: int, duration_ms: int) -> str:
    raw = f"{stimulus_key}:{sample_rate}:{duration_ms}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _prepare_efference(
    *,
    stimulus: "StimulusSpec",
    out_dir: Path,
    sample_rate: int,
    duration_ms: int,
) -> tuple[str, Path, np.ndarray]:
    """Synthesize and write the planned efference WAV; return its id, path, PCM.

    ``stimulus`` selects the probe: a vowel (analysis-by-synthesis), a pure tone,
    or a chirp (the standard latency probe). All are explicitly test stimuli.
    """
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    eid = _efference_id(
        stimulus_key=stimulus.key, sample_rate=sample_rate, duration_ms=duration_ms
    )
    efference_wav = out_dir / f"efference_{eid}_{stimulus.key}.wav"
    if stimulus.kind == "vowel":
        pcm = synthesize_vowel(
            stimulus.vowel,  # type: ignore[arg-type]
            sample_rate=sample_rate,
            duration_ms=duration_ms,
        )
    elif stimulus.kind == "tone":
        pcm = synthesize_tone(
            stimulus.freq_hz, duration_ms=duration_ms, sample_rate=sample_rate,
            amplitude=stimulus.amplitude,
        )
    elif stimulus.kind == "chirp":
        pcm = synthesize_chirp(
            stimulus.f0_hz, stimulus.f1_hz, duration_ms=duration_ms, sample_rate=sample_rate
        )
    else:
        raise ValueError(f"unknown stimulus kind: {stimulus.kind!r}")
    write_wav(str(efference_wav), pcm, sample_rate)
    return eid, efference_wav, pcm


def _read_wav_mono_float32(path: str) -> tuple[np.ndarray, int]:
    """Read 16-bit PCM WAV frames and return a normalized mono stream."""
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM WAV, got {wav.getsampwidth() * 8}-bit")
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if channels <= 0:
        raise ValueError("WAV must contain at least one channel")
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float32)
    mono = pcm.mean(axis=1) / 32768.0
    return mono.astype(np.float32, copy=False), int(sample_rate)


def play_wav_to_device(
    path: str, *, device_index: int, sample_rate: int = 48000, volume_db: float = 12.0
) -> dict[str, object]:
    """Play a WAV to an explicit output device via ffmpeg audiotoolbox.

    Blocking: returns only after ffmpeg exits successfully. Raises RuntimeError
    on startup or playback failure; there is no silent fallback.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required to play the efference WAV")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", path,
        "-af", f"volume={volume_db}dB",
        "-f", "audiotoolbox", "-audio_device_index", str(device_index),
        "/tmp/_cora_speak_sink.wav",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise RuntimeError(f"efference playback failed to start: {exc}") from exc
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"playback failed with ffmpeg return code {proc.returncode}")
    return {
        "backend": "ffmpeg_audiotoolbox",
        "device_index": device_index,
        "sample_rate": sample_rate,
        "started": True,
        "returncode": proc.returncode,
    }


def capture_to_wav(
    path: str,
    *,
    device: str,
    sample_rate: int = 48000,
    duration_ms: int,
    ready_event: Event | None = None,
    cancel_event: Event | None = None,
    capture_timeline: dict[str, object] | None = None,
    startup_timeout_ms: int = 3000,
) -> dict[str, object]:
    """Capture a bounded window from an input device to WAV via ffmpeg avfoundation.

    Blocks until the capture duration elapses. Raises RuntimeError on failure
    (no silent fallback). XVF3800 exposes 2 fused channels; we pin to 2 so we
    never report phantom channels.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required to capture the mic")
    seconds = max(0.001, duration_ms / 1000.0)
    channels = 2
    bytes_per_frame = channels * 2
    raw_path = f"{path}.capture.raw"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "avfoundation", "-i", device,
        "-t", f"{seconds:.3f}", "-ac", str(channels), "-ar", str(sample_rate),
        # The raw muxer otherwise buffers roughly 256 KiB before the output
        # file grows.  At 16 kHz stereo that is longer than the bounded
        # startup deadline, so a healthy AVFoundation stream looks stalled.
        # Flush packets as they arrive: file size is our verified stream
        # position and the coordinator must see it before starting playback.
        "-flush_packets", "1",
        "-f", "s16le", "-acodec", "pcm_s16le", raw_path,
    ]
    for output_path in (path, raw_path):
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass
    timeline = capture_timeline if capture_timeline is not None else {}

    def _frame_position() -> int:
        try:
            return os.path.getsize(raw_path) // bytes_per_frame
        except OSError:
            return 0

    def _cleanup_raw() -> None:
        try:
            os.remove(raw_path)
        except FileNotFoundError:
            pass

    timeline["frame_position_fn"] = _frame_position
    timeline["position_source"] = "raw_pcm_byte_count"
    with tempfile.TemporaryFile() as stderr_file:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_file)
        except OSError as exc:
            _cleanup_raw()
            raise RuntimeError(f"live capture failed to start on {device!r}: {exc}") from exc
        deadline = time.monotonic() + startup_timeout_ms / 1000.0
        while _frame_position() <= 0:
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                proc.wait()
                _cleanup_raw()
                raise RuntimeError(f"live capture on {device!r} was cancelled")
            if proc.poll() is not None:
                stderr_file.seek(0)
                detail = stderr_file.read().decode("utf-8", errors="replace").strip()
                _cleanup_raw()
                raise RuntimeError(
                    f"live capture failed on {device!r} with ffmpeg return code "
                    f"{proc.returncode}: {detail or 'ffmpeg error'}"
                )
            if time.monotonic() >= deadline:
                proc.kill()
                proc.wait()
                _cleanup_raw()
                raise RuntimeError(f"live capture on {device!r} did not become ready")
            time.sleep(0.01)
        if ready_event is not None:
            ready_event.set()
        proc.wait()
        if proc.returncode != 0:
            stderr_file.seek(0)
            detail = stderr_file.read().decode("utf-8", errors="replace").strip()
            _cleanup_raw()
            raise RuntimeError(
                f"live capture failed on {device!r} with ffmpeg return code "
                f"{proc.returncode}: {detail or 'ffmpeg error'}"
            )
    try:
        raw = np.fromfile(raw_path, dtype="<i2")
        complete_samples = raw[: raw.size - (raw.size % channels)]
        with wave.open(path, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(complete_samples.tobytes())
    finally:
        _cleanup_raw()
    return {
        "backend": "ffmpeg_avfoundation",
        "device": device,
        "sample_rate": sample_rate,
        "frames": complete_samples.size // channels,
        "completed": True,
    }


def _capture_during_playback(
    *,
    capture_path: str,
    playback_path: str,
    sample_rate: int,
    capture_duration_ms: int,
    capture_fn: CaptureFn,
    playback_fn: PlaybackFn,
    capture_preroll_ms: int = 100,
    capture_ready_timeout_ms: int = 3500,
) -> dict[str, object]:
    """Run capture around the full playback interval using an explicit ready signal."""
    if capture_preroll_ms < 0 or capture_ready_timeout_ms <= 0:
        raise ValueError("capture timing values must be positive/non-negative")
    ready_event = Event()
    cancel_event = Event()
    callback_start = time.monotonic()
    capture_timeline: dict[str, object] = {
        "frame_position_fn": lambda: int(
            (time.monotonic() - callback_start) * sample_rate
        ),
        "position_source": "wall_clock_fallback",
    }

    def _run_capture() -> tuple[dict[str, object], float]:
        capture_kwargs: dict[str, object] = {
            "sample_rate": sample_rate,
            "duration_ms": capture_duration_ms,
            "ready_event": ready_event,
        }
        parameters = list(inspect.signature(capture_fn).parameters.values())
        if any(
            parameter.name == "capture_timeline"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            capture_kwargs["capture_timeline"] = capture_timeline
        if any(
            parameter.name == "cancel_event"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            capture_kwargs["cancel_event"] = cancel_event
        result = capture_fn(capture_path, **capture_kwargs)
        return result, time.monotonic_ns() / 1_000_000.0

    executor = ThreadPoolExecutor(max_workers=1)
    abandon_capture = False
    try:
        capture_future = executor.submit(_run_capture)
        deadline = time.monotonic() + capture_ready_timeout_ms / 1000.0
        while not ready_event.wait(timeout=0.01):
            if capture_future.done():
                capture_future.result()
                raise RuntimeError("capture ended before signaling readiness")
            if time.monotonic() >= deadline:
                abandon_capture = True
                cancel_event.set()
                capture_future.cancel()
                raise RuntimeError("capture did not become ready before timeout")
        capture_start_ms = time.monotonic_ns() / 1_000_000.0
        if capture_preroll_ms:
            time.sleep(capture_preroll_ms / 1000.0)
        if capture_future.done():
            capture_future.result()
            raise RuntimeError("capture ended before playback started")
        frame_position_fn = capture_timeline.get("frame_position_fn")
        if not callable(frame_position_fn):
            raise RuntimeError("capture did not publish a stream frame position")
        frame_position = frame_position_fn()
        if not isinstance(frame_position, (int, float)):
            raise RuntimeError("capture stream frame position must be numeric")
        playback_offset_samples = int(frame_position)
        if playback_offset_samples < 0:
            raise RuntimeError("capture stream frame position cannot be negative")
        position_source = str(capture_timeline.get("position_source", "unknown"))
        stream_position_verified = position_source in {
            "raw_pcm_byte_count",
            "synthetic_frame_count",
        }
        playback_onset_ms = time.monotonic_ns() / 1_000_000.0
        playback_result = playback_fn(playback_path)
        playback_end_ms = time.monotonic_ns() / 1_000_000.0
        capture_result, capture_end_ms = capture_future.result()
        if capture_end_ms < playback_end_ms:
            raise RuntimeError("capture ended before playback finished")
    finally:
        executor.shutdown(wait=not abandon_capture, cancel_futures=abandon_capture)
    return {
        "capture": capture_result,
        "playback": playback_result,
        "capture_start_ms": capture_start_ms,
        "capture_end_ms": capture_end_ms,
        "playback_onset_ms": playback_onset_ms,
        "playback_end_ms": playback_end_ms,
        "playback_offset_samples": playback_offset_samples,
        "playback_offset_from_capture_ms": round(
            playback_offset_samples / sample_rate * 1000.0, 3
        ),
        "overlap_verified": True,
        "stream_position_verified": stream_position_verified,
        "timestamp_source": position_source,
    }


def _frame_tags(
    *,
    efference_id: str,
    duration_ms: int,
    max_capture_ms: int,
    latency_aligned_ms: float | None,
    playback_offset_ms: float = 0.0,
) -> list[dict[str, object]]:
    # Always give RECOVERING a positive-width interval so frame tagging never
    # collapses to a zero-length tag, even when capture barely exceeds the
    # speak window.
    playback_start = max(0, min(max_capture_ms, int(round(playback_offset_ms))))
    playback_end = min(max_capture_ms, playback_start + duration_ms)
    post_end = max(playback_end, min(max_capture_ms, playback_end + 150))
    recovering_start = post_end
    recovering_end = max(recovering_start + 1, max_capture_ms)
    windows = [
        ("READY_TO_SPEAK", 0, playback_start, False),
        ("SPEAKING", playback_start, playback_end, True),
        ("POST_SPEAK_SUPPRESSION", playback_end, post_end, True),
        ("RECOVERING", recovering_start, recovering_end, False),
    ]
    tags: list[dict[str, object]] = []
    for state, start, end, expected_self in windows:
        if end < start:
            end = start
        tags.append(
            {
                "speak_state": state,
                "start_ms": start,
                "end_ms": end,
                "speaker_output_active": state == "SPEAKING",
                "efference_id": efference_id,
                "expected_self_audio": expected_self,
                "self_suppression_applied": False,
                "latency_aligned_ms": latency_aligned_ms,
                "contamination_policy": "tag_only",
                "usable_for_world_learning": not expected_self,
            }
        )
    return tags


def run_live_speak_smoke_trial(
    *,
    vowel: Vowel = "a",
    out_dir: str,
    playback_fn: PlaybackFn | None = None,
    capture_fn: CaptureFn | None = None,
    stimulus: "StimulusSpec | None" = None,
    sample_rate: int = 16000,
    duration_ms: int = 300,
    max_capture_ms: int = 1500,
) -> dict[str, object]:
    """Build a bounded live-SPEAK smoke report using injected I/O only.

    With no injected functions, this intentionally returns a blocked report after
    writing the efference WAV. That gives tests and future callers the artifact
    path and report shape without touching speakers or microphones.
    """
    if duration_ms <= 0 or max_capture_ms <= 0:
        raise ValueError("duration_ms and max_capture_ms must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stim = _stimulus_for(stimulus=stimulus, vowel=vowel)
    eid, efference_wav, pcm = _prepare_efference(
        stimulus=stim, out_dir=out, sample_rate=sample_rate, duration_ms=duration_ms
    )
    captured_wav = out / f"captured_{eid}_{stim.key}.wav"

    base: dict[str, object] = {
        "mode": "bounded_live_speak_smoke",
        "verdict": "blocked_missing_injected_io",
        "real_hardware": False,
        "live_playback": False,
        "live_playback_allowed": False,
        "contamination_policy": "tag_only",
        "efference": {
            "id": eid,
            "vowel": vowel,
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "samples": len(pcm),
        },
        "artifacts": {
            "efference_wav": str(efference_wav),
            "captured_wav": None,
        },
        "frame_tags": _frame_tags(
            efference_id=eid,
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
            latency_aligned_ms=None,
        ),
        "playback": None,
        "capture": None,
    }
    if playback_fn is None or capture_fn is None:
        return base

    playback = playback_fn(str(efference_wav))
    capture = capture_fn(str(captured_wav), sample_rate=sample_rate, duration_ms=max_capture_ms)
    base["verdict"] = "fake_io_smoke_complete"
    base["artifacts"] = {
        "efference_wav": str(efference_wav),
        "captured_wav": str(captured_wav),
    }
    base["playback"] = playback
    base["capture"] = capture
    return base


def run_real_io_speak_trial(
    *,
    vowel: Vowel = "a",
    out_dir: str,
    playback_fn: PlaybackFn,
    capture_fn: CaptureFn,
    readiness_gate_fn: GateFn | None = None,
    latency_estimator: LatencyEstimator | None = None,
    stimulus: "StimulusSpec | None" = None,
    causal_window_ms: float = 20.0,
    sample_rate: int = 48000,
    duration_ms: int = 300,
    max_capture_ms: int = 1500,
    capture_preroll_ms: int = 100,
    bounded_live_io_authorized: bool = False,
) -> dict[str, object]:
    """Bounded REAL-hardware SPEAK smoke, gated by the speaking-readiness gate.

    Safety contract:
    - If the readiness gate is not promoted, abort BEFORE any I/O (no speakers,
      no mic) with ``verdict=blocked_gates_not_promoted``.
    - ``playback_fn`` / ``capture_fn`` are injected so unit tests use fakes; the
      CLI wires the real ffmpeg primitives (``play_wav_to_device`` /
      ``capture_to_wav``).
    - The first real slice is ``tag_only``: it measures latency and tags frames,
      but does NOT claim signal-level self-subtraction.
    - Any I/O error marks the trial ``failed`` and ``usable=False``; playback is
      awaited/terminated and capture still ends.
    """
    readiness_gate_fn = readiness_gate_fn or run_speaking_readiness_gate
    gate = readiness_gate_fn()
    if gate.get("verdict") != "promote_simulation_only":
        return {
            "mode": "bounded_live_speak_smoke",
            "verdict": "blocked_gates_not_promoted",
            "real_hardware": False,
            "live_playback": False,
            "live_playback_allowed": bool(gate.get("live_playback_allowed", False)),
            "contamination_policy": "tag_only",
            "readiness_gate": gate,
            "artifacts": {"efference_wav": None, "captured_wav": None},
            "frame_tags": [],
            "playback": None,
            "capture": None,
            "error": None,
        }

    if not bounded_live_io_authorized:
        return {
            "mode": "bounded_live_speak_smoke",
            "verdict": "blocked_bounded_live_io_not_authorized",
            "real_hardware": False,
            "live_playback": False,
            "live_playback_allowed": False,
            "contamination_policy": "tag_only",
            "readiness_gate": gate,
            "artifacts": {"efference_wav": None, "captured_wav": None},
            "frame_tags": [],
            "playback": None,
            "capture": None,
            "error": None,
        }

    if duration_ms <= 0 or max_capture_ms <= 0:
        raise ValueError("duration_ms and max_capture_ms must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stim = _stimulus_for(stimulus=stimulus, vowel=vowel)
    eid, efference_wav, pcm = _prepare_efference(
        stimulus=stim, out_dir=out, sample_rate=sample_rate, duration_ms=duration_ms
    )
    captured_wav = out / f"captured_{eid}_{stim.key}.wav"

    latency_aligned_ms: float | None = None
    error: str | None = None
    playback_result: dict[str, object] | None = None
    capture_result: dict[str, object] | None = None
    io_timing: dict[str, object] | None = None
    playback_offset_from_capture_ms = 0.0
    verdict = "real_io_complete"
    try:
        io_result = _capture_during_playback(
            capture_path=str(captured_wav),
            playback_path=str(efference_wav),
            sample_rate=sample_rate,
            capture_duration_ms=max_capture_ms,
            capture_fn=capture_fn,
            playback_fn=playback_fn,
            capture_preroll_ms=capture_preroll_ms,
        )
        capture_result = io_result["capture"]  # type: ignore[assignment]
        playback_result = io_result["playback"]  # type: ignore[assignment]
        playback_offset_ms = io_result["playback_offset_from_capture_ms"]
        if not isinstance(playback_offset_ms, (int, float)):
            raise TypeError("playback offset must be numeric")
        playback_offset_from_capture_ms = float(playback_offset_ms)
        io_timing = {
            "capture_start_ms": io_result["capture_start_ms"],
            "playback_onset_ms": io_result["playback_onset_ms"],
            "playback_offset_samples": io_result["playback_offset_samples"],
            "playback_offset_from_capture_ms": playback_offset_ms,
            "overlap_verified": io_result["overlap_verified"],
            "stream_position_verified": io_result["stream_position_verified"],
            "timestamp_source": io_result["timestamp_source"],
        }
        # Read captured PCM back and estimate roundtrip latency. Injected
        # estimators are used as-is; the default XCORR estimator applies the
        # causal window around the known playback onset (t=0).
        captured_pcm, captured_sample_rate = _read_wav_mono_float32(str(captured_wav))
        if captured_sample_rate != sample_rate:
            raise ValueError(
                f"capture sample rate mismatch: expected {sample_rate}, got {captured_sample_rate}"
            )
        onset_samples = int(round(float(playback_offset_ms) * sample_rate / 1000.0))
        if latency_estimator is not None:
            captured_pcm = captured_pcm[max(0, onset_samples) :]
            latency_aligned_ms = latency_estimator(
                efference_pcm=pcm, captured_pcm=captured_pcm, sample_rate=sample_rate
            )
        else:
            detected_onset_ms = estimate_playback_latency_ms(
                efference_pcm=pcm,
                captured_pcm=captured_pcm,
                sample_rate=sample_rate,
                expected_onset_ms=float(playback_offset_ms),
                causal_window_ms=causal_window_ms,
            )
            if detected_onset_ms is not None:
                latency_aligned_ms = round(detected_onset_ms - float(playback_offset_ms), 3)
        if latency_aligned_ms is None:
            verdict = "measurement_rejected"
    except Exception as exc:  # noqa: BLE001
        verdict = "failed"
        error = f"{type(exc).__name__}: {exc}"

    return {
        "mode": "bounded_live_speak_smoke",
        "verdict": verdict,
        "real_hardware": True,
        "live_playback": True,
        "live_playback_allowed": bool(gate.get("live_playback_allowed", False)),
        "contamination_policy": "tag_only",
        "usable": verdict == "real_io_complete",
        "efference": {
            "id": eid,
            "vowel": vowel,
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "samples": len(pcm),
        },
        "artifacts": {
            "efference_wav": str(efference_wav),
            "captured_wav": str(captured_wav),
        },
        "frame_tags": _frame_tags(
            efference_id=eid,
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
            latency_aligned_ms=latency_aligned_ms,
            playback_offset_ms=playback_offset_from_capture_ms,
        ),
        "playback": playback_result,
        "capture": capture_result,
        "io_timing": io_timing,
        "latency_aligned_ms": latency_aligned_ms,
        "readiness_gate": gate,
        "error": error,
    }


def run_real_io_speak_trials(
    *,
    vowel: Vowel = "a",
    out_dir: str,
    playback_fn: PlaybackFn,
    capture_fn: CaptureFn,
    trials: int = 3,
    readiness_gate_fn: GateFn | None = None,
    latency_estimator: LatencyEstimator | None = None,
    stimulus: "StimulusSpec | None" = None,
    causal_window_ms: float = 20.0,
    latency_jitter_ms: float = 2.0,
    save_calibration: str | None = None,
    volume_db: float = 12.0,
    sample_rate: int = 48000,
    duration_ms: int = 300,
    max_capture_ms: int = 1500,
    capture_preroll_ms: int = 100,
    bounded_live_io_authorized: bool = False,
) -> dict[str, object]:
    """Repeat the bounded real-I/O smoke and report the latency spread.

    This is the honesty gate before any signal-level self-subtraction: we only
    report ``latency_stable=True`` when the per-trial latency std dev is within
    ``latency_jitter_ms``. A single noisy trial (as seen in the first live run,
    where captured audio was mostly room tone) must NOT be promoted to a stable
    alignment.

    The harness delegates each trial to ``run_real_io_speak_trial``; it does not
    touch hardware itself, so the real playback/capture behavior is identical to
    a single run. Unit tests inject fakes for both I/O and the latency estimator.
    """
    import numpy as _np

    if trials <= 0:
        raise ValueError("trials must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trial_dir = out / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    per_trial: list[dict[str, object]] = []
    latencies: list[float] = []
    trials_ok = 0
    for n in range(trials):
        report = run_real_io_speak_trial(
            vowel=vowel,
            out_dir=str(trial_dir / f"t{n + 1}"),
            playback_fn=playback_fn,
            capture_fn=capture_fn,
            stimulus=stimulus,
            readiness_gate_fn=readiness_gate_fn,
            latency_estimator=latency_estimator,
            causal_window_ms=causal_window_ms,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
            capture_preroll_ms=capture_preroll_ms,
            bounded_live_io_authorized=bounded_live_io_authorized,
        )
        per_trial.append(report)
        if report.get("verdict") == "real_io_complete" and report.get("latency_aligned_ms") is not None:
            trials_ok += 1
            latencies.append(float(report["latency_aligned_ms"]))  # type: ignore[arg-type]

    stats: dict[str, object] = {"n": len(latencies), "mean": None, "std": None, "min": None, "max": None}
    stable = False
    canonical_latency_ms: float | None = None
    rejected_latency_outliers = 0
    if latencies:
        arr = _np.asarray(latencies, dtype=_np.float64)
        raw_stats = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
        }
        stats = raw_stats
        # CoreAudio occasionally contributes one device-open outlier. Accept a
        # calibration only when at least two thirds of trials form a tight cluster
        # around the median; a broadly scattered run still fails the gate.
        median = float(_np.median(arr))
        inliers = arr[_np.abs(arr - median) <= 2.0 * latency_jitter_ms]
        required_inliers = max(2, int(_np.ceil(arr.size * 2.0 / 3.0)))
        stable = inliers.size >= required_inliers and float(inliers.std()) <= latency_jitter_ms
        if stable:
            rejected_latency_outliers = int(arr.size - inliers.size)
            stats = {
                "n": int(inliers.size),
                "mean": round(float(inliers.mean()), 3),
                "std": round(float(inliers.std()), 3),
                "min": round(float(inliers.min()), 3),
                "max": round(float(inliers.max()), 3),
            }
            canonical_latency_ms = float(stats["mean"])  # type: ignore[arg-type]
    # Only publish a canonical tag set when latency is genuinely stable. This is
    # the measured, repeatable playback->capture delay that frame tagging (and
    # future self-subtraction) should anchor to — never a single noisy trial.
    canonical_frame_tags: list[dict[str, object]] | None = None
    playback_offsets: list[float] = []
    for report in per_trial:
        timing = report.get("io_timing")
        if isinstance(timing, dict):
            offset = timing.get("playback_offset_from_capture_ms")
            if isinstance(offset, (int, float)):
                playback_offsets.append(float(offset))
    canonical_playback_offset_ms = (
        float(np.mean(playback_offsets)) if len(playback_offsets) == trials else 0.0
    )
    if canonical_latency_ms is not None:
        canonical_frame_tags = _frame_tags(
            efference_id="aggregate",
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
            latency_aligned_ms=canonical_latency_ms,
            playback_offset_ms=canonical_playback_offset_ms,
        )
        for tag in canonical_frame_tags:
            tag["latency_source"] = "stable_mean"
            tag["latency_stable"] = True

    # Persist a calibration artifact when requested and the measurement is stable.
    # This is the anchor the future imitation loop / self-subtraction reads, so it
    # must only be written for a genuine, repeatable latency — never a noisy one.
    calibration_path: str | None = None
    timing_records = [report.get("io_timing") for report in per_trial]
    synchronized_capture = trials_ok == trials and all(
        isinstance(timing, dict)
        and bool(timing.get("overlap_verified"))
        and bool(timing.get("stream_position_verified"))
        for timing in timing_records
    )
    if (
        save_calibration is not None
        and canonical_latency_ms is not None
        and synchronized_capture
    ):
        calibration_path = save_latency_calibration(
            save_calibration,
            latency_ms=canonical_latency_ms,
            stimulus_key=stimulus.key if stimulus is not None else "unknown",
            volume_db=volume_db,
            causal_window_ms=causal_window_ms,
            sample_rate=sample_rate,
            trials=stats["n"],  # type: ignore[arg-type]
            std_ms=stats["std"],  # type: ignore[arg-type]
            synchronized_capture=True,
        )

    return {
        "mode": "bounded_live_speak_smoke_trials",
        "verdict": "trials_complete",
        "real_hardware": True,
        "live_playback_allowed": False,
        "contamination_policy": "tag_only",
        "trials_run": trials,
        "trials_ok": trials_ok,
        "latency_stats": stats,
        "rejected_latency_outliers": rejected_latency_outliers,
        "latency_stable": stable,
        "synchronized_capture": synchronized_capture,
        "latency_aligned_ms": canonical_latency_ms,
        "latency_jitter_ms": latency_jitter_ms,
        "causal_window_ms": causal_window_ms,
        "canonical_frame_tags": canonical_frame_tags,
        "calibration_path": calibration_path,
        "per_trial": per_trial,
    }
