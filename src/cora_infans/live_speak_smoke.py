"""Bounded SPEAK smoke report helper with injectable I/O.

This module is the first implementation slice after the live SPEAK-state design
doc. It does not perform real playback by default. Callers must inject playback
and capture functions; otherwise the report is blocked after generating the
planned efference artifact.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .io.motor_plan import Vowel, synthesize_vowel, write_wav
from .speaking_simulator import run_speaking_readiness_gate

PlaybackFn = Callable[[str], dict[str, object]]
CaptureFn = Callable[..., dict[str, object]]
GateFn = Callable[[], dict[str, object]]
LatencyEstimator = Callable[..., float | None]


def _default_latency_estimator(
    *, efference_pcm: object, captured_pcm: object, sample_rate: int
) -> float | None:
    return estimate_playback_latency_ms(
        efference_pcm=efference_pcm, captured_pcm=captured_pcm, sample_rate=sample_rate
    )

_DEFAULT_INPUT_DEVICE = ":0"  # avfoundation index for the reSpeaker XVF3800
_DEFAULT_OUTPUT_DEVICE_INDEX = 2  # External Headphones (audiotoolbox)


def estimate_playback_latency_ms(
    *,
    efference_pcm: object,
    captured_pcm: object,
    sample_rate: int,
) -> float | None:
    """Deterministic cross-correlation latency estimate between efference and capture.

    This is a pure-math stub for the future real-hardware slice. It is NOT wired
    to any live playback path: the ``--real-io`` flag that would call it does not
    exist yet, and this function never triggers speaker output or microphone input.
    It recovers the sample shift that best aligns ``captured_pcm`` onto ``efference_pcm``
    (normalized cross-correlation over a bounded lag window) and converts it to ms.
    """
    import numpy as np

    ref = np.asarray(efference_pcm, dtype=np.float64)
    cap = np.asarray(captured_pcm, dtype=np.float64)
    if ref.size == 0 or cap.size == 0:
        return None
    n = min(ref.size, cap.size)
    ref = ref[:n]
    cap = cap[:n]
    # Center both signals so the correlation measures alignment, not DC offset.
    ref = ref - ref.mean()
    cap = cap - cap.mean()
    norm_ref = float(np.sqrt(np.dot(ref, ref))) or 1.0
    norm_cap = float(np.sqrt(np.dot(cap, cap))) or 1.0
    best_lag = 0
    best_corr = float("-inf")
    max_lag = min(n, 500)
    for lag in range(-max_lag, max_lag + 1):
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
    return round(best_lag / float(sample_rate) * 1000.0, 3)


def _efference_id(*, vowel: str, sample_rate: int, duration_ms: int) -> str:
    raw = f"{vowel}:{sample_rate}:{duration_ms}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _prepare_efference(
    *,
    vowel: Vowel,
    out_dir: Path,
    sample_rate: int,
    duration_ms: int,
) -> tuple[str, Path, np.ndarray]:
    """Synthesize and write the planned efference WAV; return its id, path, PCM."""
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    eid = _efference_id(vowel=vowel, sample_rate=sample_rate, duration_ms=duration_ms)
    efference_wav = out_dir / f"efference_{eid}_{vowel}.wav"
    pcm = synthesize_vowel(vowel, sample_rate=sample_rate, duration_ms=duration_ms)
    write_wav(str(efference_wav), pcm, sample_rate)
    return eid, efference_wav, pcm


def _read_wav_pcm_int16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2")


def play_wav_to_device(path: str, *, device_index: int, sample_rate: int = 48000) -> dict[str, object]:
    """Play a WAV to an explicit output device via ffmpeg audiotoolbox.

    Non-blocking: returns a Popen handle the caller must wait on. Raises
    RuntimeError if ffmpeg cannot start (no silent fallback). Mirrors the
    existing ParenteseStimulus.play contract but plays an arbitrary WAV.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required to play the efference WAV")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", path,
        "-af", "volume=12dB",
        "-f", "audiotoolbox", "-audio_device_index", str(device_index),
        "/tmp/_cora_speak_sink.wav",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"efference playback failed to start: {exc}") from exc
    proc.wait()
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
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "avfoundation", "-i", device,
        "-t", f"{seconds:.3f}", "-ac", "2", "-ar", str(sample_rate),
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"live capture failed on {device!r}: {proc.stderr.strip() or 'ffmpeg error'}"
        )
    return {"backend": "ffmpeg_avfoundation", "device": device, "sample_rate": sample_rate, "completed": True}


def _frame_tags(
    *,
    efference_id: str,
    duration_ms: int,
    max_capture_ms: int,
    latency_aligned_ms: float | None,
) -> list[dict[str, object]]:
    # Always give RECOVERING a positive-width interval so frame tagging never
    # collapses to a zero-length tag, even when capture barely exceeds the
    # speak window.
    post_end = max(duration_ms, min(max_capture_ms, duration_ms + 150))
    recovering_start = post_end
    recovering_end = max(recovering_start + 1, max_capture_ms)
    windows = [
        ("READY_TO_SPEAK", 0, 0, False),
        ("SPEAKING", 0, duration_ms, True),
        ("POST_SPEAK_SUPPRESSION", duration_ms, post_end, True),
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
    eid, efference_wav, pcm = _prepare_efference(
        vowel=vowel, out_dir=out, sample_rate=sample_rate, duration_ms=duration_ms
    )
    captured_wav = out / f"captured_{eid}_{vowel}.wav"

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
    sample_rate: int = 48000,
    duration_ms: int = 300,
    max_capture_ms: int = 1500,
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

    if duration_ms <= 0 or max_capture_ms <= 0:
        raise ValueError("duration_ms and max_capture_ms must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eid, efference_wav, pcm = _prepare_efference(
        vowel=vowel, out_dir=out, sample_rate=sample_rate, duration_ms=duration_ms
    )
    captured_wav = out / f"captured_{eid}_{vowel}.wav"

    latency_aligned_ms: float | None = None
    error: str | None = None
    playback_result: dict[str, object] | None = None
    capture_result: dict[str, object] | None = None
    verdict = "real_io_complete"
    try:
        capture_result = capture_fn(
            str(captured_wav), sample_rate=sample_rate, duration_ms=max_capture_ms
        )
        playback_result = playback_fn(str(efference_wav))
        # Read captured PCM back and estimate roundtrip latency via the injected
        # estimator (defaults to the XCORR stub).
        captured_pcm = _read_wav_pcm_int16(str(captured_wav))
        est = latency_estimator or _default_latency_estimator
        latency_aligned_ms = est(
            efference_pcm=pcm, captured_pcm=captured_pcm, sample_rate=sample_rate
        )
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
        "usable": verdict != "failed",
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
        ),
        "playback": playback_result,
        "capture": capture_result,
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
    latency_jitter_ms: float = 2.0,
    sample_rate: int = 48000,
    duration_ms: int = 300,
    max_capture_ms: int = 1500,
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
            readiness_gate_fn=readiness_gate_fn,
            latency_estimator=latency_estimator,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
        )
        per_trial.append(report)
        if report.get("verdict") == "real_io_complete" and report.get("latency_aligned_ms") is not None:
            trials_ok += 1
            latencies.append(float(report["latency_aligned_ms"]))  # type: ignore[arg-type]

    stats: dict[str, object] = {"n": len(latencies), "mean": None, "std": None, "min": None, "max": None}
    stable = False
    if latencies:
        arr = _np.asarray(latencies, dtype=_np.float64)
        stats = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
        }
        # Stable only if repeatable across trials AND within the jitter bound.
        stable = len(latencies) >= 2 and float(arr.std()) <= latency_jitter_ms

    return {
        "mode": "bounded_live_speak_smoke_trials",
        "verdict": "trials_complete",
        "real_hardware": True,
        "live_playback_allowed": False,
        "contamination_policy": "tag_only",
        "trials_run": trials,
        "trials_ok": trials_ok,
        "latency_stats": stats,
        "latency_stable": stable,
        "latency_jitter_ms": latency_jitter_ms,
        "per_trial": per_trial,
    }
