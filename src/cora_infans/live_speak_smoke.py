"""Bounded SPEAK smoke report helper with injectable I/O.

This module is the first implementation slice after the live SPEAK-state design
doc. It does not perform real playback by default. Callers must inject playback
and capture functions; otherwise the report is blocked after generating the
planned efference artifact.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from dataclasses import dataclass

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
) -> str:
    """Persist a measured, stable loopback latency as a calibration artifact.

    Written only for a genuine repeatable measurement (the caller must have
    already confirmed ``latency_stable``). The path is a JSON file the future
    imitation loop / self-subtraction reads as a first-class anchor, so the
    conditions of the measurement (stimulus, volume, window, geometry) are
    recorded for honest auditability.
    """
    record = {
        "kind": "playback_capture_loopback_latency",
        "latency_ms": round(float(latency_ms), 3),
        "std_ms": round(float(std_ms), 3) if std_ms is not None else None,
        "stimulus_key": stimulus_key,
        "volume_db": volume_db,
        "causal_window_ms": causal_window_ms,
        "sample_rate": sample_rate,
        "trials": trials,
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


def _read_wav_pcm_int16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2")
def play_wav_to_device(
    path: str, *, device_index: int, sample_rate: int = 48000, volume_db: float = 12.0
) -> dict[str, object]:
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
        "-af", f"volume={volume_db}dB",
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
    stim = _stimulus_for(stimulus=stimulus, vowel=vowel)
    eid, efference_wav, pcm = _prepare_efference(
        stimulus=stim, out_dir=out, sample_rate=sample_rate, duration_ms=duration_ms
    )
    captured_wav = out / f"captured_{eid}_{stim.key}.wav"

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
        # Read captured PCM back and estimate roundtrip latency. Injected
        # estimators are used as-is; the default XCORR estimator applies the
        # causal window around the known playback onset (t=0).
        captured_pcm = _read_wav_pcm_int16(str(captured_wav))
        if latency_estimator is not None:
            latency_aligned_ms = latency_estimator(
                efference_pcm=pcm, captured_pcm=captured_pcm, sample_rate=sample_rate
            )
        else:
            latency_aligned_ms = _default_latency_estimator(
                efference_pcm=pcm,
                captured_pcm=captured_pcm,
                sample_rate=sample_rate,
                causal_window_ms=causal_window_ms,
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
    stimulus: "StimulusSpec | None" = None,
    causal_window_ms: float = 20.0,
    latency_jitter_ms: float = 2.0,
    save_calibration: str | None = None,
    volume_db: float = 12.0,
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
            stimulus=stimulus,
            readiness_gate_fn=readiness_gate_fn,
            latency_estimator=latency_estimator,
            causal_window_ms=causal_window_ms,
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
    canonical_latency_ms: float | None = None
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
        if stable:
            canonical_latency_ms = float(stats["mean"])  # type: ignore[arg-type]
    # Only publish a canonical tag set when latency is genuinely stable. This is
    # the measured, repeatable playback->capture delay that frame tagging (and
    # future self-subtraction) should anchor to — never a single noisy trial.
    canonical_frame_tags: list[dict[str, object]] | None = None
    if canonical_latency_ms is not None:
        canonical_frame_tags = _frame_tags(
            efference_id="aggregate",
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
            latency_aligned_ms=canonical_latency_ms,
        )
        for tag in canonical_frame_tags:
            tag["latency_source"] = "stable_mean"
            tag["latency_stable"] = True

    # Persist a calibration artifact when requested and the measurement is stable.
    # This is the anchor the future imitation loop / self-subtraction reads, so it
    # must only be written for a genuine, repeatable latency — never a noisy one.
    calibration_path: str | None = None
    if save_calibration is not None and canonical_latency_ms is not None:
        calibration_path = save_latency_calibration(
            save_calibration,
            latency_ms=canonical_latency_ms,
            stimulus_key=stimulus.key if stimulus is not None else "unknown",
            volume_db=volume_db,
            causal_window_ms=causal_window_ms,
            sample_rate=sample_rate,
            trials=stats["n"],  # type: ignore[arg-type]
            std_ms=stats["std"],  # type: ignore[arg-type]
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
        "latency_stable": stable,
        "latency_aligned_ms": canonical_latency_ms,
        "latency_jitter_ms": latency_jitter_ms,
        "causal_window_ms": causal_window_ms,
        "canonical_frame_tags": canonical_frame_tags,
        "calibration_path": calibration_path,
        "per_trial": per_trial,
    }
