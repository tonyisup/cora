"""M9 — convergence and prototype refinement helpers.

These helpers support M9b (repeatability and convergence):
- summarize produced audio into compact descriptors
- refine an AcousticNode prototype over repeated trials
- compute repeatability/convergence statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from sag.acoustic_models import AcousticNode


@dataclass
class RepeatabilityStats:
    """Summary of trial-to-trial repeatability."""

    n_trials: int
    mean: float
    std: float
    cv: float
    first: float
    last: float
    improvement: float
    best: float
    worst: float

    @property
    def converged(self) -> bool:
        return self.cv <= 0.2


def _mel_centroid(audio: np.ndarray, sr: int, n_mels: int = 13) -> list[float]:
    """Compute a compact log-energy spectral centroid.

    This is a lightweight stand-in for a mel centroid. It is intentionally
    simple and dependency-free so M9b can run in the local venv.
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return [0.0] * n_mels

    n_fft = 512
    hop = n_fft // 4
    n_frames = max(1, (len(audio) - n_fft) // hop + 1)
    bin_edges = np.linspace(0, n_fft // 2, n_mels + 1, dtype=int)
    acc = np.zeros(n_mels, dtype=np.float32)
    count = 0
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * np.hanning(n_fft)
        magnitude = np.abs(np.fft.rfft(windowed))
        for j in range(n_mels):
            lo, hi = bin_edges[j], bin_edges[j + 1]
            acc[j] += float(np.sum(magnitude[lo:hi] ** 2))
        count += 1
    if count:
        acc /= count
    return np.log1p(acc).astype(np.float32).tolist()


def summarize_audio(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Summarize an audio waveform into compact descriptors."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    duration_s = len(audio) / float(sr) if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2)) if len(audio) else 0.0)
    loudness_db = float(20.0 * np.log10(rms + 1e-10))

    # Zero-crossing-rate pitch proxy.
    if len(audio) > 1:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio).astype(float)))))
    else:
        zcr = 0.0
    pitch_hz = float(zcr * sr / 2.0)

    summary = {
        "duration_s": duration_s,
        "rms": rms,
        "loudness_db": loudness_db,
        "pitch_hz": pitch_hz,
        "mfcc_centroid": _mel_centroid(audio, sr),
    }
    return summary


def compute_repeatability_stats(scores: list[float]) -> RepeatabilityStats:
    """Compute convergence-style statistics over a sequence of scores."""
    if not scores:
        return RepeatabilityStats(
            n_trials=0,
            mean=0.0,
            std=0.0,
            cv=1.0,
            first=0.0,
            last=0.0,
            improvement=0.0,
            best=0.0,
            worst=0.0,
        )

    arr = np.asarray(scores, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = float(std / max(mean, 1e-9))
    first = float(arr[0])
    last = float(arr[-1])
    return RepeatabilityStats(
        n_trials=len(scores),
        mean=mean,
        std=std,
        cv=cv,
        first=first,
        last=last,
        improvement=float(last - first),
        best=float(np.max(arr)),
        worst=float(np.min(arr)),
    )


def refine_prototype(
    prototype: AcousticNode,
    *,
    trial_id: str,
    similarity: float,
    audio_summary: dict[str, Any],
) -> AcousticNode:
    """Update a prototype node from one repeated production trial."""
    return prototype.with_update(
        similarity=similarity,
        trial_id=trial_id,
        pitch_mean_hz=float(audio_summary.get("pitch_hz", prototype.pitch_mean_hz or 0.0)),
        duration_mean_s=float(audio_summary.get("duration_s", prototype.duration_mean_s or 0.0)),
        loudness_mean_db=float(audio_summary.get("loudness_db", prototype.loudness_mean_db or 0.0)),
        mfcc_centroid=list(audio_summary.get("mfcc_centroid", prototype.mfcc_centroid)),
    )
