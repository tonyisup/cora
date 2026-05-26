"""M9 — Acoustic similarity metrics.

Measures distance between target and produced sounds using basic
spectral features extracted with numpy/scipy (no librosa required for M9a).

Usage:
    sim = AcousticSimilarity(sr=16000)
    score = sim.overall_score(target_pcm, produced_pcm)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class SimilarityResult:
    """Result of a similarity comparison."""
    overall: float  # 0.0 (different) to 1.0 (identical)
    mfcc_distance: float
    pitch_distance: float
    duration_distance: float
    loudness_distance: float
    components: Dict[str, float] = field(default_factory=dict)


class AcousticSimilarity:
    """Compute acoustic similarity between two audio signals.

    Uses FFT-based mel spectrogram and zero-crossing rate for M9a.
    No librosa dependency — pure numpy/scipy.
    """

    def __init__(self, sr: int = 16000, n_mels: int = 13) -> None:
        self._sr = sr
        self._n_mels = n_mels

    def overall_score(self, target: np.ndarray, produced: np.ndarray) -> float:
        """Return composite similarity score (0.0 to 1.0)."""
        result = self.detailed_score(target, produced)
        return result.overall

    def detailed_score(self, target: np.ndarray, produced: np.ndarray) -> SimilarityResult:
        """Return detailed similarity comparison."""
        target = target.astype(np.float32).flatten()
        produced = produced.astype(np.float32).flatten()

        mfcc_dist = self._mel_distance(target, produced)
        pitch_dist = self._pitch_distance(target, produced)
        dur_dist = self._duration_distance(target, produced)
        loud_dist = self._loudness_distance(target, produced)

        overall = float(1.0 - np.clip(
            0.4 * mfcc_dist + 0.2 * pitch_dist + 0.2 * dur_dist + 0.2 * loud_dist,
            0.0, 1.0
        ))

        return SimilarityResult(
            overall=overall,
            mfcc_distance=float(mfcc_dist),
            pitch_distance=float(pitch_dist),
            duration_distance=float(dur_dist),
            loudness_distance=float(loud_dist),
            components={
                "mel": float(mfcc_dist),
                "pitch": float(pitch_dist),
                "duration": float(dur_dist),
                "loudness": float(loud_dist),
            }
        )

    def _mel_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Mel-spectrogram cosine distance (numpy FFT-based)."""
        n_fft = 512
        hop = n_fft // 4
        def _mel_spec(x: np.ndarray) -> np.ndarray:
            # Simple mel-like band energies using FFT
            n_frames = max(1, (len(x) - n_fft) // hop + 1)
            spec = np.zeros((n_frames, self._n_mels), dtype=np.float32)
            # Mel-spaced frequency bins (simplified — linear spacing as proxy)
            bin_edges = np.linspace(0, n_fft // 2, self._n_mels + 1, dtype=int)
            for i in range(n_frames):
                start = i * hop
                frame = x[start:start + n_fft]
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                # Window + FFT
                windowed = frame * np.hanning(n_fft)
                magnitude = np.abs(np.fft.rfft(windowed))
                # Bin energies
                for j in range(self._n_mels):
                    lo, hi = bin_edges[j], bin_edges[j + 1]
                    spec[i, j] = np.sum(magnitude[lo:hi] ** 2)
            # Log compress
            spec = np.log1p(spec)
            # Mean over time
            return np.mean(spec, axis=0) if n_frames > 0 else np.zeros(self._n_mels)

        sa = _mel_spec(a)
        sb = _mel_spec(b)
        # Cosine distance
        norm = np.linalg.norm(sa) * np.linalg.norm(sb)
        if norm < 1e-9:
            return 1.0
        cosine = float(np.clip(np.dot(sa, sb) / norm, 0.0, 1.0))
        return 1.0 - cosine

    def _pitch_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Zero-crossing rate difference as pitch proxy."""
        zcr_a = float(np.mean(np.abs(np.diff(np.signbit(a).astype(float)))))
        zcr_b = float(np.mean(np.abs(np.diff(np.signbit(b).astype(float)))))
        max_zcr = max(zcr_a, zcr_b, 1e-9)
        return float(abs(zcr_a - zcr_b) / max_zcr)

    def _duration_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Normalized duration difference."""
        dur_a = len(a) / self._sr
        dur_b = len(b) / self._sr
        max_dur = max(dur_a, dur_b, 0.01)
        return float(abs(dur_a - dur_b) / max_dur)

    def _loudness_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """RMS energy envelope distance."""
        frame_n = max(1, self._sr // 100)  # 100 Hz frame rate

        def _rms_env(x: np.ndarray) -> np.ndarray:
            n_frames = max(1, len(x) // frame_n)
            env = np.zeros(n_frames, dtype=np.float32)
            for i in range(n_frames):
                frame = x[i * frame_n:(i + 1) * frame_n]
                env[i] = np.sqrt(np.mean(frame ** 2))
            return env

        env_a = _rms_env(a)
        env_b = _rms_env(b)
        max_len = max(len(env_a), len(env_b))
        if len(env_a) < max_len:
            env_a = np.pad(env_a, (0, max_len - len(env_a)))
        if len(env_b) < max_len:
            env_b = np.pad(env_b, (0, max_len - len(env_b)))
        norm = np.linalg.norm(env_a) * np.linalg.norm(env_b)
        if norm < 1e-9:
            return 1.0
        cosine = float(np.clip(np.dot(env_a, env_b) / norm, 0.0, 1.0))
        return 1.0 - cosine
