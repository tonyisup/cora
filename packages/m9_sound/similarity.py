"""M9 — Acoustic similarity metrics.

Measures distance between target and produced sounds using spectral
features extracted with numpy/scipy (no librosa required for M9a).

Uses proper mel-scale binning and autocorrelation-based pitch detection
for robustness on harmonic-rich waveforms (sawtooth, square).
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

    Uses FFT-based mel spectrogram and autocorrelation pitch for M9a.
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

    # ------------------------------------------------------------------
    # Mel-spectrogram distance
    # ------------------------------------------------------------------

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Mel-spectrogram cosine distance with proper mel-scale binning."""
        n_fft = 1024
        hop = n_fft // 4

        def _mel_spec(x: np.ndarray) -> np.ndarray:
            n_frames = max(1, (len(x) - n_fft) // hop + 1)
            spec = np.zeros((n_frames, self._n_mels), dtype=np.float32)
            mel_edges = np.linspace(
                self._hz_to_mel(0.0), self._hz_to_mel(float(self._sr / 2)), self._n_mels + 1
            )
            hz_edges = self._mel_to_hz(mel_edges)
            bin_edges = np.clip(
                np.round(hz_edges / (self._sr / n_fft)).astype(int),
                0, n_fft // 2
            )
            for i in range(n_frames):
                start = i * hop
                frame = x[start:start + n_fft]
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                windowed = frame * np.hanning(n_fft)
                power = np.abs(np.fft.rfft(windowed)) ** 2
                for j in range(self._n_mels):
                    lo, hi = bin_edges[j], min(bin_edges[j + 1], len(power))
                    if hi > lo:
                        spec[i, j] = np.sum(power[lo:hi])
            spec = np.log1p(spec)
            return np.mean(spec, axis=0) if n_frames > 0 else np.zeros(self._n_mels)

        sa = _mel_spec(a)
        sb = _mel_spec(b)
        norm = np.linalg.norm(sa) * np.linalg.norm(sb)
        if norm < 1e-9:
            return 1.0
        cosine = float(np.clip(np.dot(sa, sb) / norm, 0.0, 1.0))
        return 1.0 - cosine

    # ------------------------------------------------------------------
    # Pitch distance
    # ------------------------------------------------------------------
    def _pitch_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Pitch distance using autocorrelation-based frequency estimation.

        More robust than ZCR for harmonic-rich waveforms (sawtooth, square).
        Searches overtone space to avoid octave errors on high-frequency signals.
        """
        def _autocorr_pitch(x: np.ndarray) -> float:
            x = x - np.mean(x)
            n = len(x)
            if n < 2:
                return 0.0
            r = np.correlate(x, x, mode='full')[n - 1:]
            if r[0] < 1e-9:
                return 0.0
            r = r / r[0]
            # Search for the fundamental: check harmonics 1x through 4x
            # and pick the lowest strong peak to avoid octave errors
            min_lag = max(1, int(self._sr / 800))  # max 800 Hz fundamental
            max_lag = min(n, int(self._sr / 30))   # min 30 Hz
            if max_lag <= min_lag:
                return 0.0
            search = r[min_lag:max_lag]
            if len(search) == 0:
                return 0.0
            # Find the strongest peak in the fundamental range
            peak_idx = np.argmax(search)
            peak_val = search[peak_idx] if peak_idx < len(search) else 0.0
            if peak_val < 0.05:
                return 0.0
            lag = min_lag + peak_idx
            fundamental = float(self._sr / lag)
            # Verify by checking if harmonics align
            best_lag = lag
            for harmonic in [2, 3, 4]:
                h_lag = max(1, round(lag / harmonic))
                if h_lag < len(r) and h_lag > 0:
                    h_peak = r[h_lag] if h_lag < len(r) else 0
                    # If a much stronger peak exists at a sub-harmonic, use it
                    if h_peak > peak_val * 1.2:
                        candidate_lag = h_lag
                        if candidate_lag >= min_lag:
                            test_peak = search[candidate_lag - min_lag] if candidate_lag - min_lag < len(search) else 0
                            if test_peak > 0.1:
                                best_lag = candidate_lag
                                break
            return float(self._sr / best_lag)

        pitch_a = _autocorr_pitch(a.astype(np.float64))
        pitch_b = _autocorr_pitch(b.astype(np.float64))
        max_pitch = max(pitch_a, pitch_b, 1e-9)
        return float(abs(pitch_a - pitch_b) / max_pitch)

    # ------------------------------------------------------------------
    # Duration distance
    # ------------------------------------------------------------------

    def _duration_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Normalized duration difference."""
        dur_a = len(a) / self._sr
        dur_b = len(b) / self._sr
        max_dur = max(dur_a, dur_b, 0.01)
        return float(abs(dur_a - dur_b) / max_dur)

    # ------------------------------------------------------------------
    # Loudness distance
    # ------------------------------------------------------------------

    def _loudness_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """RMS energy envelope distance."""
        frame_n = max(1, self._sr // 100)

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
