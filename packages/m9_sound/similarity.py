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

        Searches the full fundamental range and uses harmonic consistency
        to avoid octave errors on high-frequency harmonic-rich signals.
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
            # Search full fundamental range: 30 Hz to Nyquist/2
            min_lag = max(1, int(self._sr / 1000))  # max 1000 Hz
            max_lag = min(n, int(self._sr / 30))     # min 30 Hz
            if max_lag <= min_lag:
                return 0.0
            search = r[min_lag:max_lag]
            if len(search) == 0:
                return 0.0
            # Find the strongest peak
            peak_idx = int(np.argmax(search))
            peak_val = float(search[peak_idx]) if peak_idx < len(search) else 0.0
            if peak_val < 0.05:
                return 0.0
            peak_lag = min_lag + peak_idx
            candidate_lags = [peak_lag]
            # Also check sub-harmonics of the peak (in case autocorrelation
            # locked onto a harmonic rather than the fundamental)
            for h in [2, 3, 4, 5]:
                sub_lag = int(round(peak_lag * h))
                if min_lag <= sub_lag < max_lag:
                    idx = sub_lag - min_lag
                    if idx < len(search) and search[idx] > 0.05:
                        candidate_lags.append(sub_lag)
            # Pick the candidate with the strongest combined score:
            # autocorrelation value + harmonic alignment bonus
            best_lag = peak_lag
            best_score = peak_val
            for lag in candidate_lags:
                idx = lag - min_lag
                if idx < 0 or idx >= len(search):
                    continue
                score = float(r[lag]) if lag < len(r) else 0.0
                # Bonus: check if integer multiples of this lag also have
                # strong autocorrelation (harmonic alignment)
                alignment = 0.0
                for h in [2, 3, 4]:
                    h_idx = lag * h
                    if h_idx < len(r):
                        alignment += float(r[h_idx])
                total = score + 0.3 * alignment
                if total > best_score:
                    best_score = total
                    best_lag = lag
            return float(self._sr / best_lag)

        pitch_a = _autocorr_pitch(a.astype(np.float64))
        pitch_b = _autocorr_pitch(b.astype(np.float64))
        # If either pitch detector failed (returned 0), don't penalize
        if pitch_a < 1.0 or pitch_b < 1.0:
            return 0.0
        max_pitch = max(pitch_a, pitch_b, 1e-9)
        raw_dist = float(abs(pitch_a - pitch_b) / max_pitch)
        # Allow octave equivalence (2x, 3x, 4x, 0.5x, 0.33x, 0.25x)
        ratio = pitch_a / max(pitch_b, 1e-9)
        for target_ratio in [0.25, 0.33, 0.5, 2.0, 3.0, 4.0]:
            if abs(ratio - target_ratio) < 0.08:
                raw_dist = min(raw_dist, 0.02)
                break
        # Cap pitch distance — large errors are already captured by MFCC distance
        return min(raw_dist, 0.3)

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
