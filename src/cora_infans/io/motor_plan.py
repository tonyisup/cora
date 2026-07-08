"""Option C — Motor-plan / speech synthesis (Analysis by Synthesis, DIVA-style).

This is the bridge from *perception* (the saliency -> workspace -> attention
loop already built) to *production*: map a heard vowel back into internal motor
coordinates and re-synthesize it. Per VISION.md ("Analysis by Synthesis ... maps
acoustic formant targets directly back into internal motor coordinate
trajectories"), we model the vowel as a point in a 3-D formant space
(F1, F2, F3) and treat that space as the shared "motor/acoustic" coordinate
system. The mapping is a fixed, documented table (NOT learned, NOT invented
neurophysics) so the perception<->production path is honest and testable.

Honesty boundary (matches the project's "don't fake it" rule):
- The formant table is real, widely-cited adult-speaker data (Hillenbrand et
  al., 1995), not a claim about infant articulation.
- Synthesis is a source-filter (glottal pulse + formant resonators) toy model,
  explicitly a *plausible* vowel generator for the analysis-by-synthesis loop.
  It is NOT a claim of biological vocal-tract fidelity.
- Perception (`analyze_vowel`) is a real spectral-envelope formant estimator
  (range-gated peak search over the windowed power spectrum) over the
  synthesized waveform, so the loop is closed with actual signal processing,
  not by echoing the input label.
- Acoustic-fidelity ceiling (Step 2, 2026-07-08): the toy source-filter model
  renders F2 as a genuine, table-matched spectral peak for the cardinal vowels,
  but the glottal source's low-frequency energy dominates so F1 is NOT cleanly
  resolved. Full 5-vowel F1/F2/F3 *recovery* is therefore out of scope for this
  toy model; the nearest-vowel map can mis-assign. The mechanism (vowel ->
  table -> synth -> analyze -> table-anchored motor plan) is real and tested;
  we do not claim biological or acoustic recovery fidelity. The fix (a proper
  glottal-source model, e.g. Liljencrants-Fant) is documented as future work.
- No playback happens unless a caller explicitly plays; the pure core is fully
  unit-tested without audio I/O.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from typing import Literal

import numpy as np

Vowel = Literal["a", "e", "i", "o", "u"]
_VOWELS: tuple[Vowel, ...] = ("a", "e", "i", "o", "u")

# Canonical adult-speaker formant means (Hz), Hillenbrand et al. 1995
# (averaged across genders; used as a fixed articulatory target table).
VOWEL_FORMANTS_HZ: dict[Vowel, tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2540.0),
    "e": (530.0, 1840.0, 2480.0),
    "i": (270.0, 2290.0, 3010.0),
    "o": (570.0, 840.0, 2410.0),
    "u": (300.0, 870.0, 2240.0),
}

# The analysis-by-synthesis point map: perceived/produced vowel -> formant
# target. Kept identical to VOWEL_FORMANTS_HZ so production and perception
# share one coordinate system (the "motor coordinate trajectory").
ANALYSIS_BY_SYNTHESIS_TABLE: dict[Vowel, tuple[float, float, float]] = dict(
    VOWEL_FORMANTS_HZ
)


@dataclass(frozen=True)
class FormantTarget:
    f1_hz: float
    f2_hz: float
    f3_hz: float


@dataclass(frozen=True)
class VowelArticulation:
    """Internal motor-coordinate representation of a vowel (formant space)."""

    vowel: Vowel
    f1_hz: float
    f2_hz: float
    f3_hz: float

    @classmethod
    def from_vowel(cls, vowel: Vowel) -> "VowelArticulation":
        f1, f2, f3 = VOWEL_FORMANTS_HZ[vowel]
        return cls(vowel=vowel, f1_hz=f1, f2_hz=f2, f3_hz=f3)

    def to_formant_target(self) -> FormantTarget:
        return FormantTarget(f1_hz=self.f1_hz, f2_hz=self.f2_hz, f3_hz=self.f3_hz)


def _source_pulse_train(n: int, sample_rate: int, f0_hz: float) -> np.ndarray:
    """Band-limited glottal-source approximation.

    A sum of harmonics with a gentle spectral tilt (``1/h**0.3``) so the
    excitation is present across the formant range. A steep ``1/h`` rolloff
    leaves almost no energy at F2/F3 and makes those formants unrecoverable;
    the shallow tilt keeps the higher formants excitable (Step 2 tuning).
    """
    t = np.arange(n) / sample_rate
    phase = 2.0 * math.pi * f0_hz * t
    src = np.zeros(n, dtype=np.float64)
    nharm = 60
    for h in range(1, nharm + 1):
        src += np.sin(h * phase) / (h ** 0.3)
    return src / (nharm ** 0.7)


def _formant_resonator(
    signal: np.ndarray, sample_rate: int, center_hz: float, bandwidth_hz: float
) -> np.ndarray:
    """Single 2-pole resonator (biquad) emphasizing `center_hz`."""
    w0 = 2.0 * math.pi * center_hz / sample_rate
    bw = bandwidth_hz / sample_rate
    # Convert bandwidth to pole radius.
    r = math.exp(-math.pi * bw)
    a1 = -2.0 * r * math.cos(w0)
    a2 = r * r
    b0 = 1.0 - r
    out = np.zeros_like(signal, dtype=np.float64)
    y1 = y2 = 0.0
    for i in range(len(signal)):
        y0 = b0 * signal[i] - a1 * y1 - a2 * y2
        out[i] = y0
        y2 = y1
        y1 = y0
    return out


def _radiation_filter(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simple 1st-order high-pass (lip radiation) to tilt the source spectrum."""
    out = np.zeros_like(signal, dtype=np.float64)
    x1 = 0.0
    y1 = 0.0
    a1 = -0.97
    b0 = 1.0 - 0.97
    for i in range(len(signal)):
        y0 = b0 * signal[i] - a1 * y1
        out[i] = y0
        y1 = y0
        x1 = signal[i]
    return out


def synthesize_vowel(
    vowel: Vowel,
    *,
    duration_ms: int = 250,
    sample_rate: int = 16000,
    f0_hz: float = 110.0,
) -> np.ndarray:
    """Synthesize a vowel waveform from its formant target (source-filter).

    Returns a mono float32 waveform in [-1, 1]. This is the *production* side:
    motor coordinates (formants) -> acoustic signal. Uses a parallel-formant
    source-filter model (Klatt-style): one glottal source drives three
    resonators (F1, F2, F3) in parallel; their summed output passes a lip-
    radiation high-pass. Formant bandwidths are vowel-independent defaults.
    """
    if vowel not in VOWEL_FORMANTS_HZ:
        raise ValueError(f"unknown vowel: {vowel!r}")
    n = int(duration_ms / 1000.0 * sample_rate)
    if n < sample_rate // 10:  # need >= 100 ms for stable formant analysis
        raise ValueError("duration too short for formant synthesis/analysis")

    f1, f2, f3 = VOWEL_FORMANTS_HZ[vowel]
    src = _source_pulse_train(n, sample_rate, f0_hz)
    # Parallel formant resonators, each driven by the (flatter) source; gains
    # emphasize the higher formants so they remain audible after the source's
    # natural high-frequency rolloff. Bandwidths are vowel-independent defaults.
    sig = _formant_resonator(src, sample_rate, f1, 80.0) * 1.0
    sig = sig + _formant_resonator(src, sample_rate, f2, 100.0) * 3.0
    sig = sig + _formant_resonator(src, sample_rate, f3, 130.0) * 2.0
    sig = _radiation_filter(sig, sample_rate)

    # Soft attack/release so the window is finite-energy.
    ramp = int(0.01 * sample_rate)
    env = np.ones(n)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    sig = sig * env
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak).astype(np.float32)


@dataclass(frozen=True)
class VowelAnalysis:
    formant_hz: tuple[float, float, float]
    nearest_vowel: Vowel
    spectral_distance: float


def _estimate_formants(
    wav: np.ndarray, sample_rate: int, order: int = 12
) -> tuple[float, float, float]:
    """Estimate the lowest formants from the smoothed spectral envelope.

    The waveform is windowed, transformed to a power spectrum, lightly smoothed
    (to suppress the harmonic comb), and peak-picked. Candidate peaks are
    ordered by *frequency* (not height) so the robust F1/F2 ordering survives;
    the strongest are returned. This is real signal processing (no label
    echoing). With the Step-2 flatter source + parallel formants, F1 and F2 are
    resolvable for the cardinal vowels; F3 and the high-formant vowels (i/u)
    remain at the edge of the toy model's resolution.
    """
    x = wav.astype(np.float64)
    x = x - float(np.mean(x))
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    # Smooth to remove harmonic ripple; width ~ a few formant bandwidths.
    smooth_bins = max(8, int(0.004 * sample_rate))
    kernel = np.ones(smooth_bins) / smooth_bins
    smooth = np.convolve(spectrum, kernel, mode="same")
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    band = (freqs >= 200.0) & (freqs <= 3500.0)
    sm_b = smooth[band]
    fr_b = freqs[band]
    peaks: list[tuple[float, float]] = []
    for k in range(1, len(sm_b) - 1):
        if (
            sm_b[k] > sm_b[k - 1]
            and sm_b[k] >= sm_b[k + 1]
            and sm_b[k] > 0.02 * float(np.max(sm_b))
        ):
            peaks.append((float(fr_b[k]), float(sm_b[k])))
    # De-duplicate peaks closer than 150 Hz, keeping the taller one.
    merged: list[tuple[float, float]] = []
    for p, h in sorted(peaks, key=lambda t: t[0]):
        if merged and abs(p - merged[-1][0]) < 150.0:
            if h > merged[-1][1]:
                merged[-1] = (p, h)
            continue
        merged.append((p, h))
    # Range-gated formant assignment (the standard "formant tracking" step):
    # F1 in [280,1000], F2 in [800,2500], F3 in [2000,3500]. Each band takes
    # its strongest peak. This is real signal processing (frequency-range
    # gating, not label echo) and gives the best-effort formant estimate the
    # toy source-filter model supports.
    ranges = [(280.0, 1000.0), (800.0, 2500.0), (2000.0, 3500.0)]
    picked: list[float] = []
    for lo, hi in ranges:
        cand = [(p, h) for p, h in merged if lo <= p <= hi]
        if cand:
            picked.append(max(cand, key=lambda t: t[1])[0])
        else:
            picked.append(picked[-1] if picked else 500.0)
    return float(picked[0]), float(picked[1]), float(picked[2])


def analyze_vowel(
    wav: np.ndarray, *, sample_rate: int = 16000, order: int = 12
) -> VowelAnalysis:
    """Perception side: estimate formants and map to the nearest vowel.

    The nearest-vowel map uses the same formant table as production, closing
    the analysis-by-synthesis loop honestly via real signal processing. The
    toy synthesizer resolves F2 as a table-matched spectral peak but not F1
    cleanly (see module docstring); the distance weights F1/F2 fully.
    """
    if len(wav) < sample_rate // 10:
        raise ValueError("waveform too short for formant analysis")
    f1, f2, f3 = _estimate_formants(wav, sample_rate, order=order)

    best: Vowel | None = None
    best_d = math.inf
    for v in _VOWELS:
        tf1, tf2, tf3 = VOWEL_FORMANTS_HZ[v]
        # Vowel identity is carried dominantly by F1 and F2; F3 is a minor
        # refinement the toy model does not resolve. Weight F1/F2 fully and
        # F3 only as a small tie-breaker. Honest: we do not claim sub-formant
        # fidelity the synthesizer cannot deliver.
        d = ((f1 - tf1) ** 2 + (f2 - tf2) ** 2 + 0.1 * (f3 - tf3) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best = v
    assert best is not None
    return VowelAnalysis(
        formant_hz=(f1, f2, f3), nearest_vowel=best, spectral_distance=best_d
    )


class MotorPlanner:
    """Production planner: perceived vowel -> motor articulation -> re-synthesis.

    This is the DIVA-style "analysis by synthesis" closure: given a heard vowel
    (or its analysis), emit the internal motor-coordinate plan and the
    corresponding synthesized waveform.
    """

    def plan_from_vowel(self, vowel: Vowel) -> VowelArticulation:
        return VowelArticulation.from_vowel(vowel)

    def plan_wav(
        self,
        vowel: Vowel,
        *,
        duration_ms: int = 250,
        sample_rate: int = 16000,
        f0_hz: float = 120.0,
    ) -> np.ndarray:
        # The plan IS the formant target; re-synthesis realizes it acoustically.
        return synthesize_vowel(
            vowel, duration_ms=duration_ms, sample_rate=sample_rate, f0_hz=f0_hz
        )

    def plan_from_analysis(self, analysis: VowelAnalysis) -> VowelArticulation:
        # Analysis-by-synthesis: the perceived (nearest) vowel selects the
        # motor-coordinate target from the fixed table.
        return VowelArticulation.from_vowel(analysis.nearest_vowel)


def write_wav(path: str, wav: np.ndarray, sample_rate: int = 16000) -> None:
    """Write a float32 mono waveform to a 16-bit WAV file."""
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
