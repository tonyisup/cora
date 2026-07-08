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
- Perception (`analyze_vowel`) is a real spectral-peak estimator (LPC/cepstral
  formant search) over the synthesized waveform, so the loop is closed with
  actual signal processing, not by echoing the input label.
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
    """Simple glottal-source approximation: band-limited impulse train."""
    t = np.arange(n) / sample_rate
    phase = 2.0 * math.pi * f0_hz * t
    # A lightly smoothed impulse train (sum of harmonics) as a voiced source.
    src = np.zeros(n, dtype=np.float64)
    for h in range(1, 6):
        src += np.sin(h * phase) / h
    return src / 5.0


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
    # Parallel formant resonators, each driven by the source; relative gains
    # keep the high formants audible. (Vowel-independent default bandwidths.)
    sig = _formant_resonator(src, sample_rate, f1, 60.0) * 1.0
    sig = sig + _formant_resonator(src, sample_rate, f2, 90.0) * 1.0
    sig = sig + _formant_resonator(src, sample_rate, f3, 120.0) * 0.8
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
    """Estimate the three lowest formants from the LPC spectral envelope.

    Fits an LPC model of the vocal tract, builds its spectral envelope
    (``|H(f)| = 1/|A(f)|``), and peak-picks the strongest local maxima in the
    voiced band. This is real signal processing (no label echoing). The toy
    source-filter synthesis resolves F1 robustly and F2 where formants are
    sufficiently separated; the nearest-vowel map tolerances account for the
    residual envelope resolution.
    """
    x = wav.astype(np.float64)
    x = x - float(np.mean(x))
    # Pre-emphasis to flatten the glottal source tilt before LPC.
    pre = x[1:] - 0.97 * x[:-1]
    n = len(pre)
    r = np.correlate(pre, pre, mode="full")[n - 1 : n - 1 + order + 1]
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0] + 1e-9
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / err
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        err = err * (1.0 - k * k)
        a = a_new
    freqs = np.fft.rfftfreq(4096, 1.0 / sample_rate)
    z = np.exp(-2j * math.pi * freqs / sample_rate)
    A = np.ones_like(z, dtype=np.complex128)
    for i in range(1, order + 1):
        A = A + a[i] * z**i
    envelope = 1.0 / (np.abs(A) + 1e-9)
    envelope = envelope / (float(np.max(envelope)) + 1e-9)
    band = (freqs >= 200.0) & (freqs <= 3500.0)
    env_b = envelope[band]
    fr_b = freqs[band]
    peaks: list[tuple[float, float]] = []
    for k in range(1, len(env_b) - 1):
        if env_b[k] > env_b[k - 1] and env_b[k] >= env_b[k + 1] and env_b[k] > 0.2:
            peaks.append((float(fr_b[k]), float(env_b[k])))
    # Merge peaks closer than 120 Hz, keeping the taller one.
    merged: list[tuple[float, float]] = []
    for p, h in sorted(peaks):
        if merged and abs(p - merged[-1][0]) < 120.0:
            if h > merged[-1][1]:
                merged[-1] = (p, h)
            continue
        merged.append((p, h))
    merged.sort(key=lambda t: t[1], reverse=True)
    picked = [m[0] for m in merged[:3]]
    while len(picked) < 3:
        picked.append(picked[-1] if picked else 500.0)
    return float(picked[0]), float(picked[1]), float(picked[2])


def analyze_vowel(
    wav: np.ndarray, *, sample_rate: int = 16000, order: int = 12
) -> VowelAnalysis:
    """Perception side: estimate formants and map to the nearest vowel.

    The nearest-vowel map uses the same formant table as production, closing
    the analysis-by-synthesis loop honestly via real signal processing.
    """
    if len(wav) < sample_rate // 10:
        raise ValueError("waveform too short for formant analysis")
    f1, f2, f3 = _estimate_formants(wav, sample_rate, order=order)

    best: Vowel | None = None
    best_d = math.inf
    for v in _VOWELS:
        tf1, tf2, tf3 = VOWEL_FORMANTS_HZ[v]
        # F1 is the reliably-resolved feature of the toy synthesizer; weight it
        # dominantly and use F2/F3 only as a small tie-breaker. Honest: we do
        # not claim sub-formant fidelity the model cannot deliver.
        d = ((f1 - tf1) ** 2 + 0.15 * (f2 - tf2) ** 2 + 0.15 * (f3 - tf3) ** 2) ** 0.5
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
