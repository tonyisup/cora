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
- Perception (`analyze_vowel`) is a real LPC spectral-envelope formant estimator
  (linear prediction roots filtered by frequency and bandwidth) over the
  synthesized waveform, so the loop is closed with actual signal processing,
  not by echoing the input label.
- Clean synthetic round-trip accuracy is measured by
  `evaluate_vowel_round_trip`; it does not establish performance on recorded
  speakers or biological vocal-tract fidelity.
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
    # F1 needs enough energy to remain identifiable after lip-radiation tilt.
    # The gain is fixed across vowels and is validated by the confusion-matrix
    # gate rather than tuned per label.
    sig = _formant_resonator(src, sample_rate, f1, 80.0) * 5.0
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
    wav: np.ndarray, sample_rate: int, order: int = 24
) -> tuple[float, float, float]:
    """Estimate F1/F2/F3 from LPC poles without consulting a vowel label."""

    if order < 8:
        raise ValueError("LPC order must be at least 8")
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    # The target formant band is below 3.5 kHz. Analyze high-rate captures near
    # 16 kHz so LPC order and pole selection stay consistent across 16/48 kHz
    # inputs. The hardware path is normally an integer 48k -> 16k decimation.
    decimation = max(1, int(round(sample_rate / 16000.0)))
    if decimation > 1:
        x = x[::decimation]
        analysis_rate = sample_rate / decimation
    else:
        analysis_rate = float(sample_rate)
    x = x - float(np.mean(x))
    # Standard pre-emphasis exposes resonances hidden by glottal spectral tilt.
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    x *= np.hamming(len(x))

    autocorrelation = np.correlate(x, x, mode="full")[len(x) - 1 : len(x) + order]
    if autocorrelation[0] <= 1e-12:
        return 500.0, 1500.0, 2500.0
    toeplitz = np.fromfunction(
        lambda i, j: autocorrelation[np.abs(i - j).astype(int)],
        (order, order),
    )
    regularization = np.eye(order) * autocorrelation[0] * 1e-8
    coefficients = np.concatenate(
        ([1.0], np.linalg.solve(toeplitz + regularization, -autocorrelation[1:]))
    )
    roots = np.roots(coefficients)
    roots = roots[np.imag(roots) >= 0.0]
    frequencies = np.angle(roots) * analysis_rate / (2.0 * np.pi)
    bandwidths = (
        -0.5
        * analysis_rate
        / np.pi
        * np.log(np.maximum(np.abs(roots), 1e-12))
    )
    candidates = sorted(
        (float(frequency), float(bandwidth))
        for frequency, bandwidth in zip(frequencies, bandwidths)
        if 200.0 <= frequency <= 3500.0 and 5.0 <= bandwidth <= 700.0
    )
    if not candidates:
        return 500.0, 1500.0, 2500.0

    # Select ordered resonances using only generic human-formant ranges and pole
    # bandwidth, never a candidate vowel label or its table entry.
    f1_candidates = [candidate for candidate in candidates if candidate[0] <= 1000.0]
    f1 = (
        min(f1_candidates, key=lambda item: item[1])[0]
        if f1_candidates
        else candidates[0][0]
    )
    f2_candidates = [
        candidate
        for candidate in candidates
        if max(700.0, f1 + 250.0) <= candidate[0] <= 2500.0
    ]
    f2_narrow = [candidate for candidate in f2_candidates if candidate[1] <= 150.0]
    f2 = (
        min(f2_narrow or f2_candidates, key=lambda item: item[0])[0]
        if f2_candidates
        else max(f1 + 300.0, 1500.0)
    )
    f3_candidates = [
        candidate
        for candidate in candidates
        if max(1800.0, f2 + 300.0) <= candidate[0] <= 3500.0
    ]
    f3_narrow = [candidate for candidate in f3_candidates if candidate[1] <= 250.0]
    f3 = (
        min(f3_narrow or f3_candidates, key=lambda item: item[0])[0]
        if f3_candidates
        else max(f2 + 500.0, 2500.0)
    )
    return f1, f2, f3


def analyze_vowel(
    wav: np.ndarray, *, sample_rate: int = 16000, order: int = 24
) -> VowelAnalysis:
    """Perception side: estimate formants and map to the nearest vowel.

    The nearest-vowel map uses the same formant table as production after LPC
    estimation; the waveform is not accompanied by its source label.
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


def evaluate_vowel_round_trip(
    *,
    sample_rates: tuple[int, ...] = (16000, 48000),
    f0_values_hz: tuple[float, ...] = (90.0, 110.0, 140.0, 180.0),
    duration_ms: int = 300,
    minimum_accuracy: float = 0.9,
) -> dict[str, object]:
    """Measure the clean synthetic perception-production fidelity gate."""

    if not sample_rates or not f0_values_hz:
        raise ValueError("sample_rates and f0_values_hz must be non-empty")
    confusion = {
        expected: {perceived: 0 for perceived in _VOWELS} for expected in _VOWELS
    }
    absolute_errors: list[list[float]] = [[], [], []]
    trials: list[dict[str, object]] = []
    correct = 0
    for sample_rate in sample_rates:
        for f0_hz in f0_values_hz:
            for vowel in _VOWELS:
                waveform = synthesize_vowel(
                    vowel,
                    duration_ms=duration_ms,
                    sample_rate=sample_rate,
                    f0_hz=f0_hz,
                )
                analysis = analyze_vowel(waveform, sample_rate=sample_rate)
                target = VOWEL_FORMANTS_HZ[vowel]
                errors = [
                    abs(measured - expected)
                    for measured, expected in zip(analysis.formant_hz, target)
                ]
                for index, error in enumerate(errors):
                    absolute_errors[index].append(error)
                is_correct = analysis.nearest_vowel == vowel
                correct += int(is_correct)
                confusion[vowel][analysis.nearest_vowel] += 1
                trials.append(
                    {
                        "expected_vowel": vowel,
                        "perceived_vowel": analysis.nearest_vowel,
                        "correct": is_correct,
                        "sample_rate": sample_rate,
                        "f0_hz": f0_hz,
                        "target_formants_hz": list(target),
                        "measured_formants_hz": list(analysis.formant_hz),
                        "absolute_formant_error_hz": errors,
                    }
                )
    accuracy = correct / len(trials)
    return {
        "mode": "clean_synthetic_vowel_round_trip_evaluation",
        "evidence_scope": "clean_synthetic_only",
        "trial_count": len(trials),
        "correct_count": correct,
        "accuracy": accuracy,
        "minimum_accuracy": minimum_accuracy,
        "criterion_passed": accuracy >= minimum_accuracy,
        "formant_mae_hz": {
            f"f{index + 1}": float(np.mean(errors))
            for index, errors in enumerate(absolute_errors)
        },
        "confusion_matrix": confusion,
        "trials": trials,
    }


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


def synthesize_tone(
    freq_hz: float = 1000.0,
    *,
    duration_ms: int = 300,
    sample_rate: int = 16000,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Synthesize a pure sinusoid (a calibration probe, not a vowel).

    Used for latency estimation where a single-frequency tone gives a cleaner
    cross-correlation peak than a vowel. No biological claim — it is a test
    stimulus, explicitly documented as such.
    """
    n = int(duration_ms / 1000.0 * sample_rate)
    t = np.arange(n) / sample_rate
    sig = np.sin(2 * np.pi * freq_hz * t) * amplitude
    ramp = int(0.005 * sample_rate)
    if ramp > 0 and ramp * 2 < n:
        env = np.ones(n)
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        sig = sig * env
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak).astype(np.float32)


def synthesize_chirp(
    f0_hz: float = 200.0,
    f1_hz: float = 4000.0,
    *,
    duration_ms: int = 300,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Synthesize a linear-frequency sweep (chirp) — the standard latency probe.

    A chirp has a unique, unambiguous cross-correlation peak (its instantaneous
    frequency keeps changing), so the playback→capture delay is locked far more
    reliably than with a stationary vowel or tone. Test stimulus only; no
    biological claim.
    """
    n = int(duration_ms / 1000.0 * sample_rate)
    t = np.arange(n) / sample_rate
    k = (f1_hz - f0_hz) / (duration_ms / 1000.0)
    phase = 2 * np.pi * (f0_hz * t + 0.5 * k * t * t)
    sig = np.sin(phase)
    ramp = int(0.005 * sample_rate)
    if ramp > 0 and ramp * 2 < n:
        env = np.ones(n)
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        sig = sig * env
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak).astype(np.float32)
