"""Long-window, speaker-adaptive register saliency experiments.

The Music Lab study standardized acoustic summaries within voices and modeled
speech and song separately. This module implements a deliberately small,
transparent analogue: a raw-acoustic speech/song ridge gate followed by an
IDS/ADS ridge gate over unlabeled within-speaker normalized summaries.

The calibration boundary is explicit. A held-out speaker must provide at least
two observations before within-speaker contrast is meaningful. Batch evaluation
uses all supplied observations from that speaker without labels, so it is not a
cold-start or strictly causal live detector.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .io.offline_eval import CorpusClip
from .io.motor_plan import estimate_formants
from .io.wav_frames import build_pcm_observations, read_pcm_wav


REGISTER_FEATURE_NAMES = (
    "pitch_log2_median",
    "pitch_iqr_semitones",
    "pitch_std_semitones",
    "pitch_confidence_median",
    "pitch_confidence_iqr",
    "pitch_confidence_mean",
    "intensity_db_median",
    "intensity_db_iqr",
    "intensity_db_std",
    "spectral_centroid_log_median",
    "spectral_centroid_log_iqr",
    "spectral_centroid_log_std",
    "spectral_flux_log_median",
    "spectral_flux_log_iqr",
    "modulation_peak_hz",
    "modulation_low_band_fraction",
    "modulation_speech_band_fraction",
    "modulation_low_to_speech_ratio",
    "modulation_entropy",
    "modulation_peak_fraction",
    "spectral_flatness_median",
    "spectral_flatness_iqr",
    "zero_crossing_rate_median",
    "zero_crossing_rate_iqr",
    "formant_dispersion_semitones_median",
    "formant_dispersion_semitones_iqr",
    "vowel_travel_semitones_median",
    "vowel_travel_semitones_iqr",
    "yin_pitch_median_hz",
    "yin_pitch_iqr_hz",
    "yin_pitch_std_hz",
    "yin_pitch_confidence_median",
    "yin_pitch_confidence_iqr",
    "yin_voiced_fraction",
)

SPEECH_GATE_FEATURES = (
    "pitch_log2_median",
    "pitch_iqr_semitones",
    "pitch_confidence_median",
    "pitch_confidence_iqr",
    "intensity_db_median",
    "intensity_db_iqr",
    "spectral_centroid_log_median",
    "spectral_centroid_log_iqr",
    "modulation_peak_hz",
    "modulation_low_band_fraction",
    "modulation_speech_band_fraction",
    "modulation_low_to_speech_ratio",
    "modulation_entropy",
    "modulation_peak_fraction",
    "yin_pitch_median_hz",
    "yin_pitch_iqr_hz",
    "yin_pitch_std_hz",
    "yin_pitch_confidence_median",
    "yin_pitch_confidence_iqr",
)

ADAPTIVE_IDS_FEATURES = (
    "yin_pitch_median_hz",
    "yin_pitch_iqr_hz",
    "yin_pitch_std_hz",
    "yin_pitch_confidence_median",
    "yin_pitch_confidence_iqr",
    "intensity_db_median",
    "intensity_db_iqr",
    "spectral_centroid_log_median",
    "spectral_centroid_log_iqr",
)


def _iqr(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _safe_summary(values: np.ndarray) -> tuple[float, float, float]:
    if not len(values):
        return 0.0, 0.0, 0.0
    return float(np.median(values)), _iqr(values), float(np.std(values))


def _yin_pitch_track(
    mono: np.ndarray,
    sample_rate: int,
    *,
    minimum_hz: float = 70.0,
    maximum_hz: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a compact FFT-YIN pitch/confidence track.

    The register path needs robust pitch variability, not only one dominant
    pitch per frame. Audio is decimated near 12 kHz, analyzed in overlapping
    50 ms windows, and accepted only when cumulative mean normalized difference
    confidence is at least 0.65.
    """

    decimation = max(1, int(round(sample_rate / 12000.0)))
    signal = np.asarray(mono, dtype=np.float64)[::decimation]
    analysis_rate = sample_rate / decimation
    window = max(64, int(round(analysis_rate * 0.05)))
    hop = max(16, int(round(analysis_rate * 0.01)))
    minimum_lag = max(2, int(analysis_rate / maximum_hz))
    maximum_lag = min(window - 2, int(analysis_rate / minimum_hz))
    pitches: list[float] = []
    confidences: list[float] = []
    analyzed_window_count = 0
    for start in range(0, len(signal) - window + 1, hop):
        analyzed_window_count += 1
        frame = signal[start : start + window].copy()
        frame -= float(np.mean(frame))
        if math.sqrt(float(np.mean(frame * frame))) < 1e-5:
            continue
        fft_size = 1 << (2 * window - 1).bit_length()
        spectrum = np.fft.rfft(frame, fft_size)
        autocorrelation = np.fft.irfft(
            spectrum * np.conjugate(spectrum), fft_size
        )[: maximum_lag + 1]
        cumulative_energy = np.concatenate(([0.0], np.cumsum(frame * frame)))
        lags = np.arange(maximum_lag + 1)
        difference = (
            cumulative_energy[window - lags]
            + cumulative_energy[window]
            - cumulative_energy[lags]
            - 2.0 * autocorrelation
        )
        difference = np.maximum(difference, 0.0)
        normalized = np.ones(maximum_lag + 1, dtype=np.float64)
        cumulative_difference = np.cumsum(difference[1:])
        normalized[1:] = (
            difference[1:]
            * np.arange(1, maximum_lag + 1)
            / (cumulative_difference + 1e-18)
        )
        search = normalized[minimum_lag : maximum_lag + 1]
        below_threshold = np.flatnonzero(search < 0.2)
        if len(below_threshold):
            lag = minimum_lag + int(below_threshold[0])
            while lag < maximum_lag and normalized[lag + 1] < normalized[lag]:
                lag += 1
        else:
            lag = minimum_lag + int(np.argmin(search))
        confidence = 1.0 - float(normalized[lag])
        if confidence < 0.65:
            continue
        offset = 0.0
        if 1 <= lag < maximum_lag:
            left, center, right = normalized[lag - 1 : lag + 2]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                offset = 0.5 * (left - right) / denominator
        pitches.append(float(analysis_rate / (lag + offset)))
        confidences.append(confidence)
    return (
        np.asarray(pitches, dtype=np.float64),
        np.asarray(confidences, dtype=np.float64),
        analyzed_window_count,
    )


def extract_register_features(
    samples: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    """Extract robust whole-window pitch, timbre, intensity, and rhythm summaries."""

    pcm = np.asarray(samples, dtype=np.float32)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.ndim != 2 or not len(pcm):
        raise ValueError("register features require non-empty PCM")
    frames = build_pcm_observations(
        pcm,
        sample_rate=sample_rate,
        frame_ms=25,
        source="register_feature_window",
    )
    pitch = np.asarray(
        [
            float(frame.features["pitch_hz"])
            for frame in frames
            if frame.features and frame.features.get("pitch_hz") is not None
        ],
        dtype=np.float64,
    )
    confidence = np.asarray(
        [float((frame.features or {}).get("pitch_confidence") or 0.0) for frame in frames],
        dtype=np.float64,
    )
    intensity = np.asarray(
        [max(float((frame.features or {}).get("peak_rms") or 0.0), 1e-8) for frame in frames],
        dtype=np.float64,
    )
    centroid = np.asarray(
        [
            max(
                float((frame.features or {}).get("spectral_centroid_hz") or 0.0),
                1.0,
            )
            for frame in frames
        ],
        dtype=np.float64,
    )
    flux = np.asarray(
        [max(float((frame.features or {}).get("spectral_flux") or 0.0), 1e-12) for frame in frames],
        dtype=np.float64,
    )

    pitch_log = np.log2(np.maximum(pitch, 1.0))
    pitch_median, pitch_iqr, pitch_std = _safe_summary(pitch_log)
    confidence_median, confidence_iqr, _confidence_std = _safe_summary(confidence)
    intensity_median, intensity_iqr, intensity_std = _safe_summary(
        20.0 * np.log10(intensity)
    )
    centroid_median, centroid_iqr, centroid_std = _safe_summary(np.log(centroid))
    flux_median, flux_iqr, _flux_std = _safe_summary(np.log1p(flux * 1e4))

    mono = np.mean(pcm, axis=1).astype(np.float64)
    yin_pitch, yin_confidence, yin_window_count = _yin_pitch_track(mono, sample_rate)
    yin_pitch_median, yin_pitch_iqr, yin_pitch_std = _safe_summary(yin_pitch)
    yin_confidence_median, yin_confidence_iqr, _yin_confidence_std = _safe_summary(
        yin_confidence
    )
    envelope_samples = max(1, int(round(sample_rate * 0.01)))
    envelope = np.asarray(
        [
            math.sqrt(float(np.mean(mono[start : start + envelope_samples] ** 2)))
            for start in range(0, len(mono), envelope_samples)
            if len(mono[start : start + envelope_samples])
        ],
        dtype=np.float64,
    )
    envelope -= float(np.mean(envelope))
    modulation_power = np.abs(
        np.fft.rfft(envelope * np.hanning(len(envelope)))
    ) ** 2
    modulation_hz = np.fft.rfftfreq(len(envelope), 0.01)
    selected = (modulation_hz >= 0.5) & (modulation_hz <= 10.0)
    power = modulation_power[selected]
    frequencies = modulation_hz[selected]
    total_power = float(np.sum(power)) + 1e-18
    normalized_power = power / total_power

    def band_fraction(low: float, high: float) -> float:
        return float(
            np.sum(power[(frequencies >= low) & (frequencies < high)]) / total_power
        )

    low_band = band_fraction(0.5, 2.5)
    speech_band = band_fraction(3.0, 8.0)
    modulation_peak = (
        float(frequencies[int(np.argmax(power))]) if len(power) else 0.0
    )
    modulation_entropy = (
        float(
            -np.sum(normalized_power * np.log(normalized_power + 1e-18))
            / np.log(max(len(normalized_power), 2))
        )
        if len(normalized_power)
        else 0.0
    )

    acoustic_frame_samples = max(8, int(round(sample_rate * 0.025)))
    flatness_values: list[float] = []
    crossing_values: list[float] = []
    for start in range(0, len(mono), acoustic_frame_samples):
        acoustic_frame = mono[start : start + acoustic_frame_samples]
        if len(acoustic_frame) < 8:
            continue
        centered = acoustic_frame - float(np.mean(acoustic_frame))
        power_spectrum = np.abs(
            np.fft.rfft(centered * np.hanning(len(centered)))
        ) ** 2
        arithmetic = float(np.mean(power_spectrum)) + 1e-18
        geometric = float(np.exp(np.mean(np.log(power_spectrum + 1e-18))))
        flatness_values.append(geometric / arithmetic)
        signs = np.signbit(centered)
        crossing_values.append(float(np.mean(signs[1:] != signs[:-1])))
    flatness = np.asarray(flatness_values, dtype=np.float64)
    crossings = np.asarray(crossing_values, dtype=np.float64)
    flatness_median, flatness_iqr, _flatness_std = _safe_summary(flatness)
    crossing_median, crossing_iqr, _crossing_std = _safe_summary(crossings)

    formant_window = max(8, int(round(sample_rate * 0.05)))
    formant_hop = max(formant_window, int(round(sample_rate * 0.1)))
    mono_rms = math.sqrt(float(np.mean(mono * mono)))
    formants: list[tuple[float, float, float]] = []
    for start in range(0, max(1, len(mono) - formant_window + 1), formant_hop):
        formant_frame = mono[start : start + formant_window]
        if len(formant_frame) < formant_window:
            continue
        frame_rms = math.sqrt(float(np.mean(formant_frame * formant_frame)))
        if frame_rms < max(1e-6, mono_rms * 0.2):
            continue
        formants.append(estimate_formants(formant_frame, sample_rate))
    dispersions = np.asarray(
        [12.0 * math.log2(max(f3, f1 + 1.0) / max(f1, 1.0)) for f1, _f2, f3 in formants],
        dtype=np.float64,
    )
    travels = np.asarray(
        [
            math.sqrt(
                (12.0 * math.log2(current[0] / previous[0])) ** 2
                + (12.0 * math.log2(current[1] / previous[1])) ** 2
            )
            for previous, current in zip(formants, formants[1:])
            if previous[0] > 0.0
            and previous[1] > 0.0
            and current[0] > 0.0
            and current[1] > 0.0
        ],
        dtype=np.float64,
    )
    dispersion_median, dispersion_iqr, _dispersion_std = _safe_summary(dispersions)
    travel_median, travel_iqr, _travel_std = _safe_summary(travels)
    values = (
        pitch_median,
        pitch_iqr * 12.0,
        pitch_std * 12.0,
        confidence_median,
        confidence_iqr,
        float(np.mean(confidence)) if len(confidence) else 0.0,
        intensity_median,
        intensity_iqr,
        intensity_std,
        centroid_median,
        centroid_iqr,
        centroid_std,
        flux_median,
        flux_iqr,
        modulation_peak,
        low_band,
        speech_band,
        low_band / (speech_band + 1e-9),
        modulation_entropy,
        float(np.max(normalized_power)) if len(normalized_power) else 0.0,
        flatness_median,
        flatness_iqr,
        crossing_median,
        crossing_iqr,
        dispersion_median,
        dispersion_iqr,
        travel_median,
        travel_iqr,
        yin_pitch_median,
        yin_pitch_iqr,
        yin_pitch_std,
        yin_confidence_median,
        yin_confidence_iqr,
        len(yin_pitch) / yin_window_count if yin_window_count else 0.0,
    )
    return dict(zip(REGISTER_FEATURE_NAMES, values, strict=True))


@dataclass(frozen=True)
class RegisterExample:
    clip_id: str
    speaker_id: str
    fieldsite: str
    label: str
    features: dict[str, float]


@dataclass(frozen=True)
class VocalActivityGate:
    """Transparent long-window vocal/non-vocal gate based on YIN coverage."""

    minimum_voiced_fraction: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_voiced_fraction <= 1.0:
            raise ValueError("minimum voiced fraction must be in [0, 1]")

    def score(self, features: dict[str, float]) -> float:
        return float(features["yin_voiced_fraction"])

    def predicts_vocal(self, features: dict[str, float]) -> bool:
        return self.score(features) >= self.minimum_voiced_fraction

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "yin_coverage_vocal_activity_gate",
            "feature": "yin_voiced_fraction",
            "minimum_voiced_fraction": self.minimum_voiced_fraction,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "VocalActivityGate":
        if payload.get("mode") != "yin_coverage_vocal_activity_gate":
            raise ValueError("unknown vocal activity gate mode")
        if payload.get("feature") != "yin_voiced_fraction":
            raise ValueError("unknown vocal activity gate feature")
        return cls(minimum_voiced_fraction=float(payload["minimum_voiced_fraction"]))


def build_register_example(clip: CorpusClip) -> RegisterExample:
    samples, sample_rate = read_pcm_wav(clip.wav_path)
    label = clip.annotations[0].label if clip.annotations else "unlabeled"
    return RegisterExample(
        clip_id=clip.clip_id,
        speaker_id=clip.clip_id[:-1],
        fieldsite=clip.clip_id[:3],
        label=label,
        features=extract_register_features(samples, sample_rate),
    )


@dataclass(frozen=True)
class RidgeBinaryModel:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    regularization: float

    def score(self, features: dict[str, float]) -> float:
        vector = np.asarray([features[name] for name in self.feature_names])
        normalized = (vector - np.asarray(self.mean)) / np.asarray(self.scale)
        return self.intercept + float(np.dot(normalized, self.coefficients))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["mean"] = list(self.mean)
        payload["scale"] = list(self.scale)
        payload["coefficients"] = list(self.coefficients)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RidgeBinaryModel":
        return cls(
            feature_names=tuple(str(item) for item in payload["feature_names"]),  # type: ignore[arg-type]
            mean=tuple(float(item) for item in payload["mean"]),  # type: ignore[arg-type]
            scale=tuple(float(item) for item in payload["scale"]),  # type: ignore[arg-type]
            intercept=float(payload["intercept"]),
            coefficients=tuple(float(item) for item in payload["coefficients"]),  # type: ignore[arg-type]
            regularization=float(payload["regularization"]),
        )


def _fit_ridge(
    examples: list[dict[str, object]],
    *,
    feature_names: tuple[str, ...],
    target_key: str,
    regularization: float,
) -> RidgeBinaryModel:
    matrix = np.asarray(
        [
            [float(example["features"][name]) for name in feature_names]  # type: ignore[index]
            for example in examples
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [float(example[target_key]) for example in examples], dtype=np.float64
    )
    if not bool(np.any(targets > 0.0)) or not bool(np.any(targets < 0.0)):
        raise ValueError("ridge fitting requires both positive and negative examples")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    positive = targets > 0.0
    weights = np.where(
        positive,
        0.5 / int(np.sum(positive)),
        0.5 / int(np.sum(~positive)),
    ) * len(targets)
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.diag([0.0, *([regularization] * len(feature_names))])
    beta = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty,
        design.T @ (weights * targets),
    )
    return RidgeBinaryModel(
        feature_names=feature_names,
        mean=tuple(float(item) for item in mean),
        scale=tuple(float(item) for item in scale),
        intercept=float(beta[0]),
        coefficients=tuple(float(item) for item in beta[1:]),
        regularization=regularization,
    )


def _speaker_adapt(
    examples: list[RegisterExample],
    *,
    scope: str,
) -> list[dict[str, object]]:
    if scope not in ("speaker_all_modes", "speaker_and_mode"):
        raise ValueError(f"unknown speaker adaptation scope: {scope}")
    by_speaker: dict[tuple[str, str], list[RegisterExample]] = {}
    for example in examples:
        mode = "song" if example.label.endswith("song") else "speech"
        key = (
            example.speaker_id,
            mode if scope == "speaker_and_mode" else "all_modes",
        )
        by_speaker.setdefault(key, []).append(example)
    adapted: list[dict[str, object]] = []
    for speaker_examples in by_speaker.values():
        matrix = np.asarray(
            [
                [example.features[name] for name in REGISTER_FEATURE_NAMES]
                for example in speaker_examples
            ],
            dtype=np.float64,
        )
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (matrix - mean) / scale
        for example, vector in zip(speaker_examples, normalized, strict=True):
            adapted.append(
                {
                    "example": example,
                    "features": dict(
                        zip(REGISTER_FEATURE_NAMES, vector, strict=True)
                    ),
                    "calibration_observation_count": len(speaker_examples),
                }
            )
    return adapted


@dataclass(frozen=True)
class SpeakerAdaptiveRegisterModel:
    speech_gate: RidgeBinaryModel
    ids_gate: RidgeBinaryModel
    minimum_calibration_observations: int = 2
    adaptation_scope: str = "speaker_and_mode"
    vocal_gate: VocalActivityGate | None = field(default_factory=VocalActivityGate)
    speech_score_minimum: float = 0.0
    ids_score_minimum: float = 0.0

    @classmethod
    def fit(cls, examples: list[RegisterExample]) -> "SpeakerAdaptiveRegisterModel":
        adaptation_scope = "speaker_and_mode"
        adapted_by_id = {
            item["example"].clip_id: item  # type: ignore[union-attr]
            for item in _speaker_adapt(examples, scope=adaptation_scope)
        }
        speech_rows = [
            {
                "features": example.features,
                "target": 1.0 if example.label.endswith("speech") else -1.0,
            }
            for example in examples
        ]
        ids_rows = [
            {
                "features": adapted_by_id[example.clip_id]["features"],
                "target": (
                    1.0 if example.label == "infant_directed_speech" else -1.0
                ),
            }
            for example in examples
            if example.label.endswith("speech")
        ]
        return cls(
            speech_gate=_fit_ridge(
                speech_rows,
                feature_names=SPEECH_GATE_FEATURES,
                target_key="target",
                regularization=100.0,
            ),
            ids_gate=_fit_ridge(
                ids_rows,
                feature_names=ADAPTIVE_IDS_FEATURES,
                target_key="target",
                regularization=100.0,
            ),
            adaptation_scope=adaptation_scope,
            vocal_gate=VocalActivityGate(),
        )

    def evaluate_batch(self, examples: list[RegisterExample]) -> dict[str, object]:
        adapted = _speaker_adapt(examples, scope=self.adaptation_scope)
        predictions: list[dict[str, object]] = []
        for item in adapted:
            example = item["example"]
            assert isinstance(example, RegisterExample)
            count = int(item["calibration_observation_count"])
            vocal_score = (
                self.vocal_gate.score(example.features)
                if self.vocal_gate is not None
                else None
            )
            predicted_vocal = (
                self.vocal_gate.predicts_vocal(example.features)
                if self.vocal_gate is not None
                else True
            )
            speech_score = self.speech_gate.score(example.features)
            ids_score = self.ids_gate.score(item["features"])  # type: ignore[arg-type]
            calibrated = count >= self.minimum_calibration_observations
            predictions.append(
                {
                    "clip_id": example.clip_id,
                    "speaker_id": example.speaker_id,
                    "fieldsite": example.fieldsite,
                    "label": example.label,
                    "calibration_observation_count": count,
                    "calibrated": calibrated,
                    "vocal_score": vocal_score,
                    "predicted_vocal": predicted_vocal,
                    "speech_score": speech_score,
                    "ids_score": ids_score,
                    "predicted_ids": calibrated
                    and predicted_vocal
                    and speech_score >= self.speech_score_minimum
                    and ids_score >= self.ids_score_minimum,
                }
            )
        return _register_report(
            predictions,
            adaptation_scope=self.adaptation_scope,
            vocal_gate=self.vocal_gate,
            decision_thresholds={
                "speech_score_minimum": self.speech_score_minimum,
                "ids_score_minimum": self.ids_score_minimum,
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "speaker_adaptive_register_model",
            "feature_schema_version": 4,
            "calibration_boundary": (
                "unlabeled within-speaker and vocal-mode batch normalization; "
                "not cold-start causal"
            ),
            "minimum_calibration_observations": self.minimum_calibration_observations,
            "adaptation_scope": self.adaptation_scope,
            "vocal_gate": (
                self.vocal_gate.to_dict() if self.vocal_gate is not None else None
            ),
            "decision_thresholds": {
                "speech_score_minimum": self.speech_score_minimum,
                "ids_score_minimum": self.ids_score_minimum,
            },
            "speech_gate": self.speech_gate.to_dict(),
            "ids_gate": self.ids_gate.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SpeakerAdaptiveRegisterModel":
        vocal_gate_payload = payload.get("vocal_gate")
        vocal_gate = (
            VocalActivityGate.from_dict(vocal_gate_payload)
            if isinstance(vocal_gate_payload, dict)
            else None
        )
        threshold_payload = payload.get("decision_thresholds")
        thresholds = threshold_payload if isinstance(threshold_payload, dict) else {}
        return cls(
            speech_gate=RidgeBinaryModel.from_dict(payload["speech_gate"]),  # type: ignore[arg-type]
            ids_gate=RidgeBinaryModel.from_dict(payload["ids_gate"]),  # type: ignore[arg-type]
            minimum_calibration_observations=int(
                payload["minimum_calibration_observations"]
            ),
            adaptation_scope=str(
                payload.get("adaptation_scope", "speaker_all_modes")
            ),
            vocal_gate=vocal_gate,
            speech_score_minimum=float(
                thresholds.get("speech_score_minimum", 0.0)
            ),
            ids_score_minimum=float(thresholds.get("ids_score_minimum", 0.0)),
        )


def _binary_metrics(predictions: list[dict[str, object]]) -> dict[str, object]:
    tp = fp = tn = fn = 0
    for prediction in predictions:
        actual = prediction["label"] == "infant_directed_speech"
        predicted = bool(prediction["predicted_ids"])
        if actual and predicted:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def _register_report(
    predictions: list[dict[str, object]],
    *,
    adaptation_scope: str,
    vocal_gate: VocalActivityGate | None,
    decision_thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    calibrated = [item for item in predictions if bool(item["calibrated"])]
    fieldsites = sorted({str(item["fieldsite"]) for item in predictions})
    per_fieldsite = {
        fieldsite: _binary_metrics(
            [item for item in calibrated if item["fieldsite"] == fieldsite]
        )
        for fieldsite in fieldsites
    }
    songs = [item for item in calibrated if str(item["label"]).endswith("song")]
    return {
        "mode": "speaker_adaptive_register_evaluation",
        "calibration_boundary": (
            "unlabeled within-speaker and vocal-mode batch normalization; "
            "not cold-start causal"
            if adaptation_scope == "speaker_and_mode"
            else "unlabeled within-speaker batch normalization across speech and "
            "song; not cold-start causal"
        ),
        "adaptation_scope": adaptation_scope,
        "vocal_gate": vocal_gate.to_dict() if vocal_gate is not None else None,
        "decision_thresholds": decision_thresholds or {
            "speech_score_minimum": 0.0,
            "ids_score_minimum": 0.0,
        },
        "prediction_count": len(predictions),
        "calibrated_prediction_count": len(calibrated),
        "predicted_vocal_count": sum(
            bool(item.get("predicted_vocal", True)) for item in predictions
        ),
        "metrics": _binary_metrics(calibrated),
        "song_false_positive_count": sum(bool(item["predicted_ids"]) for item in songs),
        "song_count": len(songs),
        "per_fieldsite": per_fieldsite,
        "predictions": predictions,
    }


def evaluate_leave_one_fieldsite_out(
    examples: list[RegisterExample],
) -> dict[str, object]:
    """Design-time grouped evaluation; every test fieldsite is excluded from fit."""

    predictions: list[dict[str, object]] = []
    fieldsites = sorted({example.fieldsite for example in examples})
    for fieldsite in fieldsites:
        train = [example for example in examples if example.fieldsite != fieldsite]
        held_out = [example for example in examples if example.fieldsite == fieldsite]
        model = SpeakerAdaptiveRegisterModel.fit(train)
        report = model.evaluate_batch(held_out)
        predictions.extend(report["predictions"])  # type: ignore[arg-type]
    report = _register_report(
        predictions,
        adaptation_scope="speaker_and_mode",
        vocal_gate=VocalActivityGate(),
    )
    report["mode"] = "leave_one_fieldsite_out_register_evaluation"
    report["fieldsite_count"] = len(fieldsites)
    return report


def _thresholded_predictions(
    predictions: list[dict[str, object]],
    *,
    speech_score_minimum: float,
    ids_score_minimum: float,
) -> list[dict[str, object]]:
    thresholded = []
    for prediction in predictions:
        item = dict(prediction)
        item["predicted_ids"] = (
            bool(item["calibrated"])
            and bool(item.get("predicted_vocal", True))
            and float(item["speech_score"]) >= speech_score_minimum
            and float(item["ids_score"]) >= ids_score_minimum
        )
        thresholded.append(item)
    return thresholded


def select_register_thresholds(
    examples: list[RegisterExample],
    *,
    maximum_false_positive_rate: float = 0.15,
    maximum_song_false_positive_rate: float = 0.15,
) -> dict[str, object]:
    """Select score cutoffs from out-of-field predictions only."""

    predictions: list[dict[str, object]] = []
    fieldsites = sorted({example.fieldsite for example in examples})
    if len(fieldsites) < 2:
        raise ValueError("threshold selection requires at least two fieldsites")
    for fieldsite in fieldsites:
        model = SpeakerAdaptiveRegisterModel.fit(
            [example for example in examples if example.fieldsite != fieldsite]
        )
        report = model.evaluate_batch(
            [example for example in examples if example.fieldsite == fieldsite]
        )
        predictions.extend(report["predictions"])  # type: ignore[arg-type]

    def candidates(key: str) -> list[float]:
        values = sorted({float(item[key]) for item in predictions})
        return [
            values[0] - 1e-6,
            *((left + right) / 2.0 for left, right in zip(values, values[1:])),
            values[-1] + 1e-6,
        ]

    best: tuple[tuple[float, float, float, float], float, float] | None = None
    for speech_minimum in candidates("speech_score"):
        for ids_minimum in candidates("ids_score"):
            thresholded = _thresholded_predictions(
                predictions,
                speech_score_minimum=speech_minimum,
                ids_score_minimum=ids_minimum,
            )
            report = _register_report(
                thresholded,
                adaptation_scope="speaker_and_mode",
                vocal_gate=VocalActivityGate(),
                decision_thresholds={
                    "speech_score_minimum": speech_minimum,
                    "ids_score_minimum": ids_minimum,
                },
            )
            metrics = report["metrics"]
            assert isinstance(metrics, dict)
            false_positive_rate = float(metrics["false_positive_rate"])
            song_count = int(report["song_count"])
            song_rate = (
                int(report["song_false_positive_count"]) / song_count
                if song_count
                else 0.0
            )
            if false_positive_rate > maximum_false_positive_rate:
                continue
            if song_rate > maximum_song_false_positive_rate:
                continue
            per_fieldsite = report["per_fieldsite"]
            assert isinstance(per_fieldsite, dict)
            minimum_fieldsite_f1 = min(
                float(value["f1"]) for value in per_fieldsite.values()
            )
            key = (
                float(metrics["f1"]),
                minimum_fieldsite_f1,
                -false_positive_rate,
                -song_rate,
            )
            if best is None or key > best[0]:
                best = (key, speech_minimum, ids_minimum)
    if best is None:
        raise ValueError("no register thresholds satisfy false-positive limits")
    _, speech_minimum, ids_minimum = best
    selected = _thresholded_predictions(
        predictions,
        speech_score_minimum=speech_minimum,
        ids_score_minimum=ids_minimum,
    )
    report = _register_report(
        selected,
        adaptation_scope="speaker_and_mode",
        vocal_gate=VocalActivityGate(),
        decision_thresholds={
            "speech_score_minimum": speech_minimum,
            "ids_score_minimum": ids_minimum,
        },
    )
    return {
        "protocol": "leave_one_fieldsite_out_threshold_selection",
        "fieldsite_count": len(fieldsites),
        "prediction_count": len(predictions),
        "maximum_false_positive_rate": maximum_false_positive_rate,
        "maximum_song_false_positive_rate": maximum_song_false_positive_rate,
        "speech_score_minimum": speech_minimum,
        "ids_score_minimum": ids_minimum,
        "metrics": report["metrics"],
        "song_false_positive_count": report["song_false_positive_count"],
        "song_count": report["song_count"],
        "per_fieldsite": report["per_fieldsite"],
    }


def evaluate_register_pipeline(
    model: SpeakerAdaptiveRegisterModel,
    vocal_examples: list[RegisterExample],
    environmental_examples: list[RegisterExample],
) -> dict[str, object]:
    """Evaluate the composed surface without inventing noise calibration.

    Vocal examples proceed through all three gates and retain the model's
    speaker-and-mode calibration boundary. Environmental examples are scored
    only by the clip-local vocal gate. A passed environmental clip is reported
    as an abstention requiring another policy, never converted into a negative
    merely because it lacks speaker calibration.
    """

    if model.vocal_gate is None:
        raise ValueError("register pipeline evaluation requires a vocal gate")
    if not vocal_examples:
        raise ValueError("register pipeline evaluation requires vocal examples")
    if not environmental_examples:
        raise ValueError(
            "register pipeline evaluation requires environmental examples"
        )
    vocal_report = model.evaluate_batch(vocal_examples)
    environmental_predictions = []
    for example in environmental_examples:
        score = model.vocal_gate.score(example.features)
        environmental_predictions.append(
            {
                "clip_id": example.clip_id,
                "label": example.label,
                "vocal_score": score,
                "passed_vocal_gate": model.vocal_gate.predicts_vocal(
                    example.features
                ),
            }
        )
    vocal_predictions = vocal_report["predictions"]
    assert isinstance(vocal_predictions, list)
    vocal_retained = sum(
        bool(item["predicted_vocal"]) for item in vocal_predictions
    )
    environmental_passed = sum(
        bool(item["passed_vocal_gate"]) for item in environmental_predictions
    )
    environmental_labels = sorted(
        {str(item["label"]) for item in environmental_predictions}
    )
    per_environmental_label = {
        label: {
            "count": len(items),
            "pass_through_count": sum(
                bool(item["passed_vocal_gate"]) for item in items
            ),
            "pass_through_rate": (
                sum(bool(item["passed_vocal_gate"]) for item in items) / len(items)
            ),
        }
        for label in environmental_labels
        if (
            items := [
                item
                for item in environmental_predictions
                if item["label"] == label
            ]
        )
    }
    return {
        "mode": "joint_register_pipeline_evaluation",
        "claim_boundary": (
            "environmental clips passing the vocal gate are abstentions; they are "
            "not scored by a speaker-calibrated IDS gate"
        ),
        "vocal_gate": model.vocal_gate.to_dict(),
        "vocal_count": len(vocal_predictions),
        "vocal_retained_count": vocal_retained,
        "vocal_recall": vocal_retained / len(vocal_predictions),
        "environmental_count": len(environmental_predictions),
        "environmental_rejected_count": (
            len(environmental_predictions) - environmental_passed
        ),
        "environmental_pass_through_count": environmental_passed,
        "environmental_pass_through_rate": (
            environmental_passed / len(environmental_predictions)
        ),
        "per_environmental_label": per_environmental_label,
        "register_report": vocal_report,
        "environmental_predictions": environmental_predictions,
    }


def load_register_examples(root: str | Path, clips: list[CorpusClip]) -> list[RegisterExample]:
    """Build examples while retaining a root argument in provenance-facing calls."""

    _ = Path(root)
    return [build_register_example(clip) for clip in clips]
