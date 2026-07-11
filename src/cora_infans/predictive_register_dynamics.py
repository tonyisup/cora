"""Transparent class-conditional predictive dynamics for vocal register.

The observed prosodic vector is the state. Four label-specific linear-Gaussian
models predict each next state from the previous state, and held-out sequence
likelihood supplies the speech/song and IDS/ADS scores. This remains a bounded,
batch-calibrated research model rather than a cold-start live detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .io.offline_eval import CorpusClip
from .io.wav_frames import build_pcm_observations, read_pcm_wav
from .register_saliency import (
    VocalActivityGate,
    _register_report,
    _thresholded_predictions,
    _yin_pitch_track,
)


STATE_FEATURE_NAMES = (
    "pitch_log2",
    "pitch_confidence",
    "intensity_db",
    "spectral_centroid_log",
    "spectral_flux_log",
)
CONDITION_LABELS = (
    "infant_directed_song",
    "infant_directed_speech",
    "adult_directed_song",
    "adult_directed_speech",
)
RIDGE_REGULARIZATION = 1.0
RESIDUAL_VARIANCE_FLOOR = 0.05
MINIMUM_SEQUENCE_FRAMES = 6


def _fill_pitch(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return np.zeros_like(values)
    indices = np.arange(len(values))
    return np.interp(indices, indices[finite], values[finite])


def extract_prosodic_state_sequence(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, float]:
    """Extract an ordered PCM-only state sequence and the frozen vocal score."""

    pcm = np.asarray(samples, dtype=np.float32)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.ndim != 2 or not len(pcm):
        raise ValueError("predictive register requires non-empty PCM")
    frames = build_pcm_observations(
        pcm,
        sample_rate=sample_rate,
        frame_ms=50,
        source="predictive_register_state_sequence",
    )
    if len(frames) < MINIMUM_SEQUENCE_FRAMES:
        raise ValueError("predictive register sequence is too short")
    pitch = _fill_pitch(
        np.asarray(
            [
                math.log2(float((frame.features or {}).get("pitch_hz")))
                if (frame.features or {}).get("pitch_hz")
                else np.nan
                for frame in frames
            ],
            dtype=np.float64,
        )
    )
    confidence = np.asarray(
        [float((frame.features or {}).get("pitch_confidence") or 0.0) for frame in frames],
        dtype=np.float64,
    )
    intensity = np.asarray(
        [
            20.0
            * math.log10(
                max(float((frame.features or {}).get("peak_rms") or 0.0), 1e-8)
            )
            for frame in frames
        ],
        dtype=np.float64,
    )
    centroid = np.asarray(
        [
            math.log(
                max(
                    float(
                        (frame.features or {}).get("spectral_centroid_hz") or 0.0
                    ),
                    1.0,
                )
            )
            for frame in frames
        ],
        dtype=np.float64,
    )
    flux = np.asarray(
        [
            math.log1p(
                max(float((frame.features or {}).get("spectral_flux") or 0.0), 0.0)
                * 1e4
            )
            for frame in frames
        ],
        dtype=np.float64,
    )
    states = np.column_stack((pitch, confidence, intensity, centroid, flux))
    if not bool(np.all(np.isfinite(states))):
        raise ValueError("predictive register produced non-finite state")
    mono = np.mean(pcm, axis=1).astype(np.float64)
    yin_pitch, _yin_confidence, analyzed = _yin_pitch_track(mono, sample_rate)
    vocal_score = len(yin_pitch) / analyzed if analyzed else 0.0
    return states, float(vocal_score)


@dataclass(frozen=True)
class ProsodicSequenceExample:
    clip_id: str
    speaker_id: str
    fieldsite: str
    label: str
    states: np.ndarray
    vocal_score: float


def build_prosodic_sequence_example(clip: CorpusClip) -> ProsodicSequenceExample:
    samples, sample_rate = read_pcm_wav(clip.wav_path)
    label = clip.annotations[0].label if clip.annotations else "unlabeled"
    states, vocal_score = extract_prosodic_state_sequence(samples, sample_rate)
    return ProsodicSequenceExample(
        clip_id=clip.clip_id,
        speaker_id=clip.clip_id[:-1],
        fieldsite=clip.clip_id[:3],
        label=label,
        states=states,
        vocal_score=vocal_score,
    )


@dataclass(frozen=True)
class AdaptedSequence:
    example: ProsodicSequenceExample
    states: np.ndarray
    calibration_observation_count: int


def adapt_sequences(
    examples: list[ProsodicSequenceExample],
) -> list[AdaptedSequence]:
    grouped: dict[tuple[str, str], list[ProsodicSequenceExample]] = {}
    for example in examples:
        mode = "song" if example.label.endswith("song") else "speech"
        grouped.setdefault((example.speaker_id, mode), []).append(example)
    adapted = []
    for group in grouped.values():
        combined = np.concatenate([example.states for example in group], axis=0)
        mean = np.mean(combined, axis=0)
        scale = np.std(combined, axis=0)
        scale[scale < 1e-8] = 1.0
        for example in group:
            adapted.append(
                AdaptedSequence(
                    example=example,
                    states=(example.states - mean) / scale,
                    calibration_observation_count=len(group),
                )
            )
    return adapted


@dataclass(frozen=True)
class LinearGaussianDynamics:
    intercept: tuple[float, ...]
    transition: tuple[tuple[float, ...], ...]
    residual_variance: tuple[float, ...]
    ridge_regularization: float = RIDGE_REGULARIZATION
    residual_variance_floor: float = RESIDUAL_VARIANCE_FLOOR

    @classmethod
    def fit(cls, sequences: list[np.ndarray]) -> "LinearGaussianDynamics":
        if not sequences:
            raise ValueError("predictive dynamics requires training sequences")
        previous = np.concatenate([sequence[:-1] for sequence in sequences], axis=0)
        current = np.concatenate([sequence[1:] for sequence in sequences], axis=0)
        design = np.column_stack((np.ones(len(previous)), previous))
        penalty = np.diag([0.0, *([RIDGE_REGULARIZATION] * previous.shape[1])])
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ current,
        )
        prediction = design @ coefficients
        variance = np.maximum(
            np.mean((current - prediction) ** 2, axis=0),
            RESIDUAL_VARIANCE_FLOOR,
        )
        return cls(
            intercept=tuple(float(value) for value in coefficients[0]),
            transition=tuple(
                tuple(float(value) for value in row) for row in coefficients[1:]
            ),
            residual_variance=tuple(float(value) for value in variance),
        )

    def mean_log_likelihood(self, sequence: np.ndarray) -> float:
        if len(sequence) < MINIMUM_SEQUENCE_FRAMES:
            raise ValueError("predictive dynamics sequence is too short")
        intercept = np.asarray(self.intercept)
        transition = np.asarray(self.transition)
        variance = np.asarray(self.residual_variance)
        prediction = intercept + sequence[:-1] @ transition
        residual = sequence[1:] - prediction
        log_likelihood = -0.5 * (
            np.log(2.0 * np.pi * variance) + residual * residual / variance
        )
        return float(np.mean(np.sum(log_likelihood, axis=1)))

    def to_dict(self) -> dict[str, object]:
        return {
            "intercept": list(self.intercept),
            "transition": [list(row) for row in self.transition],
            "residual_variance": list(self.residual_variance),
            "ridge_regularization": self.ridge_regularization,
            "residual_variance_floor": self.residual_variance_floor,
        }


@dataclass(frozen=True)
class PredictiveRegisterDynamicsModel:
    condition_dynamics: dict[str, LinearGaussianDynamics]
    vocal_gate: VocalActivityGate = VocalActivityGate()

    @classmethod
    def fit(
        cls, examples: list[ProsodicSequenceExample]
    ) -> "PredictiveRegisterDynamicsModel":
        adapted = adapt_sequences(examples)
        condition_dynamics = {}
        for label in CONDITION_LABELS:
            sequences = [
                item.states
                for item in adapted
                if item.example.label == label
                and item.calibration_observation_count >= 2
            ]
            condition_dynamics[label] = LinearGaussianDynamics.fit(sequences)
        return cls(condition_dynamics=condition_dynamics)

    def evaluate_batch(
        self, examples: list[ProsodicSequenceExample]
    ) -> dict[str, object]:
        predictions = []
        for item in adapt_sequences(examples):
            example = item.example
            likelihoods = {
                label: model.mean_log_likelihood(item.states)
                for label, model in self.condition_dynamics.items()
            }
            speech_score = max(
                likelihoods["infant_directed_speech"],
                likelihoods["adult_directed_speech"],
            ) - max(
                likelihoods["infant_directed_song"],
                likelihoods["adult_directed_song"],
            )
            ids_score = (
                likelihoods["infant_directed_speech"]
                - likelihoods["adult_directed_speech"]
            )
            calibrated = item.calibration_observation_count >= 2
            predicted_vocal = self.vocal_gate.predicts_vocal(
                {"yin_voiced_fraction": example.vocal_score}
            )
            predictions.append(
                {
                    "clip_id": example.clip_id,
                    "speaker_id": example.speaker_id,
                    "fieldsite": example.fieldsite,
                    "label": example.label,
                    "sequence_frame_count": len(example.states),
                    "calibration_observation_count": item.calibration_observation_count,
                    "calibrated": calibrated,
                    "vocal_score": example.vocal_score,
                    "predicted_vocal": predicted_vocal,
                    "condition_mean_log_likelihood": likelihoods,
                    "speech_score": speech_score,
                    "ids_score": ids_score,
                    "predicted_ids": calibrated
                    and predicted_vocal
                    and speech_score >= 0.0
                    and ids_score >= 0.0,
                }
            )
        return _register_report(
            predictions,
            adaptation_scope="speaker_and_mode",
            vocal_gate=self.vocal_gate,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "predictive_register_dynamics_model",
            "state_feature_names": list(STATE_FEATURE_NAMES),
            "frame_ms": 50,
            "calibration_boundary": "unlabeled within-speaker and vocal-mode batch normalization; not cold-start causal",
            "condition_dynamics": {
                label: model.to_dict()
                for label, model in self.condition_dynamics.items()
            },
            "vocal_gate": self.vocal_gate.to_dict(),
        }


def predictive_dynamics_leave_one_fieldsite_out(
    examples: list[ProsodicSequenceExample],
) -> dict[str, object]:
    fieldsites = sorted({example.fieldsite for example in examples})
    predictions = []
    for fieldsite in fieldsites:
        training = [example for example in examples if example.fieldsite != fieldsite]
        held_out = [example for example in examples if example.fieldsite == fieldsite]
        report = PredictiveRegisterDynamicsModel.fit(training).evaluate_batch(held_out)
        predictions.extend(report["predictions"])
    report = _register_report(
        predictions,
        adaptation_scope="speaker_and_mode",
        vocal_gate=VocalActivityGate(),
    )
    report["mode"] = "predictive_register_dynamics_leave_one_fieldsite_out"
    report["fieldsite_count"] = len(fieldsites)
    report["confirmation_audio_or_metadata_accessed"] = False
    report["promotion_allowed"] = False
    return report


def select_predictive_dynamics_thresholds(
    predictions: list[dict[str, object]],
    *,
    maximum_false_positive_rate: float = 0.15,
    maximum_song_false_positive_rate: float = 0.15,
) -> dict[str, object]:
    if not predictions:
        raise ValueError("predictive threshold selection requires predictions")

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
            selected = _thresholded_predictions(
                predictions,
                speech_score_minimum=speech_minimum,
                ids_score_minimum=ids_minimum,
            )
            report = _register_report(
                selected,
                adaptation_scope="speaker_and_mode",
                vocal_gate=VocalActivityGate(),
            )
            metrics = report["metrics"]
            fpr = float(metrics["false_positive_rate"])
            song_count = int(report["song_count"])
            song_fpr = (
                int(report["song_false_positive_count"]) / song_count
                if song_count
                else 0.0
            )
            if fpr > maximum_false_positive_rate or song_fpr > maximum_song_false_positive_rate:
                continue
            minimum_fieldsite_f1 = min(
                float(value["f1"]) for value in report["per_fieldsite"].values()
            )
            key = (float(metrics["f1"]), minimum_fieldsite_f1, -fpr, -song_fpr)
            if best is None or key > best[0]:
                best = (key, speech_minimum, ids_minimum)
    if best is None:
        raise ValueError("no predictive thresholds satisfy false-positive limits")
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
        "protocol": "predictive_dynamics_out_of_field_threshold_selection",
        "speech_score_minimum": speech_minimum,
        "ids_score_minimum": ids_minimum,
        "metrics": report["metrics"],
        "song_false_positive_count": report["song_false_positive_count"],
        "song_count": report["song_count"],
        "per_fieldsite": report["per_fieldsite"],
    }
