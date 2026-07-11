"""Order-preserving temporal representation for developmental register.

Unlike the whole-clip summary family, this candidate retains six successive
prosodic phases. It remains a transparent, batch-calibrated research model: it
is not cold-start, causal, or a universal parentese detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .io.offline_eval import CorpusClip
from .io.wav_frames import build_pcm_observations, read_pcm_wav
from .register_saliency import (
    RegisterExample,
    RidgeBinaryModel,
    VocalActivityGate,
    _binary_metrics,
    _fit_ridge,
    _register_report,
    _thresholded_predictions,
    _yin_pitch_track,
)


TEMPORAL_PHASE_COUNT = 6
TEMPORAL_BASE_NAMES = (
    "pitch_log2",
    "pitch_confidence",
    "intensity_db",
    "spectral_centroid_log",
    "spectral_flux_log",
)
TEMPORAL_FEATURE_NAMES = tuple(
    f"{name}_phase_{phase}"
    for name in TEMPORAL_BASE_NAMES
    for phase in range(TEMPORAL_PHASE_COUNT)
) + tuple(
    f"{name}_{summary}"
    for name in TEMPORAL_BASE_NAMES
    for summary in ("path_length", "reversal_fraction")
)


def _fill_nonfinite(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return np.zeros_like(values)
    indices = np.arange(len(values))
    return np.interp(indices, indices[finite], values[finite])


def _temporal_shape(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = _fill_nonfinite(np.asarray(values, dtype=np.float64))
    edges = np.linspace(0, len(values), TEMPORAL_PHASE_COUNT + 1, dtype=int)
    phases = np.asarray(
        [
            float(np.median(values[edges[index] : edges[index + 1]]))
            if edges[index + 1] > edges[index]
            else float(values[min(edges[index], len(values) - 1)])
            for index in range(TEMPORAL_PHASE_COUNT)
        ],
        dtype=np.float64,
    )
    scale = float(np.std(phases))
    normalized = (phases - float(np.mean(phases))) / max(scale, 1e-8)
    differences = np.diff(normalized)
    path_length = float(np.mean(np.abs(differences))) if len(differences) else 0.0
    signs = np.sign(differences)
    nonzero = signs[signs != 0.0]
    reversals = (
        float(np.mean(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0.0
    )
    return normalized, path_length, reversals


def extract_temporal_register_features(
    samples: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    """Extract order-preserving six-phase prosodic trajectories."""

    pcm = np.asarray(samples, dtype=np.float32)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.ndim != 2 or not len(pcm):
        raise ValueError("temporal register features require non-empty PCM")
    frames = build_pcm_observations(
        pcm,
        sample_rate=sample_rate,
        frame_ms=50,
        source="temporal_register_window",
    )
    if len(frames) < TEMPORAL_PHASE_COUNT:
        raise ValueError("temporal register clip is too short for six phases")

    pitch = np.asarray(
        [
            math.log2(float((frame.features or {}).get("pitch_hz")))
            if (frame.features or {}).get("pitch_hz")
            else np.nan
            for frame in frames
        ]
    )
    confidence = np.asarray(
        [float((frame.features or {}).get("pitch_confidence") or 0.0) for frame in frames]
    )
    intensity = np.asarray(
        [
            20.0 * math.log10(max(float((frame.features or {}).get("peak_rms") or 0.0), 1e-8))
            for frame in frames
        ]
    )
    centroid = np.asarray(
        [
            math.log(max(float((frame.features or {}).get("spectral_centroid_hz") or 0.0), 1.0))
            for frame in frames
        ]
    )
    flux = np.asarray(
        [
            math.log1p(max(float((frame.features or {}).get("spectral_flux") or 0.0), 0.0) * 1e4)
            for frame in frames
        ]
    )
    sequences = dict(
        zip(
            TEMPORAL_BASE_NAMES,
            (pitch, confidence, intensity, centroid, flux),
            strict=True,
        )
    )
    features: dict[str, float] = {}
    for name, values in sequences.items():
        phases, path_length, reversals = _temporal_shape(values)
        for phase, value in enumerate(phases):
            features[f"{name}_phase_{phase}"] = float(value)
        features[f"{name}_path_length"] = path_length
        features[f"{name}_reversal_fraction"] = reversals

    mono = np.mean(pcm, axis=1).astype(np.float64)
    yin_pitch, _confidence, analyzed_count = _yin_pitch_track(mono, sample_rate)
    features["yin_voiced_fraction"] = (
        len(yin_pitch) / analyzed_count if analyzed_count else 0.0
    )
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("temporal register extraction produced non-finite features")
    return features


def build_temporal_register_example(clip: CorpusClip) -> RegisterExample:
    samples, sample_rate = read_pcm_wav(clip.wav_path)
    label = clip.annotations[0].label if clip.annotations else "unlabeled"
    return RegisterExample(
        clip_id=clip.clip_id,
        speaker_id=clip.clip_id[:-1],
        fieldsite=clip.clip_id[:3],
        label=label,
        features=extract_temporal_register_features(samples, sample_rate),
    )


def _adapt_temporal(
    examples: list[RegisterExample],
) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[RegisterExample]] = {}
    for example in examples:
        mode = "song" if example.label.endswith("song") else "speech"
        grouped.setdefault((example.speaker_id, mode), []).append(example)
    adapted: dict[str, dict[str, float]] = {}
    for group in grouped.values():
        matrix = np.asarray(
            [
                [example.features[name] for name in TEMPORAL_FEATURE_NAMES]
                for example in group
            ]
        )
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-8] = 1.0
        for example, row in zip(group, (matrix - mean) / scale, strict=True):
            adapted[example.clip_id] = dict(
                zip(TEMPORAL_FEATURE_NAMES, row, strict=True)
            )
    return adapted


@dataclass(frozen=True)
class TemporalRegisterModel:
    speech_gate: RidgeBinaryModel
    ids_gate: RidgeBinaryModel
    vocal_gate: VocalActivityGate = VocalActivityGate()

    @classmethod
    def fit(cls, examples: list[RegisterExample]) -> "TemporalRegisterModel":
        adapted = _adapt_temporal(examples)
        speech_rows = [
            {
                "features": example.features,
                "target": 1.0 if example.label.endswith("speech") else -1.0,
            }
            for example in examples
        ]
        ids_rows = [
            {
                "features": adapted[example.clip_id],
                "target": 1.0 if example.label == "infant_directed_speech" else -1.0,
            }
            for example in examples
            if example.label.endswith("speech")
        ]
        return cls(
            speech_gate=_fit_ridge(
                speech_rows,
                feature_names=TEMPORAL_FEATURE_NAMES,
                target_key="target",
                regularization=100.0,
            ),
            ids_gate=_fit_ridge(
                ids_rows,
                feature_names=TEMPORAL_FEATURE_NAMES,
                target_key="target",
                regularization=100.0,
            ),
        )

    def evaluate_batch(self, examples: list[RegisterExample]) -> dict[str, object]:
        adapted = _adapt_temporal(examples)
        predictions = []
        for example in examples:
            vocal_score = float(example.features["yin_voiced_fraction"])
            predicted_vocal = self.vocal_gate.predicts_vocal(example.features)
            speech_score = self.speech_gate.score(example.features)
            ids_score = self.ids_gate.score(adapted[example.clip_id])
            calibrated = sum(
                candidate.speaker_id == example.speaker_id
                and candidate.label.endswith("song") == example.label.endswith("song")
                for candidate in examples
            ) >= 2
            predictions.append(
                {
                    "clip_id": example.clip_id,
                    "speaker_id": example.speaker_id,
                    "fieldsite": example.fieldsite,
                    "label": example.label,
                    "calibrated": calibrated,
                    "vocal_score": vocal_score,
                    "predicted_vocal": predicted_vocal,
                    "speech_score": speech_score,
                    "ids_score": ids_score,
                    "predicted_ids": calibrated
                    and predicted_vocal
                    and speech_score >= 0.0
                    and ids_score >= 0.0,
                }
            )
        calibrated_predictions = [row for row in predictions if row["calibrated"]]
        return {
            "mode": "temporal_register_evaluation",
            "representation": "six_phase_order_preserving_prosodic_trajectory",
            "calibration_boundary": "unlabeled within-speaker and vocal-mode batch normalization; not cold-start causal",
            "metrics": _binary_metrics(calibrated_predictions),
            "predictions": predictions,
        }


def evaluate_temporal_leave_one_fieldsite_out(
    examples: list[RegisterExample],
) -> dict[str, object]:
    fieldsites = sorted({example.fieldsite for example in examples})
    predictions = []
    for fieldsite in fieldsites:
        training = [example for example in examples if example.fieldsite != fieldsite]
        held_out = [example for example in examples if example.fieldsite == fieldsite]
        report = TemporalRegisterModel.fit(training).evaluate_batch(held_out)
        predictions.extend(report["predictions"])
    metrics = _binary_metrics(predictions)
    songs = [row for row in predictions if str(row["label"]).endswith("song")]
    vocal_retained = [row for row in predictions if float(row["vocal_score"]) >= 0.2]
    per_fieldsite = {
        fieldsite: _binary_metrics(
            [row for row in predictions if row["fieldsite"] == fieldsite]
        )
        for fieldsite in fieldsites
    }
    return {
        "mode": "temporal_register_leave_one_fieldsite_out",
        "representation": "six_phase_order_preserving_prosodic_trajectory",
        "fieldsite_count": len(fieldsites),
        "prediction_count": len(predictions),
        "metrics": metrics,
        "song_false_positive_rate": (
            sum(bool(row["predicted_ids"]) for row in songs) / len(songs)
            if songs
            else 0.0
        ),
        "vocal_recall": len(vocal_retained) / len(predictions) if predictions else 0.0,
        "minimum_fieldsite_f1": min(
            float(report["f1"]) for report in per_fieldsite.values()
        ),
        "per_fieldsite": per_fieldsite,
        "predictions": predictions,
        "confirmation_fieldsites_accessed": False,
        "promotion_allowed": False,
    }


def select_temporal_register_thresholds(
    predictions: list[dict[str, object]],
    *,
    maximum_false_positive_rate: float = 0.15,
    maximum_song_false_positive_rate: float = 0.15,
) -> dict[str, object]:
    """Select both cutoffs from already out-of-field temporal predictions."""

    if not predictions:
        raise ValueError("temporal threshold selection requires predictions")

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
            )
            metrics = report["metrics"]
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
            minimum_fieldsite_f1 = min(
                float(value["f1"])
                for value in report["per_fieldsite"].values()
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
        raise ValueError("no temporal thresholds satisfy false-positive limits")
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
        "protocol": "out_of_field_temporal_threshold_selection",
        "speech_score_minimum": speech_minimum,
        "ids_score_minimum": ids_minimum,
        "metrics": report["metrics"],
        "song_false_positive_count": report["song_false_positive_count"],
        "song_count": report["song_count"],
        "per_fieldsite": report["per_fieldsite"],
    }
