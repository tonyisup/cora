"""Label-free cohort alignment for predictive vocal-register dynamics.

This module implements the frozen development protocol: a held-out fieldsite's
mode-specific transition covariance is aligned to training-cohort geometry,
then scored by training-only class dynamics. It is transductive batch
adaptation, not cold-start or causal inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .predictive_register_dynamics import (
    CONDITION_LABELS,
    AdaptedSequence,
    PredictiveRegisterDynamicsModel,
    ProsodicSequenceExample,
    adapt_sequences,
    select_predictive_dynamics_thresholds,
)
from .register_saliency import VocalActivityGate, _register_report


COVARIANCE_SHRINKAGE = 0.1
EIGENVALUE_FLOOR = 1e-4
MINIMUM_TARGET_SPEAKERS = 3
MINIMUM_CALIBRATION_OBSERVATIONS = 2


def vocal_mode(label: str) -> str:
    return "song" if label.endswith("song") else "speech"


def transition_rows(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("cohort alignment requires at least one transition")
    return np.column_stack((states[:-1], states[1:]))


def _regularized_covariance(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or len(rows) < 2:
        raise ValueError("cohort covariance requires at least two rows")
    covariance = np.atleast_2d(np.cov(rows, rowvar=False, ddof=1))
    diagonal = np.diag(np.diag(covariance))
    shrunk = (
        (1.0 - COVARIANCE_SHRINKAGE) * covariance
        + COVARIANCE_SHRINKAGE * diagonal
    )
    values, vectors = np.linalg.eigh(shrunk)
    values = np.maximum(values, EIGENVALUE_FLOOR)
    return (vectors * values) @ vectors.T


def _symmetric_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, EIGENVALUE_FLOOR)
    return (vectors * np.power(values, exponent)) @ vectors.T


@dataclass(frozen=True)
class CoralAlignment:
    target_mean: tuple[float, ...]
    source_mean: tuple[float, ...]
    transform: tuple[tuple[float, ...], ...]
    source_transition_count: int
    target_transition_count: int

    @classmethod
    def fit(cls, target_rows: np.ndarray, source_rows: np.ndarray) -> "CoralAlignment":
        target_rows = np.asarray(target_rows, dtype=np.float64)
        source_rows = np.asarray(source_rows, dtype=np.float64)
        if target_rows.ndim != 2 or source_rows.ndim != 2:
            raise ValueError("CORAL inputs must be matrices")
        if target_rows.shape[1] != source_rows.shape[1]:
            raise ValueError("CORAL source and target feature counts must match")
        target_covariance = _regularized_covariance(target_rows)
        source_covariance = _regularized_covariance(source_rows)
        transform = _symmetric_power(target_covariance, -0.5) @ _symmetric_power(
            source_covariance, 0.5
        )
        return cls(
            target_mean=tuple(float(value) for value in np.mean(target_rows, axis=0)),
            source_mean=tuple(float(value) for value in np.mean(source_rows, axis=0)),
            transform=tuple(tuple(float(value) for value in row) for row in transform),
            source_transition_count=len(source_rows),
            target_transition_count=len(target_rows),
        )

    def apply(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float64)
        return (
            (rows - np.asarray(self.target_mean)) @ np.asarray(self.transform)
            + np.asarray(self.source_mean)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_mean": list(self.target_mean),
            "source_mean": list(self.source_mean),
            "transform": [list(row) for row in self.transform],
            "source_transition_count": self.source_transition_count,
            "target_transition_count": self.target_transition_count,
        }


def coral_target_to_source(target_rows: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    """Fit and apply the frozen target-to-source CORAL transform."""

    return CoralAlignment.fit(target_rows, source_rows).apply(target_rows)


def _eligible_adapted(examples: list[ProsodicSequenceExample]) -> list[AdaptedSequence]:
    return [
        item
        for item in adapt_sequences(examples)
        if item.calibration_observation_count >= MINIMUM_CALIBRATION_OBSERVATIONS
    ]


def evaluate_cohort_adaptive_fold(
    training: list[ProsodicSequenceExample],
    target: list[ProsodicSequenceExample],
) -> dict[str, object]:
    """Fit training dynamics and align one unlabeled target fieldsite batch."""

    target_fieldsites = {example.fieldsite for example in target}
    if len(target_fieldsites) != 1:
        raise ValueError("cohort fold requires exactly one target fieldsite")
    target_speakers = {example.speaker_id for example in target}
    if len(target_speakers) < MINIMUM_TARGET_SPEAKERS:
        raise ValueError("cohort fold requires at least three target speakers")

    model = PredictiveRegisterDynamicsModel.fit(training)
    source_adapted = _eligible_adapted(training)
    target_adapted = _eligible_adapted(target)
    predictions: list[dict[str, object]] = []
    alignments: dict[str, object] = {}
    for mode in ("speech", "song"):
        source_items = [
            item for item in source_adapted if vocal_mode(item.example.label) == mode
        ]
        target_items = [
            item for item in target_adapted if vocal_mode(item.example.label) == mode
        ]
        if not target_items:
            continue
        source_rows = np.concatenate(
            [transition_rows(item.states) for item in source_items], axis=0
        )
        target_rows = np.concatenate(
            [transition_rows(item.states) for item in target_items], axis=0
        )
        alignment = CoralAlignment.fit(target_rows, source_rows)
        alignments[mode] = alignment.to_dict()
        for item in target_items:
            example = item.example
            aligned = alignment.apply(transition_rows(item.states))
            feature_count = item.states.shape[1]
            previous, current = aligned[:, :feature_count], aligned[:, feature_count:]
            likelihoods = {
                label: dynamics.mean_log_likelihood_pairs(previous, current)
                for label, dynamics in model.condition_dynamics.items()
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
            predicted_vocal = model.vocal_gate.predicts_vocal(
                {"yin_voiced_fraction": example.vocal_score}
            )
            predictions.append(
                {
                    "clip_id": example.clip_id,
                    "speaker_id": example.speaker_id,
                    "fieldsite": example.fieldsite,
                    "label": example.label,
                    "vocal_mode": mode,
                    "sequence_frame_count": len(example.states),
                    "transition_count": len(aligned),
                    "calibration_observation_count": item.calibration_observation_count,
                    "calibrated": True,
                    "vocal_score": example.vocal_score,
                    "predicted_vocal": predicted_vocal,
                    "condition_mean_log_likelihood": likelihoods,
                    "speech_score": speech_score,
                    "ids_score": ids_score,
                    "predicted_ids": predicted_vocal
                    and speech_score >= 0.0
                    and ids_score >= 0.0,
                }
            )

    report = _register_report(
        predictions,
        adaptation_scope="speaker_and_mode",
        vocal_gate=VocalActivityGate(),
    )
    report.update(
        {
            "mode": "cohort_adaptive_predictive_register_fold",
            "target_fieldsite": next(iter(target_fieldsites)),
            "target_speaker_count": len(target_speakers),
            "cohort_alignment": alignments,
            "alignment_uses_target_ids_ads_labels": False,
            "known_vocal_mode_grouping": True,
        }
    )
    return report


def cohort_adaptive_leave_one_fieldsite_out(
    examples: list[ProsodicSequenceExample],
) -> dict[str, object]:
    fieldsites = sorted({example.fieldsite for example in examples})
    eligible: list[str] = []
    ineligible: dict[str, str] = {}
    predictions: list[dict[str, object]] = []
    fold_alignments: dict[str, object] = {}
    for fieldsite in fieldsites:
        held_out = [example for example in examples if example.fieldsite == fieldsite]
        speaker_count = len({example.speaker_id for example in held_out})
        if speaker_count < MINIMUM_TARGET_SPEAKERS:
            ineligible[fieldsite] = f"{speaker_count} target speaker(s); minimum is 3"
            continue
        eligible.append(fieldsite)
        fold = evaluate_cohort_adaptive_fold(
            [example for example in examples if example.fieldsite != fieldsite],
            held_out,
        )
        predictions.extend(fold["predictions"])  # type: ignore[arg-type]
        fold_alignments[fieldsite] = fold["cohort_alignment"]
    report = _register_report(
        predictions,
        adaptation_scope="speaker_and_mode",
        vocal_gate=VocalActivityGate(),
    )
    report.update(
        {
            "mode": "cohort_adaptive_predictive_register_leave_one_fieldsite_out",
            "calibration_boundary": "unlabeled within-speaker and known-vocal-mode normalization plus unlabeled fieldsite-and-mode CORAL; transductive batch, not cold-start causal",
            "eligible_fieldsites": eligible,
            "eligible_fieldsite_count": len(eligible),
            "ineligible_fieldsites": ineligible,
            "ineligible_fieldsite_count": len(ineligible),
            "fieldsite_count": len(fieldsites),
            "fold_alignments": fold_alignments,
            "alignment_uses_target_ids_ads_labels": False,
            "known_vocal_mode_grouping": True,
            "confirmation_audio_or_metadata_accessed": False,
            "promotion_allowed": False,
        }
    )
    return report


def select_cohort_adaptive_thresholds(
    predictions: list[dict[str, object]],
) -> dict[str, object]:
    selection = select_predictive_dynamics_thresholds(predictions)
    selection["protocol"] = "cohort_adaptive_out_of_field_threshold_selection"
    return selection


def cohort_adaptive_artifact(
    examples: list[ProsodicSequenceExample],
) -> dict[str, object]:
    return {
        "mode": "cohort_adaptive_predictive_register",
        "parent_model": PredictiveRegisterDynamicsModel.fit(examples).to_dict(),
        "alignment": {
            "direction": "target_to_source",
            "row_features": "concatenated_previous_and_current_state",
            "grouping": "fieldsite_and_known_vocal_mode",
            "covariance_shrinkage": COVARIANCE_SHRINKAGE,
            "eigenvalue_floor": EIGENVALUE_FLOOR,
            "minimum_target_speakers": MINIMUM_TARGET_SPEAKERS,
            "minimum_calibration_observations": MINIMUM_CALIBRATION_OBSERVATIONS,
            "uses_target_ids_ads_labels": False,
        },
    }
