"""Locally learned nonlinear temporal codebook for register prediction.

The codebook is fit without condition labels by winner-local competitive
updates over causal prosodic contexts. Gaussian responses to those prototypes
form a bounded nonlinear representation; condition supervision enters only in
the four linear next-state readouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .predictive_register_dynamics import (
    CONDITION_LABELS,
    RIDGE_REGULARIZATION,
    RESIDUAL_VARIANCE_FLOOR,
    STATE_FEATURE_NAMES,
    AdaptedSequence,
    ProsodicSequenceExample,
    adapt_sequences,
    select_predictive_dynamics_thresholds,
)
from .register_saliency import VocalActivityGate, _register_report


CONTEXT_FRAMES = 4
PROTOTYPE_COUNT = 32
CODEBOOK_SEED = 20260711
LEARNING_RATE_START = 0.20
LEARNING_RATE_END = 0.02
CODEBOOK_PASSES = 1
BANDWIDTH_FLOOR = 0.1
MINIMUM_CALIBRATION_OBSERVATIONS = 2


def causal_context_targets(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return four-state causal contexts and their next-state targets."""

    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or len(states) <= CONTEXT_FRAMES:
        raise ValueError("nonlinear predictive codebook sequence is too short")
    contexts = np.concatenate(
        [
            states[offset : len(states) - CONTEXT_FRAMES + offset]
            for offset in range(CONTEXT_FRAMES)
        ],
        axis=1,
    )
    targets = states[CONTEXT_FRAMES:]
    if len(contexts) != len(targets):
        raise AssertionError("causal context/target alignment drifted")
    return contexts, targets


def _squared_distances(rows: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(rows * rows, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * rows @ centers.T
    )
    return np.maximum(distances, 0.0)


def _nearest_distances_and_assignments(
    rows: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distances = _squared_distances(rows, centers)
    assignments = np.argmin(distances, axis=1)
    nearest = np.sqrt(distances[np.arange(len(rows)), assignments])
    return nearest, assignments


def _kmeans_plus_plus(rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((PROTOTYPE_COUNT, rows.shape[1]), dtype=np.float64)
    first = int(rng.integers(len(rows)))
    centers[0] = rows[first]
    closest = np.sum((rows - centers[0]) ** 2, axis=1)
    for index in range(1, PROTOTYPE_COUNT):
        total = float(np.sum(closest))
        if total <= 0.0:
            selected = index % len(rows)
        else:
            selected = int(rng.choice(len(rows), p=closest / total))
        centers[index] = rows[selected]
        distance = np.sum((rows - centers[index]) ** 2, axis=1)
        closest = np.minimum(closest, distance)
    return centers


@dataclass(frozen=True)
class CompetitiveTemporalCodebook:
    centers: tuple[tuple[float, ...], ...]
    bandwidth: float
    training_context_count: int
    quantization_error_before: float
    quantization_error_after: float
    utilization_count: int
    winner_update_count: tuple[int, ...]

    @classmethod
    def fit(cls, contexts: np.ndarray) -> "CompetitiveTemporalCodebook":
        contexts = np.asarray(contexts, dtype=np.float64)
        expected_dimension = CONTEXT_FRAMES * len(STATE_FEATURE_NAMES)
        if contexts.ndim != 2 or contexts.shape[1] != expected_dimension:
            raise ValueError("codebook contexts have the wrong shape")
        if len(contexts) < PROTOTYPE_COUNT:
            raise ValueError("codebook requires at least one context per prototype")
        if not bool(np.all(np.isfinite(contexts))):
            raise ValueError("codebook contexts must be finite")

        rng = np.random.default_rng(CODEBOOK_SEED)
        centers = _kmeans_plus_plus(contexts, rng)
        before, _ = _nearest_distances_and_assignments(contexts, centers)
        update_counts = np.zeros(PROTOTYPE_COUNT, dtype=np.int64)
        order = rng.permutation(len(contexts))
        denominator = max(len(order) - 1, 1)
        for step, row_index in enumerate(order):
            row = contexts[row_index]
            winner = int(np.argmin(np.sum((centers - row) ** 2, axis=1)))
            fraction = step / denominator
            learning_rate = LEARNING_RATE_START * math.pow(
                LEARNING_RATE_END / LEARNING_RATE_START, fraction
            )
            centers[winner] += learning_rate * (row - centers[winner])
            update_counts[winner] += 1

        after, assignments = _nearest_distances_and_assignments(contexts, centers)
        bandwidth = max(float(np.median(after)), BANDWIDTH_FLOOR)
        utilization_count = int(len(np.unique(assignments)))
        if not bool(np.all(np.isfinite(centers))) or not math.isfinite(bandwidth):
            raise ValueError("codebook learning produced non-finite values")
        return cls(
            centers=tuple(tuple(float(value) for value in row) for row in centers),
            bandwidth=bandwidth,
            training_context_count=len(contexts),
            quantization_error_before=float(np.mean(before)),
            quantization_error_after=float(np.mean(after)),
            utilization_count=utilization_count,
            winner_update_count=tuple(int(value) for value in update_counts),
        )

    def activations(self, contexts: np.ndarray) -> np.ndarray:
        contexts = np.asarray(contexts, dtype=np.float64)
        centers = np.asarray(self.centers, dtype=np.float64)
        if contexts.ndim != 2 or contexts.shape[1] != centers.shape[1]:
            raise ValueError("codebook activation contexts have the wrong shape")
        return np.exp(
            -_squared_distances(contexts, centers) / (2.0 * self.bandwidth**2)
        )

    @property
    def utilization_fraction(self) -> float:
        return self.utilization_count / len(self.centers)

    def diagnostics(self) -> dict[str, object]:
        return {
            "training_context_count": self.training_context_count,
            "quantization_error_before": self.quantization_error_before,
            "quantization_error_after": self.quantization_error_after,
            "quantization_error_nonincreasing": self.quantization_error_after
            <= self.quantization_error_before,
            "utilization_count": self.utilization_count,
            "utilization_fraction": self.utilization_fraction,
            "winner_update_count": list(self.winner_update_count),
            "bandwidth": self.bandwidth,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "winner_local_competitive_temporal_codebook",
            "context_frames": CONTEXT_FRAMES,
            "context_dimension": CONTEXT_FRAMES * len(STATE_FEATURE_NAMES),
            "prototype_count": PROTOTYPE_COUNT,
            "seed": CODEBOOK_SEED,
            "learning_rate_start": LEARNING_RATE_START,
            "learning_rate_end": LEARNING_RATE_END,
            "passes": CODEBOOK_PASSES,
            "bandwidth_floor": BANDWIDTH_FLOOR,
            "bandwidth": self.bandwidth,
            "centers": [list(row) for row in self.centers],
            "diagnostics": self.diagnostics(),
        }


@dataclass(frozen=True)
class NonlinearPredictiveDynamics:
    intercept: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    residual_variance: tuple[float, ...]
    ridge_regularization: float = RIDGE_REGULARIZATION
    residual_variance_floor: float = RESIDUAL_VARIANCE_FLOOR

    @classmethod
    def fit(
        cls,
        contexts: np.ndarray,
        targets: np.ndarray,
        codebook: CompetitiveTemporalCodebook,
    ) -> "NonlinearPredictiveDynamics":
        contexts = np.asarray(contexts, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if contexts.ndim != 2 or targets.ndim != 2 or len(contexts) != len(targets):
            raise ValueError("nonlinear dynamics requires matched context matrices")
        features = np.column_stack(
            (contexts[:, -len(STATE_FEATURE_NAMES) :], codebook.activations(contexts))
        )
        design = np.column_stack((np.ones(len(features)), features))
        penalty = np.diag([0.0, *([RIDGE_REGULARIZATION] * features.shape[1])])
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ targets,
        )
        prediction = design @ coefficients
        variance = np.maximum(
            np.mean((targets - prediction) ** 2, axis=0),
            RESIDUAL_VARIANCE_FLOOR,
        )
        return cls(
            intercept=tuple(float(value) for value in coefficients[0]),
            coefficients=tuple(
                tuple(float(value) for value in row) for row in coefficients[1:]
            ),
            residual_variance=tuple(float(value) for value in variance),
        )

    def mean_log_likelihood(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        codebook: CompetitiveTemporalCodebook,
    ) -> float:
        features = np.column_stack(
            (
                contexts[:, -len(STATE_FEATURE_NAMES) :],
                codebook.activations(contexts),
            )
        )
        prediction = np.asarray(self.intercept) + features @ np.asarray(
            self.coefficients
        )
        variance = np.asarray(self.residual_variance)
        residual = targets - prediction
        likelihood = -0.5 * (
            np.log(2.0 * np.pi * variance) + residual * residual / variance
        )
        return float(np.mean(np.sum(likelihood, axis=1)))

    def to_dict(self) -> dict[str, object]:
        return {
            "intercept": list(self.intercept),
            "coefficients": [list(row) for row in self.coefficients],
            "residual_variance": list(self.residual_variance),
            "ridge_regularization": self.ridge_regularization,
            "residual_variance_floor": self.residual_variance_floor,
        }


def _eligible_adapted(examples: list[ProsodicSequenceExample]) -> list[AdaptedSequence]:
    return [
        item
        for item in adapt_sequences(examples)
        if item.calibration_observation_count >= MINIMUM_CALIBRATION_OBSERVATIONS
    ]


@dataclass(frozen=True)
class NonlinearPredictiveCodebookModel:
    codebook: CompetitiveTemporalCodebook
    condition_dynamics: dict[str, NonlinearPredictiveDynamics]
    vocal_gate: VocalActivityGate = VocalActivityGate()

    @classmethod
    def fit(
        cls, examples: list[ProsodicSequenceExample]
    ) -> "NonlinearPredictiveCodebookModel":
        adapted = _eligible_adapted(examples)
        all_contexts = np.concatenate(
            [causal_context_targets(item.states)[0] for item in adapted], axis=0
        )
        codebook = CompetitiveTemporalCodebook.fit(all_contexts)
        condition_dynamics = {}
        for label in CONDITION_LABELS:
            pairs = [
                causal_context_targets(item.states)
                for item in adapted
                if item.example.label == label
            ]
            contexts = np.concatenate([pair[0] for pair in pairs], axis=0)
            targets = np.concatenate([pair[1] for pair in pairs], axis=0)
            condition_dynamics[label] = NonlinearPredictiveDynamics.fit(
                contexts, targets, codebook
            )
        return cls(codebook=codebook, condition_dynamics=condition_dynamics)

    def evaluate_batch(
        self, examples: list[ProsodicSequenceExample]
    ) -> dict[str, object]:
        predictions = []
        for item in adapt_sequences(examples):
            example = item.example
            contexts, targets = causal_context_targets(item.states)
            likelihoods = {
                label: dynamics.mean_log_likelihood(
                    contexts, targets, self.codebook
                )
                for label, dynamics in self.condition_dynamics.items()
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
            calibrated = (
                item.calibration_observation_count
                >= MINIMUM_CALIBRATION_OBSERVATIONS
            )
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
                    "causal_context_count": len(contexts),
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
            "mode": "nonlinear_predictive_codebook_model",
            "state_feature_names": list(STATE_FEATURE_NAMES),
            "frame_ms": 50,
            "calibration_boundary": "unlabeled within-speaker and known-vocal-mode batch normalization; not cold-start causal",
            "codebook": self.codebook.to_dict(),
            "readout_features": {
                "last_state_dimension": len(STATE_FEATURE_NAMES),
                "rbf_activation_dimension": PROTOTYPE_COUNT,
            },
            "condition_dynamics": {
                label: dynamics.to_dict()
                for label, dynamics in self.condition_dynamics.items()
            },
            "vocal_gate": self.vocal_gate.to_dict(),
        }


def nonlinear_codebook_leave_one_fieldsite_out(
    examples: list[ProsodicSequenceExample],
) -> dict[str, object]:
    fieldsites = sorted({example.fieldsite for example in examples})
    predictions: list[dict[str, object]] = []
    fold_codebook_diagnostics: dict[str, object] = {}
    for fieldsite in fieldsites:
        model = NonlinearPredictiveCodebookModel.fit(
            [example for example in examples if example.fieldsite != fieldsite]
        )
        report = model.evaluate_batch(
            [example for example in examples if example.fieldsite == fieldsite]
        )
        predictions.extend(report["predictions"])  # type: ignore[arg-type]
        fold_codebook_diagnostics[fieldsite] = model.codebook.diagnostics()
    report = _register_report(
        predictions,
        adaptation_scope="speaker_and_mode",
        vocal_gate=VocalActivityGate(),
    )
    utilization = [
        float(item["utilization_fraction"])
        for item in fold_codebook_diagnostics.values()  # type: ignore[union-attr]
    ]
    quantization_health = [
        bool(item["quantization_error_nonincreasing"])
        for item in fold_codebook_diagnostics.values()  # type: ignore[union-attr]
    ]
    report.update(
        {
            "mode": "nonlinear_predictive_codebook_leave_one_fieldsite_out",
            "fieldsite_count": len(fieldsites),
            "fold_codebook_diagnostics": fold_codebook_diagnostics,
            "minimum_codebook_utilization_fraction": min(utilization),
            "all_fold_quantization_errors_nonincreasing": all(
                quantization_health
            ),
            "representation_uses_condition_labels": False,
            "known_vocal_mode_grouping": True,
            "confirmation_audio_or_metadata_accessed": False,
            "promotion_allowed": False,
        }
    )
    return report


def select_nonlinear_codebook_thresholds(
    predictions: list[dict[str, object]],
) -> dict[str, object]:
    selection = select_predictive_dynamics_thresholds(predictions)
    selection["protocol"] = "nonlinear_codebook_out_of_field_threshold_selection"
    return selection
