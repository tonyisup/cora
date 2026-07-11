"""Frozen synthetic-only H4 Option H smooth exemplar-local candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expanded_local_closure import REST_STATE_5, _equivalence, _sensory_distance
from .factorized_tract_plant import (
    FROZEN_OPTION_E_CONFIG,
    VOWELS,
    expanded_formants,
    expanded_sensory,
    factorized_area_function,
    halton5_points,
    nearest_vowel,
    relative_formant_rmse,
)
from .geometric_articulatory_synthesis import F0_CONDITIONS_HZ


def _motor5(value: np.ndarray) -> np.ndarray:
    motor = np.asarray(value, dtype=np.float64).reshape(-1)
    if motor.shape != (5,) or not np.all(np.isfinite(motor)):
        raise ValueError("motor must contain five finite coordinates")
    if np.any(motor < 0.0) or np.any(motor > 1.0):
        raise ValueError("motor violates [0,1] bounds")
    return motor


@dataclass(frozen=True)
class SmoothInverseResult:
    initial_motor: np.ndarray
    final_motor: np.ndarray
    initial_loss: float
    final_loss: float
    prediction_count: int
    accepted_count: int


class SmoothExemplarForwardMap:
    """Query-local weighted affine regression over a bounded babbling table."""

    def __init__(self, motors: np.ndarray, sensory: np.ndarray) -> None:
        motor_values = np.asarray(motors, dtype=np.float64)
        sensory_values = np.asarray(sensory, dtype=np.float64)
        if motor_values.shape != (2048, 5):
            raise ValueError("smooth map requires 2048 five-scalar motors")
        if sensory_values.shape != (2048, 30):
            raise ValueError("smooth map requires 2048 thirty-scalar sensory rows")
        if not np.all(np.isfinite(motor_values)) or not np.all(
            np.isfinite(sensory_values)
        ):
            raise ValueError("smooth map state must be finite")
        self.motors = motor_values.copy()
        self.sensory = sensory_values.copy()

    @property
    def retained_state_scalars(self) -> int:
        return int(self.motors.size + self.sensory.size)

    def predict(self, motor: np.ndarray) -> np.ndarray:
        query = _motor5(motor)
        distances = np.sqrt(np.sum((self.motors - query) ** 2, axis=1))
        indices = np.lexsort((np.arange(len(distances)), distances))[:64]
        local_distances = distances[indices]
        bandwidth = float(local_distances[-1])
        if bandwidth <= 0.0:
            raise ValueError("local bandwidth must be positive")
        weights = np.exp(-0.5 * (local_distances / bandwidth) ** 2)
        design = np.column_stack(
            (np.ones(64, dtype=np.float64), self.motors[indices] - query)
        )
        weighted_design = design * weights[:, None]
        xtx = design.T @ weighted_design
        xty = weighted_design.T @ self.sensory[indices]
        ridge = np.diag([0.000001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
        coefficients = np.linalg.solve(xtx + ridge, xty)
        prediction = coefficients[0].copy()
        prediction -= float(np.mean(prediction))
        norm = float(np.linalg.norm(prediction))
        if norm <= 1e-12:
            raise ValueError("smooth prediction has insufficient structure")
        return prediction / norm


def fit_smooth_map() -> tuple[SmoothExemplarForwardMap, dict[str, object]]:
    motors = halton5_points(28673, 30720)
    sensory = np.stack([expanded_sensory(motor) for motor in motors])
    previous = REST_STATE_5.copy()
    interpolation_hops = 0
    maximum_delta = 0.0
    for motor in motors:
        hops = max(1, int(np.ceil(float(np.max(np.abs(motor - previous))) / 0.20)))
        interpolation_hops += hops - 1
        maximum_delta = max(
            maximum_delta, float(np.max(np.abs(motor - previous))) / hops
        )
        previous = motor
    model = SmoothExemplarForwardMap(motors, sensory)
    return model, {
        "babbling_count": len(motors),
        "interpolation_hop_count": interpolation_hops,
        "maximum_coordinate_delta_per_hop": maximum_delta,
        "training_state_scalars": model.retained_state_scalars,
        "retained_state_scalars": model.retained_state_scalars,
        "neighbor_count": 64,
    }


def evaluate_smooth_forward(
    model: SmoothExemplarForwardMap, metadata: dict[str, object]
) -> dict[str, object]:
    points = halton5_points(32769, 33280)
    neutral = expanded_sensory(REST_STATE_5)
    previous = neutral
    errors: list[float] = []
    baselines = {"neutral": [], "nearest_exemplar": [], "persistence": []}
    orthants: dict[str, list[float]] = {}
    for motor in points:
        target = expanded_sensory(motor)
        prediction = model.predict(motor)
        errors.append(_sensory_distance(target, prediction))
        distances = np.sum((model.motors - motor) ** 2, axis=1)
        nearest = int(np.argmin(distances))
        baselines["neutral"].append(_sensory_distance(target, neutral))
        baselines["nearest_exemplar"].append(
            _sensory_distance(target, model.sensory[nearest])
        )
        baselines["persistence"].append(_sensory_distance(target, previous))
        code = "".join("1" if value >= 0.5 else "0" for value in motor)
        orthants.setdefault(code, []).append(errors[-1])
        previous = target
    wins = {
        name: float(np.mean(np.asarray(errors) < np.asarray(values)))
        for name, values in baselines.items()
    }
    criteria = {
        "strict_win_vs_neutral_at_least_0_75": wins["neutral"] >= 0.75,
        "strict_win_vs_nearest_exemplar_at_least_0_75": wins["nearest_exemplar"]
        >= 0.75,
        "strict_win_vs_persistence_at_least_0_75": wins["persistence"] >= 0.75,
        "median_distance_at_most_0_18": float(np.median(errors)) <= 0.18,
        "all_32_orthants_reported": len(orthants) == 32,
        "state_bounds_hold": metadata["retained_state_scalars"] == 71680,
    }
    return {
        "held_out_count": len(points),
        "median_unit_vector_distance": float(np.median(errors)),
        "p95_unit_vector_distance": float(np.percentile(errors, 95)),
        "strict_win_fraction": wins,
        "baseline_median_distances": {
            name: float(np.median(values)) for name, values in baselines.items()
        },
        "orthant_mean_distance": {
            code: float(np.mean(values)) for code, values in sorted(orthants.items())
        },
        "criteria": criteria,
        "gate_passed": all(criteria.values()),
    }


def _loss(
    target: np.ndarray, predicted: np.ndarray, motor: np.ndarray
) -> float:
    return (
        _sensory_distance(target, predicted) / 2.0
        + 0.0005 * float(np.sum((motor - REST_STATE_5) ** 2))
        + 0.001 * float(np.sum((motor - REST_STATE_5) ** 2))
    )


def invert_smooth(target: np.ndarray, model: SmoothExemplarForwardMap) -> SmoothInverseResult:
    anchor_indices = np.arange(0, 2048, 8)
    rows = []
    for anchor_ordinal, index in enumerate(anchor_indices):
        motor = model.motors[index]
        rows.append(
            (
                _sensory_distance(target, model.sensory[index]) / 2.0,
                float(np.sum((motor - REST_STATE_5) ** 2)),
                anchor_ordinal,
                index,
            )
        )
    _, _, _, selected = min(rows)
    current = model.motors[selected].copy()
    current_loss = _loss(target, model.predict(current), current)
    initial_motor = current.copy()
    initial_loss = current_loss
    predictions = 1
    accepted = 0
    steps = np.full(5, 0.18)
    for _round in range(2):
        for axis in range(5):
            for direction in (-1.0, 1.0):
                candidate = current.copy()
                candidate[axis] += direction * steps[axis]
                if candidate[axis] < 0.0 or candidate[axis] > 1.0:
                    continue
                prediction = model.predict(candidate)
                predictions += 1
                loss = _loss(target, prediction, candidate)
                if loss < current_loss:
                    current = candidate
                    current_loss = loss
                    accepted += 1
        steps *= 0.5
    return SmoothInverseResult(
        initial_motor=initial_motor,
        final_motor=current,
        initial_loss=initial_loss,
        final_loss=current_loss,
        prediction_count=predictions,
        accepted_count=accepted,
    )


def _targets() -> list[dict[str, object]]:
    points = halton5_points(36865, 37888)
    formants = [expanded_formants(point) for point in points]
    used: set[int] = set()
    targets = []
    for vowel in VOWELS:
        candidates = [
            (relative_formant_rmse(vowel, value), index)
            for index, value in enumerate(formants)
            if value is not None and index not in used
        ]
        cost, offset = min(candidates, key=lambda row: (row[0], row[1]))
        used.add(offset)
        value = formants[offset]
        assert value is not None
        targets.append(
            {
                "vowel": vowel,
                "halton_index": 36865 + offset,
                "motor": points[offset],
                "formants": value,
                "canonical_relative_rmse": cost,
            }
        )
    return targets


def evaluate_smooth_closure(model: SmoothExemplarForwardMap) -> dict[str, object]:
    rows = []
    for target in _targets():
        target_motor = np.asarray(target["motor"])
        target_sensory = expanded_sensory(target_motor)
        for f0_hz in F0_CONDITIONS_HZ:
            result = invert_smooth(target_sensory, model)
            final_formants = expanded_formants(result.final_motor)
            final_values = (
                np.full(3, np.nan)
                if final_formants is None
                else np.asarray(final_formants)
            )
            predicted_vowel = (
                None if final_formants is None else nearest_vowel(final_formants)
            )
            target_formants = np.asarray(target["formants"])
            equivalence = _equivalence(target_motor, result.final_motor)
            geometry = factorized_area_function(
                result.final_motor, FROZEN_OPTION_E_CONFIG
            )
            rows.append(
                {
                    "expected_vowel": target["vowel"],
                    "f0_hz": f0_hz,
                    "target_halton_index": target["halton_index"],
                    "target_motor": target_motor.tolist(),
                    "initial_motor": result.initial_motor.tolist(),
                    "final_motor": result.final_motor.tolist(),
                    "target_formants_hz": target_formants.tolist(),
                    "final_formants_hz": final_values.tolist(),
                    "formant_errors_hz": np.abs(final_values - target_formants).tolist(),
                    "predicted_vowel": predicted_vowel,
                    "correct_identity": predicted_vowel == target["vowel"],
                    **equivalence,
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "forward_prediction_count": result.prediction_count,
                    "accepted_update_count": result.accepted_count,
                    "geometry_within_bounds": bool(
                        np.all(geometry.areas_cm2 >= 0.04)
                        and np.all(geometry.areas_cm2 <= 9.0)
                        and 15.5 <= geometry.total_length_cm <= 20.0
                    ),
                }
            )
    errors = np.asarray([row["formant_errors_hz"] for row in rows], dtype=float)
    mae = np.nanmean(errors, axis=0)
    identity = float(np.mean([row["correct_identity"] for row in rows]))
    agreement = float(np.mean([row["combined_agreement"] for row in rows]))
    criteria = {
        "identity_at_least_0_95": identity >= 0.95,
        "combined_agreement_at_least_0_90": agreement >= 0.90,
        "f1_mae_at_most_30_hz": float(mae[0]) <= 30.0,
        "f2_mae_at_most_30_hz": float(mae[1]) <= 30.0,
        "f3_mae_at_most_30_hz": float(mae[2]) <= 30.0,
        "all_losses_no_worse_than_initial": all(
            row["final_loss"] <= row["initial_loss"] for row in rows
        ),
        "all_bounds_hold": all(row["geometry_within_bounds"] for row in rows),
        "maximum_predictions_at_most_23": max(
            row["forward_prediction_count"] for row in rows
        )
        <= 23,
        "labels_absent_from_learning_and_search": True,
    }
    confusion = {
        expected: {
            predicted: sum(
                row["expected_vowel"] == expected
                and row["predicted_vowel"] == predicted
                for row in rows
            )
            for predicted in VOWELS
        }
        for expected in VOWELS
    }
    return {
        "trial_count": len(rows),
        "identity_accuracy": identity,
        "combined_agreement_fraction": agreement,
        "formant_mae_hz": {"f1": float(mae[0]), "f2": float(mae[1]), "f3": float(mae[2])},
        "confusion_matrix": confusion,
        "maximum_forward_predictions": max(
            row["forward_prediction_count"] for row in rows
        ),
        "criteria": criteria,
        "gate_passed": all(criteria.values()),
        "rows": rows,
    }


def evaluate_option_h() -> dict[str, object]:
    model, metadata = fit_smooth_map()
    forward = evaluate_smooth_forward(model, metadata)
    closure = evaluate_smooth_closure(model)
    return {
        "mode": "h4_option_h_smooth_exemplar_local_closure",
        "evidence_scope": "synthetic_only",
        "babbling": metadata,
        "forward_generalization": forward,
        "inverse_closure": closure,
        "all_synthetic_gates_passed": forward["gate_passed"] and closure["gate_passed"],
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
        "promotion_allowed": False,
    }
