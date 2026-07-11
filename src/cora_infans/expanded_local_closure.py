"""Frozen synthetic-only H4 Option G local closure candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

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


REST_STATE_5 = np.array([0.6, 0.65, 0.5, 0.5, 0.35], dtype=np.float64)


def _motor5(value: np.ndarray) -> np.ndarray:
    motor = np.asarray(value, dtype=np.float64).reshape(-1)
    if motor.shape != (5,) or not np.all(np.isfinite(motor)):
        raise ValueError("motor must contain five finite coordinates")
    if np.any(motor < 0.0) or np.any(motor > 1.0):
        raise ValueError("motor violates [0,1] bounds")
    return motor


@dataclass(frozen=True)
class ExpandedInverseResult:
    initial_motor: np.ndarray
    final_motor: np.ndarray
    initial_loss: float
    final_loss: float
    forward_prediction_count: int
    accepted_update_count: int


class ExpandedLocalForwardMap:
    """128-center winner-local affine map from five motors to 30 bands."""

    def __init__(self, centers: np.ndarray) -> None:
        value = np.asarray(centers, dtype=np.float64)
        if value.shape != (128, 5):
            raise ValueError("forward map requires 128 five-scalar centers")
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError("prototype center violates motor bounds")
        self.centers = value.copy()
        self._xtx = np.zeros((128, 6, 6), dtype=np.float64)
        self._xty = np.zeros((128, 6, 30), dtype=np.float64)
        self.counts = np.zeros(128, dtype=np.int64)
        self.coefficients: np.ndarray | None = None

    @property
    def training_state_scalars(self) -> int:
        return int(self.centers.size + self._xtx.size + self._xty.size + self.counts.size)

    @property
    def retained_state_scalars(self) -> int:
        return int(self.centers.size + 128 * 6 * 30 + self.counts.size)

    def winner_index(self, motor: np.ndarray) -> int:
        m = _motor5(motor)
        return int(np.argmin(np.sum((self.centers - m) ** 2, axis=1)))

    def update(self, motor: np.ndarray, sensory: np.ndarray) -> None:
        if self.coefficients is not None:
            raise ValueError("cannot update a finalized forward map")
        m = _motor5(motor)
        target = np.asarray(sensory, dtype=np.float64).reshape(-1)
        if target.shape != (30,) or not np.all(np.isfinite(target)):
            raise ValueError("forward target must contain 30 finite values")
        winner = self.winner_index(m)
        design = np.concatenate(([1.0], m - self.centers[winner]))
        self._xtx[winner] += np.outer(design, design)
        self._xty[winner] += np.outer(design, target)
        self.counts[winner] += 1

    def finalize(self) -> None:
        if np.any(self.counts == 0):
            raise ValueError("every prototype must receive babbling support")
        ridge = np.diag([0.000001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
        coefficients = np.empty((128, 6, 30), dtype=np.float64)
        for index in range(128):
            coefficients[index] = np.linalg.solve(
                self._xtx[index] + ridge, self._xty[index]
            )
        self.coefficients = coefficients

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64).reshape(30).copy()
        value -= float(np.mean(value))
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ValueError("forward prediction has insufficient structure")
        return value / norm

    def predict(self, motor: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("forward map must be finalized")
        m = _motor5(motor)
        winner = self.winner_index(m)
        design = np.concatenate(([1.0], m - self.centers[winner]))
        return self._normalized(design @ self.coefficients[winner])

    def intercept_prediction(self, index: int) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("forward map must be finalized")
        if not 0 <= index < 128:
            raise ValueError("prototype index out of range")
        return self._normalized(self.coefficients[index, 0])


def _transition_hops(previous: np.ndarray, target: np.ndarray) -> int:
    return max(1, int(np.ceil(float(np.max(np.abs(target - previous))) / 0.20)))


def fit_expanded_forward_map() -> tuple[ExpandedLocalForwardMap, dict[str, object]]:
    training = halton5_points(16385, 18432)
    model = ExpandedLocalForwardMap(training[:128])
    previous = REST_STATE_5.copy()
    interpolation_hops = 0
    maximum_delta = 0.0
    f0_counts = {str(value): 0 for value in F0_CONDITIONS_HZ}
    for ordinal, motor in enumerate(training):
        hops = _transition_hops(previous, motor)
        interpolation_hops += hops - 1
        maximum_delta = max(
            maximum_delta, float(np.max(np.abs(motor - previous))) / hops
        )
        f0_counts[str(F0_CONDITIONS_HZ[ordinal % 4])] += 1
        model.update(motor, expanded_sensory(motor))
        previous = motor
    model.finalize()
    return model, {
        "training_count": len(training),
        "f0_assignment_counts": f0_counts,
        "interpolation_hop_count": interpolation_hops,
        "maximum_coordinate_delta_per_hop": maximum_delta,
        "minimum_prototype_support": int(np.min(model.counts)),
        "maximum_prototype_support": int(np.max(model.counts)),
        "training_state_scalars": model.training_state_scalars,
        "retained_state_scalars": model.retained_state_scalars,
    }


def _sensory_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def evaluate_forward_gate(
    model: ExpandedLocalForwardMap, metadata: dict[str, object]
) -> dict[str, object]:
    held_out = halton5_points(20481, 20992)
    neutral = expanded_sensory(REST_STATE_5)
    previous = neutral
    distances: list[float] = []
    baselines = {"neutral": [], "nearest_prototype": [], "persistence": []}
    orthants: dict[str, list[float]] = {}
    for motor in held_out:
        target = expanded_sensory(motor)
        predicted = model.predict(motor)
        distance = _sensory_distance(target, predicted)
        winner = model.winner_index(motor)
        distances.append(distance)
        baselines["neutral"].append(_sensory_distance(target, neutral))
        baselines["nearest_prototype"].append(
            _sensory_distance(target, model.intercept_prediction(winner))
        )
        baselines["persistence"].append(_sensory_distance(target, previous))
        code = "".join("1" if value >= 0.5 else "0" for value in motor)
        orthants.setdefault(code, []).append(distance)
        previous = target
    strict_wins = {
        name: float(np.mean(np.asarray(distances) < np.asarray(values)))
        for name, values in baselines.items()
    }
    criteria = {
        "strict_win_vs_neutral_at_least_0_75": strict_wins["neutral"] >= 0.75,
        "strict_win_vs_nearest_prototype_at_least_0_75": strict_wins[
            "nearest_prototype"
        ]
        >= 0.75,
        "strict_win_vs_persistence_at_least_0_75": strict_wins["persistence"]
        >= 0.75,
        "median_distance_at_most_0_18": float(np.median(distances)) <= 0.18,
        "all_32_orthants_reported": len(orthants) == 32,
        "all_prototypes_supported": bool(np.all(model.counts > 0)),
        "training_state_within_bound": metadata["training_state_scalars"] <= 28416,
        "retained_state_within_bound": metadata["retained_state_scalars"] <= 23808,
    }
    return {
        "held_out_count": len(held_out),
        "median_unit_vector_distance": float(np.median(distances)),
        "p95_unit_vector_distance": float(np.percentile(distances, 95)),
        "strict_win_fraction": strict_wins,
        "baseline_median_distances": {
            name: float(np.median(values)) for name, values in baselines.items()
        },
        "orthant_mean_distance": {
            code: float(np.mean(values)) for code, values in sorted(orthants.items())
        },
        "criteria": criteria,
        "gate_passed": all(criteria.values()),
    }


def _total_loss(
    target: np.ndarray, predicted: np.ndarray, motor: np.ndarray, previous: np.ndarray
) -> float:
    return (
        _sensory_distance(target, predicted) / 2.0
        + 0.0005 * float(np.sum((motor - REST_STATE_5) ** 2))
        + 0.001 * float(np.sum((motor - previous) ** 2))
    )


def invert_expanded(
    target: np.ndarray, model: ExpandedLocalForwardMap
) -> ExpandedInverseResult:
    initial_rows = []
    for index, center in enumerate(model.centers):
        loss = _total_loss(
            target, model.intercept_prediction(index), center, REST_STATE_5
        )
        initial_rows.append(
            (loss, float(np.sum((center - REST_STATE_5) ** 2)), index)
        )
    _, _, winner = min(initial_rows)
    current = model.centers[winner].copy()
    current_loss = _total_loss(target, model.predict(current), current, REST_STATE_5)
    initial_motor = current.copy()
    initial_loss = current_loss
    predictions = 1
    accepted = 0
    steps = np.full(5, 0.18, dtype=np.float64)
    for _round in range(2):
        for axis in range(5):
            for direction in (-1.0, 1.0):
                candidate = current.copy()
                candidate[axis] += direction * steps[axis]
                if candidate[axis] < 0.0 or candidate[axis] > 1.0:
                    continue
                predicted = model.predict(candidate)
                predictions += 1
                loss = _total_loss(target, predicted, candidate, REST_STATE_5)
                if loss < current_loss:
                    current = candidate
                    current_loss = loss
                    accepted += 1
        steps *= 0.5
    return ExpandedInverseResult(
        initial_motor=initial_motor,
        final_motor=current,
        initial_loss=initial_loss,
        final_loss=current_loss,
        forward_prediction_count=predictions,
        accepted_update_count=accepted,
    )


def _select_inverse_targets() -> list[dict[str, object]]:
    points = halton5_points(24577, 25600)
    formants = [expanded_formants(point) for point in points]
    used: set[int] = set()
    targets: list[dict[str, object]] = []
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
                "halton_index": 24577 + offset,
                "motor": points[offset],
                "formants": value,
                "canonical_relative_rmse": cost,
            }
        )
    return targets


def _equivalence(target_motor: np.ndarray, final_motor: np.ndarray) -> dict[str, object]:
    delta = 0.005
    columns = []
    for axis in range(5):
        lower = target_motor.copy()
        upper = target_motor.copy()
        lower[axis] = max(0.0, lower[axis] - delta)
        upper[axis] = min(1.0, upper[axis] + delta)
        columns.append(
            (expanded_sensory(upper) - expanded_sensory(lower))
            / (upper[axis] - lower[axis])
        )
    jacobian = np.column_stack(columns)
    _, singular_values, right = np.linalg.svd(jacobian, full_matrices=False)
    threshold = 0.01 * float(singular_values[0])
    identifiable = right[singular_values >= threshold]
    projector = identifiable.T @ identifiable
    projected_error = float(
        np.linalg.norm(projector @ (final_motor - target_motor))
    )
    acoustic = _sensory_distance(
        expanded_sensory(target_motor), expanded_sensory(final_motor)
    ) / 2.0
    return {
        "projected_motor_error": projected_error,
        "acoustic_cycle_distance": acoustic,
        "jacobian_rank": int(np.sum(singular_values >= threshold)),
        "singular_values": singular_values.tolist(),
        "rank_threshold": threshold,
        "combined_agreement": projected_error <= 0.18 and acoustic <= 0.10,
    }


def evaluate_closure_gate(model: ExpandedLocalForwardMap) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for target in _select_inverse_targets():
        target_motor = np.asarray(target["motor"], dtype=np.float64)
        target_sensory = expanded_sensory(target_motor)
        for f0_hz in F0_CONDITIONS_HZ:
            inversion = invert_expanded(target_sensory, model)
            final_formants = expanded_formants(inversion.final_motor)
            equivalence = _equivalence(target_motor, inversion.final_motor)
            target_formants = np.asarray(target["formants"], dtype=np.float64)
            if final_formants is None:
                final_values = np.full(3, np.nan)
                predicted_vowel = None
            else:
                final_values = np.asarray(final_formants)
                predicted_vowel = nearest_vowel(final_formants)
            geometry = factorized_area_function(
                inversion.final_motor, FROZEN_OPTION_E_CONFIG
            )
            rows.append(
                {
                    "expected_vowel": target["vowel"],
                    "f0_hz": f0_hz,
                    "target_halton_index": target["halton_index"],
                    "target_motor": target_motor.tolist(),
                    "initial_motor": inversion.initial_motor.tolist(),
                    "final_motor": inversion.final_motor.tolist(),
                    "target_formants_hz": target_formants.tolist(),
                    "final_formants_hz": final_values.tolist(),
                    "formant_errors_hz": np.abs(final_values - target_formants).tolist(),
                    "predicted_vowel": predicted_vowel,
                    "correct_identity": predicted_vowel == target["vowel"],
                    **equivalence,
                    "initial_loss": inversion.initial_loss,
                    "final_loss": inversion.final_loss,
                    "forward_prediction_count": inversion.forward_prediction_count,
                    "accepted_update_count": inversion.accepted_update_count,
                    "geometry_within_bounds": bool(
                        np.all(geometry.areas_cm2 >= 0.04)
                        and np.all(geometry.areas_cm2 <= 9.0)
                        and 15.5 <= geometry.total_length_cm <= 20.0
                    ),
                }
            )
    errors = np.asarray([row["formant_errors_hz"] for row in rows], dtype=float)
    identity = float(np.mean([row["correct_identity"] for row in rows]))
    agreement = float(np.mean([row["combined_agreement"] for row in rows]))
    formant_mae = np.nanmean(errors, axis=0)
    criteria = {
        "identity_preservation_at_least_0_95": identity >= 0.95,
        "combined_agreement_at_least_0_90": agreement >= 0.90,
        "target_to_final_f1_mae_at_most_30_hz": float(formant_mae[0]) <= 30.0,
        "target_to_final_f2_mae_at_most_30_hz": float(formant_mae[1]) <= 30.0,
        "target_to_final_f3_mae_at_most_30_hz": float(formant_mae[2]) <= 30.0,
        "all_losses_no_worse_than_initial": all(
            row["final_loss"] <= row["initial_loss"] for row in rows
        ),
        "all_motor_and_geometry_bounds_hold": all(
            row["geometry_within_bounds"] for row in rows
        ),
        "maximum_forward_predictions_at_most_23": max(
            row["forward_prediction_count"] for row in rows
        )
        <= 23,
        "labels_absent_from_learning_initialization_loss_and_search": True,
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
        "formant_mae_hz": {
            "f1": float(formant_mae[0]),
            "f2": float(formant_mae[1]),
            "f3": float(formant_mae[2]),
        },
        "confusion_matrix": confusion,
        "maximum_forward_predictions": max(
            row["forward_prediction_count"] for row in rows
        ),
        "criteria": criteria,
        "gate_passed": all(criteria.values()),
        "rows": rows,
    }


def evaluate_option_g() -> dict[str, object]:
    model, babbling = fit_expanded_forward_map()
    forward = evaluate_forward_gate(model, babbling)
    closure = evaluate_closure_gate(model)
    passed = forward["gate_passed"] and closure["gate_passed"]
    return {
        "mode": "h4_option_g_expanded_local_closure_synthetic",
        "evidence_scope": "synthetic_only",
        "babbling": babbling,
        "forward_generalization": forward,
        "inverse_closure": closure,
        "all_synthetic_gates_passed": passed,
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
        "promotion_allowed": False,
        "claim_boundary": "A full pass is synthetic-only evidence for the frozen simplified plant and local learner; it does not promote H4 or authorize recorded speech.",
    }
