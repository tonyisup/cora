"""Focused contracts for the H4 Option D geometric articulatory candidate."""

import json
from pathlib import Path

import numpy as np
import pytest

from cora_infans.geometric_articulatory_synthesis import (
    F0_CONDITIONS_HZ,
    LocalAffineForwardMap,
    REST_STATE,
    area_function,
    bounded_transition,
    halton_motors,
    sensory_vector,
    synthesize_tract,
    transfer_function,
)


def test_area_function_is_bounded_and_motor_dependent() -> None:
    rest = area_function(REST_STATE)
    constricted = area_function(np.array([0.1, 0.7, 0.8]))

    assert rest.areas_cm2.shape == (10,)
    assert np.all(rest.areas_cm2 >= 0.05)
    assert np.all(rest.areas_cm2 <= 8.0)
    assert rest.total_length_cm == pytest.approx(16.4)
    assert constricted.total_length_cm == pytest.approx(17.6)
    assert not np.allclose(rest.areas_cm2, constricted.areas_cm2)


def test_transfer_and_waveform_are_finite_and_deterministic() -> None:
    frequencies = np.linspace(0.0, 4000.0, 257)
    response = transfer_function(REST_STATE, frequencies)
    first = synthesize_tract(REST_STATE, f0_hz=F0_CONDITIONS_HZ[1])
    second = synthesize_tract(REST_STATE, f0_hz=F0_CONDITIONS_HZ[1])

    assert np.all(np.isfinite(response))
    assert first.shape == (2560,)
    assert first.dtype == np.float32
    assert float(np.max(np.abs(first))) < 1.0
    np.testing.assert_array_equal(first, second)


def test_halton_splits_are_disjoint_and_transitions_are_bounded() -> None:
    training = halton_motors(1, 1024)
    held_out = halton_motors(2049, 2304)
    transition = bounded_transition(training[0], training[1])

    assert training.shape == (1024, 3)
    assert held_out.shape == (256, 3)
    assert not {tuple(row) for row in training} & {tuple(row) for row in held_out}
    hops = np.diff(np.vstack((training[0], transition)), axis=0)
    assert float(np.max(np.abs(hops))) <= 0.25 + 1e-12
    np.testing.assert_allclose(transition[-1], training[1])


def test_winner_local_affine_map_learns_supported_linear_surface() -> None:
    centers = halton_motors(1, 64)
    model = LocalAffineForwardMap(centers)
    for center in centers:
        for offset in (
            np.zeros(3),
            np.array([0.01, 0.0, 0.0]),
            np.array([0.0, 0.01, 0.0]),
            np.array([0.0, 0.0, 0.01]),
        ):
            motor = np.clip(center + offset, 0.0, 1.0)
            model.update(motor, sensory_vector(motor))
    model.finalize()

    prediction = model.predict(centers[10])
    coefficients = model.coefficients.copy()
    first_intercept = model.intercept_prediction(10)
    second_intercept = model.intercept_prediction(10)

    assert prediction.shape == (24,)
    assert np.linalg.norm(prediction) == pytest.approx(1.0)
    np.testing.assert_array_equal(first_intercept, second_intercept)
    np.testing.assert_array_equal(model.coefficients, coefficients)
    assert model.training_state_scalars == 7424
    assert model.retained_state_scalars == 6400
    assert np.all(model.counts > 0)


def test_invalid_motor_and_f0_fail_closed() -> None:
    with pytest.raises(ValueError, match="bounds"):
        area_function(np.array([-0.1, 0.5, 0.5]))
    with pytest.raises(ValueError, match="F0"):
        synthesize_tract(REST_STATE, f0_hz=123.0)


def test_retained_option_d_result_preserves_failed_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-11-h4-option-d-synthetic-result.json"
        ).read_text()
    )

    assert result["plant"]["gate_passed"] is False
    assert result["forward_generalization"]["gate_passed"] is True
    assert result["inverse_closure"]["gate_passed"] is False
    assert result["all_synthetic_gates_passed"] is False
    assert result["promotion_allowed"] is False
    assert result["recorded_audio_accessed"] is False
    assert result["protected_data_accessed"] is False
    assert result["provenance"]["implementation_commit"] == "62c9535"
