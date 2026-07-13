"""Contracts for H4 Option H smooth exemplar-local closure."""

import json
from pathlib import Path

import numpy as np
import pytest

from cora_infans.factorized_tract_plant import expanded_sensory, halton5_points
from cora_infans.smooth_local_closure import fit_smooth_map, invert_smooth


@pytest.fixture(scope="module")
def smooth_model():
    return fit_smooth_map()


def test_smooth_map_has_frozen_memory_and_unit_prediction(smooth_model) -> None:
    model, metadata = smooth_model
    motor = halton5_points(28673, 28673)[0]
    prediction = model.predict(motor)

    assert metadata["babbling_count"] == 2048
    assert metadata["retained_state_scalars"] == 71680
    assert metadata["maximum_coordinate_delta_per_hop"] <= 0.20 + 1e-12
    assert prediction.shape == (30,)
    assert np.linalg.norm(prediction) == pytest.approx(1.0)


def test_smooth_inverse_is_bounded(smooth_model) -> None:
    model, _ = smooth_model
    motor = halton5_points(30720, 30720)[0]
    result = invert_smooth(expanded_sensory(motor), model)

    assert result.prediction_count <= 21
    assert result.final_loss <= result.initial_loss
    assert np.all(result.final_motor >= 0.0)
    assert np.all(result.final_motor <= 1.0)


def test_retained_option_h_result_preserves_failed_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-11-h4-option-h-smooth-local-closure-result.json"
        ).read_text()
    )

    assert result["forward_generalization"]["gate_passed"] is False
    assert result["inverse_closure"]["gate_passed"] is False
    assert result["inverse_closure"]["identity_accuracy"] == 1.0
    assert result["all_synthetic_gates_passed"] is False
    assert result["recorded_audio_accessed"] is False
    assert result["protected_data_accessed"] is False
    assert result["provenance"]["implementation_commit"] == "f9c2136"
