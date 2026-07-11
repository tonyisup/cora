"""Contracts for H4 Option H smooth exemplar-local closure."""

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
