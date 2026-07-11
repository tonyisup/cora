"""Contracts for the frozen H4 Option G local closure candidate."""

import numpy as np
import pytest

from cora_infans.expanded_local_closure import (
    ExpandedLocalForwardMap,
    REST_STATE_5,
    fit_expanded_forward_map,
    invert_expanded,
)
from cora_infans.factorized_tract_plant import expanded_sensory, halton5_points


def test_expanded_local_map_fits_frozen_babbling_with_bounded_state() -> None:
    model, metadata = fit_expanded_forward_map()
    prediction = model.predict(halton5_points(16385, 16385)[0])

    assert prediction.shape == (30,)
    assert np.linalg.norm(prediction) == pytest.approx(1.0)
    assert model.training_state_scalars == 28416
    assert model.retained_state_scalars == 23808
    assert metadata["training_count"] == 2048
    assert metadata["maximum_coordinate_delta_per_hop"] <= 0.20 + 1e-12
    assert np.all(model.counts > 0)


def test_expanded_inverse_respects_prediction_budget_and_bounds() -> None:
    model, _ = fit_expanded_forward_map()
    target_motor = halton5_points(18432, 18432)[0]
    result = invert_expanded(expanded_sensory(target_motor), model)

    assert result.forward_prediction_count <= 21
    assert result.final_loss <= result.initial_loss
    assert np.all(result.final_motor >= 0.0)
    assert np.all(result.final_motor <= 1.0)


def test_expanded_local_map_fails_closed() -> None:
    centers = halton5_points(16385, 16512)
    model = ExpandedLocalForwardMap(centers)
    with pytest.raises(ValueError, match="finalized"):
        model.predict(REST_STATE_5)
