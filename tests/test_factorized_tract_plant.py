"""Contracts for the Option E design-stage factorized tract plant."""

import json
from pathlib import Path

import numpy as np

from cora_infans.factorized_tract_plant import (
    FROZEN_OPTION_E_CONFIG,
    design_family,
    factorized_area_function,
    factorized_formants,
    halton5_points,
    synthesize_factorized,
)


def test_option_e_family_and_halton_contract() -> None:
    family = design_family()
    design = halton5_points(4097, 5120)
    confirmation = halton5_points(8193, 9216)

    assert len(family) == 64
    assert len({tuple(config.__dict__.values()) for config in family}) == 64
    assert design.shape == (1024, 5)
    assert not {tuple(row) for row in design} & {tuple(row) for row in confirmation}


def test_option_e_geometry_is_bounded_and_factorized() -> None:
    config = design_family()[0]
    base = np.full(5, 0.5)
    geometry = factorized_area_function(base, config)

    assert np.all(np.isfinite(geometry.areas_cm2))
    assert np.all(geometry.areas_cm2 >= 0.04)
    assert np.all(geometry.areas_cm2 <= 9.0)
    assert 15.5 <= geometry.total_length_cm <= config.maximum_length_cm

    for axis in range(5):
        changed = base.copy()
        changed[axis] += 0.1
        other = factorized_area_function(changed, config)
        if axis == 4:
            assert other.total_length_cm > geometry.total_length_cm
        else:
            assert not np.allclose(other.areas_cm2, geometry.areas_cm2)


def test_option_e_formant_analysis_is_deterministic() -> None:
    config = design_family()[0]
    motor = halton5_points(4097, 4097)[0]
    assert factorized_formants(motor, config) == factorized_formants(motor, config)


def test_option_e_frozen_confirmation_config_and_waveform() -> None:
    assert FROZEN_OPTION_E_CONFIG == design_family()[23]
    motor = halton5_points(8193, 8193)[0]
    first = synthesize_factorized(motor, 110.0)
    second = synthesize_factorized(motor, 110.0)

    assert first.shape == (2560,)
    assert float(np.max(np.abs(first))) < 1.0
    np.testing.assert_array_equal(first, second)


def test_retained_option_e_confirmation_preserves_stop_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-11-h4-option-e-plant-confirmation-result.json"
        ).read_text()
    )

    assert result["p0_numerical_physical"]["gate_passed"] is False
    assert result["p2_expressivity"]["gate_passed"] is True
    assert result["all_plant_gates_passed"] is False
    assert result["forward_learning_executed"] is False
    assert result["inverse_search_executed"] is False
    assert result["recorded_audio_accessed"] is False
    assert result["protected_data_accessed"] is False
    assert result["provenance"]["implementation_commit"] == "6ce2dc1"
