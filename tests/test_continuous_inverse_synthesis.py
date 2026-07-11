"""Focused contract tests for preregistered H4 continuous inverse synthesis."""

import numpy as np
import pytest

from cora_infans.continuous_inverse_synthesis import (
    MotorCoordinates,
    infer_motor_coordinates,
    nearest_motor_identity,
    spectral_shape,
    synthesize_motor_coordinates,
)
from cora_infans.io.motor_plan import VOWEL_FORMANTS_HZ, synthesize_vowel


def test_canonical_coordinates_preserve_existing_synthesizer() -> None:
    for vowel in ("a", "e", "i", "o", "u"):
        f1, f2, f3 = VOWEL_FORMANTS_HZ[vowel]
        continuous = synthesize_motor_coordinates(
            MotorCoordinates(110.0, f1, f2, f3), duration_ms=300
        )
        existing = synthesize_vowel(
            vowel, duration_ms=300, sample_rate=16_000, f0_hz=110.0
        )
        np.testing.assert_array_equal(continuous, existing)


def test_continuous_coordinates_produce_bounded_non_silent_audio() -> None:
    waveform = synthesize_motor_coordinates(
        MotorCoordinates(137.0, 645.0, 1515.0, 2765.0), duration_ms=180
    )

    assert waveform.shape == (2880,)
    assert waveform.dtype == np.float32
    assert 0.0 < float(np.max(np.abs(waveform))) <= 1.0


def test_continuous_synthesizer_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="frozen bounds"):
        synthesize_motor_coordinates(
            MotorCoordinates(110.0, 900.0, 1000.0, 2400.0), duration_ms=300
        )


def test_spectral_shape_is_gain_invariant() -> None:
    waveform = synthesize_vowel("e", duration_ms=300, f0_hz=140.0)

    np.testing.assert_allclose(
        spectral_shape(waveform), spectral_shape(waveform * 0.17), atol=2e-7
    )


def test_inverse_synthesis_obeys_frozen_search_budget() -> None:
    waveform = synthesize_motor_coordinates(
        MotorCoordinates(140.0, 650.0, 1450.0, 2700.0), duration_ms=300
    )

    result = infer_motor_coordinates(waveform)

    assert result.synthesis_count <= 23
    assert result.final_loss <= result.initial_loss
    assert result.final_identity in VOWEL_FORMANTS_HZ
    assert result.reperceived_identity in VOWEL_FORMANTS_HZ


def test_post_hoc_identity_uses_continuous_motor_coordinates() -> None:
    plan = MotorCoordinates(110.0, 280.0, 2210.0, 2980.0)

    assert nearest_motor_identity(plan) == "i"
