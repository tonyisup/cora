"""Contract tests for the preregistered H3 same-source forward model."""

import numpy as np
import pytest

from cora_infans.same_source_forward_model import (
    SameSourceForwardModel,
    _validate_result_modes,
    centered_log_bands,
    forward_distance,
    raw_efference_distance,
)


def test_running_mean_learns_one_shared_path_signature() -> None:
    first = np.linspace(-1.0, 1.0, 12)
    first -= np.mean(first)
    second = np.cos(np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False))
    second -= np.mean(second)
    path = np.linspace(-0.3, 0.3, 12)
    path -= np.mean(path)
    model = SameSourceForwardModel()

    model.update(first, first + path)
    model.update(second, second + path)

    assert model.count == 2
    assert model.state_scalars == 13
    np.testing.assert_allclose(model.path_signature, path, atol=1e-12)
    assert forward_distance(
        model, command_centered=first, current_centered=first + path
    ) == pytest.approx(0.0, abs=1e-12)


def test_forward_model_generalizes_to_unseen_command_vector() -> None:
    path = np.sin(np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)) * 0.2
    model = SameSourceForwardModel()
    for offset in (0.0, 0.3, 0.7):
        command = np.cos(
            np.linspace(offset, offset + 2.0 * np.pi, 12, endpoint=False)
        )
        command -= np.mean(command)
        model.update(command, command + path)
    unseen = np.linspace(0.8, -0.7, 12)
    unseen -= np.mean(unseen)

    predicted = forward_distance(
        model, command_centered=unseen, current_centered=unseen + path
    )
    raw = raw_efference_distance(
        command_centered=unseen, current_centered=unseen + path
    )

    assert predicted < 1e-12
    assert predicted < raw


def test_log_band_shape_is_gain_invariant() -> None:
    rng = np.random.default_rng(4401)
    time = np.arange(9600) / 16_000.0
    waveform = (
        np.sin(2.0 * np.pi * 173.0 * time)
        + 0.4 * np.sin(2.0 * np.pi * 911.0 * time)
        + 0.2 * np.sin(2.0 * np.pi * 2401.0 * time)
        + 0.01 * rng.standard_normal(len(time))
    ).astype(np.float32)

    np.testing.assert_allclose(
        centered_log_bands(waveform),
        centered_log_bands(waveform * 0.13),
        atol=1e-6,
    )


def test_model_rejects_use_before_calibration() -> None:
    with pytest.raises(ValueError, match="no calibration"):
        SameSourceForwardModel().predict(np.arange(12, dtype=np.float64))


def test_retained_result_modes_fail_closed() -> None:
    _validate_result_modes(
        {"mode": "natural_speech_workspace_template_v2_real_failure_audit"},
        {"mode": "natural_speech_workspace_spectral_v3_real_result"},
    )
    with pytest.raises(ValueError, match="template-v2"):
        _validate_result_modes(
            {"mode": "natural_speech_workspace_template_v2_protocol"},
            {"mode": "natural_speech_workspace_spectral_v3_real_result"},
        )
