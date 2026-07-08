"""Tests for the XVF3800 linearity calibration probe (hardware-gated)."""

from cora_infans.io.calibration import calibrate_linearity
from cora_infans.io.hardware_tuner import HardwareTuner, HardwareTunerError


def test_calibrate_runs_on_real_device() -> None:
    # Hardware-gated: only runs when the XVF3800 is present + tuner importable.
    try:
        HardwareTuner()
    except (HardwareTunerError, Exception):  # noqa: BLE001 - no device here
        import pytest

        pytest.skip("XVF3800 not present in this environment")

    result = calibrate_linearity(levels=(0.2, 0.4), freq_hz=500.0)
    assert result["levels"] == [0.2, 0.4]
    assert len(result["factory_rms"]) == 2
    assert len(result["raw_linear_rms"]) == 2
    # Both firmware states must have produced positive, finite RMS readings.
    for v in result["factory_rms"] + result["raw_linear_rms"]:
        assert v > 0.0 and v < 1.0
