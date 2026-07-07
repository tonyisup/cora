from cora_infans.predictive import PredictiveCoder



def test_predictive_coder_reports_residual_against_previous_prediction() -> None:
    coder = PredictiveCoder(initial_prediction_hz=1000.0, learning_rate=0.5)

    result = coder.update(observed_frequency_hz=1200.0)

    assert result.previous_prediction_hz == 1000.0
    assert result.residual_hz == 200.0
    assert result.updated_prediction_hz == 1100.0



def test_predictive_coder_second_update_uses_updated_state() -> None:
    coder = PredictiveCoder(initial_prediction_hz=1000.0, learning_rate=0.5)

    coder.update(observed_frequency_hz=1200.0)
    result = coder.update(observed_frequency_hz=900.0)

    assert result.previous_prediction_hz == 1100.0
    assert result.residual_hz == -200.0
    assert result.updated_prediction_hz == 1000.0
