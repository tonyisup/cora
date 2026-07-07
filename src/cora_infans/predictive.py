from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionUpdate:
    previous_prediction_hz: float
    residual_hz: float
    updated_prediction_hz: float


class PredictiveCoder:
    def __init__(self, initial_prediction_hz: float, learning_rate: float = 0.5) -> None:
        self.prediction_hz = initial_prediction_hz
        self.learning_rate = learning_rate

    def update(self, observed_frequency_hz: float) -> PredictionUpdate:
        previous_prediction_hz = self.prediction_hz
        residual_hz = observed_frequency_hz - previous_prediction_hz
        self.prediction_hz = previous_prediction_hz + (residual_hz * self.learning_rate)
        return PredictionUpdate(
            previous_prediction_hz=previous_prediction_hz,
            residual_hz=residual_hz,
            updated_prediction_hz=self.prediction_hz,
        )
