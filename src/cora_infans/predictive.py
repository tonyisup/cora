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


@dataclass(frozen=True)
class FocusTrajectoryUpdate:
    previous_prediction_label: str | None
    observed_label: str
    residual_miss: bool
    updated_prediction_label: str


class FocusTrajectoryCoder:
    """Higher predictive-coding level: predicts the next focus label.

    This is a deliberately minimal top-down predictor over the focus trajectory
    (which channel attention moves to next). It keeps a tally of the last
    observed label and predicts a repeat of it. The residual is binary: a hit
    (predicted label matches observation) or a miss (prediction error to be
    passed upward). No spatial physics, no global backprop -- just a local
    error signal over the auditory attention path, as the vision's hierarchy
    requires without overclaiming biological fidelity.
    """

    def __init__(self, channels: tuple[str, ...], repeat_bias: bool = True) -> None:
        self.channels = channels
        self.repeat_bias = repeat_bias
        self._last_label: str | None = None

    def update(self, observed_focus_label: str) -> FocusTrajectoryUpdate:
        previous_prediction_label = self._last_label
        if previous_prediction_label is None:
            residual_miss = True
        elif self.repeat_bias and previous_prediction_label == observed_focus_label:
            residual_miss = False
        else:
            residual_miss = previous_prediction_label != observed_focus_label

        self._last_label = observed_focus_label
        return FocusTrajectoryUpdate(
            previous_prediction_label=previous_prediction_label,
            observed_label=observed_focus_label,
            residual_miss=residual_miss,
            updated_prediction_label=observed_focus_label,
        )


class PredictiveHierarchy:
    """Two-level predictive-coding hierarchy over one frame observation.

    Lower level: frequency residual (existing PredictiveCoder behavior).
    Higher level: focus-trajectory prediction (FocusTrajectoryCoder).

    Each level is an independent local error minimizer. The hierarchy only
    bundles them so the pipeline can advance both streams on a single salient
    frame without faking a deeper stack than actually exists.
    """

    def __init__(
        self,
        frequency_coder: PredictiveCoder,
        focus_coder: FocusTrajectoryCoder,
    ) -> None:
        self.frequency_coder = frequency_coder
        self.focus_coder = focus_coder

    def update_frequency(self, observed_frequency_hz: float) -> PredictionUpdate:
        return self.frequency_coder.update(observed_frequency_hz)

    def update_focus(self, observed_focus_label: str) -> FocusTrajectoryUpdate:
        return self.focus_coder.update(observed_focus_label)
