"""Bounded online predictive learning and transparent baseline evaluation.

The online learner uses only local errors. A lower bounded trend level predicts
the next scalar observation. A higher level tracks the lower residual and sends
its next residual estimate back as a top-down correction on the following step.
A separate bounded phase template learns repeated experiment structure through
local running means on development clips. There is no global backpropagation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .io.offline_eval import CorpusClip, FrameObservation, FrameProvider


@dataclass(frozen=True)
class LocalPredictionStep:
    index: int
    observed: float
    prediction: float
    residual: float
    top_down_correction: float


class LocalPredictiveHierarchy:
    """Two coupled, bounded, locally updated predictive levels."""

    def __init__(
        self,
        *,
        lower_learning_rate: float = 0.2,
        higher_learning_rate: float = 0.25,
        top_down_gain: float = 0.5,
    ) -> None:
        if not 0.0 < lower_learning_rate <= 1.0:
            raise ValueError("lower_learning_rate must be in (0, 1]")
        if not 0.0 < higher_learning_rate <= 1.0:
            raise ValueError("higher_learning_rate must be in (0, 1]")
        if not 0.0 <= top_down_gain <= 1.0:
            raise ValueError("top_down_gain must be in [0, 1]")
        self.lower_learning_rate = lower_learning_rate
        self.higher_learning_rate = higher_learning_rate
        self.top_down_gain = top_down_gain
        self._previous: float | None = None
        self._previous_delta = 0.0
        self._trend_prediction = 0.0
        self._predicted_residual = 0.0
        self._index = 0

    @property
    def state_size(self) -> int:
        """Number of scalar state values retained independent of sequence length."""

        return 5

    def update(self, observed: float) -> LocalPredictionStep | None:
        value = float(observed)
        if not math.isfinite(value):
            raise ValueError("observation must be finite")
        if self._previous is None:
            self._previous = value
            self._index += 1
            return None

        correction_limit = max(1.0, abs(self._previous) * 0.25)
        top_down = self.top_down_gain * self._predicted_residual
        combined_correction = float(
            np.clip(
                self._trend_prediction + top_down,
                -correction_limit,
                correction_limit,
            )
        )
        prediction = self._previous + combined_correction
        residual = value - prediction

        observed_delta = value - self._previous
        clipped_delta = float(
            np.clip(observed_delta, -correction_limit, correction_limit)
        )
        # Lower level learns a bounded local velocity. Huber-style clipping
        # prevents one frequency-bin jump from destabilizing future frames.
        self._trend_prediction += self.lower_learning_rate * (
            clipped_delta - self._trend_prediction
        )
        # Higher-level local residual prediction. Its output is consumed by the
        # lower level on the next step as ``top_down``.
        clipped_residual = float(
            np.clip(residual, -correction_limit, correction_limit)
        )
        self._predicted_residual += self.higher_learning_rate * (
            clipped_residual - self._predicted_residual
        )

        step = LocalPredictionStep(
            index=self._index,
            observed=value,
            prediction=prediction,
            residual=residual,
            top_down_correction=top_down,
        )
        self._previous_delta = observed_delta
        self._previous = value
        self._index += 1
        return step


def _error_summary(residuals: list[float]) -> dict[str, float | int]:
    if not residuals:
        return {"count": 0, "mae": 0.0, "rmse": 0.0}
    array = np.asarray(residuals, dtype=np.float64)
    return {
        "count": len(residuals),
        "mae": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(array * array))),
    }


def evaluate_predictor_sequence(
    values: list[float],
    *,
    ema_learning_rate: float = 0.5,
) -> dict[str, object]:
    """Prequential comparison: every model predicts before seeing each value."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        raise ValueError("predictive evaluation requires at least two finite values")
    hierarchy = LocalPredictiveHierarchy()
    persistence_residuals: list[float] = []
    ema_residuals: list[float] = []
    hierarchy_residuals: list[float] = []
    hierarchy_steps: list[dict[str, object]] = []
    previous = finite[0]
    ema = finite[0]
    hierarchy.update(finite[0])

    for value in finite[1:]:
        persistence_residuals.append(value - previous)
        ema_residuals.append(value - ema)
        step = hierarchy.update(value)
        assert step is not None
        hierarchy_residuals.append(step.residual)
        hierarchy_steps.append(asdict(step))
        previous = value
        ema += ema_learning_rate * (value - ema)

    baselines = {
        "persistence": _error_summary(persistence_residuals),
        "exponential_moving_average": _error_summary(ema_residuals),
    }
    learned = _error_summary(hierarchy_residuals)
    best_baseline_rmse = min(float(item["rmse"]) for item in baselines.values())
    learned_rmse = float(learned["rmse"])
    return {
        "mode": "prequential_local_predictive_evaluation",
        "observation_count": len(finite),
        "bounded_state_scalars": hierarchy.state_size,
        "baselines": baselines,
        "local_predictive_hierarchy": learned,
        "hierarchy_beats_best_baseline": learned_rmse < best_baseline_rmse,
        "rmse_improvement_fraction": (
            (best_baseline_rmse - learned_rmse) / best_baseline_rmse
            if best_baseline_rmse > 0.0
            else 0.0
        ),
        "steps": hierarchy_steps,
    }


class PhaseTemplatePredictor:
    """Bounded higher-level expectation learned by local phase-wise means."""

    def __init__(self, *, max_phases: int = 512) -> None:
        if max_phases <= 0:
            raise ValueError("max_phases must be positive")
        self.max_phases = max_phases
        self.means = np.zeros(max_phases, dtype=np.float64)
        self.counts = np.zeros(max_phases, dtype=np.int64)

    def observe(self, phase: int, value: float) -> None:
        if not 0 <= phase < self.max_phases:
            return
        self.counts[phase] += 1
        self.means[phase] += (float(value) - self.means[phase]) / self.counts[phase]

    def observe_sequence(self, values: list[float]) -> None:
        for phase, value in enumerate(values[: self.max_phases]):
            self.observe(phase, value)

    def predict(self, phase: int, fallback: float) -> float:
        if 0 <= phase < self.max_phases and self.counts[phase] > 0:
            return float(self.means[phase])
        return float(fallback)

    @property
    def learned_phase_count(self) -> int:
        return int(np.count_nonzero(self.counts))


def _phase_template_residuals(
    values: list[float],
    template: PhaseTemplatePredictor,
    *,
    blend: float,
) -> list[float]:
    residuals: list[float] = []
    for phase in range(1, len(values)):
        persistence = values[phase - 1]
        top_down = template.predict(phase, persistence)
        prediction = (1.0 - blend) * persistence + blend * top_down
        residuals.append(values[phase] - prediction)
    return residuals


def _select_phase_blend(dev_sequences: list[list[float]]) -> float:
    if len(dev_sequences) < 2:
        return 0.5
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    best_blend = 0.0
    best_rmse = math.inf
    for blend in candidates:
        residuals: list[float] = []
        for validation_index, validation in enumerate(dev_sequences):
            template = PhaseTemplatePredictor()
            for training_index, training in enumerate(dev_sequences):
                if training_index != validation_index:
                    template.observe_sequence(training)
            residuals.extend(
                _phase_template_residuals(validation, template, blend=blend)
            )
        rmse = float(_error_summary(residuals)["rmse"])
        if rmse < best_rmse:
            best_rmse = rmse
            best_blend = blend
    return best_blend


def evaluate_predictive_corpus(
    clips: list[CorpusClip],
    frames_for: FrameProvider,
    *,
    frame_ms: int = 250,
    feature: str = "dominant_frequency_hz",
    split: str = "held_out",
) -> dict[str, object]:
    selected = [clip for clip in clips if clip.split == split]
    dev_clips = [clip for clip in clips if clip.split == "dev"]
    if not selected:
        raise ValueError(f"predictive evaluation requires at least one {split} clip")
    sequences: dict[str, list[float]] = {}
    for clip in [*dev_clips, *selected]:
        raw_frames = frames_for(clip, frame_ms)
        observations = [
            frame
            for frame in raw_frames
            if isinstance(frame, FrameObservation)
        ]
        values = [
            float(frame.features[feature])
            for frame in observations
            if frame.features
            and frame.features.get(feature) is not None
            and math.isfinite(float(frame.features[feature]))
        ]
        sequences[clip.clip_id] = values

    dev_sequences = [
        sequences[clip.clip_id]
        for clip in dev_clips
        if len(sequences[clip.clip_id]) >= 2
    ]
    phase_blend = _select_phase_blend(dev_sequences)
    phase_template = PhaseTemplatePredictor()
    for sequence in dev_sequences:
        phase_template.observe_sequence(sequence)

    reports: list[dict[str, object]] = []
    for clip in selected:
        values = sequences[clip.clip_id]
        if len(values) < 2:
            reports.append(
                {
                    "clip_id": clip.clip_id,
                    "status": "insufficient_observations",
                    "observation_count": len(values),
                }
            )
            continue
        report = evaluate_predictor_sequence(values)
        phase_summary = _error_summary(
            _phase_template_residuals(values, phase_template, blend=phase_blend)
        )
        best_baseline_rmse = min(
            float(item["rmse"]) for item in report["baselines"].values()  # type: ignore[union-attr]
        )
        phase_rmse = float(phase_summary["rmse"])
        report["phase_template_top_down"] = phase_summary
        report["phase_template_beats_best_baseline"] = phase_rmse < best_baseline_rmse
        report["phase_template_rmse_improvement_fraction"] = (
            (best_baseline_rmse - phase_rmse) / best_baseline_rmse
            if best_baseline_rmse > 0.0
            else 0.0
        )
        report["clip_id"] = clip.clip_id
        report["status"] = "evaluated"
        reports.append(report)

    evaluated = [report for report in reports if report["status"] == "evaluated"]
    if not evaluated:
        raise ValueError("no selected clip had enough feature observations")
    online_wins = sum(
        bool(report["hierarchy_beats_best_baseline"]) for report in evaluated
    )
    wins = sum(
        bool(report["phase_template_beats_best_baseline"]) for report in evaluated
    )
    return {
        "mode": "held_out_predictive_corpus_evaluation",
        "feature": feature,
        "split": split,
        "frame_ms": frame_ms,
        "evaluated_clip_count": len(evaluated),
        "development_clip_count": len(dev_sequences),
        "selected_phase_blend": phase_blend,
        "learned_phase_count": phase_template.learned_phase_count,
        "online_local_win_count": online_wins,
        "hierarchy_win_count": wins,
        "hierarchy_win_rate": wins / len(evaluated),
        "promotion_criterion": "hierarchy_win_rate > 0.5",
        "criterion_passed": wins / len(evaluated) > 0.5,
        "reports": reports,
    }
