from __future__ import annotations

import numpy as np

from cora_infans.predictive_register_dynamics import (
    CONDITION_LABELS,
    LinearGaussianDynamics,
    PredictiveRegisterDynamicsModel,
    ProsodicSequenceExample,
    predictive_dynamics_leave_one_fieldsite_out,
    select_predictive_dynamics_thresholds,
)


def test_linear_dynamics_rewards_matching_ordered_transition() -> None:
    rising = np.column_stack(
        [np.linspace(0.0, 1.0, 20) + feature for feature in range(5)]
    )
    model = LinearGaussianDynamics.fit([rising, rising + 0.01])

    assert model.mean_log_likelihood(rising) > model.mean_log_likelihood(rising[::-1])


def _examples(fieldsite: str, offset: float) -> list[ProsodicSequenceExample]:
    examples = []
    for condition_index, label in enumerate(CONDITION_LABELS):
        time = np.linspace(0.0, 1.0, 30)
        slope = 1.0 if label.startswith("infant") else -1.0
        mode = 2.0 if label.endswith("speech") else -2.0
        states = np.column_stack(
            [
                offset + mode + slope * time * (feature + 1)
                for feature in range(5)
            ]
        )
        examples.append(
            ProsodicSequenceExample(
                clip_id=f"{fieldsite}01{'ABCD'[condition_index]}",
                speaker_id=f"{fieldsite}01",
                fieldsite=fieldsite,
                label=label,
                states=states,
                vocal_score=0.5,
            )
        )
    return examples


def test_predictive_dynamics_uses_grouped_folds_and_frozen_thresholds() -> None:
    examples = [
        *_examples("AAA", 0.0),
        *_examples("BBB", 5.0),
        *_examples("CCC", 10.0),
    ]
    model = PredictiveRegisterDynamicsModel.fit(examples[:8])
    batch = model.evaluate_batch(examples[8:])
    report = predictive_dynamics_leave_one_fieldsite_out(examples)
    selection = select_predictive_dynamics_thresholds(report["predictions"])

    assert batch["calibrated_prediction_count"] == 4
    assert report["fieldsite_count"] == 3
    assert report["prediction_count"] == 12
    assert report["confirmation_audio_or_metadata_accessed"] is False
    assert report["promotion_allowed"] is False
    assert selection["protocol"] == "predictive_dynamics_out_of_field_threshold_selection"
    assert selection["metrics"]["false_positive_rate"] <= 0.15
