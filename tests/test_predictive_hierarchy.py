"""Failing tests for the next predictive-coding slice: a two-level hierarchy.

Lower level (already shipped): residual against predicted dominant frequency.
Higher level (this slice): predicts the next focus label from the recent focus
trajectory and computes its own residual when the prediction misses.

Both streams stay honest: each is a deterministic local error minimizer, not a
global backprop. The higher level does not invent spatial physics; it only
models "which channel the infant's attention is likely to move to next" as a
simple last-k transition tally.
"""

from cora_infans.predictive import (
    FocusTrajectoryCoder,
    PredictiveCoder,
    PredictiveHierarchy,
)


def test_focus_trajectory_coder_predicts_most_recent_focus_when_no_history() -> None:
    coder = FocusTrajectoryCoder(channels=("a", "b", "c", "d"))

    update = coder.update(observed_focus_label="b")

    # With no history, the prior is "no prediction"; residual is a full miss.
    assert update.previous_prediction_label is None
    assert update.observed_label == "b"
    assert update.residual_miss is True
    assert update.updated_prediction_label == "b"


def test_focus_trajectory_coder_learns_simple_repeat() -> None:
    coder = FocusTrajectoryCoder(channels=("a", "b", "c", "d"))

    coder.update(observed_focus_label="a")
    coder.update(observed_focus_label="a")
    update = coder.update(observed_focus_label="a")

    # After two repeats, the coder predicts "a" and the third repeat is a hit.
    assert update.previous_prediction_label == "a"
    assert update.observed_label == "a"
    assert update.residual_miss is False


def test_focus_trajectory_coder_handles_alternation() -> None:
    coder = FocusTrajectoryCoder(channels=("a", "b", "c", "d"))

    coder.update(observed_focus_label="a")
    coder.update(observed_focus_label="b")
    coder.update(observed_focus_label="a")
    coder.update(observed_focus_label="b")
    update = coder.update(observed_focus_label="a")

    # Alternating a/b: last observed was b, so it predicts b; a is a miss.
    assert update.previous_prediction_label == "b"
    assert update.observed_label == "a"
    assert update.residual_miss is True


def test_predictive_hierarchy_carries_both_streams() -> None:
    hierarchy = PredictiveHierarchy(
        frequency_coder=PredictiveCoder(initial_prediction_hz=1000.0, learning_rate=0.5),
        focus_coder=FocusTrajectoryCoder(channels=("a", "b", "c", "d")),
    )

    lower = hierarchy.update_frequency(observed_frequency_hz=1200.0)
    upper = hierarchy.update_focus(observed_focus_label="a")

    assert lower.residual_hz == 200.0
    assert upper.observed_label == "a"
    assert upper.residual_miss is True  # no history yet
