"""Pipeline integration for the two-level predictive hierarchy.

Confirms the hierarchy path emits both the lower (frequency) residual and the
higher (focus-trajectory) residual without disturbing the existing
single-coder contract.
"""

from cora_infans.config import CoraInfansConfig
from cora_infans.io.mic_array import SyntheticMicArray
from cora_infans.pipeline import CoraInfansPipeline
from cora_infans.predictive import FocusTrajectoryCoder, PredictiveCoder, PredictiveHierarchy


def _make_pipeline() -> CoraInfansPipeline:
    return CoraInfansPipeline(
        config=CoraInfansConfig(
            saliency_threshold=0.7,
            workspace_window_ms=250,
            mic_channel_geometry=(
                "front_left",
                "front_right",
                "rear_right",
                "rear_left",
            ),
        ),
        predictive_hierarchy=PredictiveHierarchy(
            frequency_coder=PredictiveCoder(initial_prediction_hz=1000.0, learning_rate=0.5),
            focus_coder=FocusTrajectoryCoder(
                channels=(
                    "front_left",
                    "front_right",
                    "rear_right",
                    "rear_left",
                ),
            ),
        ),
    )


def test_hierarchy_pipeline_emits_lower_and_higher_residuals() -> None:
    mic_array = SyntheticMicArray(channel_count=4)
    pipeline = _make_pipeline()

    # First salient frame: front_right. Higher level has no prior -> miss.
    frame = mic_array.capture_frame(
        timestamp_ms=125,
        channel_energies=[0.2, 0.9, 0.4, 0.1],
        dominant_frequency_hz=1200.0,
    )
    event = pipeline.process_frame(frame)
    assert event is not None
    state = event.workspace_state
    assert state["predictive_residual_hz"] == 200.0
    assert state["updated_prediction_hz"] == 1100.0
    assert state["higher_predictive_previous_label"] == ""
    assert state["higher_predictive_residual_miss"] is True

    # Second salient frame on the same channel: higher level predicts repeat -> hit.
    frame2 = mic_array.capture_frame(
        timestamp_ms=175,
        channel_energies=[0.2, 0.9, 0.4, 0.1],
        dominant_frequency_hz=900.0,
    )
    event2 = pipeline.process_frame(frame2)
    assert event2 is not None
    state2 = event2.workspace_state
    assert state2["higher_predictive_previous_label"] == "front_right"
    assert state2["higher_predictive_residual_miss"] is False
