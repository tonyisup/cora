from cora_infans.config import CoraInfansConfig
from cora_infans.io.mic_array import SyntheticMicArray
from cora_infans.pipeline import CoraInfansPipeline
from cora_infans.predictive import PredictiveCoder



def test_pipeline_records_workspace_event_with_predictive_residual() -> None:
    mic_array = SyntheticMicArray(channel_count=4)
    predictive_coder = PredictiveCoder(initial_prediction_hz=1000.0, learning_rate=0.5)
    pipeline = CoraInfansPipeline(
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
        predictive_coder=predictive_coder,
    )

    frame = mic_array.capture_frame(
        timestamp_ms=125,
        channel_energies=[0.2, 0.9, 0.4, 0.1],
        dominant_frequency_hz=1200.0,
    )

    event = pipeline.process_frame(frame)

    assert event is not None
    assert event.workspace_state["focus_channel"] == 1
    assert event.workspace_state["focus_label"] == "front_right"
    assert event.workspace_state["saliency_reward"] == 0.5
    assert event.workspace_state["predictive_residual_hz"] == 200.0
    assert event.workspace_state["updated_prediction_hz"] == 1100.0

    snapshot = pipeline.workspace.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0] == event
