from cora_infans.pipeline import AudioFrame, CoraInfansPipeline


def test_pipeline_emits_workspace_event_when_salient() -> None:
    pipeline = CoraInfansPipeline(saliency_threshold=0.7)

    event = pipeline.process_frame(
        AudioFrame(
            timestamp_ms=125,
            channel_energies=[0.2, 0.9, 0.4, 0.1],
            dominant_frequency_hz=1900.0,
        )
    )

    assert event is not None
    assert event.kind == "salient_audio"
    assert event.saliency_score == 0.9
    assert event.workspace_state["focus_channel"] == 1


def test_pipeline_drops_non_salient_frame() -> None:
    pipeline = CoraInfansPipeline(saliency_threshold=0.7)

    event = pipeline.process_frame(
        AudioFrame(
            timestamp_ms=250,
            channel_energies=[0.2, 0.3, 0.4, 0.1],
            dominant_frequency_hz=900.0,
        )
    )

    assert event is None
