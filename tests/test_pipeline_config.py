from cora_infans.config import CoraInfansConfig
from cora_infans.pipeline import AudioFrame, CoraInfansPipeline



def test_pipeline_uses_config_threshold_when_no_explicit_threshold_is_passed() -> None:
    pipeline = CoraInfansPipeline(
        config=CoraInfansConfig(saliency_threshold=0.95)
    )

    event = pipeline.process_frame(
        AudioFrame(
            timestamp_ms=500,
            channel_energies=[0.2, 0.9, 0.4, 0.1],
            dominant_frequency_hz=1700.0,
        )
    )

    assert event is None



def test_pipeline_explicit_threshold_overrides_config() -> None:
    pipeline = CoraInfansPipeline(
        saliency_threshold=0.7,
        config=CoraInfansConfig(saliency_threshold=0.95),
    )

    event = pipeline.process_frame(
        AudioFrame(
            timestamp_ms=500,
            channel_energies=[0.2, 0.9, 0.4, 0.1],
            dominant_frequency_hz=1700.0,
        )
    )

    assert event is not None
    assert event.saliency_score == 0.9
