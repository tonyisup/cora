from cora_infans.io.mic_array import SyntheticMicArray



def test_synthetic_mic_array_emits_audio_frame_for_four_channels() -> None:
    mic_array = SyntheticMicArray(channel_count=4)

    frame = mic_array.capture_frame(
        timestamp_ms=10,
        channel_energies=[0.1, 0.3, 0.8, 0.2],
        dominant_frequency_hz=1500.0,
    )

    assert frame.timestamp_ms == 10
    assert frame.channel_energies == [0.1, 0.3, 0.8, 0.2]
    assert frame.dominant_frequency_hz == 1500.0



def test_synthetic_mic_array_rejects_wrong_channel_count() -> None:
    mic_array = SyntheticMicArray(channel_count=4)

    try:
        mic_array.capture_frame(
            timestamp_ms=10,
            channel_energies=[0.1, 0.3, 0.8],
            dominant_frequency_hz=1500.0,
        )
    except ValueError as exc:
        assert "4 channels" in str(exc)
    else:
        raise AssertionError("expected ValueError for wrong channel count")
