from cora_infans.config import CoraInfansConfig



def test_default_config_matches_first_slice_contract() -> None:
    config = CoraInfansConfig()

    assert config.mic_channel_count == 4
    assert config.mic_channel_geometry == (
        "channel_0",
        "channel_1",
        "channel_2",
        "channel_3",
    )
    assert config.saliency_threshold == 0.7
    assert config.workspace_window_ms == 250



def test_config_allows_threshold_override() -> None:
    config = CoraInfansConfig(saliency_threshold=0.85)

    assert config.saliency_threshold == 0.85



def test_config_rejects_geometry_length_that_does_not_match_channel_count() -> None:
    try:
        CoraInfansConfig(
            mic_channel_count=4,
            mic_channel_geometry=("left", "right"),
        )
    except ValueError as exc:
        assert "geometry" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched geometry")
