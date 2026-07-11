from pathlib import Path

import numpy as np
import pytest

from cora_infans.natural_speech_workspace_binding import (
    PAIRS,
    build_natural_speech_protocol,
    build_natural_speech_trials,
    prepare_natural_speech_panel,
    render_natural_speech_tracks,
    run_spent_natural_speech_design_gate,
)


SPENT_ROOT = Path("/private/tmp/cora-ids-spent-200")


def test_natural_speech_protocol_keeps_overlap_held_out() -> None:
    protocol = build_natural_speech_protocol()
    trials = build_natural_speech_trials()

    assert protocol["trial_count"] == 15
    assert protocol["maximum_total_playback_seconds"] == 19.5
    assert all(
        trial.split == "held_out"
        for trial in trials
        if trial.condition == "overlap"
    )
    assert protocol["promotion_allowed"] is False


@pytest.mark.skipif(not SPENT_ROOT.exists(), reason="external spent corpus unavailable")
def test_spent_panel_hashes_and_excerpt_rendering() -> None:
    prepared = prepare_natural_speech_panel(SPENT_ROOT)
    overlap = next(
        trial
        for trial in build_natural_speech_trials()
        if trial.condition == "overlap"
    )
    rendered = render_natural_speech_tracks(overlap, prepared)
    start, end = rendered.stimulus_window

    assert len(prepared) == len(PAIRS) == 6
    assert rendered.chirp_source == "self"
    assert np.any(rendered.self_pcm[:start])
    assert not np.any(rendered.world_pcm[:start])
    assert np.max(np.abs(rendered.self_pcm[start:end])) == pytest.approx(0.12)
    assert np.max(np.abs(rendered.world_pcm[start:end])) == pytest.approx(0.8)


@pytest.mark.skipif(not SPENT_ROOT.exists(), reason="external spent corpus unavailable")
def test_preregistered_spent_design_gate_passes_without_promotion() -> None:
    report = run_spent_natural_speech_design_gate(SPENT_ROOT)
    selected = report["reports"]["3"]

    assert report["verdict"] == "spent_design_gate_passed"
    assert report["promotion_allowed"] is False
    assert all(report["criteria"].values())
    assert selected["scheduled_identity_plus_world_residual"]["metrics"]["f1"] == 1.0
    assert selected["flat"]["metrics"]["f1"] == 0.8
    assert selected["fixed_self_window"]["metrics"]["world_recall"] == 0.5
    assert (
        report["reports"]["1"]["scheduled_identity_plus_world_residual"][
            "metrics"
        ]["f1"]
        < 1.0
    )
