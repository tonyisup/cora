from pathlib import Path

import numpy as np
import pytest

from cora_infans.natural_speech_workspace_template_v2 import (
    PAIRS,
    build_template_protocol,
    build_template_trials,
    prepare_template_panel,
    render_template_tracks,
    run_spent_template_design_gate,
)


SPENT_ROOT = Path("/private/tmp/cora-ids-spent-200")


def test_template_protocol_is_bounded_and_calibration_is_not_scored() -> None:
    protocol = build_template_protocol()
    trials = build_template_trials()

    assert protocol["trial_count"] == 21
    assert protocol["held_out_scored_trial_count"] == 9
    assert protocol["maximum_total_playback_seconds"] == pytest.approx(19.32)
    assert all(
        trial.scored is False
        for trial in trials
        if trial.condition == "self_calibration"
    )
    assert all(
        trial.split == "held_out"
        for trial in trials
        if trial.condition == "overlap"
    )


@pytest.mark.skipif(not SPENT_ROOT.exists(), reason="external spent corpus unavailable")
def test_template_panel_hashes_rms_normalizes_and_renders() -> None:
    prepared = prepare_template_panel(SPENT_ROOT)
    overlap = next(
        trial for trial in build_template_trials() if trial.condition == "overlap"
    )
    rendered = render_template_tracks(overlap, prepared)
    start, end = rendered.stimulus_window

    assert len(prepared) == len(PAIRS) == 6
    assert rendered.chirp_source == "self"
    assert np.max(np.abs(rendered.self_pcm[: rendered.chirp_window[1]])) == pytest.approx(0.12)
    assert not np.any(rendered.world_pcm[: rendered.chirp_window[1]])
    assert np.sqrt(np.mean(rendered.self_pcm[start:end] ** 2)) <= 0.04 + 1e-7
    assert np.sqrt(np.mean(rendered.world_pcm[start:end] ** 2)) <= 0.10 + 1e-7
    assert np.max(np.abs(rendered.self_pcm[start:end])) <= 0.20 + 1e-7
    assert np.max(np.abs(rendered.world_pcm[start:end])) <= 0.80 + 1e-7


@pytest.mark.skipif(not SPENT_ROOT.exists(), reason="external spent corpus unavailable")
def test_template_spent_design_gate_passes_without_promotion() -> None:
    report = run_spent_template_design_gate(SPENT_ROOT)
    selected = report["reports"]["3"]

    assert report["verdict"] == "spent_design_gate_passed"
    assert report["promotion_allowed"] is False
    assert all(report["criteria"].values())
    assert selected["scheduled_identity_plus_template_residual"]["metrics"]["f1"] == 1.0
    assert selected["flat"]["metrics"]["f1"] == 0.8
    assert selected["fixed_self_window"]["metrics"]["world_recall"] == 0.5
    assert (
        report["reports"]["1"]["scheduled_identity_plus_template_residual"][
            "metrics"
        ]["f1"]
        < 1.0
    )
