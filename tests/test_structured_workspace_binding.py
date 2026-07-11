import json
from pathlib import Path
from threading import Barrier

import numpy as np

from cora_infans.structured_workspace_binding import (
    WorkspaceBindingConfig,
    build_workspace_binding_protocol,
    build_workspace_binding_trials,
    evaluate_workspace_binding_captures,
    play_workspace_tracks_concurrently,
    render_workspace_binding_tracks,
    run_fake_workspace_binding_protocol,
)


def test_workspace_binding_protocol_is_bounded_and_label_scoped() -> None:
    protocol = build_workspace_binding_protocol()

    assert protocol["trial_count"] == 15
    assert protocol["maximum_total_playback_seconds"] < 25.0
    assert protocol["live_io_wired"] is False
    assert protocol["capacity_sweep"] == [1, 2, 3, 4]


def test_fake_workspace_binding_exposes_baselines_and_capacity() -> None:
    report = run_fake_workspace_binding_protocol()

    selected = report["reports"]["3"]
    assert report["verdict"] == "fake_io_plumbing_passed"
    assert report["promotion_allowed"] is False
    assert selected["scheduled_identity_plus_world_evidence"]["metrics"]["f1"] == 1.0
    assert selected["flat"]["metrics"]["f1"] == 0.8
    assert selected["fixed_self_window"]["metrics"]["world_recall"] == 0.5
    assert report["reports"]["1"]["scheduled_identity_plus_world_evidence"]["metrics"]["f1"] < 1.0


def test_overlap_tracks_keep_sources_separate_and_choose_self_chirp() -> None:
    config = WorkspaceBindingConfig()
    overlap = next(
        trial
        for trial in build_workspace_binding_trials()
        if trial.condition == "overlap"
    )
    rendered = render_workspace_binding_tracks(overlap, config)
    start, end = rendered.stimulus_window
    chirp_end = int(config.alignment_chirp_ms * config.sample_rate / 1000)

    assert rendered.chirp_source == "self"
    assert np.any(rendered.self_pcm[:chirp_end])
    assert not np.any(rendered.world_pcm[:chirp_end])
    assert np.any(rendered.self_pcm[start:end])
    assert np.any(rendered.world_pcm[start:end])


def test_explicit_source_playbacks_start_concurrently() -> None:
    barrier = Barrier(2)

    def playback(path):
        barrier.wait(timeout=1.0)
        return {"started": True, "path": path}

    report = play_workspace_tracks_concurrently(
        self_path="self.wav",
        world_path="world.wav",
        self_playback=playback,
        world_playback=playback,
    )

    assert report["started"] is True
    assert set(report["sources"]) == {"self", "world"}


def test_chirp_aligned_capture_evaluator_is_device_agnostic() -> None:
    config = WorkspaceBindingConfig()
    captures = {}
    alignments = {}
    delay = 320
    for trial in build_workspace_binding_trials():
        rendered = render_workspace_binding_tracks(trial, config)
        combined = rendered.self_pcm + rendered.world_pcm
        capture = np.zeros(len(combined) + delay + 32, dtype=np.float32)
        capture[delay : delay + len(combined)] = 0.3 * combined
        captures[trial.trial_id] = np.column_stack((capture, capture))
        alignments[trial.trial_id] = delay

    report = evaluate_workspace_binding_captures(
        captures=captures,
        alignment_samples=alignments,
        config=config,
    )

    assert report["verdict"] == "capture_acceptance_passed"
    assert report["reports"]["3"]["scheduled_identity_plus_world_evidence"][
        "metrics"
    ]["f1"] == 1.0


def test_retained_structured_workspace_result_preserves_failed_first_rule() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-10-structured-workspace-binding-fake-result.json"
        ).read_text()
    )

    assert result["promotion_allowed"] is False
    assert result["verdict"] == "fake_io_plumbing_passed"
    assert result["design_history"]["initial_midpoint_rule"]["scheduled_capacity_three_f1"] == 0.8
    assert result["reports"]["3"]["scheduled_identity_plus_world_evidence"]["metrics"]["f1"] == 1.0
    assert result["reports"]["3"]["fixed_self_window"]["metrics"]["world_recall"] == 0.5
