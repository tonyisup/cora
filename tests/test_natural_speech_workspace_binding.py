from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

from cora_infans.natural_speech_workspace_binding import (
    PAIRS,
    audit_natural_speech_preregistration,
    build_natural_speech_protocol,
    build_natural_speech_trials,
    prepare_natural_speech_panel,
    render_natural_speech_tracks,
    run_natural_speech_acquisition,
    run_spent_natural_speech_design_gate,
)
from cora_infans.io.motor_plan import write_wav
from cora_infans.io.wav_frames import read_pcm_wav


SPENT_ROOT = Path("/private/tmp/cora-ids-spent-200")


def _ready_preregistration(hashes):
    return {
        "mode": "natural_speech_workspace_binding_real_preregistration",
        "status": "preregistered_before_live_execution",
        "bounded_playback_capture_authorized": True,
        "spent_design_gate_passed": True,
        "frozen_hashes": dict(hashes),
        "hardware": {
            "input": {
                "avfoundation_device": ":0",
                "device": "reSpeaker XVF3800 4-Mic Array",
                "channels": 2,
                "sample_rate": 16000,
            },
            "self_source": {
                "output_device_index": 3,
                "output_device": "Mac mini Speakers",
                "placement": "centered_directly_below_xvf3800",
                "vertical_distance_cm": 4,
                "system_volume_percent": 100,
                "playback_gain_db": -18.0,
            },
            "world_source": {
                "output_device_index": 2,
                "output_device": "External Headphones",
                "angle_degrees": 180,
                "distance_cm": 50,
                "hardware_volume": "approximately_50_percent_mechanical_travel",
                "playback_gain_db": -18.0,
            },
            "environment": {"room": "semi-quiet", "known_background": "laptop fan"},
        },
        "bounds": {
            "trial_count": 15,
            "per_trial_playback_seconds_max": 2.0,
            "total_playback_seconds_max": 25.0,
            "total_capture_seconds_max": 45.0,
            "planned_playback_seconds": 19.5,
            "planned_capture_seconds": 28.5,
        },
        "acceptance": {
            "selected_capacity": 3,
            "scheduled_f1_min": 0.9,
            "scheduled_minus_flat_f1_min": 0.2,
            "scheduled_world_recall_min": 0.9,
            "scheduled_self_suppression_specificity_min": 0.9,
            "fixed_window_world_recall_must_be_below_scheduled": True,
            "capacity_one_f1_must_be_below_capacity_three": True,
            "capacity_three_f1_must_equal_capacity_four": True,
            "all_15_captures_and_alignments_required": True,
            "no_clipped_capture_samples_required": True,
        },
        "unresolved_fields": [],
    }


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


def test_live_preregistration_audit_fails_closed() -> None:
    hashes = {"implementation_sha256": "abc"}
    ready = audit_natural_speech_preregistration(
        _ready_preregistration(hashes), current_hashes=hashes
    )
    assert ready["ready"] is True

    changed = _ready_preregistration(hashes)
    changed["hardware"]["world_source"]["distance_cm"] = 51
    blocked = audit_natural_speech_preregistration(changed, current_hashes=hashes)
    assert blocked["ready"] is False
    assert "world route mismatch: distance_cm" in blocked["blockers"]


@pytest.mark.skipif(not SPENT_ROOT.exists(), reason="external spent corpus unavailable")
def test_injected_acquisition_preserves_tracks_and_scores(tmp_path) -> None:
    playback_seen = Event()

    def playback(_path):
        playback_seen.set()
        return {"backend": "injected_playback", "started": True}

    def capture(path, *, sample_rate, duration_ms, ready_event, capture_timeline, **_kwargs):
        playback_seen.clear()
        capture_timeline["frame_position_fn"] = lambda: sample_rate // 10
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        assert playback_seen.wait(timeout=1.0)
        time.sleep(0.005)
        trial_id = Path(path).name.removesuffix("_capture.wav")
        self_pcm, self_rate = read_pcm_wav(tmp_path / f"{trial_id}_self.wav")
        world_pcm, world_rate = read_pcm_wav(tmp_path / f"{trial_id}_world.wav")
        assert self_rate == world_rate == sample_rate
        delay = sample_rate // 10
        self_impulse = np.zeros(128)
        self_impulse[0], self_impulse[61] = 0.35, 0.07
        world_impulse = np.zeros(128)
        world_impulse[0], world_impulse[79] = 0.25, 0.04
        observed = np.convolve(np.mean(self_pcm, axis=1), self_impulse)
        observed += np.convolve(np.mean(world_pcm, axis=1), world_impulse)
        captured = np.concatenate((np.zeros(delay), observed, np.zeros(sample_rate // 10)))
        write_wav(path, captured, sample_rate)
        return {
            "backend": "injected_capture",
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "completed": True,
        }

    report = run_natural_speech_acquisition(
        root=SPENT_ROOT,
        out_dir=tmp_path,
        self_playback=playback,
        world_playback=playback,
        capture_fn=capture,
    )

    assert report["verdict"] == "injected_acquisition_complete"
    assert report["capture_acceptance_verdict"] == "capture_acceptance_passed"
    assert report["completed_trial_count"] == 15
    assert report["promotion_allowed"] is False
    assert len(report["artifacts"]) == 15
