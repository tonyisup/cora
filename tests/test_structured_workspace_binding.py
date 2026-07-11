import json
from pathlib import Path
from threading import Barrier, Event
import time

import numpy as np

from cora_infans.io.motor_plan import write_wav
from cora_infans.io.wav_frames import read_pcm_wav
from cora_infans.structured_workspace_binding import (
    WorkspaceBindingConfig,
    build_workspace_binding_protocol,
    build_workspace_binding_trials,
    evaluate_workspace_binding_captures,
    play_workspace_tracks_concurrently,
    render_workspace_binding_tracks,
    run_fake_workspace_binding_protocol,
    run_workspace_binding_acquisition,
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


def test_injected_workspace_acquisition_retains_all_tracks_and_capture(tmp_path) -> None:
    config = WorkspaceBindingConfig()
    playback_seen = Event()
    playback_calls = []

    def playback(path):
        playback_calls.append(path)
        playback_seen.set()
        return {"backend": "injected_playback", "started": True, "path": path}

    def capture(
        path,
        *,
        sample_rate,
        duration_ms,
        ready_event,
        capture_timeline,
        **_kwargs,
    ):
        playback_seen.clear()
        capture_timeline["frame_position_fn"] = lambda: sample_rate // 10
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        assert playback_seen.wait(timeout=1.0)
        time.sleep(0.01)
        trial_id = Path(path).name.removesuffix("_capture.wav")
        self_pcm, self_rate = read_pcm_wav(tmp_path / f"{trial_id}_self.wav")
        world_pcm, world_rate = read_pcm_wav(tmp_path / f"{trial_id}_world.wav")
        assert self_rate == world_rate == sample_rate
        combined = np.mean(self_pcm + world_pcm, axis=1)
        delay = sample_rate // 10
        observed = np.zeros(delay + len(combined) + sample_rate // 10)
        observed[delay : delay + len(combined)] = 0.3 * combined
        write_wav(path, observed, sample_rate)
        return {
            "backend": "injected_capture",
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "completed": True,
        }

    report = run_workspace_binding_acquisition(
        out_dir=tmp_path,
        self_playback=playback,
        world_playback=playback,
        capture_fn=capture,
        config=config,
    )

    assert report["verdict"] == "injected_acquisition_complete"
    assert report["completed_trial_count"] == 15
    assert report["real_hardware"] is False
    assert report["live_playback"] is False
    assert report["playback_callback_started"] is True
    assert report["promotion_allowed"] is False
    assert report["capture_acceptance_verdict"] == "capture_acceptance_passed"
    assert report["measured_total_playback_seconds"] == 19.5
    assert report["bounded_total_capture_seconds"] == 28.5
    assert len(playback_calls) == 18
    assert all(Path(artifact["self_wav"]).exists() for artifact in report["artifacts"])
    assert all(Path(artifact["world_wav"]).exists() for artifact in report["artifacts"])
    assert all(Path(artifact["captured_wav"]).exists() for artifact in report["artifacts"])
    assert all(artifact["io_timing"]["stream_position_verified"] for artifact in report["artifacts"])


def test_workspace_acquisition_rejects_unfrozen_hardware_and_oversized_run(tmp_path) -> None:
    callback = lambda _path, **_kwargs: {"started": True}

    try:
        run_workspace_binding_acquisition(
            out_dir=tmp_path,
            self_playback=callback,
            world_playback=callback,
            capture_fn=callback,
            real_hardware_claim=True,
        )
    except ValueError as exc:
        assert "preregistration" in str(exc)
    else:
        raise AssertionError("unfrozen hardware execution must be rejected")

    try:
        run_workspace_binding_acquisition(
            out_dir=tmp_path,
            self_playback=callback,
            world_playback=callback,
            capture_fn=callback,
            config=WorkspaceBindingConfig(stimulus_ms=2000),
        )
    except ValueError as exc:
        assert "2 second bound" in str(exc)
    else:
        raise AssertionError("oversized workspace playback must be rejected")


def test_workspace_acquisition_aborts_on_source_start_failure(tmp_path) -> None:
    playback_seen = Event()

    def playback(_path):
        playback_seen.set()
        return {"backend": "injected_playback", "started": False}

    def capture(path, *, sample_rate, ready_event, capture_timeline, **_kwargs):
        capture_timeline["frame_position_fn"] = lambda: sample_rate // 10
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        assert playback_seen.wait(timeout=1.0)
        write_wav(path, np.zeros(sample_rate // 2), sample_rate)
        return {"backend": "injected_capture", "completed": True}

    report = run_workspace_binding_acquisition(
        out_dir=tmp_path,
        self_playback=playback,
        world_playback=playback,
        capture_fn=capture,
    )

    assert report["verdict"] == "acquisition_failed"
    assert report["failed_trial_id"] == "development_self_only_0"
    assert report["completed_trial_count"] == 0
    assert report["live_playback"] is False
    assert report["playback_callback_started"] is True
    assert "did not start" in report["error"]
