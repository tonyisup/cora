import hashlib
import json
from pathlib import Path
from threading import Barrier, Event
import time

import numpy as np

from cora_infans.io.motor_plan import write_wav
from cora_infans.io.wav_frames import read_pcm_wav
from cora_infans.structured_workspace_binding import (
    WorkspaceBindingConfig,
    audit_workspace_binding_preregistration,
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
    assert report["criteria"]["scheduled_world_recall_at_least_0_90"] is True
    assert (
        report["criteria"]["scheduled_self_suppression_specificity_at_least_0_90"]
        is True
    )
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
        assert "bounded authorization" in str(exc)
    else:
        raise AssertionError("real hardware execution must require explicit authorization")

    try:
        run_workspace_binding_acquisition(
            out_dir=tmp_path,
            self_playback=callback,
            world_playback=callback,
            capture_fn=callback,
            real_hardware_claim=True,
            bounded_playback_authorized=True,
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


def _ready_preregistration(hashes):
    return {
        "mode": "structured_workspace_binding_real_preregistration",
        "status": "preregistered_before_live_execution",
        "bounded_playback_capture_authorized": True,
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
                "horizontal_offset_cm": 0,
                "elevation_degrees": -90,
                "vertical_distance_cm": 25,
                "distance_reference": "XVF3800 array center to Mac mini chassis center",
                "system_volume_percent": 100,
                "playback_gain_db": -18.0,
            },
            "world_source": {
                "output_device_index": 2,
                "output_device": "External Headphones",
                "angle_degrees": 180,
                "distance_cm": 50,
                "hardware_volume": "max",
                "playback_gain_db": 0.0,
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


def test_workspace_preregistration_audit_requires_hashes_and_complete_self_geometry() -> None:
    hashes = {"implementation_sha256": "impl", "cli_sha256": "cli"}
    preregistration = _ready_preregistration(hashes)

    ready = audit_workspace_binding_preregistration(
        preregistration, current_hashes=hashes
    )
    assert ready == {"verdict": "preregistration_ready", "ready": True, "blockers": []}

    preregistration["hardware"]["self_source"]["elevation_degrees"] = None
    preregistration["frozen_hashes"]["cli_sha256"] = "stale"
    preregistration["acceptance"]["scheduled_f1_min"] = 0.5
    blocked = audit_workspace_binding_preregistration(
        preregistration, current_hashes=hashes
    )
    assert blocked["ready"] is False
    assert "self source elevation must remain -90 degrees" in blocked["blockers"]
    assert "frozen hash mismatch: cli_sha256" in blocked["blockers"]
    assert "acceptance criteria do not match the frozen H3 gate" in blocked["blockers"]


def test_successful_workspace_preregistration_is_retained_with_its_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    preregistration_path = root / (
        "docs/evidence/2026-07-10-structured-workspace-real-attempt-3-preregistration.json"
    )
    preregistration = json.loads(preregistration_path.read_text())
    audit = json.loads(
        (
            root
            / "docs/evidence/2026-07-10-structured-workspace-real-attempt-3-audit.json"
        ).read_text()
    )

    assert hashlib.sha256(preregistration_path.read_bytes()).hexdigest() == audit[
        "preregistration_sha256"
    ]
    assert preregistration["hardware"]["self_source"]["vertical_distance_cm"] == 4
    assert preregistration["hardware"]["self_source"]["playback_gain_db"] == -18.0
    assert preregistration["acceptance"]["no_clipped_capture_samples_required"] is True


def test_retained_real_workspace_result_and_artifacts_pass_frozen_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root / "docs/evidence/2026-07-10-structured-workspace-real-result.json"
        ).read_text()
    )
    artifact_dir = (
        root / "docs/evidence/2026-07-10-structured-workspace-real-artifacts"
    )

    assert result["verdict"] == "real_io_acceptance_passed"
    assert result["capture_acceptance_verdict"] == "capture_acceptance_passed"
    assert result["real_hardware"] is True
    assert result["live_playback"] is True
    assert result["promotion_allowed"] is True
    assert result["completed_trial_count"] == 15
    assert result["error"] is None
    assert all(result["criteria"].values())
    selected = result["reports"]["3"]
    assert selected["scheduled_identity_plus_world_evidence"]["metrics"] == {
        "tp": 6,
        "fp": 0,
        "tn": 3,
        "fn": 0,
        "world_recall": 1.0,
        "self_suppression_specificity": 1.0,
        "f1": 1.0,
    }
    assert selected["flat"]["metrics"]["f1"] == 0.8
    assert selected["fixed_self_window"]["metrics"]["world_recall"] == 0.5
    assert len(result["artifacts"]) == 15
    assert len(list(artifact_dir.glob("*.wav"))) == 45

    for artifact in result["artifacts"]:
        paths = {
            "self": artifact_dir / Path(artifact["self_wav"]).name,
            "world": artifact_dir / Path(artifact["world_wav"]).name,
            "captured": artifact_dir / Path(artifact["captured_wav"]).name,
        }
        for name, path in paths.items():
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact[
                f"{name}_sha256"
            ]
        self_pcm, self_rate = read_pcm_wav(paths["self"])
        world_pcm, world_rate = read_pcm_wav(paths["world"])
        captured_pcm, captured_rate = read_pcm_wav(paths["captured"])
        assert self_rate == world_rate == captured_rate == 16000
        assert self_pcm.shape == world_pcm.shape == (20800, 1)
        assert captured_pcm.shape == (30400, 2)
        assert artifact["io_timing"]["stream_position_verified"] is True
        assert artifact["capture"]["backend"] == "ffmpeg_avfoundation"
        assert artifact["capture"]["device"] == ":0"
        sources = artifact["playback"]["sources"]
        expected_sources = (
            {"self", "world"}
            if artifact["condition"] == "overlap"
            else {"self"}
            if artifact["condition"] == "self_only"
            else {"world"}
        )
        assert set(sources) == expected_sources
        if "self" in sources:
            assert sources["self"]["device_index"] == 3
        if "world" in sources:
            assert sources["world"]["device_index"] == 2


def test_retained_attempt_3_passes_classification_quality_and_artifact_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-10-structured-workspace-real-attempt-3-result.json"
        ).read_text()
    )
    artifact_dir = root / (
        "docs/evidence/2026-07-10-structured-workspace-real-attempt-3-artifacts"
    )

    assert result["verdict"] == "real_io_acceptance_passed"
    assert result["promotion_allowed"] is True
    assert result["completed_trial_count"] == 15
    assert all(result["criteria"].values())
    assert result["capture_quality"] == {
        "captures_with_clipping": 0,
        "total_clipped_samples": 0,
        "maximum_clipped_sample_fraction": 0.0,
        "no_clipping_passed": True,
    }
    scheduled = result["reports"]["3"][
        "scheduled_identity_plus_world_evidence"
    ]["metrics"]
    assert scheduled["f1"] == 1.0
    assert scheduled["world_recall"] == 1.0
    assert scheduled["self_suppression_specificity"] == 1.0
    assert len(result["artifacts"]) == 15
    assert len(list(artifact_dir.glob("*.wav"))) == 45

    for artifact in result["artifacts"]:
        for name in ("self", "world", "captured"):
            path = artifact_dir / Path(artifact[f"{name}_wav"]).name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact[
                f"{name}_sha256"
            ]
        capture_path = artifact_dir / Path(artifact["captured_wav"]).name
        capture, rate = read_pcm_wav(capture_path)
        assert rate == 16000
        assert capture.shape == (30400, 2)
        assert artifact["capture_quality"]["clipped_sample_count"] == 0
        assert artifact["io_timing"]["stream_position_verified"] is True
        assert artifact["capture"]["backend"] == "ffmpeg_avfoundation"


def test_real_workspace_acquisition_rejects_unverified_backend_provenance(tmp_path) -> None:
    hashes = {"implementation_sha256": "impl"}
    playback_seen = Event()

    def playback(_path):
        playback_seen.set()
        return {"backend": "ffmpeg_audiotoolbox", "started": True}

    def capture(path, *, sample_rate, ready_event, capture_timeline, **_kwargs):
        capture_timeline["frame_position_fn"] = lambda: sample_rate // 10
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        assert playback_seen.wait(timeout=1.0)
        time.sleep(0.01)
        write_wav(path, np.zeros(sample_rate // 2), sample_rate)
        return {"backend": "unverified_capture", "completed": True}

    report = run_workspace_binding_acquisition(
        out_dir=tmp_path,
        self_playback=playback,
        world_playback=playback,
        capture_fn=capture,
        real_hardware_claim=True,
        bounded_playback_authorized=True,
        hardware_preregistration=_ready_preregistration(hashes),
        current_hashes=hashes,
    )

    assert report["verdict"] == "acquisition_failed"
    assert report["real_hardware"] is True
    assert report["live_playback"] is True
    assert report["completed_trial_count"] == 0
    assert "capture backend provenance was not verified" in report["error"]


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
