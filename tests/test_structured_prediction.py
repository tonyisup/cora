from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from cora_infans.cli import structured_prediction_cli
from cora_infans.structured_prediction import (
    DEVELOPMENT_SEEDS,
    HELD_OUT_SEEDS,
    LocalAcousticCodebook,
    build_structured_protocol_manifest,
    run_structured_prediction_protocol,
)
from cora_infans.io.motor_plan import write_wav
from cora_infans.io.wav_frames import read_pcm_wav


def test_structured_manifest_has_disjoint_fixed_trials() -> None:
    manifest = build_structured_protocol_manifest()

    assert set(DEVELOPMENT_SEEDS).isdisjoint(HELD_OUT_SEEDS)
    assert manifest["trial_count"] == 12
    assert manifest["maximum_total_playback_seconds"] < 90.0
    assert manifest["real_io_requires_explicit_authorization"] is True
    for trial in manifest["trials"]:
        sequence = trial["grammar_sequence"]
        assert len(sequence) == 20
        assert all(
            current != following
            for current, following in zip(sequence, sequence[1:])
        )


def test_frozen_protocol_and_implementation_hashes_match_source() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root / "docs/evidence/2026-07-10-structured-prediction-protocol.json"
    )
    fake_result_path = (
        root / "docs/evidence/2026-07-10-structured-prediction-fake-result.json"
    )
    preregistration_path = root / (
        "docs/evidence/"
        "2026-07-10-structured-prediction-real-preregistration-draft.json"
    )
    retry_preregistration_path = root / (
        "docs/evidence/"
        "2026-07-10-structured-prediction-real-attempt-2-preregistration.json"
    )
    implementation_path = root / "src/cora_infans/structured_prediction.py"
    io_runner_path = root / "src/cora_infans/live_speak_smoke.py"
    stored = json.loads(protocol_path.read_text())
    generated = json.loads(json.dumps(build_structured_protocol_manifest()))

    for key, value in generated.items():
        assert stored[key] == value
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    implementation_hash = hashlib.sha256(implementation_path.read_bytes()).hexdigest()
    fake_result = json.loads(fake_result_path.read_text())
    preregistration = json.loads(preregistration_path.read_text())
    retry_preregistration = json.loads(retry_preregistration_path.read_text())
    assert fake_result["protocol_sha256"] == protocol_hash
    assert preregistration["protocol_sha256"] == protocol_hash
    historical_implementation_hash = (
        "51307274e9582885470b2ed81950da137a93cebfc7e5add5aa9ec42c1707093c"
    )
    # Historical artifacts remain immutable after the pre-playback I/O fix.
    assert fake_result["implementation_sha256"] == historical_implementation_hash
    assert preregistration["implementation_sha256"] == historical_implementation_hash
    assert retry_preregistration["protocol_sha256"] == protocol_hash
    assert retry_preregistration["predictor_implementation_sha256"] == implementation_hash
    assert retry_preregistration["io_runner_sha256"] == hashlib.sha256(
        io_runner_path.read_bytes()
    ).hexdigest()


def test_retained_real_result_has_complete_hashed_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / (
        "docs/evidence/"
        "2026-07-10-structured-prediction-real-attempt-2-result.json"
    )
    artifact_root = root / (
        "docs/evidence/"
        "2026-07-10-structured-prediction-real-attempt-2-artifacts"
    )
    result = json.loads(result_path.read_text())

    assert result["verdict"] == "real_io_acceptance_failed"
    assert result["all_acceptance_gates_passed"] is False
    assert len(result["artifacts"]) == 12
    assert all(result["acceptance"]["tones"].values())
    assert not all(result["acceptance"]["vowels"].values())
    for artifact in result["artifacts"]:
        assert artifact["playback"]["backend"] == "ffmpeg_audiotoolbox"
        assert artifact["capture"]["backend"] == "ffmpeg_avfoundation"
        assert artifact["io_timing"]["stream_position_verified"] is True
        for kind in ("efference", "captured"):
            retained = artifact_root / Path(artifact[f"{kind}_wav"]).name
            assert retained.exists()
            assert hashlib.sha256(retained.read_bytes()).hexdigest() == artifact[
                f"{kind}_sha256"
            ]


def test_real_runner_reports_whether_playback_actually_started(tmp_path) -> None:
    playback_calls = 0

    def playback(path: str) -> dict[str, object]:
        nonlocal playback_calls
        playback_calls += 1
        return {"backend": "injected_playback"}

    def capture(path: str, **kwargs) -> dict[str, object]:
        raise RuntimeError("capture unavailable")

    report = run_structured_prediction_protocol(
        out_dir=tmp_path,
        real_io=True,
        bounded_playback_authorized=True,
        playback_fn=playback,
        capture_fn=capture,
        geometry={"test_double": True},
        real_hardware_claim=False,
    )

    assert report["verdict"] == "real_io_failed"
    assert report["live_playback"] is False
    assert playback_calls == 0


def test_structured_protocol_default_and_real_gate_touch_no_io(tmp_path) -> None:
    planned = run_structured_prediction_protocol(out_dir=tmp_path)
    blocked = run_structured_prediction_protocol(
        out_dir=tmp_path,
        real_io=True,
        bounded_playback_authorized=False,
    )

    assert planned["verdict"] == "planned_no_io"
    assert blocked["verdict"] == "blocked_bounded_live_io_not_authorized"
    assert list(tmp_path.iterdir()) == []
    assert planned["live_playback"] is False
    assert blocked["live_playback"] is False


def test_cli_requires_geometry_before_authorized_real_io(tmp_path) -> None:
    blocked = structured_prediction_cli(
        [
            "--evaluate-structured-prediction",
            "--out-dir",
            str(tmp_path),
            "--real-io",
        ]
    )
    assert blocked["verdict"] == "blocked_bounded_live_io_not_authorized"

    with pytest.raises(ValueError, match="speaker-distance-cm"):
        structured_prediction_cli(
            [
                "--evaluate-structured-prediction",
                "--out-dir",
                str(tmp_path),
                "--real-io",
                "--authorize-bounded-playback",
            ]
        )

    assert list(tmp_path.iterdir()) == []


def test_fake_structured_protocol_exercises_measured_transition_path(tmp_path) -> None:
    report = run_structured_prediction_protocol(out_dir=tmp_path, fake_io=True)

    assert report["verdict"] == "fake_io_plumbing_passed"
    assert report["evidence_class"] == "synthetic_fake_io"
    assert report["promotion_allowed"] is False
    assert report["real_hardware"] is False
    assert len(report["artifacts"]) == 12
    assert len(list(tmp_path.glob("*.wav"))) == 24
    for artifact in report["artifacts"]:
        assert Path(artifact["efference_wav"]).exists()
        assert Path(artifact["captured_wav"]).exists()
        assert len(artifact["captured_sha256"]) == 64
        assert artifact["capture_provenance"] == "synthetic_fake_acoustic_channel"
    for family in report["family_reports"].values():
        assert family["held_out_codebook_occupancy"] == 4
        assert family["post_hoc_token_cluster_purity"] == 1.0
        assert family["accuracy_improvement"] >= 0.2
        assert family["top_down_prediction_consumed"] is True
        assert family["bounded_state_scalars"] == 68


def test_local_codebook_updates_only_nearest_bounded_centroid() -> None:
    initial = np.eye(4, dtype=np.float64)
    codebook = LocalAcousticCodebook(initial)
    before = codebook.centroids.copy()

    state = codebook.observe(np.asarray([0.9, 0.1, 0.0, 0.0]))

    assert state == 0
    assert codebook.state_scalars == 20
    assert not np.array_equal(codebook.centroids[0], before[0])
    assert np.array_equal(codebook.centroids[1:], before[1:])


def test_injected_real_runner_uses_synchronized_capture_and_alignment(tmp_path) -> None:
    playback_ready = Event()
    shared: dict[str, object] = {}

    def playback(path: str) -> dict[str, object]:
        pcm, sample_rate = read_pcm_wav(path)
        shared["pcm"] = np.mean(pcm, axis=1)
        shared["sample_rate"] = sample_rate
        playback_ready.set()
        return {"backend": "injected_playback", "completed": True}

    def capture(
        path: str,
        *,
        sample_rate: int,
        duration_ms: int,
        ready_event,
        capture_timeline,
        **kwargs,
    ) -> dict[str, object]:
        playback_offset = int(0.1 * sample_rate)
        acoustic_delay = int(0.03 * sample_rate)
        capture_timeline["frame_position_fn"] = lambda: playback_offset
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        assert playback_ready.wait(timeout=2.0)
        efference = np.asarray(shared["pcm"], dtype=np.float32)
        playback_ready.clear()
        captured = np.zeros(
            playback_offset + acoustic_delay + len(efference) + sample_rate // 5,
            dtype=np.float32,
        )
        start = playback_offset + acoustic_delay
        captured[start : start + len(efference)] = efference * 0.5
        write_wav(path, captured, sample_rate)
        time.sleep(0.15)
        return {
            "backend": "injected_capture",
            "frames": len(captured),
            "completed": True,
        }

    report = run_structured_prediction_protocol(
        out_dir=tmp_path,
        real_io=True,
        bounded_playback_authorized=True,
        playback_fn=playback,
        capture_fn=capture,
        geometry={"test_double": True},
        real_hardware_claim=False,
    )

    assert report["error"] is None, report["error"]
    assert report["verdict"] == "injected_real_runner_test_complete"
    assert report["real_hardware"] is False
    assert all(
        abs(float(artifact["alignment_onset_ms"]) - 130.0) <= 1.0
        for artifact in report["artifacts"]
    ), json.dumps(report["artifacts"], indent=2)
    assert report["all_acceptance_gates_passed"] is True, json.dumps(
        report["family_reports"], indent=2
    )
    assert len(report["artifacts"]) == 12
    assert all(
        artifact["capture_provenance"] == "injected_io_test_double"
        for artifact in report["artifacts"]
    )
    assert all(
        artifact["io_timing"]["stream_position_verified"] is True
        for artifact in report["artifacts"]
    )
