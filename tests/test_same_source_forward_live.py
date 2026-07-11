"""Tests for the fresh no-replay H3 same-source live gate."""

import json
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from cora_infans.io.motor_plan import write_wav
from cora_infans.io.wav_frames import read_pcm_wav
from cora_infans.same_source_forward_live import (
    FROZEN_THRESHOLD,
    audit_fresh_live_preregistration,
    build_fresh_protocol,
    run_fresh_live_acquisition,
    score_fresh_evidence,
)


SPENT_ROOT = Path("/private/tmp/cora-ids-spent-200")


def test_fresh_protocol_has_no_new_utterance_calibration() -> None:
    protocol = build_fresh_protocol()

    assert protocol["trial_count"] == 9
    assert protocol["self_calibration_trial_count"] == 0
    assert protocol["planned_total_playback_seconds"] == pytest.approx(8.28)
    assert protocol["shared_model_scalars"] == 13
    assert protocol["per_new_utterance_sensory_templates"] == 0
    assert protocol["frozen_threshold"] == FROZEN_THRESHOLD


def test_fresh_scoring_preserves_workspace_ablation() -> None:
    evidence = {}
    for index in range(3):
        evidence[f"fresh_self_probe_{index}"] = FROZEN_THRESHOLD - 0.01
        evidence[f"fresh_world_only_{index}"] = 2.0
        evidence[f"fresh_overlap_{index}"] = FROZEN_THRESHOLD + 0.01

    report = score_fresh_evidence(evidence)

    selected = report["reports"]["3"]
    scheduled = selected["scheduled_identity_plus_same_source_forward_model"]
    assert report["classification_gates_passed"] is True
    assert scheduled["metrics"]["f1"] == 1.0
    assert selected["flat"]["metrics"]["f1"] == 0.8
    assert selected["fixed_self_window"]["metrics"]["world_recall"] == 0.5
    assert report["reports"]["1"][
        "scheduled_identity_plus_same_source_forward_model"
    ]["metrics"]["f1"] < 1.0


def test_live_preregistration_fails_closed_on_hash_mismatch() -> None:
    preregistration = {
        "mode": "h3_same_source_forward_model_real_preregistration",
        "status": "preregistered_before_live_execution",
        "bounded_playback_capture_authorized": True,
        "frozen_hashes": {"implementation_sha256": "wrong"},
        "unresolved_fields": [],
    }

    audit = audit_fresh_live_preregistration(
        preregistration, current_hashes={"implementation_sha256": "actual"}
    )

    assert audit["ready"] is False
    assert "frozen hash mismatch: implementation_sha256" in audit["blockers"]


@pytest.mark.skipif(not SPENT_ROOT.exists(), reason="spent source corpus unavailable")
def test_injected_fresh_acquisition_is_bounded_without_promotion(tmp_path) -> None:
    playback_seen = Event()

    def playback(_path):
        playback_seen.set()
        return {"backend": "injected_playback", "started": True}

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
        time.sleep(0.003)
        trial_id = Path(path).name.removesuffix("_capture.wav")
        self_pcm, _ = read_pcm_wav(tmp_path / f"{trial_id}_self.wav")
        world_pcm, _ = read_pcm_wav(tmp_path / f"{trial_id}_world.wav")
        impulse = np.zeros(128)
        impulse[0], impulse[43] = 0.35, 0.05
        observed = np.convolve(np.mean(self_pcm, axis=1), impulse)
        observed += np.convolve(np.mean(world_pcm, axis=1), impulse)
        retained = np.concatenate(
            (np.zeros(sample_rate // 10), observed, np.zeros(sample_rate // 10))
        )
        write_wav(path, retained, sample_rate)
        return {
            "backend": "injected_capture",
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "completed": True,
        }

    report = run_fresh_live_acquisition(
        root=SPENT_ROOT,
        out_dir=tmp_path,
        self_playback=playback,
        world_playback=playback,
        capture_fn=capture,
    )

    assert report["verdict"] == "injected_acquisition_complete"
    assert report["completed_trial_count"] == 9
    assert report["promotion_allowed"] is False


def test_live_design_preregistration_matches_frozen_constants() -> None:
    design = json.loads(
        Path(
            "docs/evidence/2026-07-11-h3-same-source-forward-model-live-design-preregistration.json"
        ).read_text()
    )

    assert design["frozen_model"]["selected_world_evidence_threshold"] == FROZEN_THRESHOLD
    assert design["protocol"]["self_calibration_trials"] == 0
